"""Byte-Pair Encoding (BPE) tokenizer training."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Chunk boundary helper (assignment starter code, reproduced for self-containment)
# ---------------------------------------------------------------------------

def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


# ---------------------------------------------------------------------------
# Pre-tokenization — runs in worker processes
# ---------------------------------------------------------------------------

def _pre_tokenize_chunk(args: tuple[bytes, list[str]]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize one text chunk, returning a pre-token frequency table.

    A pre-token is one regex match, stored as a tuple of single-byte objects
    (e.g. b'low' -> (b'l', b'o', b'w')).  Special tokens are stripped first
    so no merge ever crosses a document boundary.
    """
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(s) for s in sorted_specials)
        pieces = re.split(split_pattern, text)
    else:
        pieces = [text]

    freq: dict[tuple[bytes, ...], int] = defaultdict(int)
    for piece in pieces:
        for match in regex.finditer(GPT2_PAT, piece):
            word_bytes = match.group(0).encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in word_bytes)
            freq[token_tuple] += 1
    return dict(freq)


def _pre_tokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int = 8,
) -> dict[tuple[bytes, ...], int]:
    """Read corpus, chunk at special-token boundaries, pre-tokenize in parallel."""
    primary_special = (
        b"<|endoftext|>" if "<|endoftext|>" in special_tokens
        else special_tokens[0].encode("utf-8") if special_tokens
        else b"<|endoftext|>"
    )

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_workers, primary_special)
        print(f"Chunk boundaries: {boundaries}")
        chunks: list[bytes] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start))

    args = [(chunk, special_tokens) for chunk in chunks]
    with Pool(processes=min(num_workers, len(chunks))) as pool:
        results = pool.map(_pre_tokenize_chunk, args)

    combined: dict[tuple[bytes, ...], int] = defaultdict(int)
    for result in results:
        for token_tuple, count in result.items():
            combined[token_tuple] += count
    return dict(combined)


# ---------------------------------------------------------------------------
# Incremental pair-count data structures
# ---------------------------------------------------------------------------

def _build_pair_index(
    token_freqs: dict[tuple[bytes, ...], int],
) -> tuple[
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
]:
    """
    Build two indexes over all pre-tokens:

    pair_counts[(A, B)]
        Total weighted occurrences of the adjacent pair (A, B) across every
        pre-token, where each occurrence is weighted by that pre-token's corpus
        frequency.

    pair_to_tokens[(A, B)]
        Set of pre-token tuples that contain the pair (A, B) at least once.
        Used to find which pre-tokens to update after a merge without scanning
        the entire frequency table.
    """
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for token_tuple, freq in token_freqs.items():
        for pair in zip(token_tuple, token_tuple[1:]):
            pair_counts[pair] += freq
            pair_to_tokens[pair].add(token_tuple)

    return dict(pair_counts), dict(pair_to_tokens)


def _apply_merge(
    token_freqs: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair: tuple[bytes, bytes],
    merged: bytes,
) -> None:
    """
    Apply one BPE merge, updating all three data structures in-place.

    Only the pre-tokens containing *pair* are visited — O(k) per merge where k
    is the number of distinct affected pre-tokens, rather than O(N) for the
    whole table.

    Steps for each affected pre-token:
      1. Subtract its weighted pair counts from pair_counts.
      2. Remove it from pair_to_tokens for every pair it contained.
      3. Construct the merged pre-token.
      4. Add the new pre-token's weighted pair counts.
      5. Register the new pre-token in pair_to_tokens.
      6. Update token_freqs (delete old key, upsert new key).
    """
    a, b = pair
    affected = list(pair_to_tokens.get(pair, set()))

    for old_token in affected:
        freq = token_freqs.get(old_token)
        if freq is None:
            continue

        # -- Build new pre-token ------------------------------------------ #
        new_token_list: list[bytes] = []
        i = 0
        while i < len(old_token):
            if i < len(old_token) - 1 and old_token[i] == a and old_token[i + 1] == b:
                new_token_list.append(merged)
                i += 2
            else:
                new_token_list.append(old_token[i])
                i += 1
        new_token = tuple(new_token_list)

        # -- Remove old token's pair contributions ------------------------- #
        del token_freqs[old_token]
        for p in zip(old_token, old_token[1:]):
            pair_counts[p] -= freq
            if pair_counts[p] == 0:
                del pair_counts[p]
            if p in pair_to_tokens:
                pair_to_tokens[p].discard(old_token)
                if not pair_to_tokens[p]:
                    del pair_to_tokens[p]

        # -- Add new token's pair contributions ---------------------------- #
        # If new_token already exists (two old tokens collapsed into the same
        # new form), we must first back out its existing pair contributions
        # before combining, then re-add with the combined frequency.
        existing = token_freqs.get(new_token, 0)
        if existing:
            for p in zip(new_token, new_token[1:]):
                pair_counts[p] -= existing
                if pair_counts[p] == 0:
                    del pair_counts[p]
                if p in pair_to_tokens:
                    pair_to_tokens[p].discard(new_token)
                    if not pair_to_tokens[p]:
                        del pair_to_tokens[p]

        token_freqs[new_token] = existing + freq
        combined_freq = token_freqs[new_token]

        for p in zip(new_token, new_token[1:]):
            pair_counts[p] = pair_counts.get(p, 0) + combined_freq
            if p not in pair_to_tokens:
                pair_to_tokens[p] = set()
            pair_to_tokens[p].add(new_token)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Algorithm
    ---------
    1. Initialise vocabulary with the 256 single-byte tokens, then append
       special tokens (each gets the next sequential ID).
    2. Pre-tokenise the corpus using the GPT-2 regex, building a frequency
       table of pre-tokens.  Each pre-token is a tuple of single-byte objects
       so the merge loop can operate directly on it.
    3. Build pair_counts and a pair->pre-tokens reverse index once.
    4. Repeat until ``vocab_size`` is reached:
       a. Pick the most frequent pair; break ties by choosing the
          lexicographically *greater* pair (matching the assignment spec).
       b. Apply the merge incrementally — only affected pre-tokens are updated.
       c. Record the merge and add the merged bytes to the vocabulary.

    Parameters
    ----------
    input_path  : path to the raw text corpus.
    vocab_size  : target total vocabulary size.
    special_tokens : strings that act as hard document boundaries and are never
                     split during training.
    num_workers : parallel worker processes for pre-tokenisation.

    Returns
    -------
    vocab  : dict[int, bytes]  — token ID to its byte representation.
    merges : list[tuple[bytes, bytes]]  — ordered BPE merge operations.
    """
    # 1. Vocabulary initialisation ---------------------------------------- #
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # 2. Pre-tokenise corpus ---------------------------------------------- #
    token_freqs = _pre_tokenize_file(input_path, special_tokens, num_workers)

    # 3. Build pair index once -------------------------------------------- #
    pair_counts, pair_to_tokens = _build_pair_index(token_freqs)

    # 4. Merge loop -------------------------------------------------------- #
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Tie-break: lexicographically greater pair wins.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged_bytes = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        vocab[next_id] = merged_bytes
        next_id += 1

        _apply_merge(token_freqs, pair_counts, pair_to_tokens, best_pair, merged_bytes)

    return vocab, merges
