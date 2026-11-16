import os
import multiprocessing
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from dataclasses import dataclass

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class Node:

    def __init__(self, value: bytes, ID: int):
        self.value: bytes = value
        self.previous: Node | None = None
        self.next: Node | None = None
        self.id: int = ID

def sequence_to_nodes(sequence: tuple[bytes, ...], ID: int) -> Node | None:
    head = None
    previous = None
    for value in sequence:
        node = Node(value, ID)
        if head is None:
            head = node
        else:
            assert previous is not None
            previous.next = node
            node.previous = previous
        previous = node
    return head

@dataclass
class Row:
    ID: int
    head: Node | None
    frequency: int

class BPEMergeState:

    def __init__(self, counter: Counter):
        self.data: list[Row] = []
        self.pair_counts = Counter()
        self.pair_locations: dict[tuple[bytes, bytes], list[Node]] = defaultdict(list)

        for i, (sequence, count) in enumerate(counter.items()):
            head = sequence_to_nodes(sequence, i)
            self.data.append(Row(i, head, count))

            node = head
            if node is not None:
                while node.next:
                    pair = (node.value, node.next.value)
                    self.pair_counts[pair] += count
                    self.pair_locations[pair].append(node)
                    node = node.next

    def merge(self):
        pair = self.get_pair()
        if pair is None:
            return None
        a, b = pair
        merged_value = a + b
        locations = self.pair_locations[pair].copy()
        for left in locations:

            right = left.next

            # sanity check
            if right is None:
                continue
            if not (left.value == a and right.value == b):
                continue

            before = left.previous
            after = right.next

            row = self.data[left.id]
            frequency = row.frequency

            if before is not None:
                old_left = (before.value, left.value)
                self.remove_occurence(old_left, before, frequency)

            current = (left.value, right.value)
            self.remove_occurence(current, left, frequency)

            if after is not None:
                old_right = (right.value, after.value)
                self.remove_occurence(old_right, right, frequency)

            # left.value = left.value + right.value
            left.value = merged_value
            left.next = after
            if after is not None:
                after.previous = left
            right.previous = None
            right.next = None

            if before is not None:
                new_left = (before.value, left.value)
                self.add_occurence(new_left, before, frequency)
            if after is not None:
                new_right = (left.value, after.value)
                self.add_occurence(new_right, left, frequency)

            # if row.head is left:
            #     new_row = Row(row.ID, left, frequency)
            #     self.data[left.id] = new_row

        return pair

    def add_occurence(self, pair, left_node, frequency):
        self.pair_counts[pair] += frequency
        self.pair_locations[pair].append(left_node)

    def remove_occurence(self, pair, left_node, frequency):
        self.pair_counts[pair] -= frequency
        self.pair_locations[pair].remove(left_node)

        if not self.pair_locations[pair]:
            self.pair_locations.pop(pair)
            self.pair_counts.pop(pair)

    def get_pair(self) -> tuple[bytes, bytes] | None:
        # ? implement a heap structure for faster computation 
        if not self.pair_counts:
            return None
        pair, _ = max(self.pair_counts.items(), key=lambda x:(x[1], x[0]))
        return pair


def get_counts(text_parts: list[str], PAT: str) -> Counter:
    # ? do this with multiprocessing for faster computation
    counts = Counter()
    for part in text_parts:

        if part == "":
            continue

        vocab_part = re.finditer(PAT, part)
        token_list = []
        for token in vocab_part:
            token_encode = token.group().encode("utf-8")
            token_tuple = tuple(bytes([x]) for x in token_encode)

            token_list.append(token_tuple)
        
        counts.update(token_list)

    return counts

def worker(chunk):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return get_counts(chunk, pattern)

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int, 
        special_tokens: list[str]
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # init vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # old that works, but now with multiprocessing
    # # get input
    # with open(input_path, "r", encoding="utf-8") as f:
    #     input_text = f.read()

    # # split at special tokens
    # special_tokens = sorted(special_tokens, key=len, reverse=True)
    # if len(special_tokens) > 0:
    #     splits = "|".join([re.escape(special) for special in special_tokens])
    #     input_splitted = re.split(splits, input_text)
    # else:
    #     input_splitted = [input_text]

    # # pretokenize and get counts
    # pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # counts = get_counts(input_splitted, pattern)

    # pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # create special token split pattern
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    if len(special_tokens) > 0:
        splits = "|".join([re.escape(special) for special in special_tokens])
    else:
        splits = None

    with open(input_path, "rb") as f:
        num_processes = 4

        if special_tokens:
            split_special_token = special_tokens[0].encode("utf-8")
        else:
            split_special_token = b"<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        # create chunks for multiprocessing and split each junk for get_counts()
        counts = Counter()
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if splits is not None:
                splitted_text = re.split(splits, chunk)
            else:
                splitted_text = [chunk]
            chunks.append(splitted_text)
            # Run pre-tokenization on your chunk and store the counts for each pre-token

        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_count = pool.map(worker, chunks)

        for count in chunk_count:
            counts.update(count)



    # merges
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []
    bpe_state = BPEMergeState(counts)
    for i in range(num_merges):
        merged_pair = bpe_state.merge()
        if merged_pair is None:
            break

        merges.append(merged_pair)
        vocab[256 + len(special_tokens) + i] = merged_pair[0] + merged_pair[1]

    # TODO write save to pickle if necessary
    return vocab, merges
