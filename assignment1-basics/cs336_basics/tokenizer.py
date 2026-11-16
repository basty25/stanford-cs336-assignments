import pickle
import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
            ):
        self.vocab = vocab.copy()
        # add special tokens to vocab
        if special_tokens:
            values = set(self.vocab.values())
            for special_token in sorted(special_tokens, key=len, reverse=True):
                token = special_token.encode("utf-8")
                if token in values:
                    continue
                else:
                    # gets bigger ids, because vocab gets larger
                    self.vocab[len(self.vocab)] = token
                    values.add(token)

        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {sequence: ids for ids, sequence in self.vocab.items()}
        self.merge_ranks = {pair: index for index, pair in enumerate(self.merges)}


    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
            ):

        with open(vocab_filepath, "rb") as f:
            loaded_vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f: 
            loaded_merges = pickle.load(f)

        return cls(loaded_vocab, loaded_merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        # 1. split on special tokens

        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # get splits for the special tokens
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            splits = "|".join([re.escape(special) for special in special_tokens])
        else:
            splits = None

        # text splitted on special tokens
        if splits is not None:
            splitted = re.split(f"({splits})", text)
            # splitted = re.split(splits, text)
        else:
            splitted = [text]

        # 2. Pre-kokenize each none-special piece
        # pretokenize with pattern as in train_bpe
        pre_token_list = []
        for part in splitted:
            if part == "":
                continue
            # handle special tokens
            if self.special_tokens is not None and part in self.special_tokens:
                pre_token_list.append(part.encode("utf-8"))
                continue

            # handle other parts
            scanner = re.finditer(pattern, part)
            for matching in scanner:
                encoded = matching.group().encode("utf-8")
                pre_token_list.append(encoded)

        # 3. Apply merges
        after_merges = []
        for pre_token in pre_token_list:
            # check special token

            if self.special_tokens is not None and pre_token.decode("utf-8") in self.special_tokens:
                after_merges.append(pre_token)
                continue

            as_list = list(bytes([x]) for x in pre_token)

            # find merge
            for merge in self.merges:
                if len(as_list) == 1:
                    break
                indices = find_merge_indices(as_list, merge) 
                if indices is not None:
                    as_list = apply_merge(as_list, indices)
            after_merges.extend(as_list)

        # 4. Map bytes to Ids
        ids = [self.reverse_vocab[value] for value in after_merges]
        return ids
        


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # think of yield
        # maybe line by line
        for step in iterable:
            ids = self.encode(step)
            yield from ids


    def decode(self, ids: list[int]) -> str:
        # check if not valid utf-8. see bytes.decode(argument for thsi)
        # check if id is not in vocab
        word = b''
        for Id in ids: 
            word += self.vocab[Id] 
        return word.decode("utf-8", errors='replace')


def find_merge_indices(token_list: list[bytes], merge: tuple[bytes, bytes]) -> list[int] | None:
    # indices = [i for i in range(len(token_list)-2+1) if tuple(token_list[i:i+2]) == merge]
    indices = []
    i = 0
    while i <= len(token_list) - 2:
        if tuple(token_list[i:i+2]) == merge:
            indices.append(i)
            i += 2
        else:
            i += 1
    if indices:
        return indices
    else:
        return None

def apply_merge(token_list: list[bytes], indices: list[int]) -> list[bytes]:
    new_token = []
    for index in indices:
        token_list[index] = token_list[index] + token_list[index+1]
        remove_indices = [i + 1 for i in indices]
        new_token = [val for i, val in enumerate(token_list) if i not in set(remove_indices)]
    return new_token
