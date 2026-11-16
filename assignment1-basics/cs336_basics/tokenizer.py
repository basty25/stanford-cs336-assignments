import pickle
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

        # 2. Pre-kokenize each none-special piece

        # 3. Apply merges

        # 4. Map bytes to Ids
        pass


    def encode_iterable(self, interable: Iterable[str]) -> Iterator[int]:
        # think of yield
        # maybe line by line
        pass


    def decode(self, ids: list[int]) -> str:
        # check if not valid utf-8. see bytes.decode(argument for thsi)
        # check if id is not in vocab
        pass
