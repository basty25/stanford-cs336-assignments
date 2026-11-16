import time
from datetime import timedelta
from cs336_basics.train_bpe import train_bpe



path_smoke = "data/test_corpus.txt"
path_valid = "data/TinyStoriesV2-GPT4-valid.txt"
path_train = "data/TinyStoriesV2-GPT4-train.txt"

def main():
    print("start")
    start = time.perf_counter()
    vocab, merges = train_bpe(
        # input_path=path_smoke,
        input_path=path_valid,
        # input_path=path_train,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"]
    )

    elapsed = time.perf_counter() - start
    print("end")
    print("time", str(timedelta(seconds=round(elapsed))))
    print("first 10 merges", merges[:10])
    _, longest_word = max(vocab.items(), key=lambda item: len(item[1]))
    print(longest_word)

if __name__ == "__main__":
    main()
