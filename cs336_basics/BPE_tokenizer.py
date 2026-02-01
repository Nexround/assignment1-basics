from collections.abc import Iterator, Iterable
import json
import base64
from functools import reduce
import operator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens

        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab: dict[int, bytes]
        self.reverse_vocab: dict[bytes, int]
        self.merges: list[tuple[bytes, bytes]]
        self.vocab, self.merges = vocab, merges
        self.special_tokens: list[str] | None = special_tokens
        self.rank: dict[tuple[bytes, bytes], int] = {}
        for idx, merge in enumerate(self.merges):
            self.rank[merge] = idx

        # 支持用户自定义特殊标记
        if self.special_tokens is not None:
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                # 若词典中尚未收录，则将其追加至词典
                if special_token_bytes not in self.vocab.values():
                    self.vocab[next_id] = special_token_bytes
                    next_id += 1
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
        self.reverse_vocab = {value: key for key, value in self.vocab.items()}
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Classmethod that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:

        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab, merges = cls._load_bpe_data(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # for idx, part in enumerate():

        if self.special_tokens is not None:
            segments = re.split(self.special_pattern, text)
        else:
            segments = [text]

        # 首先预分词，然后逐个处理预分词结果
        encoded_ids = []
        for segment_idx, segment in enumerate(segments):
            if not segment:
                continue
            if segment_idx % 2 == 0:
                # 处理普通文本片段
                pretokenized_matches = list(re.finditer(PAT, segment))
                for match in pretokenized_matches:
                    # 将匹配的文本转换为字节序列
                    byte_sequence = list(bytes([b]) for b in match.group().encode())
                    while True:
                        # 查找所有可以合并的字节对
                        indexed_pairs = enumerate(zip(byte_sequence, byte_sequence[1:]))
                        mergeable_pairs = [(pair_idx, byte_pair) for pair_idx, byte_pair in indexed_pairs if byte_pair in self.rank]
                        if mergeable_pairs == []:
                            break
                        # ranked_pairs = sorted(mergeable_pairs, key=lambda x: self.rank[x[1]])
                        # pair_to_merge = ranked_pairs[0]
                        # 选择rank最小的字节对进行合并
                        pair_to_merge = min(mergeable_pairs, key=lambda x: self.rank[x[1]])
                        pair_position = pair_to_merge[0]
                        merged_bytes = reduce(operator.add, pair_to_merge[1])
                        byte_sequence[pair_position : pair_position + 2] = [merged_bytes]

                    # 将字节序列转换为token IDs
                    segment_token_ids = [self.reverse_vocab[token_bytes] for token_bytes in byte_sequence]
                    encoded_ids.extend(segment_token_ids)
            if segment_idx % 2 != 0:
                # 处理特殊token
                special_token_id = self.reverse_vocab[segment.encode()]
                encoded_ids.append(special_token_id)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory
        """
        ids_generator = (self.encode(string) for string in iterable)
        g = (id for group in ids_generator for id in group)
        return g

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        if ids == []:
            return ''
        decoded_text = [self.vocab[id] for id in ids]
        text = reduce(operator.add, decoded_text)
        return text.decode("utf-8", errors="replace")

    @classmethod
    def _load_bpe_data(cls, vocab_path: str, merges_path: str):
        # 1. 加载 Merges
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        # 还原 Merges: str(base64) -> bytes
        merges = [(base64.b64decode(p[0]), base64.b64decode(p[1])) for p in merges_data]

        # 2. 加载 Vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # 还原 Vocab:
        #   Key: str -> int
        #   Value: str(base64) -> bytes
        vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

        return vocab, merges


if __name__ == "__main__":
    # 测试 tokenizer
    print("Loading tokenizer from vocab.json and merges.json...")
    tok = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])

    print(f"\nVocab size: {len(tok.vocab)}")
    print(f"Number of merges: {len(tok.merges)}")

    # 测试文本
    test_texts = [
        "Hello, world!",
        "Once upon a time, there was a little girl named Lily.",
        "The cat sat on the mat.",
        "<|endoftext|>",
    ]

    print("\n" + "=" * 60)
    print("Testing encode():")
    print("=" * 60)
    for text in test_texts:
        print(f"\nText: {text!r}")
        try:
            result = tok.encode(text)
            print(f"Encoded: {result}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Testing decode():")
    print("=" * 60)
    # 测试一些简单的 token IDs
    test_ids = [
        [257, 263],  # "he" + " the"
        [256],  # "<|endoftext|>"
    ]
    for ids in test_ids:
        print(f"\nIDs: {ids}")
        try:
            result = tok.decode(ids)
            print(f"Decoded: {result!r}")
        except Exception as e:
            print(f"Error: {e}")
