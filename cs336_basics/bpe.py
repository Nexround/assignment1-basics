from collections import defaultdict
from utils import run_pretokenization
from pprint import pprint


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 预分词,获取 token 序列及其出现次数
    token_counts = run_pretokenization(input_path, special_tokens=special_tokens)
    token_seq = {token: tuple(bytes([x]) for x in token) for token in token_counts}
    # 统计所有相邻 字节对 的全局频率
    pair_frequencies: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # 记录每个字节对出现在哪些**原始 token**中
    pair_to_original_tokens: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    next_vocab_id = len(vocab)
    for special_token in special_tokens:
        vocab[next_vocab_id] = bytes(special_token, encoding="utf-8")
        next_vocab_id += 1

    num_merges = vocab_size - next_vocab_id

    # 初始化:统计所有相邻字节对
    for token, count in token_counts.items():
        byte_tuple = tuple(bytes([b]) for b in token)

        for left_bytes, right_bytes in zip(byte_tuple, byte_tuple[1:]):
            pair_frequencies[(left_bytes, right_bytes)] += count
            pair_to_original_tokens[(left_bytes, right_bytes)].add(token)

    for merge_step in range(num_merges):
        print(f"Merge step: {merge_step}")

        # 找到频率最高的字节对
        most_frequent_pair, _ = max(pair_frequencies.items(), key=lambda x: (x[1], x[0]))
        del pair_frequencies[most_frequent_pair]

        left_token, right_token = most_frequent_pair
        merged_token = left_token + right_token
        new_token_id = next_vocab_id + merge_step

        merges.append(most_frequent_pair)
        vocab[new_token_id] = merged_token
        # 更新所有包含该字节对的 token 序列
        for token in pair_to_original_tokens[most_frequent_pair].copy():
            occurrence_count = token_counts[token]  # 目标token的出现次数
            current_token_seq = token_seq[token]

            # 检查most_frequent_pair是否真的存在于current_token_seq中
            pair_exists = False
            # 减法阶段：先遍历旧序列，减去所有相关的 Pair（排除掉当前正在合并的那个）。
            for i in range(len(current_token_seq) - 1):
                current_pair = (current_token_seq[i], current_token_seq[i + 1])
                if current_pair != most_frequent_pair:
                    pair_frequencies[current_pair] -= occurrence_count
                    pair_to_original_tokens[current_pair].discard(token)
                else:
                    pair_exists = True

            if not pair_exists:
                pair_to_original_tokens[most_frequent_pair].discard(token)
                continue
            # 构建new_token_seq
            new_token_seq = []
            idx = 0
            # 重构阶段：专注于构建正确的 new_seq，处理好尾部元素。
            while idx < len(current_token_seq):
                left_bytes = current_token_seq[idx]
                right_bytes = current_token_seq[idx + 1] if idx + 1 < len(current_token_seq) else None

                # 找到要合并的字节对位置
                if (left_bytes, right_bytes) == most_frequent_pair:
                    new_token_seq.append(merged_token)
                    idx += 2
                else:
                    new_token_seq.append(
                        left_bytes
                    )  # 在 else 分支中，只应该追加 left_bytes（即 current_token_seq[idx]），因为 right_bytes 会在下一次循环作为 left 被处理（或者作为合并对的一部分）。
                    idx += 1  # 更新频率计数

            # 加法阶段：遍历构建好的 new_seq，加上所有新 Pair。
            # 若产生合并，则添加新的字节对计数
            for i in range(len(new_token_seq) - 1):
                current_pair = (new_token_seq[i], new_token_seq[i + 1])
                pair_frequencies[current_pair] += occurrence_count
                pair_to_original_tokens[current_pair].add(token)

            token_seq[token] = tuple(new_token_seq)  # 更新当前token的编码方式

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStories-train.txt", 10000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("data/TinyStories-valid.txt", 10000, ["<|endoftext|>"])
    pprint(vocab)
