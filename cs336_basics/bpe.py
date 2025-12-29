from collections import defaultdict
from dataclasses import dataclass
from utils import run_pretokenization
from pprint import pprint

count: dict[tuple[bytes], int] = {}


@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 预分词,获取 token 序列及其出现次数
    token_counts = run_pretokenization(input_path, special_tokens)
    
    # 统计所有相邻 字节对 的全局频率
    pair_frequencies: dict[tuple[bytes, bytes], int] = defaultdict(int)
    
    # 记录每个字节对出现在哪些**原始 token**中
    pair_to_original_tokens: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)
    
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    next_vocab_id = len(vocab)
    for special_token in special_tokens:
        vocab[next_vocab_id] = bytes(special_token, encoding="utf-8")
        next_vocab_id += 1
    
    num_merges = vocab_size - next_vocab_id

    # 初始化:统计所有相邻字节对
    for token_seq, count in token_counts.items():
        for left_token, right_token in zip(token_seq, token_seq[1:]):
            pair_frequencies[(left_token, right_token)] += count
            pair_to_original_tokens[(left_token, right_token)].add(token_seq)

    for merge_step in range(num_merges):
        print(f"Merge step: {merge_step}")
        
        # 找到频率最高的字节对
        most_frequent_pair, _ = max(pair_frequencies.items(), key=lambda x: x[1])
        del pair_frequencies[most_frequent_pair]

        left_token, right_token = most_frequent_pair
        merged_token = left_token + right_token
        new_token_id = next_vocab_id + merge_step
        
        merges.append(most_frequent_pair)
        vocab[new_token_id] = merged_token
        
        # 更新所有包含该字节对的 token 序列
        for old_token_seq in pair_to_original_tokens[most_frequent_pair]:
            token_list = list(old_token_seq) # 正在处理的目标token
            occurrence_count = token_counts[old_token_seq] # 目标token的出现次数
            
            # 找到要合并的字节对位置
            token_bytes = b"".join(token_list)
            merge_position = token_bytes.find(merged_token)
            if merge_position == -1:
                raise ValueError("substring not found")
            
            # 记录合并前后的相邻对
            prev_pair = None if merge_position == 0 else token_list[merge_position - 1 : merge_position + 1]
            next_pair = None if merge_position + 2 == len(token_list) else token_list[merge_position + 1 : merge_position + 3]
            
            # 执行合并
            del token_list[merge_position : merge_position + len(merged_token)]
            token_list.insert(merge_position, merged_token) # 用合并后的字节对替换原来内容
            
            # 添加新产生的字节对
            for left_token, right_token in zip(token_list, token_list[1:]):
                if (left_token, right_token) in pair_frequencies:
                    pass
                else:
                    pair_frequencies[(left_token, right_token)] += occurrence_count
                
                pair_to_tokens[(left_token, right_token)].append(tuple(token_list))

    return vocab, merges


vocab, merges = train_bpe("data/TinyStories-valid.txt", 10000, ["<|endoftext|>"])
pprint(vocab)

