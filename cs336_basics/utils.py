from collections import defaultdict, Counter
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
            # 若找到special_token，则将其作为该chunk的end
            initial_position += mini_chunk_size
            # 否则继续循环直到在mini_chunk中找到special_token

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenization(chunk: str, special_token: list[str]):
    """
    先处理special_token，切成多段
    将正则表达式匹配到的token转换为UTF-8编码的字节，然后拆分成单字节对象
    :param chunk: 说明
    :type chunk: str
    :param special_token: 说明
    :type special_token: tuple[str]
    """
    pre: dict[bytes, int] = defaultdict(int)
    # re.split 的行为 按 pattern 匹配到的位置切, 匹配到的 token 本身会被丢弃
    if special_token:
        pattern = "|".join(re.escape(token) for token in special_token)
        segments = re.split(pattern, chunk)
    else:
        segments = [chunk]
    for segment in segments:
        for match in re.finditer(PAT, segment):
            token_bytes = match.group().encode("utf-8")
            # byte_tuple = tuple(bytes([b]) for b in token_bytes)
            pre[token_bytes] += 1
    return pre


def run_pretokenization(input_path, special_tokens: list[str]):
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = os.process_cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        pre_results = []
        with Pool(processes=num_processes) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                async_pre = pool.apply_async(pretokenization, (chunk, special_tokens))
                pre_results.append(async_pre)

                # Run pre-tokenization on your chunk and store the counts for each pre-token
            pool.close()
            pool.join()
        total_counts = Counter()
        for result in pre_results:
            total_counts.update(result.get())
        return total_counts
