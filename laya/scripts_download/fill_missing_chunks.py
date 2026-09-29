#!/usr/bin/env python3
import sys
sys.path.insert(0, "scripts_download")
from multi_range_download import resolve_final_url, fetch_chunk

url = "https://hf-mirror.com/jhu-clsp/mmBERT-base/resolve/main/pytorch_model.bin"
out_path = "models/mmBERT-base/pytorch_model.bin"
size = 1231188142
n = 32
chunk_size = (size + n - 1) // n
missing = [23, 26]

final_url, resolved_size = resolve_final_url(url)
assert resolved_size == size, (resolved_size, size)
print("resolved:", final_url[:80], "...")

from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=len(missing)) as ex:
    futs = []
    for i in missing:
        start = i * chunk_size
        end = min(start + chunk_size - 1, size - 1)
        futs.append(ex.submit(fetch_chunk, final_url, start, end, out_path, i, n))
    for fut in as_completed(futs):
        fut.result()
print("ALL MISSING CHUNKS DONE")
