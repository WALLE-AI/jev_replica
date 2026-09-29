#!/usr/bin/env python3
"""Multi-connection range downloader to work around a slow single-connection
path to the HF xet CDN mirror for a specific large LFS file."""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def resolve_final_url(url):
    r = requests.head(url, allow_redirects=True, timeout=20)
    r.raise_for_status()
    return r.url, int(r.headers["Content-Length"])

def fetch_chunk(url, start, end, out_path, idx, n_chunks, retries=60):
    expected = end - start + 1
    pos = start  # resume point; advances across retries, never resets to `start`
    for attempt in range(retries):
        headers = {"Range": f"bytes={pos}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                fd = os.open(out_path, os.O_WRONLY)
                try:
                    for data in resp.iter_content(chunk_size=1 << 18):
                        if not data:
                            continue
                        os.pwrite(fd, data, pos)
                        pos += len(data)
                finally:
                    os.close(fd)
                got = pos - start
                if got != expected:
                    raise IOError(f"chunk {idx}: got {got} expected {expected}")
                print(f"[chunk {idx}/{n_chunks}] done ({expected} bytes)", flush=True)
                return
        except Exception as e:
            done_frac = (pos - start) / expected * 100
            print(f"[chunk {idx}/{n_chunks}] attempt {attempt+1} failed at {done_frac:.1f}%: {e}", flush=True)
            time.sleep(min(1.5 ** min(attempt, 12), 15))
    raise RuntimeError(f"chunk {idx} failed after {retries} retries, at byte {pos}")

def main():
    url, out_path, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(f"Resolving {url} ...", flush=True)
    final_url, size = resolve_final_url(url)
    print(f"Final URL resolved, size={size} bytes ({size/1e6:.1f} MB)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.truncate(size)

    chunk_size = (size + n_workers - 1) // n_workers
    ranges = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size - 1, size - 1)
        if start > end:
            continue
        ranges.append((start, end))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
        futs = {
            ex.submit(fetch_chunk, final_url, s, e, out_path, i, len(ranges)): i
            for i, (s, e) in enumerate(ranges)
        }
        for fut in as_completed(futs):
            fut.result()

    elapsed = time.time() - t0
    actual_size = os.path.getsize(out_path)
    print(f"Downloaded {actual_size} bytes in {elapsed:.1f}s ({actual_size/1e6/elapsed:.2f} MB/s)", flush=True)
    if actual_size != size:
        print(f"WARNING: size mismatch, expected {size} got {actual_size}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
