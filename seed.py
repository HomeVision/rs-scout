import argparse
import json
import time

import requests

default_host = "http://164.90.253.179/"
default_index = "qc_rules"
default_data = "qc_rules.json"
default_batch = 100

headers = {"Content-Type": "application/json"}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="seed", description="Seeds a scout index")
    parser.add_argument("--host", type=str, default=default_host,
                        help=f"Scout index host (default: {default_host})")
    parser.add_argument("--index", type=str, default=default_index,
                        help=f"Index name to create (default: {default_index})")
    parser.add_argument("--data", type=str, default=default_data,
                        help=f"Data file to load from (default: {default_data})")
    parser.add_argument("--batch", type=int, default=default_batch,
                        help=f"Batch size to load data with (default: {default_batch})")

    return parser.parse_args()


def batch_rows(rows: list[dict], batch_size) -> list[list[dict]]:
    return [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]


def index_url(host: str, index_name: str) -> str:
    return f"{host}/index/{index_name}"


def clear_index(host: str, index_name: str):
    print("Calling clear_index")
    tstart = time.time()
    url = index_url(host, index_name)
    requests.delete(url, headers=headers)

    elapsed = time.time() - tstart
    print(f"Called clear_index in {elapsed:7.3f} seconds")


def create_index(host: str, index_name: str):
    print("calling create index")
    url = index_url(host, index_name)
    requests.post(url, headers=headers)


def process_batch(host: str, index_name: str, batch: list[dict]) -> int:
    url = index_url(host, index_name)
    response = requests.put(url, headers=headers, data=json.dumps(batch))

    return response.status_code


def main():
    args = parse_args()
    with open(args.data) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} rows")

    clear_index(args.host, args.index)
    create_index(args.host, args.index)

    for idx, batch in enumerate(batch_rows(data, args.batch)):
        print(f"Batch {idx:3d}: Calling process_batch")
        tstart = time.time()
        status_code = process_batch(
            host=args.host, index_name=args.index, batch=batch)
        elapsed = time.time() - tstart
        print(
            f"Batch {idx:3d}: Processed {len(batch):3d} in {elapsed:7.3f} seconds with status: {status_code:3d}")


if __name__ == "__main__":
    main()
