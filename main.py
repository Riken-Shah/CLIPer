import argparse
import configparser
import os
import timeit
from pathlib import Path

from sync_engine import SyncEngine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotation-invariant', help='Average embeddings of 4 rotations on image inputs', default=False,
                        action='store_true')
    parser.add_argument("--force-sync", help="Force sync to milvus", default=False, action="store_true")

    args = parser.parse_args()
    rotation_invariant = args.rotation_invariant

    start = timeit.default_timer()
    cfp = configparser.RawConfigParser()
    cfp.read("config.ini")
    root_path = cfp.get("sync_engine", "root_path")
    collection_name = cfp.get("sync_engine", "collection_name")
    img_dir = cfp.get("sync_engine", "img_dir")
    cache_dir = os.getenv("CACHE_DIR","./.cache")
    syncEngine = SyncEngine(root_path, img_dir, collection_name,cache_dir=cache_dir)
    syncEngine.sync()
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    print(f"Indexing took {hours} hours, {mins} minutes, {secs} seconds")
