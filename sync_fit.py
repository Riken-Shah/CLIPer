import json
from datetime import datetime
from itertools import chain

import torch
import argparse
import configparser
import timeit
import os
from pathlib import Path

from caption import ImageCaption
from clip import ImagesIndexer
from milvus import Milvus

EXTENSIONS_LIST = ["*.jpeg"]


class SyncDir:
    def __init__(self, images_folder, cache_dir="", collection_name=""):
        # Cache Dir for models
        self.cache_dir = cache_dir

        self.device = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Torch will use self.device: {self.device}")

        # The root directory encapsulates all the images
        self.images_path = Path(images_folder)
        assert (
            self.images_path.exists()
        ), f"Image folder {self.images_path.resolve().absolute()} does not exist"

        # Indexer
        self.indexer = ImagesIndexer(self.images_path, do_rotate_images=True, cache_dir=cache_dir, device=self.device)
        # Captioner
        self.captioner = ImageCaption(self.indexer.model, self.indexer.preprocess_image, cache_dir=self.cache_dir,
                                      device=self.device)

        # VectorDB which acts as the index for the images
        self.milvus = Milvus(collection_name=collection_name)

    def _rglob_extension(self, extension):
        for fname in chain.from_iterable(
                [self.images_path.rglob(extension), self.images_path.rglob(extension.upper())]):
            yield fname.relative_to(self.images_path)

    def empty_function(self, *args, **kwargs):
        return

    def create_record(self, fname, width, height, emb, caption, tags):
        aspect_ratio = width / height
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "fname": fname,
            "width": width,
            "height": height,
            "emb": emb,
            "aspect_ratio": aspect_ratio,
            "last_synced_on": now,
            "caption": caption,
            "tags": tags
        }

    def create_records_for_milvus(self, records):
        records_for_db = []
        for record in records:
            record = {
                "embedding": record["emb"],
                "fname": str(record["fname"]),
                "metadata": json.dumps({
                    "width": record["width"],
                    "height": record["height"],
                    "caption": record["caption"],
                    "tags": record["tags"]
                })
            }
            records_for_db.append(record)
        return records_for_db

    def add_bulk(self):
        # images_files = sorted(
        #     map(str,  chain(*map(self._rglob_extension, EXTENSIONS_LIST)))
        # )
        # images_files = os.scandir(self.images_path)

        # print("{} images found".format(len(images_files)))

        bulk = 10000
        images_files = []
        for entry in os.scandir(self.images_path):
            if entry.name.endswith(".jpeg"):
                images_files.append(entry.path)

            if len(images_files) >= bulk:
                records = self.indexer.add_bulk(images_files, self.create_record, self.empty_function,
                                                self.captioner.get_caption_and_tags)
                # insert into milvus
                self.milvus.upsert(self.create_records_for_milvus(records))
                images_files = []

        if len(images_files) > 0:
            records = self.indexer.add_bulk(images_files, self.create_record, self.empty_function,
                                            self.captioner.get_caption_and_tags)
            # insert into milvus
            self.milvus.upsert(self.create_records_for_milvus(records))



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
    cache_dir = os.getenv("CACHE_DIR", "./.cache")
    syncEngine = SyncDir(img_dir, collection_name=collection_name, cache_dir=cache_dir)
    syncEngine.add_bulk()
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    print(f"Indexing took {hours} hours, {mins} minutes, {secs} seconds")
