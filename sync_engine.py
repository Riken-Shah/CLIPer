import asyncio

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from itertools import chain

import torch
from tinydb import Query
from twisted.internet import task, reactor
from PIL import Image
from milvus import Milvus

from clip import ImagesIndexer
from pathlib import Path

from thumb import Thumbnails
from db import LocalDB

from caption import ImageCaption
EXTENSIONS_LIST = ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"]
EXTENSIONS_LIST.extend([e.upper() for e in EXTENSIONS_LIST])
Image.MAX_IMAGE_PIXELS = None

def time_the_function_decorator(func):
    def wrapper(*args, **kwargs):

        start = datetime.now()
        func(*args, **kwargs)
        end = datetime.now()
        print(f"Time taken: {(end - start).total_seconds()} seconds")

    return wrapper


class SyncEngine:
    def __init__(self, root_path, images_folder, collection_name, sync_interval=10 * 60, cache_dir=""):
        # Cache Dir for models
        self.cache_dir = cache_dir

        self.device = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Torch will use self.device: {self.device}")

        # All the necessary data for the sync engine is stored in the root_path.
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            os.makedirs(self.root_path)

        # The root directory encapsulates all the images
        self.images_path = Path(images_folder)
        assert (
            self.images_path.exists()
        ), f"Image folder {self.images_path.resolve().absolute()} does not exist"

        # Local JSON DB which keeps track of all the images
        self.__db_client = LocalDB(self.root_path / "db.json")
        self.__table = self.__db_client.get_or_create_table("images")
        self.__internal_table = self.__db_client.get_or_create_table("internal")

        # Thumbnail generator and appender
        self.thumbs = Thumbnails(self.root_path)
        self.appender = self.thumbs.appender()


        # Indexer
        self.indexer = ImagesIndexer(self.images_path, do_rotate_images=True, cache_dir=cache_dir, device=self.device)
        # Captioner
        self.captioner = ImageCaption(self.indexer.model, self.indexer.preprocess_image, cache_dir=self.cache_dir, device=self.device)


        # VectorDB which acts as the index for the images
        self.milvus = Milvus(collection_name=collection_name)

        # Sync interval
        self.sync_interval = sync_interval

    def register_sync(self):
        task.LoopingCall(self.sync).start(self.sync_interval)  # call every 10 minutes
        reactor.run()

    def list_file_fast(self):
        jpg_files = self.images_path.rglob("*.jpg")
        tif_files = self.images_path.rglob("*.tif")
        return len(jpg_files) + len(tif_files)


    def sync(self):
        images_files_found = sorted(
            map(str, chain(*map(self._rglob_extension, EXTENSIONS_LIST)))
        )
        print("{} images found".format(len(images_files_found)))

        sync_metadata = self.__internal_table.all()

        if len(sync_metadata) == 0:
            self.__internal_table.insert({"is_syncing": False, "attempts": 0, "last_synced_on": None})
            sync_metadata = self.__internal_table.all()

        attempts = 0
        is_syncing = False
        last_synced_on = None

        if len(sync_metadata):
            attempts = sync_metadata[0].get("attempts") or 0
            is_syncing = sync_metadata[0].get("is_syncing") or False
            last_synced_on = sync_metadata[0].get("last_synced_on")
            last_synced_on = datetime.strptime(last_synced_on, "%Y-%m-%d %H:%M:%S") if last_synced_on else None

        if attempts > 3:
            print("Sync failed 3 times. Overriding the is_syncing")
            is_syncing = False
            attempts = 0

        # if is_syncing:
        #     print("Sync is already in progress. Skipping this sync")
        #     return

        # If it's not synced in last 10 minutes, sync it
        # if is_syncing and not (last_synced_on is None or (datetime.now() - last_synced_on).total_seconds() > self.sync_interval):
        #     print("Already synced in last 10 minutes. Skipping this sync")
        #     return

        self.__internal_table.update({"is_syncing": True, "attempts": attempts + 1}, doc_ids=[sync_metadata[0].doc_id])

        should_sync_to_milvus = False
        should_sync_to_local_db = False

        # Cleanup: Delete the files which are not in the local db
        self.__table.remove(~Query().fname.one_of(images_files_found))
        self.milvus.delete_not_found_in(images_files_found)

        files_in_local_db = self.__table.all()

        if len(files_in_local_db) != len(images_files_found):
            should_sync_to_local_db = True

        if self.milvus.get_total_count() < len(images_files_found):
            should_sync_to_milvus = True

        print("files_in_local_db", len(files_in_local_db), "images_files_found", len(images_files_found),
              "files_found_in_milvus", self.milvus.get_total_count())

        if should_sync_to_local_db:
            self.add_bulk(True)

        self.__internal_table.update({"is_syncing": False, "attempts": 0, "last_synced_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, doc_ids=[sync_metadata[0].doc_id])

        print(f"Everything is in sync. Local DB: {should_sync_to_local_db}, Milvus: {should_sync_to_milvus}")

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

    def create_records_for_db(self, records):
        records_for_db = []
        for record in records:
            record = record.copy()
            record["fname"] = str(record["fname"])
            delete_keys = ["emb"]
            for key in delete_keys:
                del record[key]
            records_for_db.append(record)
        return records_for_db

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

    async def generate_thumbnail(self, fname, image):
        # respect aspect ratio of image and resize to closest of 512x512
        width, height = image.size
        aspect_ratio = width / height

        if aspect_ratio > 1:
            width = 512
            height = int(width / aspect_ratio)
        else:
            height = 512
            width = int(height * aspect_ratio)

        image = image.resize((width, height), Image.BICUBIC)
        image = image.convert("RGB")

        image.save(await self.appender.append(fname), format='JPEG')
        return image

    def _rglob_extension(self, extension):
        for fname in chain.from_iterable(
                [self.images_path.rglob(extension), self.images_path.rglob(extension.upper())]):
            yield fname.relative_to(self.images_path)

    def add_bulk_without_embedding(self, image_files):
        async def process_image(fname):
            with Image.open(self.images_path / fname) as image:
                width, height = image.size
                caption, tags = self.captioner.get_caption_and_tags(image)
                record = self.create_record(fname, width, height, None, caption ,tags)
                await self.generate_thumbnail(fname, image)
                return record

        with ThreadPoolExecutor() as executor:
            records = list(executor.map(process_image, image_files))
        return records

    def add_bulk(self, should_sync_to_milvus=True):
        assert (
            self.images_path.exists()
        ), f"Image folder {self.images_path.resolve().absolute()} does not exist"

        images_files = sorted(
            map(str, chain(*map(self._rglob_extension, EXTENSIONS_LIST)))
        )

        print("{} images found".format(len(images_files)))

        # Check if the following files are already in the database
        lists = self.__table.search(Query().fname.one_of(images_files))

        images_files_found_in_db = [f["fname"] for f in lists]

        # Remove them from the list
        images_files = [f for f in images_files if f not in images_files_found_in_db]

        print("{} images to add".format(len(images_files)))

        if len(images_files) == 0:
            print("No new images to add")
            return

        self.appender.__enter__()
        if should_sync_to_milvus:
            records = self.indexer.add_bulk(images_files, self.create_record, self.generate_thumbnail, self.captioner.get_caption_and_tags)
        else:
            records = self.add_bulk_without_embedding(images_files)
        self.appender.__exit__(None, None, None)

        # add many records
        self.__table.insert_multiple(self.create_records_for_db(records))

        if should_sync_to_milvus:
            # insert into milvus
            self.milvus.upsert(self.create_records_for_milvus(records))

    def thumbnail(self, fname):
        return self.thumbs.thumbnail(fname)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indexer.release_the_model()
        self.__internal_table.update({"is_syncing": False}, doc_ids=[self.__internal_table.all()[0].doc_id])
