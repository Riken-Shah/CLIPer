import configparser

from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from pymilvus import connections, utility


class Milvus:
    def __init__(self, config_path="config.ini", service_name="milvus", collection_name="dev_collection"):
        cfp = configparser.RawConfigParser()
        cfp.read(config_path)
        milvus_uri = cfp.get(service_name, 'uri')
        user = cfp.get(service_name, 'user')
        password = cfp.get(service_name, 'password')
        connections.connect("default",
                            uri=milvus_uri,
                            user=user,
                            password=password)

        self.DIM = 512  # dimension of vector
        self._collection = self.setup_collection(collection_name)
        print("Successfully connected to Milvus")

    def setup_collection(self, collection_name) -> Collection:
        check_collection = utility.has_collection(collection_name)
        if check_collection:
            collection = Collection(name=collection_name)
            return collection

        image_vector = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.DIM,
                                   description="vector which represents the image",)
        fname = FieldSchema(name="fname", dtype=DataType.VARCHAR, max_length=512,
                            description="path of the image", is_primary=True, auto_id=False)

        metadata = FieldSchema(name="metadata", dtype=DataType.JSON, description="metadata of the image", )

        schema = CollectionSchema(fields=[image_vector, fname, metadata],
                                  auto_id=False,
                                  description="Textile Dev 01",
                                  enable_dynamic_field=False)

        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type': 'L2',
            'index_type': "IVF_FLAT",
            'params': {'nlist': 16384}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    def get_total_count(self):
        self.setup_collection(self._collection.name)
        return self._collection.num_entities

    def insert(self, vectors, file_paths):
        self._collection.insert([vectors, file_paths])

    def delete_not_found_in(self, file_paths):
        self._collection.delete(f"fname not in {file_paths}")

    def upsert(self, records):
        print("upserted records to milvus", len(records))
        self._collection.upsert(records)

    def save(self):
        self._collection.flush()

    def search(self, vector, top_n=5, output_fields=None):
        if output_fields is None:
            output_fields = ["fname", "metadata"]
        return self._collection.search(vector, anns_field="embedding", param={"nprobe": 256}, limit=top_n,
                                       output_fields=output_fields)
