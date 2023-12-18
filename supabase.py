import configparser

import vecs


class Supabase:
    def __init__(self, config_path="supabase.ini", service_name="default", collection_name="dev_collection"):
        self._collection = None
        cfp = configparser.RawConfigParser()
        cfp.read(config_path)
        uri = cfp.get(service_name, 'uri')
        self.client = vecs.create_client(
            uri)
        self.DIM = 512  # dimension of vector
        self.setup_collection("demo-10k")
        print("Successfully connected to Milvus")

    def setup_collection(self, collection_name):
        self._collection = self.client.get_or_create_collection(collection_name, dimension=self.DIM)
        return self._collection

    def insert(self, vectors, file_paths):
        insets = []
        for i in range(len(vectors)):
            insets.append((file_paths[i], vectors[i], {"type": "png"}))
            print(insets[i])

        self._collection.upsert(records=insets)
        print("Successfuly inserted")

    def save(self):
        self._collection.create_index()

    def search(self, vector, top_n=5, output_fields=None):
        if output_fields is None:
            output_fields = ['image_path']
        return self._collection.search(vector, anns_field="image_embedding", param={"nprobe": 256}, limit=top_n,
                                       output_fields=output_fields)
