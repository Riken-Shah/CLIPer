from tinydb import TinyDB


def to_rgb(image):
    return image.convert("RGB")


class LocalDB:
    def __init__(self, db_path="db.json", version=0.1):
        self.__db = TinyDB(db_path)
        self.__version = version

    def get_or_create_table(self, table_name):
        table_name = f"v{self.__version}-{table_name}"
        table = self.__db.table(table_name)
        if table is None:
            table = self.__db.table(table_name)
        return table


