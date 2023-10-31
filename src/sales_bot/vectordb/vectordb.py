import os
import sys
from abc import ABC, abstractmethod

from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding import ChineseEmbedding


def defaultDocTransformer():
    return CharacterTextSplitter(        
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )

class VectorDb(ABC):
    _db: VectorStore
    _transformer: TextSplitter

    def __init__(self, dbfile: str = "resources/electronic_devices_sales_qa.txt", transformer: TextSplitter = defaultDocTransformer(),rebuild: bool = False):
        self._transformer = transformer
        if not os.path.exists(dbfile.replace(".txt", ".db")) or rebuild:
            self._db = self._initDb(dbfile)

    @abstractmethod
    def _initDb(self, dbfile: str) -> VectorStore:
        pass
        

    @property
    def db(self):
        return self._db
