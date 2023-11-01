import os
import sys
from abc import ABC, abstractmethod

from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema.embeddings import Embeddings
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
    _embedding: Embeddings

    def __init__(self, dbfile: str = "resources/electronic_devices_sales_qa.txt", embedding: Embeddings =  ChineseEmbedding().embeddings,transformer: TextSplitter = defaultDocTransformer(),rebuild: bool = False):
        self._transformer = transformer
        self._embedding = embedding
        self._db = self._initDb(dbfile, embedding, rebuild)

    @abstractmethod
    def _initDb(self, dbfile: str, embedding: Embeddings ,rebuild: bool) -> VectorStore:
        pass
    
    @abstractmethod
    def createMemory(self) -> VectorStoreRetrieverMemory:
        pass

    @property
    def db(self):
        return self._db
