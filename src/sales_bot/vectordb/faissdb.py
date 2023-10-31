import os
import sys
from abc import ABC, abstractmethod

from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain.vectorstores import FAISS

from vectordb import VectorDb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding import ChineseEmbedding


class FaissDb(VectorDb):
    """_summary_

    The filename is not allowed to be named as faiss.py, 
    otherwise it will pop up strange error like 'module 'faiss' has no attribute 'IndexFlatL2''
    See [issue](https://github.com/facebookresearch/faiss/issues/1195)
    """
    def __init__(self, *args, **kwargs):
        super(FaissDb, self).__init__(*args, **kwargs)
        
    def _initDb(self, dbfile: str, rebuild: bool) -> VectorStore:
        _db: FAISS = None
        if not os.path.exists(dbfile.replace(".txt", ".db")) or rebuild:
            try:
                with open(dbfile, 'r', encoding='utf-8-sig') as f:
                    docs = f.read()
                docs = self._transformer.create_documents([docs])
                _db = FAISS.from_documents(docs, ChineseEmbedding().embeddings)
                _db.save_local(dbfile.replace(".txt", ".db"))
                return _db
            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(e)
        else:
            _db = FAISS.load_local(dbfile.replace(".txt", ".db"), ChineseEmbedding().embeddings)
        return _db
    
if __name__ == "__main__":
    v = FaissDb()
    retriever = v.db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}
    )
    query = "你们价格怎么这么贵，是不是在坑人？"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content + "\n")