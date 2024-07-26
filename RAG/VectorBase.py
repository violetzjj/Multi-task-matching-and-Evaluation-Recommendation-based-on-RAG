import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, ZhipuEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        if document is None:
            document = []
        self.document = document
        self.vectors = []  # 初始化向量存储列表

    def get_vector(self, EmbeddingModel: OpenAIEmbedding) -> List[List[float]]:

        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return OpenAIEmbedding.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: OpenAIEmbedding, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()