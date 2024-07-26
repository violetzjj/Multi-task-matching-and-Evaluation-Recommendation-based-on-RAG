import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = "https://ganjiuwanshi.com/v1/"

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            # Ensure that text is properly formatted as a single line without newlines
            text = text.replace("\n", " ")
            try:
                response = self.client.embeddings.create(input=[text], model=model)
                return response.data[0].embedding
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
        else:
            raise NotImplementedError

    # def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
    #     if self.is_api:
    #         text = text.replace("\n", " ")
    #         return self.client.embeddings.create(input=[text], model=model).data[0].embedding
    #     else:
    #         raise NotImplementedError


class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-2",
        input=text,
        )
        return response.data[0].embedding




