from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings 
from transformers import AutoTokenizer
import numpy as np 
import logging 
from typing import List
import torch 
import requests
import traceback
import time
import re 

logging.basicConfig(level=logging.INFO)


class EmbeddingClient:
    def __init__(self, endpoint_url: str, endpoint_port: int, model_name: str, max_tokens: int = 512, batch_size=32, test_connection=True):
        self.embed_client = HuggingFaceEndpointEmbeddings(model=f"{endpoint_url}:{endpoint_port}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        if test_connection:
            self.test_connection()
        
        
    def canonicalize_whitespace(self, text: str) -> str:
        # multiple \n -> single \n
        text = re.sub(r'\n+', '\n', text)
        return text
    
    def pre_process_documents(self, documents: List[str]) -> List[str]:
        
        processed_docs = []
        documents = [doc.strip() for doc in documents]
        # documents = [self.canonicalize_whitespace(doc) for doc in documents]
        documents = filter(lambda x: len(x) > 0, documents)
        
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            tokens = tokens[:self.max_tokens-8] # 8 truncated tokens
            assert len(tokens) <= self.max_tokens
            tokens_as_string = self.tokenizer.convert_tokens_to_string(tokens).strip()
            tokens_as_string = self.canonicalize_whitespace(tokens_as_string)
            processed_docs.append(tokens_as_string)
        assert all(len(self.tokenizer.tokenize(doc)) <= self.max_tokens for doc in processed_docs)
        return processed_docs
        
        
    # def get_embeddings(self, docuemnts: List[str]) -> List[List[float]]:
    #     processed_docs = self.pre_process_documents(docuemnts)
    #     if len(processed_docs) == 0:
    #         return []
    #     # embeddings = self.embed_client.embed_documents(processed_docs)
    #     embeddings = []
    #     for i in range(0, len(processed_docs), self.batch_size):
    #         batch_docs = processed_docs[i:i+self.batch_size]
    #         assert all(len(self.tokenizer.tokenize(doc)) <= self.max_tokens for doc in batch_docs)
    #         batch_embeddings = self.embed_client.embed_documents(batch_docs)
    #         embeddings.extend(batch_embeddings)
    #     return embeddings
    
    def get_embeddings(self, documents: List[str]) -> List[List[float]]:
        processed_docs = self.pre_process_documents(documents)
        if len(processed_docs) == 0:
            return []
        embeddings = []
        max_retries = 10
        backoff_factor = 0.5
        for i in range(0, len(processed_docs), self.batch_size):
            batch_docs = processed_docs[i:i+self.batch_size]
            assert all(len(self.tokenizer.tokenize(doc)) <= self.max_tokens for doc in batch_docs)
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.embed_client.embed_documents(batch_docs)
                    embeddings.extend(batch_embeddings)
                    break  # If successful, break out of the retry loop
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    logging.error(f"Request failed with error {e}. Traceback: {traceback_str}")
                    # log the lengths 
                    logging.error(f"Length of each document in the batch: {[len(self.tokenizer.tokenize(doc)) for doc in batch_docs]}")
                    # truncate length by 8 tokens
                    batch_docs = self.pre_process_documents(batch_docs) 
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor * (2 ** attempt)
                        logging.info(f"Retrying in {sleep_time} seconds")
                        time.sleep(sleep_time)
                    else:
                        raise  # Re-raise the last exception if all retries fail
        return embeddings
    
    def _average_cosine_distance_of_embeddings(self, embeddings: List[List[float]]) -> float:
        # Normalize vectors
        embeddings = torch.tensor(embeddings)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / norms
        cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)
        cosine_distance_matrix = 1 - cosine_similarity_matrix
        # exclude diagonal and duplicate entries using triu with diagonal=1
        mask = torch.triu(torch.ones_like(cosine_distance_matrix), diagonal=1)
        masked_distances = cosine_distance_matrix * mask
        average_distance = masked_distances.sum() / mask.sum()
        return average_distance.item()
    
    def average_cosine_distance(self, documents: List[str]) -> float:
        embeddings = self.get_embeddings(documents)
        if len(embeddings) == 0:
            return np.nan
        else: 
            return self._average_cosine_distance_of_embeddings(embeddings)
    
    def test_connection(self):
        docs_same = ["Hello World", "Hello World", "Hello World"]
        docs_diff = ["Ipsum Lorem", "Hello World", "Dolor Sit"]
        logging.info(f"Testing connection with the following documents: {docs_same} and {docs_diff}")
        same_distance = self.average_cosine_distance(docs_same)
        diff_distance = self.average_cosine_distance(docs_diff)
        assert np.isclose(same_distance, 0.0, atol=1e-3), f"the three documents {docs_same} should have a positive distance"
        assert diff_distance > 0, f"the three documents {docs_diff} should have a positive distance"
        logging.info("Connection test passed")
    
    
    
    
    