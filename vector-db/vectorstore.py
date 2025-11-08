import numpy as np
import json

class CustomVectorStore():
    def __init__(
        self,
        vectors: list[dict] = []
    ) -> None:
        """
        Initialize vectors data from locally persisted data or given values.
        """
        with open("persisted_vectors.json") as f:
            persisted_vectors = json.loads(f.read())
        self.vectors_data = vectors or persisted_vectors
    
    def insert_vector(
        self,
        vector_id: str,
        text: str,
        vector: list[float],
        metadata: dict = {}
    ) -> None:
        """
        Insert vectors data with id, text, vector, and optional metadata.
        """
        data = {
            "id": vector_id,
            "text": text,
            "vector": vector,
            "metadata": metadata
        }
        self.vectors_data.append(data)
        self._persist_vectors()
        
    def delete_vector(
        self,
        vector_id: str
    ) -> None:
        """
        Delete vectors data based on vector id.
        """
        self.vectors_data = [data for data in self.vectors_data if data["id"] != vector_id]
        self._persist_vectors()
    
    def _calculate_cosine_similarity(
        self,
        vector1: list[float],
        vector2: list[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        
        dot_product = np.dot(vector1, vector2)
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        
        if vector1_norm == 0 or vector2_norm == 0:
            return 0
        
        cos_sim = dot_product / (vector1_norm * vector2_norm)
        
        return cos_sim
    
    def search_similar_vectors(
        self,
        vector: list[float],
        top_k: int = 5
    ) -> list[dict]:
        """
        Search the most similar vectors based on cosine similarity score.
        """
        cos_sim_results = []
        for data in self.vectors_data:
            vector2 = data["vector"]
            cos_sim = self._calculate_cosine_similarity(vector, vector2)
            
            cos_sim_results.append({
                **data,
                "sim_score": cos_sim
            })
        
        cos_sim_results.sort(key=lambda x: x["sim_score"], reverse=True)
        return cos_sim_results[:top_k]
    
    def _persist_vectors(self):
        """
        Simulate local data persistence.
        """
        with open("persisted_vectors.json", "w") as f:
            f.write(json.dumps(self.vectors_data))