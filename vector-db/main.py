import uuid
from google import genai
from google.genai.types import EmbedContentConfig
from vectorstore import CustomVectorStore

client = genai.Client(
    vertexai=True
)

def generate_embeddings(text_contents: list | str) -> list:
    """
    Generate embeddings for a list of text or a single text.
    """
    if isinstance(text_contents, str):
        text_contents = [text_contents]
        
    response = client.models.embed_content(
        model="text-multilingual-embedding-002",
        contents=text_contents,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )
    
    return response.embeddings

def insert_sample_data(vector_store: CustomVectorStore) -> None:
    """
    Insert sample data to the custom vector store.
    """
    sample_data = [
        {
            "id": str(uuid.uuid4()),
            "text": "Saya sedang membuat coding tentang generative AI"
        },
        {
            "id": str(uuid.uuid4()),
            "text": "Budi sedang bermain balon dengan teman-temannya."
        },
        {
            "id": str(uuid.uuid4()),
            "text": "Dani lagi main Valorant bareng temen kuliahnya"
        },
        {
            "id": str(uuid.uuid4()),
            "text": "Kalau mau makan, harus siapin piring dan sendok dulu."
        },
        {
            "id": str(uuid.uuid4()),
            "text": "RAG adalah teknologi gen AI yang mempermudah pencarian informasi."
        }
    ]
    
    print("Generating vector from text contents using Vertex AI embeddings...")
    
    text_contents = [data["text"] for data in sample_data]
    embeddings = generate_embeddings(text_contents=text_contents)
    
    print(f"Inserting {len(sample_data)} vector(s) into the custom vector store...")
    for i in range(len(sample_data)):
        sample_data[i]["vector"] = embeddings[i].values
        vector_store.insert_vector(
            vector_id=sample_data[i]["id"],
            text=sample_data[i]["text"],
            vector=embeddings[i].values
        )
    
    print(f"Finished inserting all vectors.")

def main():
    """
    Main method to test custom vector store with Vertex AI embeddings.
    """
    vector_store = CustomVectorStore()
    
    # Insert sample_data if needed
    insert_sample_data(vector_store=vector_store)
    
    query = "apa itu RAG?"
    embeddings = generate_embeddings(query)
    embedding = embeddings[0].values
    
    retrieved_vectors = vector_store.search_similar_vectors(
        vector=embedding,
        top_k=3
    )
    retrieved_text = [{
        "id": data["id"],
        "text": data["text"],
        "sim_score": data["sim_score"]
    } for data in retrieved_vectors]
    
    print(retrieved_text)

if __name__ == "__main__":
    main()