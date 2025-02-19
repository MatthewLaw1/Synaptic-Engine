"""EEG thought classification inference."""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import chromadb
from dotenv import load_dotenv
from datetime import datetime
from google.cloud import storage
import json
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

from .models import EEGEmbeddingCNN, embed_new_sample

load_dotenv()

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '300'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

THOUGHT_MAPPINGS = {
    'cfm': 'Child cries for mother',
    'dhp': 'Doctor Helps Patient',
    'sab': 'Sister argues with brother',
    'fbh': 'Fire burns house',
    'null': 'No specific thought pattern detected'
}

class CloudStorageManager:
    def __init__(self, bucket_name: str = None):
        """Initialize the Cloud Storage Manager.
        
        Args:
            bucket_name: Name of the GCS bucket (optional, defaults to env var)
        """
        self.enabled = False
        try:
            credentials = {
                "type": "service_account",
                "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("GOOGLE_PRIVATE_KEY"),
                "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GOOGLE_CLIENT_EMAIL')}",
                "universe_domain": "googleapis.com"
            }
            self.bucket_name = bucket_name or os.getenv("GOOGLE_BUCKET_NAME", "synapse-ai-bucket")
            self.client = storage.Client.from_service_account_info(credentials)
            self.bucket = self.client.get_bucket(self.bucket_name)
            self.enabled = True
        except Exception as e:
            print(f"Warning: Cloud Storage disabled - {str(e)}")

    def get_latest_thought(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            blobs = list(self.bucket.list_blobs())
            if not blobs:
                return None

            latest_blob = max(blobs, key=lambda x: x.name)
            data = json.loads(latest_blob.download_as_string())
            return data if data.get('thought') else None

        except Exception as e:
            print(f"Error getting latest thought: {e}")
            return None

    def store_result(self, thought: str, embedding: np.ndarray) -> None:
        if not self.enabled:
            return

        try:
            data = {
                'thought': thought,
                'embedding': embedding.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            blob_name = f"thought_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.bucket.blob(blob_name).upload_from_string(
                json.dumps(data),
                content_type='application/json'
            )
        except Exception as e:
            print(f"Error storing result: {e}")

storage_manager = CloudStorageManager()

@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        ssl=True,
        host=os.getenv('CHROMA_HOST'),
        tenant=os.getenv('CHROMA_TENANT'),
        database=os.getenv('CHROMA_DATABASE'),
        headers={'x-chroma-token': os.getenv('CHROMA_TOKEN')}
    )

def load_model_and_scaler() -> Tuple[EEGEmbeddingCNN, Any]:
    if not os.path.exists("eeg_model.pth") or not os.path.exists("eeg_scaler.joblib"):
        raise FileNotFoundError("Model or scaler file not found")
    
    from .eeg_processing import get_feature_vector
    input_dim = len(get_feature_vector(np.random.randn(4, 2560)))
    
    model = EEGEmbeddingCNN(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load("eeg_model.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    
    return model, scaler

def query_chroma_for_embedding(embedding, collection, k=1):
    """
    Query ChromaDB for similar embeddings.
    
    Args:
        embedding: numpy array of shape (embedding_dim,)
        collection: ChromaDB collection
        k: Number of results to return
        
    Returns:
        dict: ChromaDB query results
    """
    query_emb = [embedding.tolist()]
    return collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

def process_eeg_file(file_path: str, model: EEGEmbeddingCNN, scaler: Any) -> Tuple[str, np.ndarray]:
    df = pd.read_csv(file_path)
    if not all(ch in df.columns for ch in CHANNELS):
        raise ValueError(f"Missing required channels: {CHANNELS}")
    
    raw_eeg = df.iloc[:2560][CHANNELS].values.T
    embedding = embed_new_sample(raw_eeg, model, scaler, device=DEVICE)
    
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=os.getenv('CHROMA_COLLECTION', 'embeddings_eeg')
    )
    
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=1,
        include=['metadatas']
    )
    
    if results and results['metadatas']:
        thought_label = results['metadatas'][0][0]['thought_label']
        return THOUGHT_MAPPINGS.get(thought_label, 'Unknown thought pattern'), embedding
    
    return 'No specific thought pattern detected', embedding

def main(file_path: str) -> Optional[str]:
    try:
        model, scaler = load_model_and_scaler()
        thought, embedding = process_eeg_file(file_path, model, scaler)
        storage_manager.store_result(thought, embedding)
        return thought
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <eeg_file.csv>")
        sys.exit(1)
    main(sys.argv[1])