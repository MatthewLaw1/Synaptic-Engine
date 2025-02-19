"""
Inference script for EEG thought classification.
"""

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
from typing import Optional, Dict, Any

from .models import EEGEmbeddingCNN, embed_new_sample

# Load environment variables
load_dotenv()

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '300'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Create credentials dict from environment variables
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

    def get_latest_thought(self) -> Optional[Dict[str, Any]]:
        """Get the latest thought data from cloud storage.
        
        Returns:
            Dict containing the thought data if found, None otherwise
        """
        # List all blobs in the bucket
        blobs = list(self.bucket.list_blobs())
        if not blobs:
            print("No files found in bucket")
            return None

        # Sort by timestamp in the filename (assuming ISO format timestamps)
        latest_blob = max(blobs, key=lambda x: x.name)
        print(f"Found latest file: {latest_blob.name}")
        
        try:
            content = latest_blob.download_as_string()
            data = json.loads(content)
            
            # Only process if there's a thought
            if data.get('thought'):
                print(f"Found thought: {data['thought']}")
                return data
            else:
                print(f"No thought found in file {latest_blob.name}")
                return None
                
        except Exception as e:
            print(f"Error processing file {latest_blob.name}: {e}")
            return None

    def store_result(self, thought: str, embedding: np.ndarray) -> None:
        """Store the classification result in Google Cloud Storage.
        
        Args:
            thought: Classified thought string
            embedding: Generated embedding array
        """
        try:
            # Prepare data
            data = {
                'thought': thought,
                'embedding': embedding.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Create unique blob name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob_name = f"thought_{timestamp}.json"
            
            # Upload
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(data),
                content_type='application/json'
            )
            
            print(f"Result stored in {blob_name}")
            
        except Exception as e:
            print(f"Error storing result: {str(e)}")

# Global instance for backward compatibility
storage_manager = CloudStorageManager()

def load_model_and_scaler():
    """
    Load the trained model and scaler.
    
    Returns:
        tuple: (model, scaler)
    """
    if not os.path.exists("eeg_model.pth"):
        raise FileNotFoundError("Model file 'eeg_model.pth' not found!")
    if not os.path.exists("eeg_scaler.joblib"):
        raise FileNotFoundError("Scaler file 'eeg_scaler.joblib' not found!")
    
    # Calculate input dimension
    test_eeg = np.random.randn(4, 2560)
    from .eeg_processing import get_feature_vector
    test_features = get_feature_vector(test_eeg)
    input_dim = len(test_features)
    
    # Load model
    model = EEGEmbeddingCNN(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load("eeg_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    
    # Load scaler
    scaler = joblib.load("eeg_scaler.joblib")
    
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

def process_eeg_file(file_path, model, scaler):
    """
    Process an EEG file and classify the thought.
    
    Args:
        file_path: Path to CSV file containing EEG data
        model: Loaded EEGEmbeddingCNN model
        scaler: Loaded StandardScaler
        
    Returns:
        tuple: (thought classification, embedding)
    """
    # Load and validate CSV
    df = pd.read_csv(file_path)
    channels = ["TP9", "AF7", "AF8", "TP10"]
    if not all(ch in df.columns for ch in channels):
        raise ValueError(f"CSV must contain channels {channels}")
    
    # Process fixed window
    df = df.iloc[:2560]
    raw_eeg = df[channels].values.T
    
    # Generate embedding
    embedding = embed_new_sample(raw_eeg, model, scaler, device=DEVICE)
    
    # Query ChromaDB
    client = chromadb.HttpClient(
        ssl=True,
        host=os.getenv('CHROMA_HOST'),
        tenant=os.getenv('CHROMA_TENANT'),
        database=os.getenv('CHROMA_DATABASE'),
        headers={'x-chroma-token': os.getenv('CHROMA_TOKEN')}
    )
    collection = client.get_or_create_collection(
        name=os.getenv('CHROMA_COLLECTION', 'embeddings_eeg')
    )
    
    results = query_chroma_for_embedding(embedding, collection)
    
    # Get thought classification
    if results and results['metadatas']:
        thought_label = results['metadatas'][0][0]['thought_label']
        return THOUGHT_MAPPINGS.get(thought_label, 'Unknown thought pattern'), embedding
    
    return 'No specific thought pattern detected', embedding

def main(file_path):
    """
    Main inference function.
    
    Args:
        file_path: Path to EEG CSV file to process
    """
    try:
        # Load model and scaler
        print("Loading model and scaler...")
        model, scaler = load_model_and_scaler()
        
        # Process EEG data
        print("Processing EEG data...")
        thought, embedding = process_eeg_file(file_path, model, scaler)
        print(f"Classified thought: {thought}")
        
        # Store result
        print("Storing result...")
        storage_manager.store_result(thought, embedding)
        
        return thought
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <eeg_file.csv>")
        sys.exit(1)
    main(sys.argv[1])