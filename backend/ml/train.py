"""
Training script for EEG embedding model.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import chromadb
from dotenv import load_dotenv
from datetime import datetime
import random

from .eeg_processing import get_feature_vector
from .models import EEGEmbeddingCNN, TripletLoss

# Load environment variables
load_dotenv()

# Configuration from environment
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '300'))
EPOCHS = int(os.getenv('EPOCHS', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-3'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thought labels for classification
THOUGHT_LABELS = [
    "dhp",  # Doctor Helps Patient - care, empathy
    "sab",  # Sister argues with brother - conflict
    "fbh",  # Fire burns house - emergency, fear
    "cfm",  # Child cries for mother - sadness
    "null"  # No specific thought
]

class EEGTripletDataset(Dataset):
    """Dataset for training with triplet loss."""
    
    def __init__(self, feature_dict):
        """
        Initialize dataset.
        
        Args:
            feature_dict: {label_str: [feature_vector, ...]}
        """
        super().__init__()
        self.feature_dict = feature_dict
        self.labels = list(feature_dict.keys())
        
        self.total_samples = sum(len(feats) for feats in feature_dict.values())

    def __len__(self):
        return 999999  # Effectively infinite for random sampling

    def __getitem__(self, idx):
        """Get a triplet of anchor, positive, and negative samples."""
        # Select anchor label
        anchor_label = random.choice(self.labels)
        positives = self.feature_dict[anchor_label]
        
        # Ensure enough samples for anchor/positive pair
        if len(positives) < 2:
            anchor_label = random.choice([lbl for lbl in self.labels 
                                        if len(self.feature_dict[lbl]) >= 2])
            positives = self.feature_dict[anchor_label]

        # Get anchor and positive from same label
        anchor_feat, positive_feat = random.sample(positives, 2)
        
        # Get negative from different label
        neg_label = random.choice([lbl for lbl in self.labels if lbl != anchor_label])
        negative_feat = random.choice(self.feature_dict[neg_label])
        
        return (anchor_feat, positive_feat, negative_feat, 
                anchor_label, anchor_label, neg_label)

def load_and_process_data(data_dir):
    """
    Load and process EEG data from CSV files.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        dict: {label: [feature_vector, ...]}
    """
    feature_dict = {lbl: [] for lbl in THOUGHT_LABELS}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    for csv_file in csv_files:
        # Extract label from filename
        base = os.path.basename(csv_file)
        found_label = next((lbl for lbl in THOUGHT_LABELS if lbl in base), None)
        
        if not found_label:
            print(f"Warning: no label found for {csv_file}, skipping")
            continue
            
        # Load and validate data
        df = pd.read_csv(csv_file)
        channels = ["TP9", "AF7", "AF8", "TP10"]
        if not all(ch in df.columns for ch in channels):
            print(f"Warning: {csv_file} missing channels {channels}. Skipping.")
            continue
            
        # Process fixed window of samples
        df = df.iloc[:2560]
        raw_eeg = df[channels].values.T
        
        # Extract features
        feats = get_feature_vector(raw_eeg)
        feature_dict[found_label].append(feats)
    
    return feature_dict

def scale_features(feature_dict):
    """
    Scale all features using StandardScaler.
    
    Args:
        feature_dict: {label: [feature_vector, ...]}
        
    Returns:
        tuple: (scaled feature_dict, fitted scaler)
    """
    # Gather all features for fitting
    all_feats = []
    for feats in feature_dict.values():
        all_feats.extend(feats)
    all_feats = np.array(all_feats)
    
    # Fit and apply scaler
    scaler = StandardScaler()
    scaler.fit(all_feats)
    
    scaled_dict = {}
    for label, feats in feature_dict.items():
        scaled_dict[label] = [
            scaler.transform(f.reshape(1, -1))[0] for f in feats
        ]
    
    return scaled_dict, scaler

def store_embeddings(model, feature_dict, device=DEVICE):
    """
    Store embeddings in ChromaDB.
    
    Args:
        model: Trained EEGEmbeddingCNN
        feature_dict: Dict of scaled feature vectors
        device: torch device
    """
    # Initialize ChromaDB client
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
    
    # Generate and store embeddings
    doc_ids = []
    doc_metadatas = []
    doc_embeddings = []
    
    counter = 0
    model.eval()
    
    for label in THOUGHT_LABELS:
        for feat_scaled in feature_dict[label]:
            # Generate embedding
            with torch.no_grad():
                ft = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                emb = model(ft)
            embedding_vec = emb.cpu().numpy().reshape(-1).tolist()
            
            # Store metadata
            doc_ids.append(f"{label}_{counter}")
            doc_metadatas.append({"thought_label": label})
            doc_embeddings.append(embedding_vec)
            counter += 1
    
    # Add to collection
    collection.add(
        embeddings=doc_embeddings,
        metadatas=doc_metadatas,
        ids=doc_ids
    )
    
    print(f"Stored {counter} embeddings in ChromaDB collection '{collection.name}'")

def main():
    """Main training function."""
    data_dir = os.getenv('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    # Load and process data
    print("Loading and processing data...")
    feature_dict = load_and_process_data(data_dir)
    scaled_dict, scaler = scale_features(feature_dict)
    
    # Initialize model and training
    input_dim = next(iter(scaled_dict.values()))[0].shape[0]
    model = EEGEmbeddingCNN(input_dim=input_dim, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TripletLoss()
    
    # Create dataset and dataloader
    dataset = EEGTripletDataset(scaled_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            anchor_f, pos_f, neg_f, _, _, _ = data
            
            # Prepare data
            anchor_f = anchor_f.float().unsqueeze(1).to(DEVICE)
            pos_f = pos_f.float().unsqueeze(1).to(DEVICE)
            neg_f = neg_f.float().unsqueeze(1).to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            anchor_emb = model(anchor_f)
            pos_emb = model(pos_f)
            neg_emb = model(neg_f)
            
            # Compute loss and update
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx > 50:  # Limit batches per epoch
                break
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/num_batches:.4f}")
    
    # Save model and scaler
    print("Saving model and scaler...")
    torch.save(model.state_dict(), "eeg_model.pth")
    joblib.dump(scaler, "eeg_scaler.joblib")
    
    # Store embeddings
    print("Storing embeddings in ChromaDB...")
    store_embeddings(model, scaled_dict)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()