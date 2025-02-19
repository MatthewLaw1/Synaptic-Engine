"""EEG embedding model training with triplet loss."""

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
import random

from .eeg_processing import get_feature_vector
from .models import EEGEmbeddingCNN, TripletLoss

load_dotenv()

# Training configuration
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '300'))
EPOCHS = int(os.getenv('EPOCHS', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-3'))
PATIENCE = int(os.getenv('PATIENCE', '10'))
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
    def __init__(self, feature_dict):
        super().__init__()
        self.feature_dict = feature_dict
        self.valid_labels = [lbl for lbl in feature_dict.keys()
                           if len(feature_dict[lbl]) >= 2]
        if not self.valid_labels:
            raise ValueError("No labels with sufficient samples for triplet training")

    def __len__(self):
        return 10000  # Limited but large enough for training

    def __getitem__(self, _):
        anchor_label = random.choice(self.valid_labels)
        anchor_feat, positive_feat = random.sample(self.feature_dict[anchor_label], 2)
        
        neg_label = random.choice([l for l in self.valid_labels if l != anchor_label])
        negative_feat = random.choice(self.feature_dict[neg_label])
        
        return (anchor_feat, positive_feat, negative_feat,
                anchor_label, anchor_label, neg_label)

def load_and_process_data(data_dir):
    """Load and process EEG data from CSV files."""
    feature_dict = {lbl: [] for lbl in THOUGHT_LABELS}
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    print(f"Processing {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        found_label = next((lbl for lbl in THOUGHT_LABELS
                          if lbl in os.path.basename(csv_file)), None)
        if not found_label:
            continue
            
        try:
            df = pd.read_csv(csv_file)
            if not all(ch in df.columns for ch in CHANNELS):
                continue
                
            raw_eeg = df.iloc[:2560][CHANNELS].values.T
            feature_dict[found_label].append(get_feature_vector(raw_eeg))
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not any(len(feats) >= 2 for feats in feature_dict.values()):
        raise ValueError("No label has enough samples for triplet training")
    
    return feature_dict

def scale_features(feature_dict):
    """Scale features using StandardScaler."""
    all_feats = np.vstack([feat for feats in feature_dict.values() for feat in feats])
    scaler = StandardScaler().fit(all_feats)
    
    return {
        label: [scaler.transform(f.reshape(1, -1))[0] for f in feats]
        for label, feats in feature_dict.items()
    }, scaler

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
    
    model.eval()
    embeddings_batch = []
    
    with torch.no_grad():
        for label, features in feature_dict.items():
            for idx, feat in enumerate(features):
                ft = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                emb = model(ft).cpu().numpy().reshape(-1).tolist()
                embeddings_batch.append({
                    'id': f"{label}_{idx}",
                    'metadata': {"thought_label": label},
                    'embedding': emb
                })
    
    # Batch insert
    collection.add(
        embeddings=[e['embedding'] for e in embeddings_batch],
        metadatas=[e['metadata'] for e in embeddings_batch],
        ids=[e['id'] for e in embeddings_batch]
    )

def train_model(model, dataloader, criterion, optimizer, epochs, patience=PATIENCE):
    """Train model with early stopping."""
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchor_f, pos_f, neg_f, _, _, _) in enumerate(dataloader):
            if batch_idx > 50:  # Limit batches per epoch
                break
                
            # Prepare batch
            anchor_f = anchor_f.float().unsqueeze(1).to(DEVICE)
            pos_f = pos_f.float().unsqueeze(1).to(DEVICE)
            neg_f = neg_f.float().unsqueeze(1).to(DEVICE)
            
            # Forward pass and loss
            optimizer.zero_grad()
            loss = criterion(
                model(anchor_f),
                model(pos_f),
                model(neg_f)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "eeg_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    return best_loss

def main():
    """Train EEG embedding model."""
    data_dir = os.getenv('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    print("Loading and processing data...")
    feature_dict = load_and_process_data(data_dir)
    scaled_dict, scaler = scale_features(feature_dict)
    joblib.dump(scaler, "eeg_scaler.joblib")
    
    # Initialize model and training
    input_dim = next(iter(scaled_dict.values()))[0].shape[0]
    model = EEGEmbeddingCNN(input_dim=input_dim, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TripletLoss()
    
    # Train model
    dataset = EEGTripletDataset(scaled_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    best_loss = train_model(model, dataloader, criterion, optimizer, EPOCHS)
    
    print(f"Training completed with best loss: {best_loss:.4f}")
    print("Storing embeddings in ChromaDB...")
    store_embeddings(model, scaled_dict)

if __name__ == "__main__":
    main()