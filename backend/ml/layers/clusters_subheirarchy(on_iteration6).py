import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import chromadb


def main(): 
    # Connect to ChromaDB
    client = chromadb.HttpClient(
        ssl=True,
        host='api.trychroma.com',
        tenant='768c13d0-c1fb-4e76-bd7c-ae5c8732fe8d',
        database='synapse-ai',
        headers={'x-chroma-token': 'ck-CoaMuowHnZq62X3jgLyFKzS5qPvxzPTaERDXbEuWNiTi'}
    )

    # Retrieve embeddings from the ChromaDB collection
    collection = client.get_or_create_collection(name="embeddings_eeg")
    results = collection.get(include=['embeddings', 'metadatas'])

    # Extract embeddings and labels
    embeddings = np.array(results['embeddings'])
    labels = [meta['thought_label'] for meta in results['metadatas']]

    # Convert labels into numerical categories
    unique_labels = list(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    label_numbers = np.array([label_mapping[label] for label in labels])

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Reduce dimensions using t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_scaled)

    # Create DataFrame for visualization
    df_embeddings = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    df_embeddings['Label'] = label_numbers

    # Plot the t-SNE results with clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Dim1', y='Dim2', hue=df_embeddings['Label'].map(lambda x: unique_labels[x]), palette="viridis", data=df_embeddings)
    plt.title("t-SNE Cluster Visualization of EEG Thought Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Thought Label")
    plt.savefig('frontend/public/images/tsne_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
