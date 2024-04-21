import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

IMG_EMBEDDINGS = "/ghome/group01/MCV-C5-G1/Week6/feature_embeddings/image/best_augmented_embedding.pkl"
IMG_PLOTS = "/ghome/group01/MCV-C5-G1/Week6/plots/embeddings/image/"

try:
    with open(IMG_EMBEDDINGS, "rb") as f:
        data = pickle.load(f)
        embeddings, labels = data["embeddings"], data["labels"]
except FileNotFoundError:
    print(f"Error: File '{IMG_EMBEDDINGS}' not found.")
    exit(1)
except KeyError:
    print(f"Error: The pickle file '{IMG_EMBEDDINGS}' does not contain 'embeddings' or 'labels' keys.")
    exit(1)


def plot_embeddings_2d(embeddings, labels):
    # Move the embeddings tensor to CPU and convert to numpy array
    embeddings = embeddings.cpu().numpy()
    
    # Reshape the embeddings to ensure it has at least two dimensions
    embeddings = embeddings.reshape(-1, 1)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in set(labels):
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, s=10)
    plt.title('2D Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(IMG_PLOTS + "TSNE.jpg")

def get_embedding_shape(embeddings):
    #embeddings = embeddings.cpu().numpy()
    print(embeddings)
    print(len(embeddings))
    print(len(labels[0]))

get_embedding_shape(embeddings=embeddings)
plot_embeddings_2d(embeddings, labels)
