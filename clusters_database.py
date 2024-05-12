import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Defina o caminho do arquivo CSV
caminho_arquivo = r'C:\Users\Miguel\OneDrive\Ambiente de Trabalho\DataSet Suporte.csv'

# Lista para armazenar as strings da sexta coluna
strings_coluna_seis = []

# Abra o arquivo CSV e leia as strings da sexta coluna a partir da terceira linha
with open(caminho_arquivo, newline='', encoding='utf-8') as arquivo_csv:
    leitor_csv = csv.reader(arquivo_csv, delimiter=';')
    # Pula as duas primeiras linhas
    next(leitor_csv)  # Pula a linha de cabeçalho
    next(leitor_csv)  # Pula a segunda linha
    # Agora, comece a ler as linhas restantes
    for linha in leitor_csv:
        sexta_coluna = linha[5]  # A sexta coluna é indexada em 5 (começando de 0)
        strings_coluna_seis.append(sexta_coluna)

# Exibe o conteúdo da lista
#print(strings_coluna_seis)


# Embedding model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Obtaining embeddings
embeddings = model.encode(strings_coluna_seis)

# Defining the range of clusters to try
min_clusters = 3
max_clusters = 5

# Initializing lists to store silhouette scores
silhouette_scores = []

# Loop over different numbers of clusters
for k in range(min_clusters, max_clusters + 1):
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit KMeans to the embeddings
    kmeans.fit(embeddings)

    # Obtain cluster labels
    cluster_labels = kmeans.labels_

    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)

    # Append silhouette score to list
    silhouette_scores.append((k, silhouette))

# Plot silhouette scores
plt.plot(*zip(*silhouette_scores))
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette scores for K-means clustering')
plt.show()

# Choose the number of clusters with highest silhouette score
best_k, _ = max(silhouette_scores, key=lambda x: x[1])
print("Best number of clusters:", best_k)

# Initialize KMeans with best number of clusters
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
best_kmeans.fit(embeddings)

# Perform t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # Modifique perplexity para um valor adequado
tsne_embeddings = tsne.fit_transform(embeddings)


# Visualize clusters
plt.figure(figsize=(10, 8))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=best_kmeans.labels_, cmap='viridis')
plt.title('t-SNE embeddings for clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.show()
