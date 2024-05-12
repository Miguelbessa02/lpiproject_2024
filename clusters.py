import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Caminho do arquivo CSV
caminho_arquivo = r'C:\Users\sebas\PycharmProjects\lpiproject2024\DataSet.csv'
# Ler o arquivo CSV usando a primeira linha como cabeçalho (se necessário ajuste de acordo com o seu CSV)
df = pd.read_csv(caminho_arquivo, delimiter=';')

# Verifique se todas as colunas necessárias estão presentes
print("Colunas disponíveis no DataFrame:", df.columns)

# Remover colunas desnecessárias se existirem
df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')

# Limpeza de dados: Garantindo que não há valores NaN nas colunas usadas para análise
df.dropna(subset=['DescricaoProb', 'ComoReproduzir'], inplace=True)

# Conversão de descrições para embeddings
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
embeddings_descricao_prob = model.encode(df['DescricaoProb'].tolist())

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Classificacao'] = kmeans.fit_predict(embeddings_descricao_prob)

# Mapeamento dos rótulos dos clusters para descrições significativas
cluster_labels = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C', 3: 'Cluster D'}
df['Departamento'] = df['Classificacao'].map(cluster_labels)

# Salvar o DataFrame atualizado em um arquivo CSV
df.to_csv('updated_dataset.csv', sep=';', index=False, encoding='utf-8')

# Visualização t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings_descricao_prob)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=df['Classificacao'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Classificação')
plt.title('Visualização t-SNE dos Clusters')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.show()

# Preparação dos dados para classificação
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Classificacao'])
X = embeddings_descricao_prob

# Divisão dos dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Classificador - RandomForest
classifier = RandomForestClassifier(max_depth=10, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm,
                x=[f'Cluster {cls}' for cls in label_encoder.classes_],
                y=[f'Cluster {cls}' for cls in label_encoder.classes_],
                text_auto=True,
                labels=dict(x="Predicted Label", y="True Label"),
                color_continuous_scale='Blues',
                title="Matriz de Confusão")
fig.show()