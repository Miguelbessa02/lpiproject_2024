from joblib import load
from sentence_transformers import SentenceTransformer
import numpy as np

# Carregar o modelo
rf = load('random_forest_model.joblib')

# Carregar o modelo de embeddings, que deve ser o mesmo usado durante o treinamento
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Aqui entra a nova descrição do problema
nova_descricao = ""

# Converter a descrição em embeddings
embedding_nova_descricao = model.encode(nova_descricao)

# Reshape do embedding para compatibilidade de dimensão esperada pelo modelo
embedding_nova_descricao = np.array([embedding_nova_descricao])

# Usar o modelo para prever o cluster
predicted_cluster = rf.predict(embedding_nova_descricao)[0]

# Converter o identificador do cluster para o nome do departamento (ajuste conforme teu mapeamento real)
cluster_label = {
    'Cluster A': 'Cluster A',
    'Cluster B': 'Cluster B',
    'Cluster C': 'Cluster C',
    'Cluster D': 'Cluster D'
}
departamento = cluster_label[predicted_cluster]

print(f"O departamento indicado para a descrição fornecida é: {departamento}")