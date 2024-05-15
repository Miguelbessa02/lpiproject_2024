from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar o modelo
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Definindo 'df' no escopo global para que possa ser acessado em outras partes do aplicativo
global df

# Caminho do arquivo e leitura do CSV
caminho_arquivo = r'C:\Users\Miguel\PycharmProjects\lpiproject_2024\DataSet.csv'
df = pd.read_csv(caminho_arquivo, delimiter=';')

# Limpar colunas desnecessárias
df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')

# Gerar embeddings
embeddings = np.array([model.encode(desc) for desc in df['DescricaoProb']], dtype=np.float64)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(embeddings)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global df  # Correção: certificando-se de declarar 'global df' antes de usá-lo
    description = request.form['description']
    embedding = np.array(model.encode(description), dtype=np.float64).reshape(1, -1)
    cluster = kmeans.predict(embedding)
    cluster_label = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C', 3: 'Cluster D'}[cluster[0]]

    # Criar e adicionar nova entrada
    new_entry = pd.DataFrame({
        'DataHoraAbertura': [pd.Timestamp.now()],
        'NumProcesso': [df['NumProcesso'].max() + 1 if not df['NumProcesso'].empty else 1],
        'Cliente': [0],
        'Prioridade': [1],
        'Tecnico': ['Online Submission'],
        'DescricaoProb': [description],
        'ComoReproduzir': ['Not Applicable'],
        'Classificacao': [cluster[0]],
        'Departamento': [cluster_label]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(caminho_arquivo, sep=';', index=False)

    return render_template('result.html', cluster=cluster_label)

if __name__ == '__main__':
    app.run(debug=True)