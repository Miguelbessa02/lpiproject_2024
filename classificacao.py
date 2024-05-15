import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE  # Para balanceamento de dados
from joblib import dump

# Configurações para mostrar mais linhas e colunas
#pd.set_option('display.max_rows', None)  # Substitui None por um número específico se o DataFrame for muito grande
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_colwidth', None)

# Carregando os dados
caminho_arquivo = 'C:\\Users\\Miguel\\PycharmProjects\\lpiproject_2024\\updated_dataset.csv'
df = pd.read_csv(caminho_arquivo, delimiter=';')

# Eliminar linhas onde 'Departamento' ou 'DescricaoProb' possam conter NaN
df.dropna(subset=['Departamento', 'DescricaoProb'], inplace=True)

# Verifica se ainda existem NaNs nos dados
#print("NaN em Departamento:", df['Departamento'].isnull().any())
#print("NaN em Descrição:", df['DescricaoProb'].isnull().any())

# Mostrar todo o DataFrame
#print(df)

# Gerar embeddings para as descrições
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
embeddings = np.array([model.encode(desc) for desc in df['DescricaoProb']])

# Preparando a coluna alvo
y = df['Departamento']
# Transformando rótulos categóricos em numéricos, se necessário
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42, stratify=y)

# Balanceamento de dados usando SMOTE
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Treinamento do modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)

# Avaliação do modelo
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Opcional: Salvar o modelo treinado para uso futuro
# import joblib
# joblib.dump(rf, 'random_forest_model.pkl')

# Supondo que 'rf' seja o teu modelo RandomForest treinado
dump(rf, 'random_forest_model.joblib')


