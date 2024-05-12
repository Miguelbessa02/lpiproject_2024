import pandas as pd

# Vamos simular a leitura do seu arquivo CSV que você carregou
# Nota: No seu ambiente, você deve usar o caminho do arquivo local como você especificou antes
df = pd.read_csv(r'C:\Users\Miguel\PycharmProjects\lpiproject2024\DataSet.csv', delimiter=';', header=1)

# Garantindo que toda a coluna seja tratada como string
df['DescricaoProb'] = df['DescricaoProb'].astype(str)
df['ComoReproduzir'] = df['ComoReproduzir'].astype(str)

# Limpar as colunas para remover caracteres de ponto e vírgula extras
df['DescricaoProb'] = df['DescricaoProb'].str.replace(';', ',')
df['ComoReproduzir'] = df['ComoReproduzir'].str.replace(';', ',')

# Limitar o tamanho do texto nas colunas "DescricaoProb" e "ComoReproduzir" para 20 caracteres
df['DescricaoProb'] = df['DescricaoProb'].apply(lambda x: x if len(x) <= 20 else x[:17] + '...')
df['ComoReproduzir'] = df['ComoReproduzir'].apply(lambda x: x if len(x) <= 20 else x[:17] + '...')

# Agora vamos salvar este DataFrame limpo em um novo arquivo CSV
df.to_csv(r'C:\Users\Miguel\PycharmProjects\lpiproject2024\novoupdate.csv', sep=';', index=False, encoding='utf-8')
