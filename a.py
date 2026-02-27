import sqlite3

# Conectar ao banco de dados
conn = sqlite3.connect("aircraft_stability.db")  # Substitua pelo caminho correto do seu arquivo
cursor = conn.cursor()

# Escolha a tabela que deseja visualizar
tabela = "analises de estabilidade"  # Troque pelo nome da tabela desejada

# Buscar os primeiros 10 registros da tabela
cursor.execute(f"SELECT * FROM {tabela} LIMIT 10;")
dados = cursor.fetchall()

# Exibir os dados
for linha in dados:
    print(linha)


import os
print("Banco criado?", os.path.exists("aircraft_stability.db"))

# Fechar conexão
conn.close()
