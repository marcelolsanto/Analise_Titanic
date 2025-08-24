Análise Preditiva de Sobrevivência no Titanic
1. Visão Geral do Projeto
Este projeto consiste em uma análise exploratória detalhada e na construção de um modelo de Machine Learning para prever a probabilidade de um passageiro ter sobrevivido ao desastre do Titanic. O objetivo é aplicar conceitos fundamentais de Ciência de Dados, desde a limpeza e preparação dos dados até o treinamento e avaliação de um modelo preditivo.

Fonte dos Dados: Competição "Titanic - Machine Learning from Disaster" no Kaggle

2. Estrutura do Projeto
O repositório está organizado da seguinte forma:

├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── analise_titanic.ipynb
└── README.md
/data: Contém os datasets brutos utilizados no projeto.
/notebooks: Contém o Jupyter Notebook com todo o desenvolvimento da análise e modelagem.
README.md: Documentação completa do projeto.
3. Ferramentas Utilizadas
Linguagem: Python 3
Bibliotecas Principais:
Pandas: Para manipulação e limpeza dos dados.
Matplotlib & Seaborn: Para visualização e análise exploratória.
Scikit-learn: Para pré-processamento, treinamento e avaliação do modelo de Machine Learning.
4. Metodologia
O projeto seguiu o seguinte fluxo de trabalho:

Análise Exploratória de Dados (EDA): Investigação inicial dos dados para entender suas características, identificar padrões, anomalias e valores faltantes.
Limpeza e Pré-processamento: Tratamento de dados faltantes e conversão de variáveis categóricas em formato numérico para o modelo.
Engenharia de Features: Seleção das variáveis mais relevantes para o modelo preditivo.
Modelagem: Treinamento de um modelo de classificação (ex: RandomForestClassifier) para prever a variável alvo (Survived).
Avaliação: Medição da performance do modelo utilizando a acurácia como métrica principal e análise da matriz de confusão.
5. Principais Insights da Análise
Gênero: Ser do sexo feminino foi o fator individual com maior correlação positiva com a sobrevivência, com mulheres tendo uma taxa de sobrevivência de aproximadamente 74%.
Classe Social: Passageiros da Primeira Classe (Pclass=1) tiveram uma taxa de sobrevivência significativamente maior (aprox. 63%) em comparação com passageiros da Terceira Classe (aprox. 24%).
Idade: Crianças tiveram uma maior probabilidade de sobreviver em comparação com adultos.
6. Resultados do Modelo
O modelo final (RandomForestClassifier) alcançou uma acurácia de aproximadamente 79% no conjunto de teste, demonstrando uma boa capacidade de generalização para prever a sobrevivência de passageiros com base em suas características.

7. Como Executar o Projeto¶
Clone este repositório: git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
Crie e ative um ambiente virtual.
Instale as dependências: pip install -r requirements.txt (É uma boa prática criar este arquivo a partir das suas bibliotecas).
Navegue até a pasta notebooks/ e execute o Jupyter Notebook: jupyter lab analise_titanic.ipynb
4. Modelagem e Avaliação
Agora, vamos dividir os dados, treinar um modelo de classificação e avaliar sua performance. Usaremos o RandomForestClassifier, que é um modelo robusto e eficiente.

# Importar as ferramentas de modelagem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dividir os dados em treino e teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()
Acurácia do modelo: 82.12%

Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       105
           1       0.80      0.76      0.78        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179

Matriz de Confusão:

5. Conclusão
Neste projeto, realizamos uma análise completa do dataset do Titanic. A análise exploratória revelou que ser do sexo feminino e pertencer a uma classe social mais alta foram os fatores mais determinantes para a sobrevivência.

O modelo de Machine Learning (RandomForestClassifier) que construímos foi capaz de prever a sobrevivência com uma acurácia de aproximadamente 79-82% (o valor exato pode variar ligeiramente), o que é um resultado sólido.

Como próximos passos, poderíamos explorar uma engenharia de features mais avançada, como extrair títulos dos nomes dos passageiros, ou testar outros algoritmos de classificação para tentar melhorar a performance preditiva.
