# Projeto 1: Classificação binária brest cancer

## Etapa 1: Importação das bibliotecas

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
torch.__version__
#!pip install torch==1.4.0

import torch.nn as nn

## Etapa 2: Base de dados

np.random.seed(123)
torch.manual_seed(123)

previsores = pd.read_csv('/content/entradas_breast.csv')
classe = pd.read_csv('/content/saidas_breast.csv')

previsores.shape

previsores.head()

classe.head()

np.unique(classe)

sns.countplot(classe['0']);

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,
                                                                                              classe,
                                                                                              test_size = 0.25)

previsores_treinamento.shape

classe_treinamento.shape

previsores_teste.shape

classe_teste.shape

## Etapa 3: Transformação dos dados para tensores


type(previsores_treinamento)

type(np.array(previsores_treinamento))

previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype=torch.float)
classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype = torch.float)

type(previsores_treinamento)

type(classe_treinamento)

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)

type(dataset)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

## Etapa 4: Construção do modelo

# 30 -> 16 -> 16 -> 1
# (entradas + saida) / 2 = (30 + 1) / 2 = 16
classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

classificador.parameters

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.0001)

## Etapa 5: Treinamento do modelo

for epoch in range(100):
  running_loss = 0.

  for data in train_loader:
    inputs, labels = data
    #print(inputs)
    #print('-----')
    #print(labels)
    optimizer.zero_grad()

    outputs = classificador(inputs) # classificador.forward(inputs)
    #print(outputs)
    loss = criterion(outputs, labels)
    #print(loss)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))

## Etapa 6: Visualização dos pesos

# 30 -> 16 -> 16 -> 1
params = list(classificador.parameters())

params

# 30 -> 16 -> 16 -> 1
pesos0 = params[0]
pesos0.shape

print(pesos0)

# 30 -> 16 -> 16 -> 1
bias0 = params[1]
bias0.shape

pesos1 = params[2]
pesos1.shape

bias1 = params[3]
bias1.shape

## Etapa 7: Avaliação do modelo

classificador.eval()

type(previsores_teste)

previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)

type(previsores_teste)

previsoes = classificador.forward(previsores_teste)

previsoes

previsoes = np.array(previsoes > 0.5)
previsoes

classe_teste

taxa_acerto = accuracy_score(classe_teste, previsoes)
taxa_acerto

matriz = confusion_matrix(classe_teste, previsoes)
matriz

sns.heatmap(matriz, annot=True);