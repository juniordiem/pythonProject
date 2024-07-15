## Etapa 1: Instalação da bibliotecas


!pip uninstall -y tensorflow #Comando necessário, pois o TensorFlow-gpu não desinstala a versão mais recente do Tensorflow, pode gerar conflitos.

!pip install tensorflow==2.8.0 # Atualizado 14/03/2022 Algumas bibliotecas ainda não tem compatibilidade com versões acima, favor utilizar somente essa versão.

!pip install -q tensorflow-gpu==2.8.0 # Atualizado 14/03/2022 Algumas bibliotecas ainda não tem compatibilidade com versões acima, favor utilizar somente essa versão.

!pip install pandas-datareader

## Etapa 2: Importação das bibliotecas

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
from pandas.util.testing import assert_frame_equal #import alterado

from tqdm import tqdm_notebook, tqdm
from collections import deque

tf.__version__

## Etapa 3: Construção da IA para negociação de ações

class AI_Trader():

  def __init__(self, state_size, action_space = 3, model_name = "AITrader"):
    self.state_size = state_size
    self.action_space = action_space
    self.memory = deque(maxlen = 2000)
    self.model_name = model_name

    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_final = 0.01
    self.epsilon_decay = 0.995
    self.model = self.model_builder()

  def model_builder(self):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(self.state_size,)))
    model.add(tf.keras.layers.Dense(units = 32, activation = "relu"))
    model.add(tf.keras.layers.Dense(units = 64, activation = "relu"))
    model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
    model.add(tf.keras.layers.Dense(units = self.action_space, activation = "linear"))
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr = 0.001))
    return model


  def trade(self, state):
    if random.random() <= self.epsilon:
      return random.randrange(self.action_space)

    actions = self.model.predict(state[0]) #Atualizado 12/12/2022
    return np.argmax(actions[0])

  def batch_train(self, batch_size):
    batch = []
    for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
      batch.append(self.memory[i])

    for state, action, reward, next_state, done in batch:
      if not done:
        reward = reward + self.gamma * np.amax(self.model.predict(next_state[0]))

      target = self.model.predict(state[0]) #Atualizado 12/12/2022
      target[0][action] = reward

      self.model.fit(state[0], target, epochs=1, verbose=0) #Atualizado 12/12/2022

    if self.epsilon > self.epsilon_final:
      self.epsilon *= self.epsilon_decay


## Etapa 4: Pré-processamento da base de dados

### Definição de funções auxiliares

#### Sigmoid

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid(0.5)

#### Formatação de preços

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

stocks_price_format(100)

#### Carregador da base de dados

!pip install -q yfinance

import yfinance as yf

#dataset = data_reader.DataReader("AAPL", data_source = "yahoo")
dataset = yf.download("AAPL", start='2016-06-02')

dataset.head()

str(dataset.index[0]).split()[0]

dataset.index[-1]

dataset['Close']

def dataset_loader(stock_name):
  #dataset = data_reader.DataReader(stock_name, data_source = "yahoo")
  dataset = yf.download(stock_name, start='2016-06-02')
  start_date = str(dataset.index[0]).split()[0]
  end_date = str(dataset.index[-1]).split()[0]
  close = dataset['Close']
  return close

### Criador de estados

0 - 5 + 1

20 - 5 + 1

dataset[16:21]

def state_creator(data, timestep, window_size):
  starting_id = timestep - window_size + 1

  if starting_id >= 0:
    # windowed_data = data[starting_id:timestep + 1] # Atualizado 14/03/2022
    windowed_data = np.array(data[starting_id:timestep + 1]) # Atualizado 14/03/2022
  else:
    # windowed_data = - starting_id * [data[0]] + list(data[0:timestep + 1]) # Atualizado 14/03/2022
    windowed_data = np.array(- starting_id * [data[0]] + list(data[0:timestep + 1])) # Atualizado 14/03/2022

  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

  return np.array([state]), windowed_data

### Carregando a base de dados

stock_name = "AAPL"
data = dataset_loader(stock_name)

s, w = state_creator(data, 0, 5)

s

w

## Etapa 5: Treinando a IA

### Configuração dos hyper parâmetros

window_size = 10
episodes = 1000
batch_size = 32
data_samples = len(data) - 1

data_samples

### Definição do modelo

trader = AI_Trader(window_size)

trader.model.summary()

### Loop de treinamento

for episode in range(1, episodes + 1):
  print("Episode: {}/{}".format(episode, episodes))
  state = state_creator(data, 0, window_size + 1)
  total_profit = 0
  trader.inventory = []
  for t in tqdm(range(data_samples)):
    action = trader.trade(state)
    next_state = state_creator(data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # Comprando uma ação
      trader.inventory.append(data[t])
      print("AI Trader bought: ", stocks_price_format(data[t]))
    elif action == 2 and len(trader.inventory) > 0: # Vendendo uma ação
      buy_price = trader.inventory.pop(0)

      reward = max(data[t] - buy_price, 0)
      total_profit += data[t] - buy_price
      print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))

    if t == data_samples - 1:
      done = True - deploy
    else:
      done = False

    trader.memory.append((state, action, reward, next_state, done))

    state = next_state

    if done:
      print("########################")
      print("Total profit: {}".format(total_profit))
      print("########################")

    if len(trader.memory) > batch_size:
      trader.batch_train(batch_size)

  if episode % 10 == 0:
    trader.model.save("ai_trader_{}.h5".format(episode))
