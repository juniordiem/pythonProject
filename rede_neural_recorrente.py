## Etapa 1: Instalando o TensorFlow 2.0

# pip uninstall -y tensorflow #Comando necessário, pois o TensorFlow-gpu não desinstala a versão mais recente do Tensorflow, pode gerar conflitos.

# pip install tensorflow-gpu==2.0

# pip install numpy==1.17

## Etapa 2: Importação das bibliotecas

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb

tf.__version__

## Etapa 3: Pré-processamento

### Configurando os parâmetros para a base de dados

number_of_words = 20000
max_len = 100

### Carregando a base de dados IMDB

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

X_train.shape

X_train

X_train[0]

Base de dados original com os textos:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

y_train

### Preenchimento das sequências (textos) para terem o mesmo tamanho

len(X_train[0])

len(X_train[1])

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)

len(X_train[0])

len(X_train[1])

X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

## Etapa 4: Construindo a Rede Neural Recorrente

### Definindo o modelo

model = tf.keras.Sequential()

### Adicionando a camada de embedding

X_train.shape[1]

model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))

Embeddings: https://www.tensorflow.org/guide/embedding
Artigo Word Embeddings: https://iaexpert.com.br/index.php/2019/04/12/word-embedding-transformando-palavras-em-numeros/

### Adicionando a camada LSTM

- units: 128
- activation: tanh

model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

### Adicionando a camada de saída

- units: 1
- activation: sigmoid

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

### Compilando o modelo

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

### Treinando o modelo

model.fit(X_train, y_train, epochs=3, batch_size=128)

### Avaliando o modelo

test_loss, test_acurracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_acurracy))

test_loss