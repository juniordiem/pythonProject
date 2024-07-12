## Etapa 2: Importando as bibliotecas e a base de dados

import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

tf.__version__

## Etapa 3: Pré-processamento



### Carregando a base de dados

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train

X_train[0]

y_train

y_train[0]


### Normalizando as imagens

##Dividimos cada pixel das imagens das bases de treinamento e teste, utilizando o maior valor que é 255

##Com isso, cada pixel estará na faixa entre 0 e 1. Dessa forma, a rede neural vai treinar mais rápida

X_train = X_train / 255.0

X_test = X_test / 255.0

X_train[0]

### Remodelando (reshaping) a base de dados

##Como estamos trabalhando com uma rede neural densa, mudamos a dimensão das bases de dados para ficarem no formato de vetor

X_train.shape

# Como a dimensão de cada imagem é 28x28, mudamos toda a base de dados para o formato [-1 (todos os elementos), altura * largura]
X_train = X_train.reshape(-1, 28*28)

X_train.shape

X_train[0]

# Mudamos também a dimensão da base de teste
X_test = X_test.reshape(-1, 28*28)

X_test.shape

## Etapa 4: Construindo a Rede Neural Artificial

### Definindo o modelo

##Definimos um objeto do tipo Sequential (sequência de camadas)

model = tf.keras.models.Sequential()

model

### Adicionando a primeira camada densa (fully-connected)

##Hyper-parâmetros da camada:
##- número de units/neurônios: 128
##- função de ativação: ReLU
##- input_shape (camada de entrada): (784, )



model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Adicionando Dropout

##Dropout é uma técnica de regularização na qual alguns neurônios da camada tem seu valor mudado para zero, ou seja, durante o treinamento esses neurônios não serão atualizados. Com isso, temos menos chances de ocorrer overfitting

model.add(tf.keras.layers.Dropout(0.2))

### Adicionando a camada de saída

##- units: número de classes (10 na base de dados Fashion MNIST)
##- função de ativação: softmax

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

### Compilando o modelo

##- Optimizer (otimizador): Adam
##- Loss (função de erro): Sparse softmax (categorical) crossentropy

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()

### Treinando o modelo

model.fit(X_train, y_train, epochs=5)

### Avaliação do modelo e previsão

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

print(f'Test Loss: {test_loss}')
