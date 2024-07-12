## Etapa 1: Instando o TensorFlow 2.0
#pip uninstall -y tensorflow #Comando necessário, pois o TensorFlow-gpu não desinstala a versão mais recente do Tensorflow, pode gerar conflitos.

#pip install tensorflow-gpu==2.0

## Etapa 2: Importando as bibliotecas

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

%matplotlib inline
tf.__version__

## Etapa 3: Pré-processamento

### Carregando a base de dados Cifar10

# Configurando o nome das classes que serão previstas
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Carregando a base de dados
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

### Normalização das imagens

X_train[0]

X_train = X_train / 255.0

X_train.shape

X_test = X_test / 255.0

plt.imshow(X_test[1])

## Etapa 4: Construindo a Rede Neural Convolucional

### Definindo o modelo

model = tf.keras.models.Sequential()

### Adicionado a primeira camada de convolução

#Hyper-parâmetros da camada de convolução:
# filters (filtros): 32
# kernel_size (tamanho do kernel): 3
# padding (preenchimento): same
# função de ativação: relu
# input_shape (camada de entrada): (32, 32, 3)


model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

### Adicionando a segunda camada de convolução e a camada de max-pooling

#Hyper-parâmetros da camada de convolução:
# filters (filtros): 32
# kernel_size (tamanho do kernel):3
# padding (preenchimento): same
# função de ativação: relu

#Hyper-parâmetros da camada de max-pooling:
# pool_size: 2
# strides: 2
# padding: valid

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

### Adicionando a terceira camada de convolução

#Hyper-parâmetros da camada de convolução:

#   filters: 64
#  kernel_size:3
# padding: same
#activation: relu


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

###  Adicionando a quarta camada de convolução e a camada de max pooling

#Hyper-parâmetros da camada de convolução:

#    filters: 64
#    kernel_size:3
#    padding: same
#    activation: relu

#Hyper-parâmetros da camada de max pooling:

#    pool_size: 2
#    strides: 2
#    padding: valid


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

### Adicionando a camada de flattening

model.add(tf.keras.layers.Flatten())

### Adicionando a primeira camada densa (fully-connected)

#Hyper-parâmetros da camada densa:
#- units/neurônios: 128
#- função de ativação: relu

model.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Adicionando a camada de saída

#Hyper-parâmetros da camada de saída:

# - units/neurônios: 10 (número de classes)
# - activation: softmax


model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

### Compilando o modelo

#### sparse_categorical_accuracy

#https://stackoverflow.com/questions/44477489/keras-difference-between-categorical-accuracy-and-sparse-categorical-accuracy

# 0 0 0 1 0 0 0 0 0 0
y_test[0]

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

### Treinando o modelo

model.fit(X_train, y_train, epochs=5)

### Avaliando o modelo

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

test_loss