## Etapa 1: Instalação das dependências

!pip uninstall -y tensorflow #Comando necessário, pois o TensorFlow-gpu não desinstala a versão mais recente do Tensorflow, pode gerar conflitos.

!pip install tensorflow-gpu==2.0

!pip install tqdm

### Fazendo o download da base de dados de gatos e cachorros

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O ./cats_and_dogs_filtered.zip

## Etapa 2: Pré-processamento

### Importação das bibliotecas

import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
tf.__version__

### Descompactando a base de dados de gatos e cachorros

dataset_path = "./cats_and_dogs_filtered.zip"

zip_object = zipfile.ZipFile(file=dataset_path, mode="r")

zip_object.extractall("./")

zip_object.close()

### Configurando os caminhos (paths)

dataset_path_new = "./cats_and_dogs_filtered"

train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

## Construindo o modelo

### Carregando o modelo pré-treinado (MobileNetV2)

img_shape = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape,
                                               include_top = False,
                                               weights = "imagenet")

base_model.summary()

### Congelando o modelo base

base_model.trainable = False

### Definindo o cabeçalho personalizado da rede neural

base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

global_average_layer

prediction_layer = tf.keras.layers.Dense(units = 1, activation = "sigmoid")(global_average_layer)

### Definindo o modelo

model = tf.keras.models.Model(inputs = base_model.input, outputs = prediction_layer)

model.summary()

### Compilando o modelo

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001),
              loss="binary_crossentropy", metrics = ["accuracy"])

### Criando geradores de dados (Data Generators)

Redimensionando as imagens

    Grandes arquiteturas treinadas suportam somente alguns tamanhos pré-definidos.

Por exemplo: MobileNet (que estamos usando) suporta: (96, 96), (128, 128), (160, 160), (192, 192), (224, 224).

data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_train.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

### Treinando o modelo

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

### Avaliação do modelo de transferência de aprendizagem

valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

valid_accuracy

## Fine tuning


Duas questões principais:

- NÃO USE Fine Tuning em toda a rede neural, pois somente em algumas camadas já é suficiente. A ideia é adotar parte específica da rede neural para nosso problema específico
- Inicie o Fine Tuning DEPOIS que você finalizou a transferência de aprendizagem. Se você tentar o Fine Tuning imediatamente, os gradientes serão muito diferentes entre o cabeçalho personalizado e algumas camadas descongeladas do modelo base

### Descongelando algumas camadas do topo do modelo base

base_model.trainable = True
len(base_model.layers)

fine_tuning_at = 100

for layer in base_model.layers[:fine_tuning_at]:
  layer.trainable = False

### Compilando o modelo para fine tuning

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001), loss="binary_crossentropy", metrics=["accuracy"])

### Fine tuning

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

### Avaliação do modelo com fine tuning

valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

valid_accuracy