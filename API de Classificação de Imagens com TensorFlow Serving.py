## Etapa 1: Instalação das bibliotecas

# echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

# Instalar TF Server 2.8.0
# wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-universal-2.8.0/t/tensorflow-model-server-universal/tensorflow-model-server-universal_2.8.0_all.deb'
# dpkg -i tensorflow-model-server-universal_2.8.0_all.deb

# pip install requests

## Etapa 2: Importação das bibliotecas

import os
import json
import random
import requests
import subprocess
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

%matplotlib inline
tf.__version__

## Etapa 3: Pré-processamento

### Carregando a base de dados

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

### Normalização das imagens

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train.shape

X_test.shape

y_test

## Etapa 4: Definição do modelo

NOTA: Estamos usando o mesmo modelo da seção sobre Redes Neurais Convolucionais

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

### Compilando o modelo

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

### Treinando o modelo

model.fit(X_train,
          y_train,
          batch_size=128,
          epochs=10)

### Avaliação do modelo

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy is {}".format(test_accuracy))

## Etapa 5: Salvando o modelo para produção

### Criando o diretório para o modelo

model_dir = "model/"
version = 1

export_path = os.path.join(model_dir, str(version))
export_path

if os.path.isdir(export_path):
  !rm -r {export_path}

### Salvando o modelo para o TensorFlow Serving

model.input

model.outputs

o = {t.name: t for t in model.outputs}

o

tf.saved_model.simple_save(tf.keras.backend.get_session(), export_dir = export_path,
                           inputs = {"input_image": model.input},
                           outputs = {t.name: t for t in model.outputs}) #Função simple_save não mais compatível com > Tensorflow 2.0, necessário comando tensorflow.compat.v1 no import

## Etapa 6: Configuração do ambiente de produção

### Exportando o MODEL_DIR para as variáveis de ambiente

os.environ["model_dir"] = os.path.abspath(model_dir)

### Executando a API TensorFlow Serving REST

%%bash --bg
nohup tensorflow_model_server --rest_api_port=8501 --model_name=cifar10 --model_base_path="${model_dir}" >server.log 2>&1

tail server.log

## Etapa 7: Criando a primeira requisição POST

random_image = np.random.randint(0, len(X_test))

random_image

### Criando o objeto JSON

data = json.dumps({"signature_name": "serving_default", "instances": [X_test[random_image].tolist()]})

data

### Enviando a primeira requisição POST para o modelo

headers = {"content-type": "application/json"}

json_response = requests.post(url="http://localhost:8501/v1/models/cifar10:predict", data = data, headers = headers)

json_response

predictions = json.loads(json_response.text)['predictions']

predictions

plt.imshow(X_test[random_image])

class_names[np.argmax(predictions[0])]

## Etapa 8: Enviando a requisição POST para um modelo específico que está armazendo no servidor

specific_json_response = requests.post(url="http://localhost:8501/v1/models/cifar10/versions/1:predict", data = data, headers = headers)

specific_json_response