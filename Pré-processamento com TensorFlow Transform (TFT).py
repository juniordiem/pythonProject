## Etapa 1: Instalação das bibliotecas

# pip install tensorflow - transform

## Etapa 2: Importação das bibliotecas

import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam

import apache_beam.io.iobase  # Adicionado novo import

from __future__ import print_function
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema, schema_utils  # Adicionado schema_utils

## Etapa 3: Pré-processamento

### Carregando a base de dados

from google.colab import files

uploaded = files.upload()

dataset = pd.read_csv("pollution-small.csv")

dataset.head()

### Apagando a coluna com a data

features = dataset.drop("Date", axis=1)

features.head()

### Conversão da base de dados (dataframe) para dicionário

Mais
sobre
dicionários: https: // iaexpert.com.br / index.php / 2019 / 0
9 / 12 / dicionarios - em - python /

dict_features = list(features.to_dict("index").values())

dict_features[0:2]

### Definição dos metadados

data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec({
    "no2": tf.io.FixedLenFeature([], tf.float32),
    "pm10": tf.io.FixedLenFeature([], tf.float32),
    "so2": tf.io.FixedLenFeature([], tf.float32),
    "soot": tf.io.FixedLenFeature([], tf.float32),
}))

data_metadata


## Etapa 4: Função para pré-processamento

def preprocessing_fn(inputs):
    no2 = inputs["no2"]
    pm10 = inputs["pm10"]
    so2 = inputs["so2"]
    soot = inputs["soot"]

    no2_normalized = no2 - tft.mean(no2)
    so2_normalized = so2 - tft.mean(so2)

    pm10_normalized = tft.scale_to_0_1(pm10)
    soot_normalized = tft.scale_by_min_max(soot)

    return {
        "no2_normalized": no2_normalized,
        "so2_normalized": so2_normalized,
        "pm10_normalized": pm10_normalized,
        "sott_normalized": soot_normalized
    }


## Etapa 5: Unindo a codificação

O
Tensorflow
Transform
usa
o ** Apache
Beam ** como
background
para
realizar as operações.

Parâmetros
a
serem
passados
para
a
função:

dict_features - Nossa
base
de
dados
convertida
para
dicionário
data_metadata - Nossos
metadados
que
criamos
anteriormente
preprocessing_fn - Função
de
pré - processamento
que
fará as transformações
coluna
por
coluna

Abaixo
temos
a
sintaxe
usada
pelo
Apache
Beam

```
result = data_to_pass | where_to_pass_the_data
```

Explicando
cada
um
dos
parâmetros:

** result **  -> `transformed_dataset, transform_fn`

** data_to_pass ** -> `(dict_features, data_metadata)`

** where_to_pass_the_data ** -> `tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)`

```
transformed_dataset, transform_fn = (
            (dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

```

Mais
sobre
essa
sintaxe:
https: // beam.apache.org / documentation / programming - guide /  # applying-transforms

LINKS:
> Mais
sobre
o
Apache
Beam: https: // beam.apache.org /


def data_transform():
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (
                    (dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

    transformed_data, transformed_metadata = transformed_dataset

    for i in range(len(transformed_data)):
        print("Initial: ", dict_features[i])
        print("Transformed: ", transformed_data[i])


data_transform()