import numpy as np

# Criando um tensor 2D (matriz)
x = np.array([[1, -2, 3],
              [-4, 5, -6]])

def naive_relu(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# Aplicando a função naive_relu
result = naive_relu(x)

print(result)
