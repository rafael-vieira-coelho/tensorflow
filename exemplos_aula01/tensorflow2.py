from tensorflow import *
import numpy as np

'''
Problem:
Y = 2 * X - 1
'''


def main():
    # cria o modelo sequencial com 1 camada (1 neurônio) e 1 dado de entrada
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) #Dense representa uma camada de neurônios conectados
    model.compile(optimizer='sgd', loss='mean_squared_error') #define a forma como vai ser calculado o erro (médio quadrático)

    # inicializa os dados de entrada como dois arrays
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # define o modelo (exemplos e épocas) para o treinamento
    model.fit(xs, ys, epochs=500)

    # faz a predição do modelo para o valor de entrada 10
    print(model.predict([10.0]))


if __name__ == '__main__':
    main()
