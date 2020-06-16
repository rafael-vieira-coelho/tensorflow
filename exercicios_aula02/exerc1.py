'''
Exercício 1:

a) Inicialmente, rode o código abaixo:

classifications = model.predict(test_images)
print(classifications[0])

Neste código, é criado um conjunto de classificações para cada imagem de teste,
e depois imprime a primeira entrada das classificações.
A saída, ao final da execução, é uma lista de números.

b) Por que você acha que isto acontece e o que esses números representam?

DICA: tente rodar print(test_labels[0]) e você obterá um 9.
Isto ajuda você a entender o porquê da lista ser do jeito que é?

c) O que a lista representa?
opção 1) São 10 valores randômicos
opção 2) São as 10 primeiras classificações que o computador fez
opção 3) É a probabilidade que It's the probability that this item is each of the 10 classes

RESPOSTA:
A opção correta é a 3.
Os 10 números são a probabilidade de que o valor classificado é o vavlor correspondente.

d) Como você sabe que esta lista diz que o item é um ankle boot?

opção 1) Não existe informação suficiente para responder esta questão.
opção 2) O décimo elemento da lista é o maior elemento. E o ankle boot tem o label 9.
opção 3) O label de ankle boot é 9 e existem 0->9 elementos na lista.

RESPOSTA:
A opção correta é a 2. A rede neural fez a predição de que o último elemento (label 9)
é o mais provável, ou seja, um ankle boot.
'''

import tensorflow as tf
import keras


def main():
    print(tf.__version__)

    # carrega os dados do BD de Keras (já dividindo em casos de teste e treino)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # mostra a representação da matriz com os tons de cores (0 a 255) para cada pixel.
    # Teste com outro valor do array (ex: 42, ao invés de 0)!
    print(train_labels[0])
    print(train_images[0])
    # normalizando os valores para 0 e 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # define a arquitetura da rede neural com 3 camadas
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # constrói o modelo da rede neural definido antes
    # define-se um otimizador, uma função de perda e métricas
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # treina a rede neural
    # define-se o número de épocas desejada
    # no final do treinamento, deve-se ver uma acurácia.
    # Ex: 0.9098 representa 91% de certeza que a imagem vai ser classsificada corretamente.
    model.fit(train_images, train_labels, epochs=5)
    # testa a rede neural
    # usa dados que a rede ainda não viu para testar sua acurácia
    # no meu caso, retornou 0.8838, ou seja, 88% de acurácia.
    model.evaluate(test_images, test_labels)

    #Exercício 1a
    classifications = model.predict(test_images)
    print(classifications[0])
    #Exercício 1b
    print(test_labels[0])


if __name__ == "__main__":
    main()

