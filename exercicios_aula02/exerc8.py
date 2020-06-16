import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 8:

Utilizaremos um conjunto de dados do MNIST com dados para identificação de escrita (handwriting)
através de dígitos de 0 a 9. Escreva um classificador com base no código abaixo que
treine com pelo menos 99% de acurácia sem um número fixo de épocas. Ou seja,
você deve parar de treinar sua rede quando alcançar esta acurácia.

DICAS:
- Deve ter sucesso em menos de 10 épocas
- Quando se atingir o sucesso no treinamento, deve-se escrever a seguinte mensagem:
"Reached 99% accuracy so cancelling training!"
'''

def main():
    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE

        # YOUR CODE SHOULD END HERE
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE SHOULD START HERE

        # YOUR CODE SHOULD END HERE
    )
    print(history.epoch, history.history['accuracy'][-1])


if __name__ == "__main__":
    main()
