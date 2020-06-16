import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 6:

Agora vamos considerar treinar a rede por mais ou menos épocas. Tente 5, 15 e 30 épocas.
O que você acha que irá ocorrer em cada um dos casos? 

RESPOSTA:
Com 15 épocas você provavelmente obteve um modelo com menos erro do que com 5 ou 30 épocas.
Com 30 ou 5 épocas você deve ter visto que o erro (loss) para de crescer ou diminuir. Isto
se chama 'overfitting' e você ter cuidado para que isto não ocorra em suas redes neurais
Não vale a pena continuar treinando sua rede se o seu erro não melhora.
'''


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=30)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[34])
    print(test_labels[34])


if __name__ == "__main__":
    main()
