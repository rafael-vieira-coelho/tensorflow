import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 3:

O que acontece quando você remove a chamada Flatten()? Por que você acha que acontece isto?

RESPOSTA:
Você obteve um erro sobre o formato (shape) dos dados. Ou seja, a camada de entrada deve ter
o mesmo formato que os seus dados. Como estamos trabalhando com imagens de 28x28, para não
termos que usar 28 camadas com 28 neurônios cada, usamos uma camada flatt que achata os dados
em uma 784x1. Quando os arrays são carregados no modelo, eles são achatados para nós.
'''


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([  # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
