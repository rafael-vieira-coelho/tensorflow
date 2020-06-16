import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 7:

Antes de começar o treinamento, os dados foram normalizados de valores entre 0-255 para 
valores entre 0-1. 

a) Qual seria o impacto de remover esta normalização?

b) Por que você acha que os resultados são diferentes?
'''


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # training_images = training_images / 255.0
    # test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
