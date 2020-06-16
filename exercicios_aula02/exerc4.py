import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 4:

a) Considere a camada final (saída). Por que temos 10 neurônios na saída?

b) O que acontece se modificarmos para 5 neurônios na saída?

RESPOSTA:
Você obtem um erro pois o número de neurônios na última camada deve ser igual ao número de
casos (classes) aos quais os dados de entrada podem ser classificados. No exemplo, temos
0 a 9 digitos possíveis de saída (cada um representando uma classe de objeto), precisamos
de 10 neurônios na camada de saída.
'''


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
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
