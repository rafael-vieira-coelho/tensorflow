import tensorflow as tf
import keras.datasets.mnist as mnist
'''
Exercício 5:

Agora vamos considerar adicionar novas camadas à rede. O que acontecerá se 
adicionarmos uma camada entre a que tem 512 neurônios e a camada final que tem 10?

RESPOSTA: Não existe um impacto significativo pois se tratam de dados simples. Com exemplos
mais complexos como, por exemplo, classificação de imagens como flores que veremos na 
próxima aula, precisaremos de mais camadas.
'''


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu),
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
