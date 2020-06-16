# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
Label	Classe
0	Camisetas/Top (T-shirt/top)
1	Calça (Trouser)
2	Suéter (Pullover)
3	Vestidos (Dress)
4	Casaco (Coat)
5	Sandálias (Sandal)
6	Camisas (Shirt)
7	Tênis (Sneaker)
8	Bolsa (Bag)
9	Botas (Ankle boot)
'''


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Grab an image from the test dataset.
    img = test_images[0]
    print(img.shape)
    # Adiciona a imagem em um batch que possui um só membro.
    img = (np.expand_dims(img, 0))
    print(img.shape)
    # Agora prediremos a label correta para essa imagem:
    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    # model.predict retorna a lista de listas — uma lista para cada imagem em um batch de dados.
    # Pegue a predição de nossa (única) imagem no batch:
    np.argmax(predictions_single[0])


if __name__ == "__main__":
    main()
