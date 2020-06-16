'''
1. Tente editar as camadas de convulação. Troque de 32 para 16 ou 64. Qual o impacto isto terá no tempo de treinamento ou acurácia?
2. Remova a camada de convulação final. Qual o impacto isto terá no tempo de treinamento ou acurácia?
3. E que tal adicionar mais convulações? Quais os impactos?
4. Remova todas as camadas de convulação, exceto a primeira. Quais os impactos?
5. Na aula passada, implementamos um callback para testar a taxa de erro e cancelar o treinamento com um determinado valor. Tente implementar isto aqui.
'''

import tensorflow as tf
import tensorflow.keras.datasets.fashion_mnist as mnist

'''
Callback é uma forma de testar se ao final de cada época o treinamento deve terminar.
'''


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.8:
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True


def main():
    callbacks = MyCallback()
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=5,  callbacks=[callbacks])
    test_loss = model.evaluate(test_images, test_labels)
    print('Error: {:02.2f}%'.format(float(test_loss[0] * 100)))
    print('Accuracy: {:02.2f}%'.format(float(test_loss[1] * 100)))


if __name__ == '__main__':
    main()
