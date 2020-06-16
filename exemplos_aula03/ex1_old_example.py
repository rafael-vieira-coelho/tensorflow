import tensorflow as tf
import tensorflow.keras.datasets.fashion_mnist as mnist


def main():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)


if __name__ == '__main__':
    main()
