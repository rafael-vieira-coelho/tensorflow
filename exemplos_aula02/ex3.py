import tensorflow as tf
import keras


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images)
    print(train_labels)
    print(test_images)
    print(test_labels)
    
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu), #hidden layer
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


if __name__ == "__main__":
    main()
