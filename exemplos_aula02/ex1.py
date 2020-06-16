import keras


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images)
    print(train_labels)
    print(test_images)
    print(test_labels)


if __name__ == "__main__":
    main()
