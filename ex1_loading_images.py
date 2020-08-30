from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    '''
     wget --no-check-certificate \
         https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
         -O /tmp/horse-or-human.zip
    '''
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = "./train_images"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )
    print(train_generator)


if __name__ == '__main__':
    main()
