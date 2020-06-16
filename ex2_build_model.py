import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


def main():
    # 0 - constants
    image_width = 150
    image_height = 150
    base_dir = './train_images_big_data'

    # 1 - loading images
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    # Directory with our training cat/dog pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    # Directory with our validation cat/dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)
    print(train_cat_fnames[:10])
    print(train_dog_fnames[:10])
    print('total training cat images :', len(os.listdir(train_cats_dir)))
    print('total training dog images :', len(os.listdir(train_dogs_dir)))
    print('total validation cat images :', len(os.listdir(validation_cats_dir)))
    print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

    # 2 - building the model (neural network)
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats')
        # and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # save the model
    model.save_weights("model.h5")


if __name__ == '__main__':
    main()
