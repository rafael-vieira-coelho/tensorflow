import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_model():
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
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def main():
    # 0 - constants
    image_width = 150
    image_height = 150
    base_dir = './train_images_big_data'
    model_name = "model_big_data.h5"
    use_previous_model = False
    validation_dir = os.path.join(base_dir, 'validation')
    train_dir = os.path.join(base_dir, 'train')

    if use_previous_model:
        model = create_model()
        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # save the model
        model.load_weights(model_name)
        model.summary()
    else:
        # 1 - loading images
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
        model = create_model()
        model.summary()
        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # save the model
        model.save(model_name)

        # 3 - Preprocessing the data (preparing)
        # All images will be rescaled by 1./255.
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        # --------------------
        # Flow training images in batches of 20 using train_datagen generator
        # --------------------
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(image_width, image_height))
        # --------------------
        # Flow validation images in batches of 20 using test_datagen generator
        # --------------------
        validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(image_width, image_height))
        # 4 - train the network
        history = model.fit(train_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=100,
                            epochs=15,
                            validation_steps=50,
                            verbose=2)
        print(history)
        # -----------------------------------------------------------
        # Retrieve a list of list results on training and test data
        # sets for each training epoch
        # -----------------------------------------------------------
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))  # Get number of epochs
        # ------------------------------------------------
        # Plot training and validation accuracy per epoch
        # ------------------------------------------------
        plt.plot(epochs, acc, color='b')
        plt.plot(epochs, val_acc, color='r')
        plt.title('Training and validation accuracy')
        plt.show()
        # ------------------------------------------------
        # Plot training and validation loss per epoch
        # ------------------------------------------------
        plt.plot(epochs, loss, color='b')
        plt.plot(epochs, val_loss, color='r')
        plt.title('Training and validation loss')
        plt.show()


if __name__ == '__main__':
    main()
