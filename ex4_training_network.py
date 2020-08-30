from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from keras.preprocessing import image
import os
from keras.preprocessing.image import load_img


def main():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = "./train_images"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )
    validation_dir = "./train_images"
    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=8,
        verbose=1)

    # Let's now take a look at actually running a prediction using the model.
    # This code will allow you to choose 1 or more files from your file system,
    # it will then upload them, and run them through the model, giving an indication
    # of whether the object is a horse or a human.
    path = './test_images/'
    files = [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(path)
             for f in filenames
             if os.path.splitext(f)[1] == '.png']
    # for every file on images folder
    for fn in files:
        print(fn)
        # load the image
        img = load_img(fn, target_size=(300, 300))
        # predicting images
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes[0])
        if classes[0] > 0.5:
            print(fn + " is a human")
        else:
            print(fn + " is a horse")


if __name__ == '__main__':
    main()
