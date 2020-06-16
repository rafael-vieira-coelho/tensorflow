'''
Below is code with a link to a happy or sad dataset which contains 80 images,
40 happy and 40 sad. Create a convolutional neural network that trains to 100% accuracy
on these images, which cancels training upon hitting training accuracy of >.999

https://jibmpxqxwljkrghcypuuvc.coursera-apps.org/notebooks/week4/Exercise4-Question.ipynb

'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from keras.preprocessing import image
import os
from keras.preprocessing.image import load_img
from PIL import Image

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def main():
    IMAGE_SIZE = 150
    callbacks = MyCallback()
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = "./train_images"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=10,
        class_mode='binary'
    )
    validation_dir = "./test_images"
    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=4,
        class_mode='binary'
    )
    # This Code Block should Define and Compile the Model
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 256 neuron hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('sad') and 1 for the other ('happy')
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
        verbose=1,
        callbacks=[callbacks])

    path = './test_images/'
    files = [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(path)
             for f in filenames
             if os.path.splitext(f)[1] == '.png']
    for fn in files:
        print(fn)
        img = load_img(fn, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)
        print(classes[0])
        if classes[0] > 0.5:
            print(fn + " is sad")
        else:
            print(fn + " is happy")


if __name__ == '__main__':
    main()
