import numpy as np
import os
from keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import load_img
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def main():
    training_dir = "./train_images/"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )
    validation_dir = "./test_images/"
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                        validation_steps=3)
    # predicting images
    test_dir = './validation_images/'
    files = [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(test_dir)
             for f in filenames
             if os.path.splitext(f)[1] == '.png']
    print(files)
    rock = 0
    paper = 0
    scissor = 0
    for fn in files:
        img = load_img(fn, target_size=(150, 150))
        y = image.img_to_array(img)
        x = np.expand_dims(y, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes[0])
        # classes[0][0] = paper
        # classes[0][1] = rock
        # classes[0][2] = scissor
        if classes[0][0] > classes[0][1] and classes[0][0] > classes[0][2]:
            print(fn + " is a paper")
            paper += 1
        elif classes[0][1] > classes[0][0] and classes[0][1] > classes[0][2]:
            print(fn + " is a rock")
            rock += 1
        else:
            print(fn + " is a scissor")
            scissor += 1
    print('TOTAL PAPER: ', paper)
    print('TOTAL ROCK: ', rock)
    print('TOTAL SCISSOR: ', scissor)
    # 6 - plot results
    plt.tight_layout()
    plt.show()  # show previous figure
    plt.title("Prediction of rock, paper or scissor")
    plt.bar(['paper (0)'], paper, color='b')
    plt.bar(['rock (1)'], rock, color='r')
    plt.bar(['scissor (2)'], scissor, color='g')
    plt.show()


if __name__ == '__main__':
    main()
