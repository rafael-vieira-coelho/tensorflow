from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def main():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = "./train_images"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )
    # Directory with our training horse pictures
    train_horse_dir = os.path.join('./train_images/horses')
    # Directory with our training human pictures
    train_human_dir = os.path.join('./train_images/humans')
    # Horses Names
    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])
    # Human Names
    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    # Index for iterating over images
    pic_index = 0
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in train_horse_names[pic_index - 8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in train_human_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    main()
