import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


def load_images(folder: str, categories, filenames):
    filenames2 = os.listdir(folder)
    for filename in filenames2:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)
        filenames.append(filename)


def main():
    # 0 - constants
    image_width = 150
    image_height = 150
    base_dir = './train_images_big_data'

    # 1 - loading data
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

    # show the images quantity
    categories = []
    filenames = []
    load_images(train_dogs_dir, categories, filenames)
    load_images(train_cats_dir, categories, filenames)
    #    load_images(validation_cats_dir, categories, filenames)
    #    load_images(validation_dogs_dir, categories, filenames)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    print(df.head())
    plt.bar(['cats (0)'], df['category'].value_counts()[0], color='r')
    plt.bar(['dogs (1)'], df['category'].value_counts()[1], color='b')
    plt.show()

    # show images
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    pic_index = 0  # Index for iterating over images
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]]
    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
