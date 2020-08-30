# example of loading an image with the Keras API
import os
from keras.preprocessing.image import load_img


def main():
    path = './train_images/'
    files = [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(path)
             for f in filenames
             if os.path.splitext(f)[1] == '.png']

    for f in files:
        # load the image
        img = load_img(f)
        # report details about the image
        print(f, type(img), img.mode, img.size)
        # show the image
        # img.show()


if __name__ == '__main__':
    main()
