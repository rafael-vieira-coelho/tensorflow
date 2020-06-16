import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    train_dir = './train_images/'
    rock_dir = os.path.join(train_dir + 'rock')
    paper_dir = os.path.join(train_dir + 'paper')
    scissors_dir = os.path.join(train_dir + 'scissors')

    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

    rock_files = os.listdir(rock_dir)
    print(rock_files[:10])
    paper_files = os.listdir(paper_dir)
    print(paper_files[:10])
    scissors_files = os.listdir(scissors_dir)
    print(scissors_files[:10])
    pic_index = 2

    next_rock = [os.path.join(rock_dir, file_name)
                 for file_name in rock_files[pic_index - 2:pic_index]]
    next_paper = [os.path.join(paper_dir, file_name)
                  for file_name in paper_files[pic_index - 2:pic_index]]
    next_scissors = [os.path.join(scissors_dir, file_name)
                     for file_name in scissors_files[pic_index - 2:pic_index]]

    for i, img_path in enumerate(next_rock + next_paper + next_scissors):
        print(img_path)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('Off')
        plt.show()


if __name__ == '__main__':
    main()
