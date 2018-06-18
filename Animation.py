#gif.py
import sys
import datetime
import imageio
import os

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":

    duration = 1
    filenames = ["Test_cases/blue_circle.png","Test_cases/Arkadi.png"]


    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)

    create_gif(filenames, duration)


try:
    module_path = os.path.dirname(__file__)
    folder_path = os.path.join(module_path,"Images")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
except Exception as e:
    print(e)