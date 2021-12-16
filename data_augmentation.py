import os
import random
from scipy import ndarray
from PIL import Image

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

# dictionary of the transformations 
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

# change with your own data path
folder_path = '/Users/dyanidillard/Desktop/DarkSkinLesions/test' 
augmented_path = '/Users/dyanidillard/Desktop/dark_malignant_aug'
num_files_desired = 500

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        try:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

            new_file_path = '%s/dark_malignant_aug6_%s.jpg' % (augmented_path, num_generated_files)

            # write image 
            io.imsave(new_file_path, transformed_image)
        except (IOError, ValueError) as e:
            pass
            #print('Bad file:', image_path)
    num_generated_files += 1