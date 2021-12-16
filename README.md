# Deep-Learning-to-Detect-Melanoma-in-Different-Skin-Types

Deep Learning model that detects melanoma amongst dark-skinned and fair-skinned images!

## Reproducing Results

To replicate this project you need to download and install

1. Python version 3.9.1 (found here: https://www.python.org/downloads/)
2. PyTorch version 1.10.0 (found here: https://pytorch.org/get-started/locally/)
3. Anaconda version 4.10.3 (found here: https://docs.anaconda.com/anaconda/install/index.html)
4. Numpy version 1.20.1 (found here: https://numpy.org/install/)
5. Pandas version 1.3.0 (found here: https://pandas.pydata.org/docs/getting_started/install.html)
6. Pillow version 8.3.1 (found here: https://anaconda.org/anaconda/pillow)
7. Scikit-image version 0.18.3 (found here: https://scikit-image.org/docs/dev/install.html)

### Datasets and Python files

Click [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/data/) to download the fair-skinned dataset from the ISIC archive. 

The dark-skinned images were developed via data augmentation techniques and stored in a custom dataset. I strongly encourage manipulating more data to generalize the model's training data! 

Download the python files for data augmentation, image resizing, and for the classification model itself. Most of the parameters and instructions are straight forward - **be sure to change the file paths so the model can read in the images correctly** . 

## Code Overview

* melanomaclassifier.zip
  * Compressed version of the fair skin/dark skin dataset used for training and testing the model.   

* fair_dark_split.csv 
  * CSV file that contains the image names and target labels for the dataset.

* resize_images.py
  * A python file that resizes all of the images in a folder to 224x224 pixels. Resizing is essential especially when using a pre-trained model because they only accept images in this format. 

* data_augmentation.py
  * A python file that generates data from images in a folder. Data augmentation techniques such as horizontal flipping, distortion, and rotating are utilized in this file but more can definitely be added. 

* MelanomaTest1.py
  * The python file that contains the model. Many parameters can be adjusted such as learning rate, batch size, epochs, etc. The pre-trained model can also be adjusted in addition to the evaluation metrics.  
