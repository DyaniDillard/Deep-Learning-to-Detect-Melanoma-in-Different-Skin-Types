import PIL
import os
import os.path
from PIL import Image

# change with your own dataset path
f = r'/Users/dyanidillard/Desktop/melanomaclassifier/fair_dark_dataset'
for file in os.listdir(f):
    try:
        f_img = f+"/"+file
        img = Image.open(f_img)
        img = img.resize((224,224)) # converts all images in the file to 224x224
        img.save(f_img)
    except Exception as e:
        pass