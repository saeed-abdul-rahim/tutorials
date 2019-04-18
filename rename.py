import os
import glob
import uuid
import random

# Set the working directory where the images exist.
dir_path = os.path.dirname("/storage/9016-4EF8/auto_message/")
os.chdir(dir_path)

img_list = []

for file in glob.glob("*.jpg"):
    os.rename(file, uuid.uuid4().hex+".jpg")

for file in glob.glob("*.jpg"):
    img_list.append(file)

# Selects the random image to be the new “message1.jpg”
rand_img = random.choice(img_list)
os.rename(rand_img, "message1.jpg")
