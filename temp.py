# import os

# print(len(os.listdir("data/train/fake_images")))
# print(len(os.listdir("data/train/real_images")))
from glob import glob
X_test = glob(f"data/test/images/*.png")

print(list(X_test)[:5])