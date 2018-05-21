# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import glob
import os, cv2
import numpy as np

if __name__ == "__main__":
    path = r'C:\Users\KinectProcessing\Desktop\training_mmse_pentagons'
    pos_im_path = os.path.join(path, 'pos_pent')
    neg_im_path = os.path.join(path, 'neg_pent')
    pos_feat_ph = os.path.join(path, "pos_feat_ph")
    neg_feat_ph = os.path.join(path, "neg_feat_ph")

    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)
    des_type = "HOG"

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)
    print("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        #print(im_path)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (128,128))
        if des_type == "HOG":
            fd, _ = hog(im, orientations, pixels_per_cell, cells_per_block, visualise = True, transform_sqrt=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print("Positive features saved in {}".format(pos_feat_ph))

    print("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (128,128))
        if des_type == "HOG":
            fd, _ = hog(im,  orientations, pixels_per_cell, cells_per_block, visualise = True, transform_sqrt=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print("Negative features saved in {}".format(neg_feat_ph))

    print("Completed calculating features from training images")
