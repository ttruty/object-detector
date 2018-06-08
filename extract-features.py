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
    # file structure for feature ies
    #path = r'\training_mmse_pentagons' #change to the base dir for files
    path = os.getcwd()
    pos_im_path = os.path.join(path, 'pos_img')
    neg_im_path = os.path.join(path, 'neg_img')
    pos_feat_ph = os.path.join(path, "pos_feat_ph")
    neg_feat_ph = os.path.join(path, "neg_feat_ph")
    #parameters for HOG
    #reference: http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)
    des_type = "HOG"
    if not os.path.isdir(pos_im_path):
        os.makedirs(pos_im_path)
        print("Positive image path was created add images in order to create features")

    if not os.path.isdir(neg_im_path):
        os.makedirs(neg_im_path)
        print("Negative image path was created, add images in order to create features")

    if os.listdir(pos_im_path) == [] or os.listdir(neg_im_path) == []:
        print("<<-- Missing image files in image path, please add images to create features -->>")
    else:
        # If feature directories don't exist, create them
        if not os.path.isdir(pos_feat_ph):
            os.makedirs(pos_feat_ph)
        # If feature directories don't exist, create them
        if not os.path.isdir(neg_feat_ph):
            os.makedirs(neg_feat_ph)
        print("Calculating the descriptors for the positive samples and saving them")
        for im_path in glob.glob(os.path.join(pos_im_path,  "*")):
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (128,128))
            if des_type == "HOG":
                fd, _ = hog(im, orientations, pixels_per_cell, cells_per_block, visualise = True, transform_sqrt=True)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph, fd_name)
            joblib.dump(fd, fd_path) #dumps feature detection in the save path
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
