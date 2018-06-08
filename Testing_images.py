import glob
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2, os, time, math, itertools, random, xlwt
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def detection(img, model_path):
    '''
    The detections methond using the HOG SVM algorithm
    '''
    clf = joblib.load(model_path) #prediciton method using SVM
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    im_reshape = cv2.resize(opening, (128,128)) # reduces noise on image
    fd, _ = hog(im_reshape,9, (8,8), (3,3), visualise = True, transform_sqrt=True)
    pred = clf.predict(fd)
    #print(pred)
    #print(clf.decision_function(fd))
    #if pred == 1:
    #    return clf.decision_function(fd)
    #else:
    #    return 0
    return pred

path = os.getcwd()
test_im_path = os.path.join(path, 'test_images')
model_path = os.path.join(path, 'models', 'svm.model')
print(test_im_path)
for im_path in glob.glob(os.path.join(test_im_path, "*")):
    print(im_path)
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    grow_im = cv2.resize(im, (512, 512))
    im = cv2.resize(im, (128, 128))
    cv2.imshow(str(im_path), grow_im)

    cd = detection(im, model_path)
    if cd == 0:
        print("Bad")
    if cd == 1:
        print("Good")

    cv2.waitKey()