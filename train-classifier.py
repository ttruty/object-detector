# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import glob
import os
import pandas as pd
import numpy as np



if __name__ == "__main__":
    path = os.getcwd()
    pos_im_path = os.path.join(path, 'pos_img')
    neg_im_path = os.path.join(path, 'neg_img')
    
    pos_feat_ph = os.path.join(path, "pos_feat_ph")
    neg_feat_ph = os.path.join(path, "neg_feat_ph")
    model_path = os.path.join(path, "models")

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    model_path =  os.path.join(path, "models", "svm.model")

    pos_feat_path =  pos_feat_ph
    neg_feat_path =  neg_feat_ph

    # Classifiers supported
    clf_type = "LIN_SVM"

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)

    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print("Training a Linear SVM Classifier")
        print(len(fds), len(labels))
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))
