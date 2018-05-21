# Import the required modules
from skimage.feature import local_binary_pattern

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


import glob
import os
import pandas as pd
import numpy as np


def test_classifiers(data, lables, clf):
    """
    create training objects
    """
    data = np.array(data)
    is_train = np.random.uniform(0, 1, len(data)) <= 0.5
    y = np.where(np.array(labels)== 1, 1, 0)

    #set training and test data
    train_x, train_y = data[is_train], y[is_train]
    test_x, test_y = data[is_train==False], y[is_train==False]


    pca = PCA(n_components=5, svd_solver='randomized')
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    clf.fit(train_x, train_y)
    df = pd.crosstab(test_y, clf.predict(test_x), rownames=["Actual"], colnames=["Predicted"])
    score = clf.score(test_x, test_y)
    print(df)
    print("Score: " + str(score))
    print("")



if __name__ == "__main__":
    path = r'E:\pent_python\training_mmse_pentagons'
    pos_im_path = os.path.join(path, 'pos_pent')
    neg_im_path = os.path.join(path, 'neg_pent')
    
    pos_feat_ph = os.path.join(path, "pos_feat_ph")
    neg_feat_ph = os.path.join(path, "neg_feat_ph")

    model_path =  os.path.join(path, "models", "randtree.model")

    pos_feat_path =  pos_feat_ph
    neg_feat_path =  neg_feat_ph

    # Classifiers supported
    

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

    clf_type = "LIN_SVM"
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print("Testing Linear SVM Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "KNN"
    if clf_type is "KNN":
        clf = KNeighborsClassifier()
        print("Testing KNN Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "RandomForest"
    if clf_type is "RandomForest":
        clf = RandomForestClassifier(n_jobs=2)
        print("Testing Random Forest Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))

    clf_type = "DecisionTreeCLassifier"
    if clf_type is "DecisionTreeCLassifier":
        clf = DecisionTreeClassifier(random_state=0)
        print("Testing Decision Tree Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "GaussianProcessClassifier"
    if clf_type is "GaussianProcessClassifier":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = GaussianProcessClassifier()
        print("Testing Gaussian Process Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "MLPClassifier"
    if clf_type is "MLPClassifier":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = MLPClassifier(alpha=1)
        print("Testing MLPClassifier Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "SVC"
    if clf_type is "SVC":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = SVC()
        print("Testing SVC Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)
        
    clf_type = "AdaBoost"
    if clf_type is "AdaBoost":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = AdaBoostClassifier()
        print("Testing AdaBoost Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "GaussianNB"
    if clf_type is "GaussianNB":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = GaussianNB()
        print("Testing GaussianNB Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)

    clf_type = "QuadraticDiscriminantAnalysis"
    if clf_type is "QuadraticDiscriminantAnalysis":
        #kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        clf = QuadraticDiscriminantAnalysis()
        print("Testing QuadraticDiscriminantAnalysis Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        test_classifiers(fds, labels, clf)
        

##        ## SAVE MODEL
##        joblib.dump(clf, model_path)
##        print("Classifier saved to {}".format(model_path))

