# - Python version: 3.8.10
'''

The following template is provided as a guideline only. You can modify it as you need.

"introduce something here" = ""

'''
# -libraries
import numpy
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition

##- visualization
import matplotlib.pyplot as plt

'''
Add here the functions need.
'''

if __name__ == "__main__":
    # loading data
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.data
    print(X.shape)
    print(Y.shape)

    # split the data 75% train and 25% test subsets
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split("")

    # Data normalization by zero mean and unit variance
    from sklearn import preprocessing

    # print(digits['DESCR'])

    scaler = preprocessing.StandardScaler().fit("")
    x_train_scaled = scaler.trasform("")

    # apply scaling on testing data having into account x_train statistics
    x_test_scaled = ""



    # Dimensionality reduction: PCA and SVD analysis
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD

    # PCA Analysis
    pca = PCA().fit(x_train_scaled)
    # the PCA explains a part of the variance. Plot the cumulative variance to get the number of principal
    # components we need.
    ""
    # SVD Analysis
    tsvd = TruncatedSVD().fit_transform(x_train_scaled)
    # Same analysis as PCA: you have to obtain the best number of components for SVD.
    ""

    # Apply a dimension reduction choosing a 95% of variance explained (x_pca_train, x_tsvd_train, x_pca_test...)
    ""

    # Find the k optimal value for the k-NN classifier using 10-fold cross-validation
    #
    # create the knn classifier
    from sklearn import KNeighborsClassifier

    clf = KNeighborsClassifier("")
    # learn the digits on the train subset

    clf.fit(x_train_scaled, y_train)  # you should also use preprocessed x data

    # predict the value of the digit on the test subset

    predicted = clf.predict(x_test_scaled)

    # print the metrics
    ""
    # k neighbours analysis: different clsssifier performances varyng the number of k
    ""
