import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from recognition.unambiguous.supervised.prepare_data import *

if __name__ == '__main__':
    train = pd.read_csv('../../../../../data/23_schemes/train.csv')
    test = pd.read_csv('../../../../../data/23_schemes/test.csv')

    train_values = train.values[:, :-1].astype(float)
    test_values = test.values[:, :-1].astype(float)

    le = LabelEncoder()
    le.fit(train.scheme)
    ss = StandardScaler()
    ss.fit(train_values)

    X_train, y_train = prepare_data(train_values, train.scheme, le, ss)
    X_test, y_test = prepare_data(test_values, test.scheme, le, ss)

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        # "RBF SVM",
        # "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        # "Neural Net",
        # "AdaBoost",
        # "Naive Bayes",
        # "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
    ]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        print(f'{name}: {clf.score(X_test, y_test)}')

        my_tests = prepare_x(np.array([[5, 6, 7, 12, 35, 33, 37, 39, 56, 60],  # 4 4 2
                                       [5, 6, 7, 12, 35, 33, 37, 56, 60, 59],  # 4 3 3
                                       [5, 6, 7, 33, 35, 33, 37, 56, 60, 59],  # 3 4 3
                                       [5, 6, 7, 35, 33, 37, 39, 36, 56, 60]]),  # 3 5 2
                             ss)

        pr = clf.predict(my_tests)
        print(le.inverse_transform(pr))
        print()
