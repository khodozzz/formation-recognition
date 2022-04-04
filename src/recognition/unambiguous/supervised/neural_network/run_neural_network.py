import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from keras.utils.np_utils import to_categorical

from recognition.unambiguous.supervised.neural_network.models import build_2layers_nn, build_7layers_nn
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

    X_train, y_train = prepare_data(train_values, train.scheme, le, ss, categorical_y=True)
    X_test, y_test = prepare_data(test_values, test.scheme, le, ss, categorical_y=True)

    print('############ Creating model ############')
    model = build_2layers_nn(10, len(le.classes_))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save('models/last_model')  # TODO: save in directory

    print('############ Evaluating model ############')
    results = model.evaluate(X_test, y_test)

    print('############ My tests ############')
    my_tests = prepare_x(np.array([[5, 6, 7, 12, 35, 33, 37, 39, 99, 108],  # 4 4 2
                                   [5, 6, 7, 12, 35, 33, 37, 88, 99, 108],  # 4 3 3
                                   [5, 6, 7, 33, 35, 33, 37, 88, 99, 108],  # 3 4 3
                                   [5, 6, 7, 46, 50, 62, 65, 66, 99, 108]]),  # 3 5 2
                         ss)

    pr = model.predict(my_tests)
    print(le.inverse_transform(np.argmax(pr, axis=1)))
