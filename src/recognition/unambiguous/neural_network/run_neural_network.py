import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from keras.utils.np_utils import to_categorical

from recognition.unambiguous.neural_network.models import dense1, dense2


def prepare_x(x_array, standard_scaler):
    x_array.sort(axis=1)
    return standard_scaler.transform(x_array)


def prepare_y(y_array, label_encoder):
    return to_categorical(label_encoder.transform(y_array))


def prepare_data(x_array, y_array, standard_scaler, label_encoder):
    return prepare_x(x_array, standard_scaler), prepare_y(y_array, label_encoder)


if __name__ == '__main__':
    train = pd.read_csv('../../../../data/train.csv')
    test = pd.read_csv('../../../../data/test.csv')

    train_values = train.values[:, :-1].astype(float)
    test_values = test.values[:, :-1].astype(float)

    le = LabelEncoder()
    le.fit(train.scheme)
    ss = StandardScaler()
    ss.fit(train_values)

    X_train, y_train = prepare_data(train_values, train.scheme, ss, le)
    X_test, y_test = prepare_data(test_values, test.scheme, ss, le)

    print('############ Creating model ############')
    model = dense1.create_model(10, len(le.classes_))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)
    model.save('models/last_model')  # TODO: save in directory

    print('############ Evaluating model ############')
    results = model.evaluate(X_test, y_test)

    print('############ My tests ############')
    my_tests = prepare_x(np.array([[5, 6, 7, 12, 35, 33, 37, 30, 99, 108],
                                   [5, 6, 7, 12, 35, 33, 37, 88, 99, 108],
                                   [5, 6, 7, 33, 35, 33, 37, 88, 99, 108],
                                   [5, 6, 7, 46, 50, 62, 65, 66, 99, 108]]),
                         ss)

    pr = model.predict(my_tests)
    print(le.inverse_transform(np.argmax(pr, axis=1)))
