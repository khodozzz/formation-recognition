from keras.utils.np_utils import to_categorical


def prepare_x(x_array, scaler=None):
    x_array.sort(axis=1)
    if scaler is None:
        return x_array
    return scaler.transform(x_array)


def prepare_y(y_array, label_encoder):
    return label_encoder.transform(y_array)


def prepare_categorical_y(y_array, label_encoder):
    return to_categorical(label_encoder.transform(y_array))


def prepare_data(x_array, y_array, label_encoder, scaler=None, categorical_y=False):
    if categorical_y:
        return prepare_x(x_array, scaler), prepare_categorical_y(y_array, label_encoder)
    return prepare_x(x_array, scaler), prepare_y(y_array, label_encoder)
