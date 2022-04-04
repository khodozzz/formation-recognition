import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input

__all__ = ['build_2layers_nn', 'build_7layers_nn']


def build_2layers_nn(input_dim: int, output_dim: int):
    """
    Build simple NN with 2 hidden layers
    :param input_dim: input dimension
    :param output_dim: output dimension
    :return: compiled model
    """
    input_layer = Input(shape=(input_dim, ))
    x = Dense(10, activation='relu', kernel_initializer='he_uniform')(input_layer)
    x = Dropout(0.05)(x)
    x = Dense(10, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.05)(x)
    output_layer = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_7layers_nn(input_dim: int, output_dim: int):
    """
    Build deep NN with 7 hidden layers
    :param input_dim: input dimension
    :param output_dim: output dimension
    :return: compiled model
    """
    model = Sequential()

    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
