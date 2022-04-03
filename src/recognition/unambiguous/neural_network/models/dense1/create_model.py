from keras.models import Model
from keras.layers import Dense, Dropout, Input


def create_model(input_dim, output_dim):
    input_layer = Input(shape=(input_dim, ))

    x = Dense(10, activation='relu', kernel_initializer='he_uniform')(input_layer)
    x = Dropout(0.05)(x)  # 0.2
    x = Dense(10, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.05)(x)  # 0.2

    output_layer = Dense(output_dim, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)
