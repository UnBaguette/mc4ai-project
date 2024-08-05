import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization # type: ignore
from keras.utils import set_random_seed # type: ignore
from keras.backend import clear_session # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

def Train_model(X_train):
    clear_session()
    set_random_seed(42)
    np.random.seed(42)

    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))

    model.add(Flatten())

    model.add(Dense(26, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model 

def fit_model(model, X_train, y_train_ohe, epochs):
    history = model.fit(X_train, y_train_ohe, epochs = epochs, verbose=1)
    return history