from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split

def prep_dataset(X, y, testsize):
    # split_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, shuffle=True)

    # normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # ohe
    y_train_ohe = to_categorical(y_train, num_classes= 26)
    y_test_ohe = to_categorical(y_test, num_classes= 26)

    return X_train, X_test, y_train_ohe, y_test_ohe