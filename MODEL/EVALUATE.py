
def evaluate(X_test, y_test_ohe, model):
    loss, accuracy = model.evaluate(X_test, y_test_ohe)

    return loss, accuracy