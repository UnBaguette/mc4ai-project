def inference(model, X_new, y):
    y_new = model.predict(X_new)
    y_new_label = y_new.argmax(axis=1)
    y[y_new_label]