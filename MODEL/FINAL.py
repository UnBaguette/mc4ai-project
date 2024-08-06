from LOAD import load_dataset 
from PREP import prep_dataset
from TRAINMODEL import Train_model

data_path = "../DATASET"

X, y = load_dataset(data_path)
X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, testsize=.2)
model = Train_model(X_train)
model.fit(X_train, y_train_ohe, epochs=100, verbose=1)

model.save('model.h5')