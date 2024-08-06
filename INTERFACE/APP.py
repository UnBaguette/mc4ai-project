import streamlit as st
import numpy as np
import sys
import os
import cv2
from streamlit_drawable_canvas import st_canvas  # Import canvas
import string

# Tạo một danh sách các chữ cái từ A đến Z
letters = list(string.ascii_uppercase)

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import funtion từ MODEL
from MODEL.LOAD import load_dataset 
from MODEL.PREP import prep_dataset
from MODEL.TRAINMODEL import Train_model, fit_model
from MODEL.EVALUATE import evaluate

# Khởi tạo trạng thái của session
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'y_label' not in st.session_state:
    st.session_state.y_label = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test_ohe' not in st.session_state:
    st.session_state.y_test_ohe = None

# Trang lựa chọn
page = st.sidebar.selectbox("Select a page", ["Dataset Loading & Training", "Model Evaluation", "Prediction"])

if page == "Dataset Loading & Training":
    st.title("Dataset Loading & Training")
    
    # Chọn nguồn dữ liệu
    data_path = "DATASET"

    # Cài đặt test size và epochs
    test_size = st.number_input("Test Set Size (0.1 - 0.5)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    epochs = st.number_input("Number of Training Epochs", min_value=1, value=10)

    if st.button("Load, Preprocess & Train Model"):
        if os.path.isdir(data_path):
            st.write("Loading and Preprocessing Dataset...")
            X, y = load_dataset(data_path)
            X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)
            
            st.session_state.X_test = X_test
            st.session_state.y_test_ohe = y_test_ohe

            st.write(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
            
            st.write("Training model...")
            model = Train_model(X_train)
            history = fit_model(model, X_train, y_train_ohe, epochs)
            
            st.session_state.model = model
            st.session_state.history = history
            #st.session_state.y_label = np.unique(y)
            st.session_state.y_label = letters
            
            if history.history.get('accuracy') and history.history.get('loss'):
                st.line_chart(history.history['accuracy'], use_container_width=True)
                st.line_chart(history.history['loss'], use_container_width=True)
            else:
                st.write("Training history not available.")

        else:
            st.error("Invalid dataset path!")

elif page == "Model Evaluation":
    if st.session_state.model is not None and st.session_state.X_test is not None and st.session_state.y_test_ohe is not None:
        st.title("Model Evaluation")
        st.write("Evaluate the model on test set:")
        loss, accuracy = evaluate(st.session_state.X_test, st.session_state.y_test_ohe, st.session_state.model)
        st.write(f"Loss: {loss}\nAccuracy: {accuracy}")
    else:
        st.error("No model or test data available for evaluation. Please train the model first.")

elif page == "Prediction":
    st.title("Model Prediction")
    
    if st.session_state.model is not None:
        st.write("Draw a letter and see the prediction:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.5)",
            stroke_color="white",
            stroke_width=15,
            width=256,
            height=256,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("Predict"):
            if canvas_result.image_data is not None:
                img = canvas_result.image_data
                img = np.mean(img, axis=-1)  # Convert to grayscale
                img = cv2.resize(img, (64, 64))  # Resize image
                img = img / 255.0  # Normalize
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                
                predictions = st.session_state.model.predict(img)
                sorted_indices = np.argsort(predictions[0])[::-1]  # Sắp xếp giảm dần
                top_n = 5  # Số lượng kết quả muốn hiển thị

                st.write("Predictions:")
                for i in range(top_n):
                    label = st.session_state.y_label[sorted_indices[i]]
                    confidence = predictions[0][sorted_indices[i]] * 100
                    st.write(f"{label}: {confidence:.2f}%")
            else:
                st.error("No drawing found on canvas!")
    else:
        st.error("No model available for prediction. Please train the model first.")
