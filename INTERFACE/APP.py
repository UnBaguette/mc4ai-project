import streamlit as st
import numpy as np
import sys
import os
import cv2
from streamlit_drawable_canvas import st_canvas  # Import canvas
import string
from PIL import Image
import matplotlib.pyplot as plt
import time

# Tạo một danh sách các chữ cái từ A đến Z
letters = list(string.ascii_uppercase)

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import funtion từ MODEL
from MODEL.LOAD import load_dataset 
from MODEL.PREP import prep_dataset
from MODEL.TRAINMODEL import Train_model, fit_model

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
page = st.sidebar.selectbox("Select a page", ["Dataset Loading & Training", "Prediction"])

if page == "Dataset Loading & Training":
    st.title("Dataset Loading & Training")
    
    # Chọn nguồn dữ liệu
    data_path = "DATASET"

    # Cài đặt num_samples, test size và epochs
    num_samples = st.number_input("Number of Samples per Folder", min_value=10, value=2000)
    test_size = st.number_input("Test Set Size (0.1 - 0.5)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    epochs = st.number_input("Number of Training Epochs", min_value=1, value=10)

    # Nút để bắt đầu quá trình load, preprocess và train
    if st.button("Load, Preprocess & Train Model"):
        with st.spinner('Loading and Preprocessing Dataset...'):
            if os.path.isdir(data_path):
                X, y = load_dataset(data_path, num_samples)
                X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)
                
                st.session_state.X_test = X_test
                st.session_state.y_test_ohe = y_test_ohe

                st.write(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

                st.write("Sample images from the dataset:")
                fig, axs = plt.subplots(nrows=26, ncols=10, figsize=(20, 50))
                axs = axs.flatten()
                for i, letter in enumerate(letters):
                    indices = np.where(y == i)[0]
                    for j in range(min(10, len(indices))):  # Hiển thị tối đa 10 mẫu cho mỗi chữ cái
                        img = X[indices[j]]
                        img = Image.fromarray(img)
                        axs[i * 10 + j].imshow(img, cmap='gray')
                        axs[i * 10 + j].axis('off')
                        if j == 0:
                            axs[i * 10 + j].set_title(letter)
                
                plt.tight_layout()
                st.pyplot(fig)

                # Training model
                with st.spinner('Training model...'):
                    start_time = time.time()
                    model = Train_model(X_train)
                    history = fit_model(model, X_train, y_train_ohe, epochs)
                    end_time = time.time()

                    # Tính thời gian chạy
                    elapsed_time = end_time - start_time
                    
                    st.session_state.model = model
                    st.session_state.history = history
                    st.session_state.y_label = letters

                    test_loss, test_accuracy = model.evaluate(st.session_state.X_test, st.session_state.y_test_ohe)

                    st.write(f"Training completed in {elapsed_time:.2f} seconds. "
                             f"Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%. "
                             f"Test Accuracy: {test_accuracy*100:.2f}%")
                    
                    # Hiển thị độ chính xác và mất mát
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                    
                    # Accuracy plot
                    ax1.plot(history.history['accuracy'], label='Train Accuracy')
                    #ax1.plot(history.history['val_accuracy'], label='Test Accuracy', linestyle='--')
                    ax1.set_title('Model Accuracy', fontsize=16)
                    ax1.set_xlabel('Epoch', fontsize=14)
                    ax1.set_ylabel('Accuracy', fontsize=14)
                    ax1.legend()

                    # Loss plot
                    ax2.plot(history.history['loss'], label='Train Loss', color='red')
                    #ax2.plot(history.history['val_loss'], label='Test Loss', color='orange', linestyle='--')
                    ax2.set_title('Model Loss', fontsize=16)
                    ax2.set_xlabel('Epoch', fontsize=14)
                    ax2.set_ylabel('Loss', fontsize=14)
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)
    
                # Thông báo hoàn tất
                st.success("Dataset loaded, preprocessed, and model trained successfully!")
            else:
                st.error("Invalid dataset path!")

#elif page == "Model Evaluation":
#    if st.session_state.model is not None and st.session_state.X_test is not None and st.session_state.y_test_ohe is not None:
#        st.title("Model Evaluation")
#        st.write("Evaluate the model on test set:")
#        loss, accuracy = evaluate(st.session_state.X_test, st.session_state.y_test_ohe, st.session_state.model)
#        st.write(f"Loss: {loss}\nAccuracy: {accuracy}")
#    else:
#        st.error("No model or test data available for evaluation. Please train the model first.")

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
