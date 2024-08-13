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
import tensorflow.keras.models # type: ignore

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
if 'y_label' not in st.session_state:
    st.session_state.y_label = None

# Trang lựa chọn
page = st.sidebar.selectbox("Select a page", ["Dataset Loading & Training", "Prediction"])

if page == "Dataset Loading & Training":
    st.title("Dataset Loading & Training")
    
    # Chọn nguồn dữ liệu
    data_path = "DATASET"

    # Cài đặt num_samples
    num_samples = st.number_input("Number of Samples per Folder", min_value=10, value=2000)

    col1, col2, col3 = st.columns(3)

    with col1:
        option = st.selectbox("Select setting:", ["Pretrained Model", "Custom Setting"])   

    with col2:
        test_size = st.number_input("Test Set Size (0.1 - 0.5)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    with col3: 
        epochs = st.number_input("Number of Training Epochs", min_value=1, value=10)

    # load
    if option == "Pretrained Model":
      if st.button("Load Dataset and Train Model"):
        if os.path.isdir(data_path):
            with st.spinner('Loading Dataset...'):
                X, y = load_dataset(data_path, num_samples)

                #st.session_state.X = X
                #st.session_state.y = y

            st.success("Dataset loaded successfully!")

            st.write(f"Dataset loaded: {X.shape[0]} samples.")

            st.write("Sample images from the dataset:")
            fig, axs = plt.subplots(nrows=26, ncols=10, figsize=(20, 50))
            axs = axs.flatten()
            for i, letter in enumerate(letters):
                ids = np.where(y == i)[0]
                for j in range(10):  # Hiển thị 10 mẫu cho mỗi chữ cái
                    img = X[ids[j]]
                    img = Image.fromarray(img)
                    axs[i * 10 + j].imshow(img, cmap='gray')
                    axs[i * 10 + j].axis('off')
                    if j == 0:
                        axs[i * 10 + j].set_title(letter)
                
            plt.tight_layout()
            st.pyplot(fig)

            if X is not None and y is not None:
                
                with st.spinner('Loading model...'):
                    X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)
                    model = tensorflow.keras.models.load_model("Model.h5")

                    st.session_state.model = model
                    st.session_state.y_label = letters

                st.success("Model loaded successfully!")

                test_loss, test_accuracy = model.evaluate(X_test, y_test_ohe)
                

                st.write(f"Test Accuracy: {test_accuracy*100:.2f}%. "
                         f"Test Loss: {test_loss*100:.2f}%")
            
            else:
                st.error("MODEL_LOADING FAILED")

        else:
            st.error("Invalid dataset path!")

    # load
    elif option == "Custom Setting":
      if st.button("Load Dataset and Train Model"):
        if os.path.isdir(data_path):
            with st.spinner('Loading Dataset...'):
                X, y = load_dataset(data_path, num_samples)

                #st.session_state.X = X
                #st.session_state.y = y

            st.success("Dataset loaded successfully!")

            st.write(f"Dataset loaded: {X.shape[0]} samples.")

            st.write("Sample images from the dataset:")
            fig, axs = plt.subplots(nrows=26, ncols=10, figsize=(20, 50))
            axs = axs.flatten()
            for i, letter in enumerate(letters):
                ids = np.where(y == i)[0]
                for j in range(10):  # Hiển thị tối đa 10 mẫu cho mỗi chữ cái
                    img = X[ids[j]]
                    img = Image.fromarray(img)
                    axs[i * 10 + j].imshow(img, cmap='gray')
                    axs[i * 10 + j].axis('off')
                    if j == 0:
                        axs[i * 10 + j].set_title(letter)
                
            plt.tight_layout()
            st.pyplot(fig)
            if X is not None and y is not None:
                with st.spinner('Training model...'):
                    X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)

                    #st.session_state.X_test = X_test
                    #st.session_state.y_test_ohe = y_test_ohe
                    start_time = time.time()
                    model = Train_model(X_train)
                    history = fit_model(model, X_train, y_train_ohe, epochs)
                    end_time = time.time()

                    # Tính thời gian chạy
                    elapsed_time = end_time - start_time
                
                    st.session_state.model = model
                    st.session_state.y_label = letters

                st.success("Model trained successfully!")

                #model.save("Model_v2.h5")

                test_loss, test_accuracy = model.evaluate(X_test, y_test_ohe)
                

                st.write(f"Training completed in {elapsed_time:.2f} seconds. "
                         f"Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%. "
                         f"Test Accuracy: {test_accuracy*100:.2f}%")
                
                # Hiển thị đồ thị cho độ chính xác và mất mát
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                
                # Accuracy plot
                ax1.plot(history.history['accuracy'], label='Train Accuracy')
                if 'val_accuracy' in history.history:
                    ax1.plot(history.history['val_accuracy'], label='Test Accuracy', linestyle='--')
                ax1.set_title('Model Accuracy', fontsize=16)
                ax1.set_xlabel('Epoch', fontsize=14)
                ax1.set_ylabel('Accuracy', fontsize=14)
                ax1.legend()

                # Loss plot
                ax2.plot(history.history['loss'], label='Train Loss', color='red')
                if 'val_loss' in history.history:
                    ax2.plot(history.history['val_loss'], label='Test Loss', color='orange', linestyle='--')
                ax2.set_title('Model Loss', fontsize=16)
                ax2.set_xlabel('Epoch', fontsize=14)
                ax2.set_ylabel('Loss', fontsize=14)
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("MODEL_TRAINING FAILED")

        else:
            st.error("Invalid dataset path!")
        


elif page == "Prediction":
    st.title("Model Prediction")

    if st.session_state.model is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
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

        with col2: 
            uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], key="uploader")
        
        if st.button("Predict"):
            if uploaded_image is not None:
                # Read and preprocess the uploaded image
                img = Image.open(uploaded_image).convert('RGB')  
                img = np.array(img)
                img = cv2.resize(img, (64, 64))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = img_gray / 255.0
                img_gray = np.expand_dims(img_gray, axis=0)

                # Create inverted image
                img_gray_inverted = 1 - img_gray

                col3, col4, col5 = st.columns([2, 1, 1])

                with col3:
                    st.write("Uploaded Image")
                    st.image(img, use_column_width= True, clamp=True)

                # Make prediction for original image
                prediction = st.session_state.model.predict(img_gray)
                predicted_class = np.argmax(prediction)
                labels = np.unique(st.session_state.y_label)
                st.write(predicted_class)
                predicted_label = labels[predicted_class]

                with col4:
                    st.write("Grayscale Image")
                    st.image(img_gray.reshape(64, 64), use_column_width=False, clamp=True)
                    #st.write(f"Predicted Letter (Normal): {predicted_label}  {prediction[0][predicted_class] * 100:.2f}%")

                    # Get the top 5 predictions for normal image
                    top5_indices = np.argsort(prediction[0])[::-1][:5]
                    for i in top5_indices:
                        st.write(f"{labels[i]}: {prediction[0][i] * 100:.2f}%")

                # Make prediction for inverted image
                prediction_inverted = st.session_state.model.predict(img_gray_inverted)
                predicted_class_inverted = np.argmax(prediction_inverted)
                predicted_label_inverted = labels[predicted_class_inverted]

                with col5:
                    st.write("Invert Grayscale Image")
                    st.image(img_gray_inverted.reshape(64, 64), use_column_width=False, clamp=True)
                    #st.write(f"Predicted Letter (Inverted): {predicted_label_inverted}  {prediction_inverted[0][predicted_class_inverted] * 100:.2f}%")

                    # Get the top 5 predictions for inverted image
                    top5_indices_inverted = np.argsort(prediction_inverted[0])[::-1][:5]
                    for i in top5_indices_inverted:
                        st.write(f"{labels[i]}: {prediction_inverted[0][i] * 100:.2f}%")

            elif canvas_result.image_data is not None:
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
        st.error("No model available for prediction. Please train the model first.")
