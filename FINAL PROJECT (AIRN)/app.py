import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configure the page layout
st.set_page_config(layout="wide", page_title="Advanced Image Recognition", page_icon="🎨")

def preprocess_image(image):
    """
    Preprocess the image for model prediction.
    - Convert to grayscale
    - Resize to 64x64
    - Normalize pixel values
    """
    try:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(grayscale_image, (64, 64))
        preprocessed_image = resized_image / 255.0  # Normalize
        return preprocessed_image
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def load_dataset(dataset_path):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    for label, class_name in enumerate(class_names):
        folder_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                processed_image = preprocess_image(cv2.imread(file_path))
                images.append(processed_image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return np.array(images), np.array(labels), class_names

# Load the pre-trained model
DATASET_PATH = "./My dataset/"
model = load_model("./Models/wildlife_classifier.keras")
_, _, class_names = load_dataset(DATASET_PATH)

def predict_image(model, image, class_names, confidence_threshold=0.7):
    """
    Predict the class of an image using the trained model.
    If confidence is below the threshold, return 'Unknown'.
    """
    try:
        image = preprocess_image(image)
        image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "Unknown", confidence
        return class_names[predicted_class], confidence
    except Exception as e:
        return f"Error predicting image: {e}", None

# Streamlit UI
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Advanced Image Recognition")
st.write("Upload an image to identify its class with confidence. The model supports multiple categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file is not None:
    try:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize image for display
        max_width = 600
        max_height = 400
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=True)

        # Perform prediction
        with st.spinner("Analyzing the image..."):
            predicted_class, confidence = predict_image(model, image, class_names)

        # Display prediction result
        if confidence is not None:
            if predicted_class == "Unknown":
                st.warning(f"The model is not confident about the prediction (Confidence: {confidence:.2f}).")
            else:
                st.success(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
        else:
            st.error(predicted_class)
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to proceed.")

st.markdown("""
    <hr>
    <p style="text-align: center; color: gray;">
        © 2024 Advanced Image Recognition - Built with ❤️ by Tushar (Project Intern- Infosys Springboard)
    </p>
""", unsafe_allow_html=True)
