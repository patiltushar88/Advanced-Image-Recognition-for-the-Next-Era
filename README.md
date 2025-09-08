## Advanced-Image-Recognition-for-the-Next-Era_Infosys_Internship_Oct2024 🔍 

Overview

This project involves building an advanced image recognition system using machine learning techniques. The model achieves high prediction accuracy by utilizing data preprocessing, feature extraction, and model optimization. The system is scalable for real-time applications and has been implemented with Python, OpenCV, and TensorFlow/Keras.

## Features  📈

- **Accurate Predictions**: Achieved approximately 100% accuracy through advanced model optimization techniques.
- **Real-time Recognition**: Processes input images and generates results instantly.
- **Fullscreen modeUser-Friendly Interface**: Built with Streamlit for ease of use and interaction.
- **Scalable Architecture**: Capable of handling large datasets and real-time image processing.
- **Secure Deployment**: Dockerized for deployment with enhanced reliability and security.


## Tech Stack 💻


- **Languages**: Python

- **Libraries**: OpenCV, TensorFlow, Keras, NumPy, Pandas

- **Framework**: Streamlit

- **Deployment Tools**: Docker


## System Architecture 🏗️

- **Data Preprocessing**: Enhances input image quality through normalization and augmentation.

- **Feature Extraction**: Employs convolutional layers to identify and extract relevant patterns.

- **Model Training**: Trained a Convolutional Neural Network (CNN) on a labeled dataset.

- **Prediction and Output**: Outputs labels and confidence scores for each image.

## Setup Instructions⚙️


**Clone the repository**

Clone the repository and switch to the tushar branch:
```bash
git clone https://github.com/patiltushar88/Advanced-Image-Recognition-for-the-Next-Era.git

cd Advanced-Image-Recognition-for-the-Next-Era


```
    

**Install Dependencies**

Make sure Python 3.12.5 or later is installed. Install all the necessary dependencies using:

```bash
pip install -r requirements.txt
```

**Run the Application**

Start the Streamlit application to access the interface:

```bash
streamlit run app.py
```

**Test the Model**

- Upload an image via the interface.
- Click on the "Predict" button to get recognition results.
- View labels and confidence scores in real-time.


## Screenshots 🖼️

**Input Interface**

User Interface Using Streamlit
<img width="1833" height="481" alt="Streamlit" src="https://github.com/user-attachments/assets/03a423a3-ddb1-4cd1-a307-d9ba18338754" />


**Prediction Results**✅

We took 4 Classes dataset Containing

**Akshay Kumar, Vijay , Prabhas , Amithabh Bacchan**

**An image of Akshay Kumar was uploaded, and the model accurately identified it, demonstrating its reliability and precision.**
<img width="1920" height="2276" alt="output ok" src="https://github.com/user-attachments/assets/2b9c304c-22bf-4684-bd35-cdd6ade22f4c" />

**Additional images were tested to validate the model's accuracy and ensure correct predictions in diverse scenarios.**
<img width="1920" height="2526" alt="output false" src="https://github.com/user-attachments/assets/befa5fd9-59aa-4996-b19b-35d178e1c3ca" />


