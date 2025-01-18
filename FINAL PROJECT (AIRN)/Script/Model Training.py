import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randint
from tensorflow.keras import Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam



# Dataset Path
DATASET_PATH = "D:/My dataset/"

# Preprocessing Function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale and resize
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (64, 64))
    
    # Normalize the image
    preprocessed_image = resized_image / 255.0
    return preprocessed_image

# Load Dataset
def load_dataset(dataset_path):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    for label, class_name in enumerate(class_names):
        folder_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                processed_image = preprocess_image(file_path)
                images.append(processed_image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return np.array(images), np.array(labels), class_names

# Load and preprocess dataset
X, y, class_names = load_dataset(DATASET_PATH)
X = X[..., np.newaxis]  # Add channel dimension for CNN
y = to_categorical(y, num_classes=len(class_names))  # One-hot encode labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Use VGG16 as the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False 

# Add a custom classification head
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


####
# Updated CNN model definition
model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile the model with Adam optimizer 
model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Accuracy metric
)

# Train the model using mini-batch gradient descent with batch size 16
history = model.fit(
    X_train,  # Training features
    y_train,  # Training labels
    validation_data=(X_test, y_test),  # Validation set
    epochs=20,  # Number of training epochs
    batch_size=16,  # Mini-batch size
    verbose=1  # Display progress during training
)


#######

# Callbacks for better training
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('./Models/best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train the model with callbacks
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.argmax(axis=1)),  # Use class indices
    y=y_train.argmax(axis=1)  # Provide class indices, not one-hot encoded labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)


# Plot training accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# Visualize predictions on test set
def visualize_predictions(model, X_test, y_test, class_names):
    plt.figure(figsize=(15, 10))
    for i in range(5):  # Show 5 random test images
        idx = randint(0, len(X_test) - 1)
        image = X_test[idx]
        true_label = np.argmax(y_test[idx])
        predicted_label = np.argmax(model.predict(image[np.newaxis, ...]))

        # Plot the image and prediction
        plt.subplot(1, 5, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Display predictions
visualize_predictions(model, X_test, y_test, class_names)

# Save the model in the native Keras format
model.save("./Models/wildlife_classifier.keras")
print("Model saved as 'wildlife_classifier.keras'")