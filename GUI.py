# Author: Randa Yoga Saputra
# Version: 1
# Purpose: Simple software for data predict based on model, can select from file or webcam
# How to start: python GUI.py

import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

class ImageApp:
    def __init__(self, root, model_path, class_names):
        self.root = root
        self.root.title("Image and Camera App")
        
        # Load TensorFlow model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names  # List of class labels
        
        # GUI Components
        self.label = Label(root, text="Select an option below:", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.image_label = Label(root)
        self.image_label.pack(pady=10)
        
        self.select_button = Button(root, text="Select Image", command=self.select_image, font=("Arial", 12))
        self.select_button.pack(pady=5)
        
        self.camera_button = Button(root, text="Open Camera", command=self.open_camera, font=("Arial", 12))
        self.camera_button.pack(pady=5)
        
        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # Define image dimensions
        self.img_height = 100  # Update based on your model's input
        self.img_width = 100  # Update based on your model's input

    def preprocess_image(self, image, file_path):
        """
        Resize, normalize, and prepare the image for the model.
        """
        print("Original Image Shape:", image.shape)
        image = cv2.resize(image, (self.img_height, self.img_width))  # Resize
        # image = image / 255.0  # Normalize pixel values
        # image = np.expand_dims(image, axis=0)  # Add batch dimension
        print("Processed Image Shape:", image.shape)
        images = keras.preprocessing.image.load_img(file_path, target_size=(self.img_height, self.img_width))
        return images

    def predict_and_display(self, processed_image):
        image_array = keras.preprocessing.image.img_to_array(processed_image)
        image_array = tf.expand_dims(image_array, 0)  # Create a batch

        predictions = self.model.predict(image_array)
        print("Raw Prediction (Logits):", predictions[0])

        score = tf.nn.softmax(predictions[0])
        print("Softmax Probabilities:", score.numpy())

        model_predicted = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        print(f"Predicted Label: {model_predicted}, Confidence: {confidence:.2f}%")
        self.result_label.config(text=f"Prediction: {model_predicted}, Confidence: {confidence:.2f}%")

    def select_image(self):
        """
        Handle the image selection from the file explorer.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            # Load and display the image
            image = cv2.imread(file_path)
            self.display_image(image)
            
            # Preprocess and predict
            processed_image = self.preprocess_image(image, file_path)
            self.predict_and_display(processed_image)

    def open_camera(self):
        """
        Open the camera, capture a frame, and make predictions.
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to capture
                self.display_image(frame)
                processed_image = self.preprocess_image(frame)
                self.predict_and_display(processed_image)
                break
        cap.release()
        cv2.destroyAllWindows()

    def display_image(self, image):
        """
        Display the selected or captured image in the GUI.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

if __name__ == "__main__":
    # Define the model path
    model_path = r".\models\models_ganas_v_jinak\ResNet\0\model_0"  # Update with your model's path
    
    # Define class names (update based on your model's training classes)
    class_names = ["Tumor Ganas (Kanker)", "Tumor Jinak"]

    # Create the main Tkinter window
    root = tk.Tk()
    app = ImageApp(root, model_path, class_names)
    root.mainloop()