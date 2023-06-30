import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, image_shape=(256, 256)):
        self.image_shape = image_shape
        self.label_encoder = LabelEncoder()
        self.class_labels = []

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_shape)
        image = image / 255.0  # Normalize pixel values between 0 and 1
        return image

    def preprocess_data(self, data_folder):
        image_paths = []
        labels = []

        for label in os.listdir(data_folder):
            label_folder = os.path.join(data_folder, label)
            if os.path.isdir(label_folder):
                self.class_labels.append(label)  # Add class label
                for image_file in os.listdir(label_folder):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(label_folder, image_file)
                        image_paths.append(image_path)
                        labels.append(label)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Preprocess images
        images = []
        for image_path in image_paths:
            image = self.preprocess_image(image_path)
            images.append(image)

        return np.array(images), np.array(encoded_labels), image_paths

    def preprocess_and_split_data(self, data_folder, test_size=0.2, val_size=0.2, random_state=42):
        train_folder = os.path.join(data_folder, "train")
        val_folder = os.path.join(data_folder, "validation")
        test_folder = os.path.join(data_folder, "test")

        # Preprocess data
        X_train, y_train, train_image_paths = self.preprocess_data(train_folder)
        X_val, y_val, val_image_paths = self.preprocess_data(val_folder)
        X_test, y_test, test_image_paths = self.preprocess_data(test_folder)

        # Print example output
        print("Number of training samples:", len(X_train))
        print("Number of validation samples:", len(X_val))
        print("Number of test samples:", len(X_test))
        print("Example training image shape:", X_train[0].shape)
        print("Example training label:", self.label_encoder.inverse_transform([y_train[0]]))

        return X_train, X_val, X_test, y_train, y_val, y_test, train_image_paths, val_image_paths, test_image_paths
