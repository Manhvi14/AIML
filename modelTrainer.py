from PreProcessor import Preprocessor
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


preprocessor = Preprocessor(image_shape=(256, 256))

data_folder = "data"  # Relative path to the data folder from the current working directory

data_folder_path = os.path.join(os.getcwd(), data_folder)

X_train, X_val, X_test, y_train, y_val, y_test, train_image_paths, val_image_paths, test_image_paths = preprocessor.preprocess_and_split_data(data_folder_path)

# Convert labels to one-hot vectors
num_classes = len(preprocessor.class_labels)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Perform inference on the test set
predicted_labels = np.argmax(model.predict(X_test), axis=1)  # Assuming it's a classification model
predicted_classes = preprocessor.label_encoder.inverse_transform(predicted_labels)

# Output the results
for i in range(len(X_test)):
    image_path = test_image_paths[i]
    predicted_class = predicted_classes[i]
    print(f"Image: {image_path} | Predicted class: {predicted_class}")
