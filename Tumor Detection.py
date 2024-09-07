#imports
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
import glob
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, Dropout, MaxPooling2D,BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.optimizers import Adam,RMSprop # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras import regularizers # type: ignore
from PIL import Image
from efficientnet.tfkeras import EfficientNetB0
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
#-------------PREPROCESSING AND VISUALIZATION--------------------
import os

#-------------PREPROCESSING AND VISUALIZATION--------------------
# Define the paths to the folders
path_Yes = r"C:\Users\Anirban Das\OneDrive\Desktop\DS ML PYTHON\MACHINE LEARNING\Notes of ML\Segmentation of Brain Tumor using MRI\Brain Tumor Data Set\Tumor"
path_No = r"C:\Users\Anirban Das\OneDrive\Desktop\DS ML PYTHON\MACHINE LEARNING\Notes of ML\Segmentation of Brain Tumor using MRI\Brain Tumor Data Set\No_Tumor"
tumor = []
no_tumor = []
random_state = 42
# List all files in each directory
files_Yes = os.listdir(path_Yes)
files_No = os.listdir(path_No)

# Create full paths for the images
files_Yes = [os.path.join(path_Yes, f) for f in files_Yes]
files_No = [os.path.join(path_No, f) for f in files_No]

# Process tumor images
for file in files_Yes:
    img = cv2.imread(file)
    if img is None:
        print(f"Failed to load image: {file}")
        continue  # Skip this file if it fails to load
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    tumor.append((img, 1))  # Label 1 indicates the presence of a tumor

# Process healthy images
for file in files_No:
    img = cv2.imread(file)
    if img is None:
        print(f"Failed to load image: {file}")
        continue  # Skip this file if it fails to load
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    no_tumor.append((img, 0))  # Label 0 indicates the absence of a tumor

# Concatenating the two lists and shuffle the data
all_data = tumor + no_tumor

# Splitting data and labels
data = np.array([item[0] for item in all_data])
labels = np.array([item[1] for item in all_data])

#---------EXPLORATORY DATA ANALYSIS---------------
plt.figure(figsize=(15, 5))

# Display tumor images with label 'yes'
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(tumor[i][0])  
    plt.title("Tumor: Yes")  
    plt.axis('off')

# Display no_tumor images with label 'no'
for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.imshow(no_tumor[i][0])  
    plt.title("Tumor: No")  
    plt.axis('off')

plt.tight_layout()
plt.show()
plt.close()

# Counting the occurrences of each class label
unique_labels, label_counts = np.unique(labels, return_counts=True)

plt.bar(unique_labels, label_counts, color=['blue', 'orange'])
plt.xticks(unique_labels, ['No Tumor', 'Tumor'])
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Distribution of Class Labels')
plt.show()
plt.close()

#---------DATA SCALING AND TRANSFORMATION------------

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
# Assuming x_train and x_test are your image datasets
# Normalize the pixel values to the range [0, 1]
x_train= x_train /255.0
x_test = x_test / 255.0
print("Minimum value of the scaled data:", x_train.min())
print("Maximum value of the scaled data:",  x_train.max())

#---------------BUILD MODEL------------------

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
            
back = myCallback()     
history = model.fit(x_train, 
                    y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                   callbacks=[back])
# plot the accuracy and loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()
# model loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.show()
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
# Assuming 'model' is your trained Keras model
model.save("brain_tumor.keras")
