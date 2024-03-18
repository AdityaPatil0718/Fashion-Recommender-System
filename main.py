import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Construct a Keras Sequential model with ResNet50 base, GlobalMaxPooling2D layer, and three hidden Dense layers
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu')
])

# Define a function to extract features from an image using the model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Get the list of file names from the 'images' directory
filenames = [os.path.join('test', file) for file in os.listdir('test')]

# Extract features for each image and store them in a list
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save the feature list and filenames into pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
