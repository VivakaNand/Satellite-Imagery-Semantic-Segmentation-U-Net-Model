import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained model
MODEL_PATH = "satellite-imagery.h5"
model = load_model(MODEL_PATH)

# Define a function to preprocess the image
def preprocess_image(image, target_size):
    """Preprocess the image to the required input size of the model."""
    image = image.resize(target_size)  # Resize image to model's input size
    image_array = img_to_array(image)  # Convert to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array

# Streamlit UI
st.title("Satellite Image Prediction App")
st.write("Upload a satellite image to see the prediction results.")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    st.write("Processing the image...")
    processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust size as per your model

    # Make prediction
    st.write("Predicting using the model...")
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions, axis=1)

    # Display the results
    st.subheader("Prediction Results:")
    st.write(f"Predicted Class: {predicted_label[0]}")
    st.write(f"Prediction Probabilities: {predictions[0]}")

   # Generate and display the model-generated image (if applicable)
    if model.outputs[0].shape[1:] == (224, 224, 3):  # Check if the model outputs an image
        st.subheader("Model-Generated Image:")
        generated_image_array = model.predict(processed_image)  # Predict
        generated_image_array = np.squeeze(generated_image_array, axis=0)  # Remove batch dimension
        generated_image = Image.fromarray((generated_image_array * 255).astype('uint8'))  # Convert to image
        st.image(generated_image, caption="Generated Image", use_column_width=True)
