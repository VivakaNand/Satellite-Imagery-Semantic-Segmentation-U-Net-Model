import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the model
@st.cache_resource
#def load_trained_model():
#    return load_model(
#        'satellite-imagery.h5',
##        custom_objects={
 #           'dice_loss_plus_1focal_loss': total_loss,
 #           'jaccard_coef': jaccard_coef
 #       }
 #   )

saved_model = load_model('satellite-imagery.h5')

# Define image preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Satellite Image Prediction App")
st.write("Upload a satellite image to see the original and predicted images.")

# File uploader
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Processing the image...")
    preprocessed_image = preprocess_image(uploaded_file)

    # Perform prediction
    st.write("Performing prediction...")
    prediction = saved_model.predict(preprocessed_image)

    # Post-process the prediction
    predicted_image = np.argmax(prediction, axis=3)[0, :, :]  # Remove batch dimension

    # Display the original and predicted images side by side
    st.subheader("Original and Predicted Images")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Original Image
    axes[0].set_title("Original Image")
    axes[0].imshow(uploaded_image)

    # Predicted Image
    axes[1].set_title("Predicted Image")
    axes[1].imshow(predicted_image, cmap='viridis')  # Adjust colormap as needed

    # Render the matplotlib figure in Streamlit
    st.pyplot(fig)

st.write("Note: Ensure the uploaded image is compatible with the model's expected input format.")
