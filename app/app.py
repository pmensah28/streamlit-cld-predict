import os
from os.path import join
import json
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from collections import deque


path = "/Users/princemensah/Desktop/deep-learning-projects/cld-pred-deployment/"
model = torch.load(join(path, "model/best_model_ResNeXt50_32X4D_10_epochs_fold_0.pt"), map_location=torch.device('cpu'))

# loading the class names
class_indices = json.load(open(join(path, "app/labels.json")))

# Initialize session state for feedback and history
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=10)  # Keep only the last 10 predictions

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to add prediction text to image
def add_text_to_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Calculate text size
    textbbox = draw.textbbox((0, 0), text, font=font)
    textwidth, textheight = textbbox[2] - textbbox[0], textbbox[3] - textbbox[1]
    width, height = image.size

    # Position the text at the center
    x = (width - textwidth) / 2
    y = (height - textheight) / 2

    # Draw semi-transparent rectangle behind text
    draw.rectangle([(x-5, y-5), (x + textwidth+5, y + textheight+5)], fill=(0, 0, 0, 128))

    # Draw the text
    draw.text((x, y), text, font=font, fill="red")

    return image

# Function to convert PIL image to bytes for download
def image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return byte_im

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = preprocess_image(image)
    with torch.no_grad():
        outputs = model(preprocessed_img)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = class_indices[str(predicted_class_index)]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        confidence = confidence[predicted_class_index].item()
    return predicted_class_name, confidence

# Function to add a prediction to the history
def add_to_history(image, prediction, confidence):
    st.session_state.history.appendleft((image, prediction, confidence))

# Display prediction history
def display_history():
    st.sidebar.title("Prediction History")
    for i, (image, prediction, confidence) in enumerate(st.session_state.history):
        st.sidebar.image(image, caption=f"Prediction {i+1}: {prediction} ({confidence:.2f}%)", use_column_width=True)



# Streamlit App
st.set_page_config(page_title='Plant Disease Classifier', layout='wide')
st.title('Cassava Leaf Disease Classifier')
# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This application uses a deep learning model to classify cassava leaf images into one of several categories to help identify common diseases. 
Cassava is a staple crop in many countries, and early detection of disease can help farmers in managing and treating the crops effectively.

### How to use:
1. Upload an image of a cassava leaf using the drag and drop feature or by clicking to browse.
2. The uploaded image will be displayed.
3. The application will automatically classify the image and display the predicted disease category and confidence.

### Categories:
- Cassava Bacterial Blight (CBB)
- Cassava Mosaic Disease (CMD)
- Cassava Brown Streak Disease (CBSD)
- Healthy
""")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
    
    with st.spinner('Classifying...'):
        prediction, confidence = predict_image_class(model, image, class_indices)
    st.success(f'Prediction: {prediction}')
    st.write(f'Confidence: {confidence:.2f}%')

    # Add to history
    add_to_history(image.copy(), prediction, confidence)
    
    # Ground truth input
    ground_truth = st.text_input("Enter the actual diagnosis (ground truth) for this image:", key=f"gt_{uploaded_image.name}")
    feedback = st.text_input("Any feedback?", key=f"feedback_{uploaded_image.name}")
    if st.button('Submit Feedback', key=f"submit_{uploaded_image.name}"):
        st.session_state.feedback.append((uploaded_image.name, prediction, ground_truth, feedback))
        st.success("Feedback submitted")

    # Add text overlay to image
    overlay_text = f'{prediction} ({confidence:.2f}%)'
    image_with_text = add_text_to_image(image.copy(), overlay_text)

    # Convert image to bytes for download
    img_bytes = image_to_bytes(image_with_text)

    # Provide download link
    st.download_button(
        label="Download Image with Prediction",
        data=img_bytes,
        file_name=f"predicted_{uploaded_image.name}",
        mime="image/jpeg"
    )

# Display prediction history
# display_history()

# Display feedback
if st.session_state.feedback:
    st.sidebar.title("Feedback")
    for i, (image_name, prediction, ground_truth, feedback) in enumerate(st.session_state.feedback):
        st.sidebar.write(f"**Image {i+1}:** {image_name}")
        st.sidebar.write(f"Prediction: {prediction}")
        st.sidebar.write(f"Ground Truth: {ground_truth}")
        st.sidebar.write(f"Feedback: {feedback}")
        st.sidebar.write("---")


# Footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            text-align: center;
        }
    </style>
    <div class="footer">
        <p>Powered by Streamlit</p>
    </div>
""", unsafe_allow_html=True)