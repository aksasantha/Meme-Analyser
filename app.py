import streamlit as st
from PIL import Image
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# Load meme classes and interpretations
meme_data = pd.read_csv("meme_classes.csv")
meme_classes = meme_data['meme_name'].tolist()

interpretation_data = pd.read_csv("meme_interpretations.csv")
interpretations = dict(zip(interpretation_data['meme_name'], interpretation_data['interpretation']))

# Set up the Streamlit app
st.title("Meme Analyser")
st.write("Upload a meme image to receive a deep, philosophical interpretation.")

uploaded_file = st.file_uploader("Choose a meme image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded meme
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Meme', use_column_width=True)

    # Prepare the inputs for CLIP
    inputs = processor(text=meme_classes, images=image, return_tensors="pt", padding=True)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    logits_per_image = outputs.logits_per_image
    predictions = logits_per_image.argmax(dim=1).item()
    predicted_class = meme_classes[predictions]
    
    st.write("Predicted Meme Class:", predicted_class)

    # Display the interpretation dynamically
    st.write("Philosophical Interpretation:", interpretations.get(predicted_class, "No interpretation available."))