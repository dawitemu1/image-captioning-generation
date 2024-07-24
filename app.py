import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

# Define the Encoder and Decoder classes as above or import them if they're in another module

# Load the model
embed_size = 256
hidden_size = 512
num_layers = 1
vocab_size = len(vocab)  # Make sure vocab is available or load it from a saved file

model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers)
model.load_state_dict(torch.load('image_captioning_model.pth'))
model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("Image Captioning with Attention Visualization")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Generate caption
    with torch.no_grad():
        features = model.encoder(image_tensor)
        output = model.decoder.generate_caption(features, vocab)
        
        # Assuming attention_map is obtained during generation
        attention_map = ...  # Replace with actual attention map extraction

    st.write("Generated Caption: " + ' '.join(output))

    # Visualize attention
    def visualize_attention(image, caption, attention_map):
        fig = plt.figure(figsize=(15, 15))
        len_s = len(caption)
        for i in range(len_s):
            ax = fig.add_subplot(len_s // 5 + 1, 5, i + 1)
            ax.imshow(image)
            ax.set_title(caption[i])
            current_alpha = attention_map[i, :].cpu().data.numpy().reshape(7, 7)
            alpha_img = skimage.transform.pyramid_expand(current_alpha, upscale=32, sigma=20)
            ax.imshow(alpha_img, alpha=0.7)
        st.pyplot(fig)

    visualize_attention(image, output, attention_map)
