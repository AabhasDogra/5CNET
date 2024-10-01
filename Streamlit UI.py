import streamlit as st
import requests
import numpy as np
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a .tif file", type="tif")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Send image to FAST API
    response = requests.post("http://localhost:8000/predict/", files={"file": uploaded_file})
    if response.status_code == 200:
        mask = np.array(response.json()["mask"])
        st.image(mask, caption='Predicted Segmentation Mask', use_column_width=True)
