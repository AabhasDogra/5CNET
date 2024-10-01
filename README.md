# Brain MRI Metastasis Segmentation

## Overview
This project aims to demonstrate proficiency in computer vision techniques by implementing and comparing Nested U-Net and Attention U-Net architectures for brain MRI metastasis segmentation.

## Dataset
The dataset consists of Brain MRI images and their corresponding metastasis segmentation masks. The dataset can be downloaded from [here](https://dicom5c.blob.core.windows.net/public/Data.zip). 

## Data Preprocessing
We implemented CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the visibility of metastases in MRI images, followed by normalization.

## Model Architectures
### Nested U-Net (U-Net++)
- An extension of the U-Net architecture, it has additional skip connections and improved feature extraction capabilities.

### Attention U-Net
- This architecture incorporates attention mechanisms to focus on the relevant features for segmentation tasks, helping to enhance performance.

## Training and Evaluation
Both models were trained on the preprocessed dataset. The primary evaluation metric used was the DICE Score.

## Web Application
The web application consists of:
- A FastAPI backend to serve the best performing model.
- A Streamlit UI for users to upload images and view segmentation results.
- 
![ex1](https://github.com/user-attachments/assets/b3976c82-0ef1-490c-bd1b-0febbe901901)


