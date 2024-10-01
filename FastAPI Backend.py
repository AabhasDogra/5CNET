from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import io
from PIL import Image

app = FastAPI()
model = load_model('best_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize to match model input
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    prediction = model.predict(image)
    pred_mask = (prediction.squeeze() > 0.5).astype(np.uint8)  # Thresholding
    return {"mask": pred_mask.tolist()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <body>
            <h1>Upload an Image</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/tiff"/>
            <input type="submit"/>
            </form>
        </body>
    </html>
    """
