import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "api/models/1" 
DL_model = tf.keras.models.load_model(model_path)

class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/prediction")
async def prediction(
    file: UploadFile = File(...)
    ):
    content = read_file_as_image(await file.read())
    img_batch = np.expand_dims(content, 0)
    predict = DL_model.predict(img_batch)
    class_predicted = class_names[np.argmax(predict[0])]
    most_accurate = np.max(predict[0])

    return {
        'class': class_predicted,
        'confidence': float(most_accurate)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

