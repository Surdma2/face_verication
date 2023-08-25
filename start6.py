
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.nasnet import preprocess_input
import base64

app = FastAPI()

# Load TensorFlow model
model = tf.keras.models.load_model('last_face_model.h5', compile=False)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Face Verification</title>
        </head>
        <body>
            <h1>Face Verification</h1>
            <video id="video" width="640" height="480" autoplay></video>
            <button id="recognize">Recognize</button>
            <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
            <script>
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const recognizeButton = document.getElementById('recognize');
                const constraints = { video: true };
                recognizeButton.addEventListener('click', () => {
                    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                    const image = canvas.toDataURL('image/jpeg');
                    fetch('/recognize', {
                        method: 'POST',
                        body: JSON.stringify({ image }),
                        headers: { 'Content-Type': 'application/json' }
                    })
                    .then(response => response.json())
                    .then(result => {
                        alert(result.prediction);
                    });
                });
                async function setupCamera() {
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = stream;
                    await video.play();
                }
                setupCamera();
            </script>
        </body>
    </html>
    """

def predict_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = prediction[0][0]
    # prediction = tf.nn.softmax(prediction)
    print(prediction)
    # prediction = np.argmax(prediction)
    return "Human" if prediction < 0.05 else "Non-Human"

@app.post("/recognize")
async def recognize(image: dict):
    image_data = image['image'].split(",")[1].encode('utf-8')
    result = predict_image(io.BytesIO(base64.b64decode(image_data)))
    if result == "Human":
        with open("saved_image.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))
    else:
        pass
    return {"prediction": result}
