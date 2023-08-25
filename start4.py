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

# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Face Verification</title>
                <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
        </head>
        <body>
        <div class="container"> 
        
		      <div class="col">
				 <h1>Face Verification</h1>
			</div>
				  
			<div class="col">
				 <video id="video" width="640" height="480" autoplay></video>
			</div>
				<iframe id="loading" src="https://media.tenor.com/zecVkmevzcIAAAAC/please-wait.gif" width="100" height="100" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>			
			<div class="col">
				 <button id="recognize" type="button" class="btn btn-lg btn-primary" >Recognize</button>
			</div>
			      <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        </div>
           
           
          
      
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
                
             <script>
				$(document).ready(function() {
					// Hide the iframe on page load
					$("#loading").hide();
				});
			</script>

            <script>
    			
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const recognizeButton = document.getElementById('recognize');
                const constraints = { video: true };
                
                

				recognizeButton.addEventListener('click', () => {
				$("#loading").show();
					canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
					const image = canvas.toDataURL('image/jpeg');
					fetch('/recognize', {
						method: 'POST',
						body: JSON.stringify(image),
						headers: { 'Content-Type': 'application/json' }
					})
					.then(response => response.json())
					.then(result => {
					$("#loading").hide();
     					console.log(result);
						if (result.prediction === 'Human') {
							
					console.log(result);
							alert('success');
							// Redirect to another page here
							// window.location.href = '/success';
						} else {
							alert('Try again');
							// Keep the video feed active for another attempt
							setupCamera();
						}
					}).catch((error) => {
					$("#loading").hide();
					console.log("showing error");
					console.log(error);
					alert(error);
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
    return "Human" if prediction < 0.02 else "Non-Human"
    
def save_img(img_bin):
	import cloudinary
          
	cloudinary.config( 
	  cloud_name = "dcysfieol", 
	  api_key = "851589193853581", 
	  api_secret = "J2FZWZLTigmfpt9VEozTm7tbzFE" 
	)
	return cloudinary.utils.cloudinary_url(img_bin, width=100, height=150, crop="fill")
	
import cloudinary
from cloudinary.uploader import upload

# Configure Cloudinary with your credentials
cloudinary.config( 
	  cloud_name = "dcysfieol", 
	  api_key = "851589193853581", 
	  api_secret = "J2FZWZLTigmfpt9VEozTm7tbzFE" 
	)

def upload_to_cloudinary(image_path):
    response = upload(image_path)
    return response['secure_url']  # Return the URL of the uploaded image

	

@app.post("/recognize")
async def recognize(image: dict):
	try:
		image_data = image['image'].split(",")[1].encode('utf-8')
		result = predict_image(io.BytesIO(base64.b64decode(image_data)))
		if result == "Human":
			with open("captured_image.jpg", "wb") as f:
			    f.write(base64.b64decode(image_data))
			cloudinary_url = upload_to_cloudinary("captured_image.jpg")
			print(cloudinary_url)
			return {"prediction": result, "message": "Image saved", "cloudinary_url": cloudinary_url}
		else:
			return {"prediction": result, "message": "Try again"}
	except Exception as err:
		return {"prediction": "no prediction", "message": str(err)}
