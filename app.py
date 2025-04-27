from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

import os

app = Flask(__name__)

# Load the trained Keras model
model = load_model('MODEL/model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from the request
    image_data = request.form['image_data']
    
    # Decode the base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])
    
    # Convert byte data to image
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Reshape the image to (1, 28, 28, 1) as expected by Keras model
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction, axis=1)[0])
    
    return jsonify({'predicted_digit': predicted_digit})

# Serve the UI
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Configure static folder for serving index.html
app._static_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templetes'))

if __name__ == '__main__':
    app.run(debug=True)
