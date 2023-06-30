from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the pre-trained model for background removal
model = tf.keras.models.load_model('D:/bg/models/bg.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove_background', methods=['POST'])
def remove_background():
    # Retrieve the uploaded image
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Preprocess the image
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize to match the input size of the model
    image_array = np.array(image) / 255.0  # Normalize pixel values
    
    # Remove the background using the model
    result = model.predict(np.expand_dims(image_array, axis=0))
    mask = (result > 0.5).astype(np.uint8) * 255  # Threshold the result to create a binary mask
    mask_image = Image.fromarray(mask[0], mode='L')
    
    # Apply the mask to the original image
    transparent_image = Image.new('RGBA', image.size)
    transparent_image.paste(image, mask=mask_image)
    
    # Convert the transparent image to bytes
    image_bytes = io.BytesIO()
    transparent_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    return image_bytes.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
