import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the TensorFlow model (assuming it's a saved .h5 model)
model = tf.keras.models.load_model('model/trained_model-2.h5')




# Image preprocessing function
def preprocess_image(image_path):
    # Load the image and resize it to the input size the model expects
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    
    # Convert the image to an array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    # Normalize the image (if that was done in Colab)
    #input_arr = input_arr / 255.0
    
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    return input_arr

# Function to make a prediction
def predict_image(image_path):
    input_arr = preprocess_image(image_path)
    
    # Predict the disease using the model
    prediction = model.predict(input_arr)

    
    # Get the predicted class index (highest probability)
    result_index = np.argmax(prediction)

    # Class names corresponding to the model's output (based on your provided list)
    class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]



    
    
# Get the disease class name corresponding to the predicted index
    class_name = class_names[result_index]

    
    # Get the prediction confidence (max probability)
    confidence = np.max(prediction) * 100  # Confidence as percentage
    
    return class_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction and confidence
            class_name, confidence = predict_image(filepath)
            
            relative_path = f'images/{filename}'
            return render_template('index.html', image=relative_path, disease=class_name, confidence=confidence)
    
    return render_template('index.html')

#if __name__ == '__main__':
   #app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
   app.run(debug=True)
