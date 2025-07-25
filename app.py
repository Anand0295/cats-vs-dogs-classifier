from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import webbrowser
import threading

app = Flask(__name__)

model = tf.keras.models.load_model("model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')
    results = []
    
    for file in files:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        prediction = model.predict(img_array)[0][0]
        
        if prediction > 0.5:
            result = "DOG"
            confidence = prediction * 100
        else:
            result = "CAT"
            confidence = (1 - prediction) * 100
        
        results.append({
            'prediction': result,
            'confidence': f"{confidence:.1f}%",
            'filename': file.filename
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True)
