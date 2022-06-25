import tensorflow as tf
from tensorflow import keras
from flask import Flask, redirect, url_for, render_template, request, flash
from flask import Flask
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
print(tf.__version__)

model = keras.models.load_model('model_weights.h5')

def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def prediction(img_path):
    img = image.load_img(img_path, target_size=(64, 64,1))
    img_array = image.img_to_array(img)
    prediction = model.predict([prepare(img_path)])       
    return prediction

app = Flask(__name__)

#get_model()


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'E:\Data set\static', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        print(file_path)
        product = prediction(file_path)
        classes = {1:'ladha',2:'leaf',3:'black pest'}
        classes_x=np.argmax(product,axis=1)
        product = classes[int(classes_x)]

        
    return render_template('predict.html', product = product)
app.run()
'''

from flask import Flask
from flask import Flask, redirect, url_for, render_template, request, flash

app = Flask(__name__)
 
  
@app.route("/")
def home():
    return render_template('home.html')
    
app.run()'''
