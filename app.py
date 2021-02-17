from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


# tensorflow & keras
#import tensorflow as tf
#from tensorflow.keras.preprocessing import image
import tflite_runtime.interpreter as tflite

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/model_1000.tflite'

# Load your trained model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_detils = interpreter.get_output_details()
print('Model loaded. Check http://127.0.0.1:5000/')



img_size = (299,299)
id_to_class = {0: 'Airplane',
				 1: 'Candle',
				 2: 'Christmas Tree',
				 3: 'Jacket',
				 4: 'Miscellaneous',
				 5: 'Snowman'}



def model_predict(img_path):
	try:
		#print("predict called")

		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		if len(img.shape)==2:
			# 1 to 3 chennels
			img = cv2.merge((img,img,img))
		elif img.shape[2] ==4:
			#remove the alpha chennel
			img = img[:,:,:3]

		img = cv2.resize(img, img_size)

		img = img/255.
		#pred = model.predict(tf.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2])))

		interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0).astype(np.float32))
		interpreter.invoke()
		pred =  interpreter.get_tensor(output_detils[0]['index'])
		lable = id_to_class[pred[0].argmax()]
		return lable

	except Exception as e:
		print(e)
		return 'Error in loading image.'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    #print("upload called")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)

        #delete the downloaded file
        os.remove(file_path)
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

