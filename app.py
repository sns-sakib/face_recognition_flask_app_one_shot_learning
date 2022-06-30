import pickle
import json
from flask import Flask, request, url_for, redirect, render_template,Response
import os
import matplotlib.pyplot as plt
import pandas as pd
import face_recognition
import tensorflow as tf
import numpy as np
# from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential,model_from_json
from tensorflow.keras.callbacks import *
from sklearn.utils import shuffle
from random import choice
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
import tensorflow.keras.backend as K
import h5py
import datetime
from PIL import Image
import cv2
import csv
import pickle
import sqlite3



app = Flask(__name__)
import io
##https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
##To save numpy array in sqlite database
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


APP_ROOT = os.path.dirname(os.path.abspath('__file__'))
@app.route('/')
def index():
    return render_template('index.html')

## Redirct to capturing image
@app.route('/CaptureImage')
def CaptureImage():
    return render_template('CaptureImage.html')

##Redirecting to add image and encogins to database
@app.route('/add', methods=['GET', 'POST'])
def add():
    return render_template('add.html')

##Redirect to removing person from database
@app.route('/delete', methods=['GET', 'POST'])
def delete():
    return render_template('delete.html')

##Redirecting to prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')


##Adding person to database
@app.route('/sucess', methods=['POST'])
def sucess():
    ##Encodings
    if os.path.exists('encodings.db'):
        conn = sqlite3.connect("encodings.db", detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()

#         en_di = json.load(enc_file)
    else:
        conn = sqlite3.connect("encodings.db", detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        cur.execute("create table enc (name TEXT UNIQUE,encodings array)")
    # Loading Pretrained model
    model_path = "Models/Inception_ResNet_v1.json"
    weights_path = "Models/facenet_keras_weights.h5"
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    enc_model = model_from_json(loaded_model_json)
    enc_model.load_weights(weights_path)
    print(APP_ROOT)
    if not os.path.exists("images"):
        os.mkdir("images")
    p_name = request.form['name']
    target = os.path.join(APP_ROOT,'images',p_name)

    print('Target:',target)
    if not os.path.exists(target):
        os.mkdir(target)
    print(target)
    file = request.files.get('file')
  
    print(file)
   
    
    file_name = file.filename
    dest = '/'.join([target,file_name])
    # print(file)
    file.save(dest)
    ##Extracting face
    i = plt.imread(file)
    
    face = face_recognition.face_locations(i)
    top, right, bottom, left = face[0]
    im = cv2.resize(i[top:bottom,left:right]/255,(160,160))
    im = im.reshape(160,160,3)

    encodings = enc_model.predict(np.expand_dims(im,0))
#     print(p_name)
#     print(np.array(encodings))
    print("done encoding")
    try:
        cur.execute("insert into enc (name,encodings) values (?,?)", (p_name,encodings))
        conn.commit()
#         data = cur.execute("select * from enc")
#         print(data.fetchall())
        return 'Sucessfully added '+str(p_name)+" encodings to database"
    except:
        return str(p_name)+" already exits in database."





# ## ref : https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
##To show the video while capturing
@app.route('/video_feed')
def video_feed():
    
    camera = cv2.VideoCapture(0)
    def gen_frames():  
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


## Capture the image from webcam and save in local drive
@app.route('/capture', methods=['POST','GET'])
def capture():
    camera = cv2.VideoCapture(0)
    n = request.form['i_name']
    ret,frame = camera.read()
    try:
        os.mkdir('images/'+str(n))
        dir_path = os.path.join('images/',n)
        img_name = "{}.jpg".format(n)
        cv2.imwrite(os.path.join(dir_path,img_name), frame)
        print("{} written!".format(img_name))
        return "Image_saved"
    except:
        return 'Images of this person are already in database.'


##Predicting the given person from database
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    min_distance = 13
    thresh = 12
    ##Encodings
    if os.path.exists('encodings.db'):
        conn = sqlite3.connect("encodings.db", detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
    else:
        return "No encodings database found!!"
    ##Loading model
    model_path = "Models/Inception_ResNet_v1.json"
    weights_path = "Models/facenet_keras_weights.h5"
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    enc_model = model_from_json(loaded_model_json)
    enc_model.load_weights(weights_path)

    file = request.files.get('pred_file')
    i = plt.imread(file)

    ##Extracting face
    face = face_recognition.face_locations(i)
    top, right, bottom, left = face[0]
    im = cv2.resize(i[top:bottom,left:right]/255,(160,160))
    im = im.reshape(160,160,3)

    pred_enc = enc_model.predict(np.expand_dims(im,0))
    cur.execute('Select * from enc')
    data = cur.fetchall()
    for k,l in data:
        print(l)
        print(pred_enc)
        distance = np.linalg.norm(l - pred_enc)
        if distance < min_distance:
            min_distance = distance
            pred_name = k
    if min_distance > thresh:
        return ("Unknown Person")
    else:
        return render_template('/prediction.html', name = pred_name)

##Deleting the person from database
@app.route('/remove', methods=['GET', 'POST'])
def remove():
    ##Encodings
    if os.path.exists('encodings.db'):
        conn = sqlite3.connect("encodings.db", detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
    else:
        return "No encodings database found!!"
    name_d = request.form['del_name']
    try:
        cur.execute('delete from enc where name = (?)',(name_d,))
        conn.commit()
        return name_d+" removed from database"
    except:
        return 'No person named '+str(name_d)+' in database.'

if __name__ == '__main__':
    app.run()