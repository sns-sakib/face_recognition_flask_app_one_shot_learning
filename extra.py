import tensorflow as tf
import os
from tensorflow.keras.models import Model,Sequential,model_from_json
from app import APP_ROOT
import sqlite3
conn = sqlite3.connect("encodings.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
cur.execute('Select * from enc')
data = cur.fetchall()
model_path = "Models/Inception_ResNet_v1.json"
weights_path = "Models/facenet_keras_weights.h5"
enc_model = model_from_json(loaded_model_json)
enc_model.load_weights(weights_path)
for k,l in data:
    print(k,l)
    print()
    #print(pred_enc)