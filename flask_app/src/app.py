#### LIBRARY IMPORTS ####
import os
import subprocess
from flask import Flask, render_template, request
import cv2
import numpy as np
import imageio
import datetime
import re
import shutil
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import logging

import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Sequential
from keras import layers
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Model

K.set_image_dim_ordering('tf')

#### SET LOGGING ####
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("BACKEND: {}".format(str(keras.backend.backend())))

handler = logging.FileHandler('logging_records.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

#### HELPER FUNCTIONS ####
def split_model(model):
    '''
    DESCRIPTION:
        - Splits the top layer of the model with the rest of the model.
    INPUT:
        - model should be pretrained.
    OUTPUT:
        - Returns the bottom_layers and a newly created top_layer.
    '''
    bottom_layers = model
    bottom_layers.layers.pop()
    bottom_layers.layers.pop()
    inp = bottom_layers.input
    out = bottom_layers.layers[-1].output

    bottom_layers = Model(inp, out)
    
    top_layer = Sequential()
    top_layer.add(Dropout(0.5, input_shape=bottom_layers.output_shape))
    top_layer.add(Dense(1))
    top_layer.add(Activation(activation='sigmoid'))

    top_layer.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                loss='binary_crossentropy', metrics=['accuracy'])
    
    return bottom_layers, top_layer


def get_frames(file_path, top_layer, bottom_layers, path, sim_threshold, good_threshold, consecutive):
    '''
    DESCRIPTION:
        - Given a video file, this function predicts frame-by-frame if the picture is "good" or "bad".
    INPUT:
        - file_path is a valid video file. Must be a string.
        - model should be fit and able to be predicted on. Ideally should be a binary classifier for this use case.
        - consecutive should be the number of consecutive good photos the model needs to see before saving the photo.
    OUTPUT:
        - prints if the frame is good or bad.
    '''
    try:
        vid = imageio.get_reader(file_path)
    except:
        logger.error("Invalid video file")
        return None
    
    feature_vec_list = []
    orig_frames = []
    good_frames = []
    curr_feat_vec = []
    
    frame_count = 0
    good_count = 0
    good_frames_count = 0
    
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        
    os.makedirs(path)

    for i in range(vid.get_length()):
        try:
            frame = vid.get_data(i)
        except:
            logger.warning("Frame could not be read.")
            continue

        resized = np.array([cv2.resize(frame, (224, 224)).astype(np.float32)])
        feat_vec = bottom_layers.predict(resized)
        
        if curr_feat_vec == []:
            curr_feat_vec = feat_vec
        if cosine_similarity(curr_feat_vec, feat_vec) > sim_threshold or len(feature_vec_list) == 0:
            feature_vec_list.append(feat_vec)
            orig_frames.append(frame)
            curr_feat_vec = feat_vec
            if len(feature_vec_list) == consecutive:
                logger.info("LENGTH OF CURRENT SCENE (CONSECUTIVE): {}".format(str(len(feature_vec_list))))
                pred = top_layer.predict(np.array(feature_vec_list))
                if pred[np.argmax(pred)] < good_threshold:
                    feature_vec_list = []
                    orig_frames = []
                    continue
                if good_frames_count < 10:
                    file_path = os.path.join(path, "000" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
                elif good_frames_count < 100:
                    file_path = os.path.join(path, "00" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
                else:
                    file_path = os.path.join(path, "0" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
                orig_frames[np.argmax(pred)] = cv2.cvtColor(orig_frames[np.argmax(pred)], cv2.COLOR_RGB2BGR)    
                cv2.imwrite(file_path, orig_frames[np.argmax(pred)])
                good_frames.append(orig_frames[np.argmax(pred)])
                good_frames_count += 1

                pred = np.delete(pred, np.argmax(pred))
                logger.info("File Path: {}".format(str(file_path)))
                feature_vec_list = []
                orig_frames = []
        else:
            logger.info("LENGTH OF CURRENT SCENE (CHANGE): {}".format(str(len(feature_vec_list))))
            pred = top_layer.predict(np.array(feature_vec_list))
            if pred[np.argmax(pred)] < good_threshold:
                feature_vec_list = []
                orig_frames = []
                continue
            if good_frames_count < 10:
                file_path = os.path.join(path, "000" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
            elif good_frames_count < 100:
                file_path = os.path.join(path, "00" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
            else:
                file_path = os.path.join(path, "0" + str(good_frames_count) + "_" + str(uuid.uuid4()) + '.jpg')
            
            orig_frames[np.argmax(pred)] = cv2.cvtColor(orig_frames[np.argmax(pred)], cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, orig_frames[np.argmax(pred)])
            good_frames.append(orig_frames[np.argmax(pred)])
            good_frames_count += 1

            pred = np.delete(pred, np.argmax(pred))
            logger.info("File Path: {}".format(str(file_path)))

            feature_vec_list = []
            orig_frames = []

    logger.info("Good Frames Count: {}".format(str(good_frames_count)))

    return good_frames


#### APP ####
__author__ = 'paul'

app = Flask(__name__)
model = keras.models.load_model('model.h5')
print('MODEL LOADED')
graph = tf.get_default_graph()
SIM_THRESHOLD = 0.50
GOOD_THRESHOLD = 0.95
CONSECUTIVE = 300

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)
@app.route("/", methods=["GET", "POST"])
def upload():
    msg = ""
    destination = ""
    list_of_destinations = []
    image_list = []
    target = os.path.join(APP_ROOT, "videos/")
    print("TARGET: ", target)

    if request.method == 'POST':
        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("video"):
            print(file.filename)
            filename = file.filename
            destination = os.path.join(target, filename)
            file.save(destination)
            print("DESTINATION: ", destination)
	
            print("VALID DESTINATION: ", os.path.isfile(destination))
            if os.path.isfile(destination):
                msg = "Here are the Best Photos"
                print(msg)
		
                print("BEFORE GET_FRAMES: ")
                print(datetime.datetime.now())
                with graph.as_default():
                    bot, top = split_model(model=model)
                    good_frames = get_frames(destination,\
					top_layer=top,\
					bottom_layers=bot,\
					path='static/images/',\
					sim_threshold=SIM_THRESHOLD,\
					good_threshold=GOOD_THRESHOLD,\
					consecutive=CONSECUTIVE) 
                    print("AFTER GET_FRAMES: ")
                    print(datetime.datetime.now())
                    image_list = ['static/images/' + f for f in os.listdir('static/images/') if os.path.isfile(os.path.join('static/images/', f))]

    return render_template('upload.html', msg=msg, image_list=image_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
