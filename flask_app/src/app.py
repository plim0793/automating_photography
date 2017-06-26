import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import keras
import skvideo.io
import datetime
import re
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')

def get_frames(file_path, top_layer, bottom_layers, path, threshold):
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
        print("Invalid video file")
        return None
    
    feature_vec_list = []
    orig_frames = []
    good_frames = []
    curr_feat_vec = []
    
    good_count = 0
    good_frames_count = 0

    for i in range(vid.get_length()):
        try:
            frame = vid.get_data(i)
        except:
            print("Frame could not be read.")
            continue

        resized = np.array([cv2.resize(frame, (224, 224)).astype(np.float32)])
        feat_vec = bottom_layers.predict(resized)
        if curr_feat_vec == []:
            curr_feat_vec = feat_vec

        if cosine_similarity(curr_feat_vec, feat_vec) > threshold or len(feature_vec_list) == 0:
            feature_vec_list.append(feat_vec)
            orig_frames.append(frame)
            curr_feat_vec = feat_vec
        else:
            print("LENGTH OF CURRENT SCENE: {}".format(len(feature_vec_list)))
            pred = top_layer.predict(np.array(feature_vec_list))
            if not os.path.isdir(path):
                os.mkdir(path)

            file_path = os.path.join(path, str(good_frames_count) + str(uuid.uuid4()) + '.jpg')
            cv2.imwrite(file_path, orig_frames[np.argmax(pred)])
            good_frames.append(orig_frames[np.argmax(pred)])
            good_frames_count += 1
            print("File Path: {}".format(file_path))

            feature_vec_list = []
            orig_frames = []

    print("Good Frames Count: {}".format(good_frames_count))

    return good_frames


__author__ = 'paul'

app = Flask(__name__)
model = keras.models.load_model('model.h5')
print('MODEL LOADED')
graph = tf.get_default_graph()


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
            msg = "Upload Complete"
            print(msg)
		
            print("BEFORE GET_FRAMES: ")
            print(datetime.datetime.now())
            with graph.as_default():
                bot, top = split_model(model=model)
                good_frames = get_frames(destination,top_layer=top, bottom_layers=bot, path='static/images/') 
                print("AFTER GET_FRAMES: ")
                print(datetime.datetime.now())
                image_list = ['static/images/' + f for f in listdir('static/images/') if isfile(join('static/images/', f))]

    return render_template('upload.html', msg=msg, image_list=image_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
