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

def mse(image_1, image_2):
    err = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    err /= float(image_1.shape[0] * image_1.shape[1])
    
    return err

def get_frames(file_path, model, consecutive, err_threshold):
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
    file_path = file_path.encode('utf-8')
    print("FILE_PATH: ", file_path)

    vid = skvideo.io.VideoCapture(file_path)
   #  vid = cv2.VideoCapture(file_path)
    pred_arr = []
    orig_frames = []
    good_frames = []
    
    good_count = 0
    good_frames_count = 1000
    
    destination_list = []
    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                vid.release()
                print("Released Video Resource")
                break

            resized = cv2.resize(frame, (224, 224)).astype(np.float32)
            if not pred_arr:
                pred_arr.append(resized)
                orig_frames.append(frame)
            
            for image in pred_arr:
                err = mse(image, resized)
                if err > err_threshold:
                    pred_arr.append(resized)
                    orig_frames.append(frame)
                    break
                
            if len(pred_arr) >= consecutive:
                pred_arr = np.array(pred_arr)
                pred = model.predict(pred_arr)
                for p in pred:
                    if p > 0.5:
                        good_count += 1
                    else: 
                        good_count = 0
                        break

                if good_count == consecutive:
                    print("PRED_ARR SHAPE: ", orig_frames[consecutive // 2].shape)
                    tar = os.path.join(APP_ROOT, 'static/images/')
                    if not os.path.isdir(tar):
                        os.mkdir(tar)

                    dest = tar + str(good_frames_count) + '.jpg'
                    destination_list.append(dest)
                    print("File Path: ", dest)

                    cv2.imwrite(dest, orig_frames[consecutive // 2])
                    
                    good_frames.append(orig_frames[consecutive // 2])
                    good_frames_count += 1

                pred_arr = []
                orig_frames = []
                
        return good_frames, destination_list

    except KeyboardInterrupt:
        vid.release()
        print("Released Video Resource")

__author__ = 'paul'

app = Flask(__name__)
model = keras.models.load_model('final_model.h5')
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
                good_frames, list_of_destinations  = get_frames(destination, model=model, consecutive=5, err_threshold=500)
                print("AFTER GET_FRAMES: ")
                print(datetime.datetime.now())
                for image in list_of_destinations:
                    temp_image = re.search('[0-9]*.jpg', image)
                    image_list.append(temp_image.group(0))

    return render_template('upload.html', msg=msg, image_list=image_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
