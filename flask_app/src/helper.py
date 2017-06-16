import cv2
import numpy as np
import keras
import skvideo.io


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
#     vid = skvideo.io.VideoCapture(file_path)
    vid = cv2.VideoCapture(file_path)
    pred_arr = []
    orig_frames = []
    good_frames = []
    
    good_count = 0
    good_frames_count = 0

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
                    file_path = 'data/good_photos/' + str(good_frames_count) + '.jpg'
                    print("File Path: ", file_path)
                    cv2.imwrite(file_path, orig_frames[consecutive // 2])
                    
                    good_frames.append(orig_frames[consecutive // 2])
                    good_frames_count += 1

                pred_arr = []
                orig_frames = []
                
        return good_frames

    except KeyboardInterrupt:
        vid.release()
        print("Released Video Resource")