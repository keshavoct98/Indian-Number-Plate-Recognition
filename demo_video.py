# Required libraries
import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode

from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
import time

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # limits gpu memory usage

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline() # downloads pretrained weights for text detector and recognizer

tf.keras.backend.clear_session()

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

flags.DEFINE_string('input', 'inputs/demo1.mp4', 'path to input video')
flags.DEFINE_string('output', 'results/output.avi', 'path to save results')
flags.DEFINE_integer('size', 608, 'resize images to')

def platePattern(string):
    '''Returns true if passed string follows
    the pattern of indian license plates,
    returns false otherwise.
    '''
    if len(string) < 9 or len(string) > 10:
        return False
    elif string[:2].isalpha() == False:
        return False
    elif string[2].isnumeric() == False:
        return False
    elif string[-4:].isnumeric() == False:
        return False
    elif string[-6:-4].isalpha() == False:
        return False
    else:
        return True
    
def drawText(img, plates):
    '''Draws recognized plate numbers on the
    top-left side of frame
    '''
    string  = 'plates detected :- ' + plates[0]
    for i in range(1, len(plates)):
        string = string + ', ' + plates[i]
    
    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    (text_width, text_height) = cv2.getTextSize(string, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((1, 30), (10 + text_width, 20 - text_height))
    
    cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(img, string, (5, 25), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
    
def plateDetect(frame, input_size, model):
    '''Preprocesses image and pass it to
    trained model for license plate detection.
    Returns bounding box coordinates.
    '''
    frame_size = frame.shape[:2]
    image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    
    return bboxes

def main(_argv):
    input_layer = tf.keras.layers.Input([FLAGS.size, FLAGS.size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, 'data/YOLOv4-obj_1000.weights')
    
    vid = cv2.VideoCapture(FLAGS.input) # Reading input
    return_value, frame = vid.read()
    
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out = cv2.VideoWriter(FLAGS.output, fourcc, 10.0, (frame.shape[1],frame.shape[0]), True)

    plates = []

    n = 0
    Sum = 0
    while True:
        start = time.time()
        n += 1
        return_value, frame = vid.read()
        if frame is None:
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = plateDetect(frame, FLAGS.size, model) # License plate detection
        for i in range(len(bboxes)):
            img = frame[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])]
            prediction_groups = pipeline.recognize([img]) # Text detection and recognition on license plate
            string = ''
            for j in range(len(prediction_groups[0])):
                string = string+ prediction_groups[0][j][0].upper()
            
            if platePattern(string) == True and string not in plates:
                plates.append(string)
        
        if len(plates) > 0:
            drawText(frame, plates)
        
        frame = utils.draw_bbox(frame, bboxes) # Draws bounding box around license plate
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        Sum += time.time()-start
        print('Avg fps:- ', Sum/n)

        out.write(frame)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27: break
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass