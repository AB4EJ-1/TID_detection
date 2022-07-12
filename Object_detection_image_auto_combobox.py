
######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Modified by W. Engelke, Universty of Alabama, Jan. 2022
# Now uses compatibility code for tf.GraphDef, tf.gfile.GFile, and tf.Session
# Note: run under anaconda; will use GPU as long as V10.0 of CUDA installed and V7 of CuDNN.
# The dlls must be in \Windows\System32 to be found.

# This uses a modified version of /utils/visualization_utils.py - this program includes a feature to
# identify a bounding box around all detected TIDs, outputting #TIDs, TID start, TID end, and TID total time
# to a csv file. TODO: there is too much hard coded in the csv file creation: needs to be improved
# to make this more flexible. 

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

def proc_image(image_path, switchMonth, output_path):
    global currentMonth
    global monthArray
    global hoursArray
    global monthTotal
    global monthInProcess
    global prev_fdate
    global fdate
    global monthlyCount
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

#add this part to count objects
    final_score = np.squeeze(scores)    
    count = 0
    for i in range(100):
      #  print("score ",i,final_score[i])
        if scores is None or final_score[i] > 0.5:
          count = count + 1

    print("count of TIDs detected:", count)
    monthlyCount = monthlyCount + count
    # subtract 1 to use array starting with zero
    monthArray[currentMonth-1] = monthArray[currentMonth-1] + count
    #df = open('D:\\images\\files_analysis_40m_NA.csv','a+')
    df = open(output_path,'a+')
    df.write(str(count) + ",")


 #   print("Box limits:",boxes[0,3][0],boxes[0,3][1])
 #   print("boxes type:", type(boxes[0,0]))
 #   print("Boxes shape:",boxes[0,0].size)

# Draw the results of the detection (aka 'visulaize the results')
    TID_time = 0
    image, TID_time = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        TID_time,
        output_path,
        use_normalized_coordinates=True,
        line_thickness=0,
        min_score_thresh=0.60
        )
    hoursArray[currentMonth - 1] = hoursArray[currentMonth - 1] + TID_time
    if switchMonth:
      #  print("at switch, prev_fdate="+prev_fdate)
        df.write(prev_fdate + "," + str(monthTotal) + "," + str (monthlyCount) + "," + "\n")
        monthTotal = 0     # hours of TID activity
        monthlyCount = 0   # count of TIDs for the month
     #   print("switch, month=", currentMonth)
        monthInProcess = currentMonth
    else:
        df.write("\n")
        prev_fdate = fdate
     #   print("fdate now =" + fdate)
     #   d = input() # temporary
        monthTotal = monthTotal + TID_time
    df.close()
    print("*** TID time ***" , TID_time)

 #   cv2.imwrite("C:\\temp\\checked2020-01-17.png",image)
# All the results have been drawn on image. Now display the image.
  #  cv2.imshow('Object detector', image)

# Press any key to close the image
# comment this out to race thru images as fast as possible
 #   cv2.waitKey(0)

# Clean up
  #  cv2.destroyAllWindows()

## end of function

#########################################################################
# Start of main program

global currentMonth
global monthArray
global hoursArray
global monthTotal
global monthInProcess
global prev_fdate
global fdate
global monthCount

prev_fdate = ''
monthTotal = 0
monthlyCount = 0
monthArray = np.array((0,0,0,0,0,0,0,0,0,0,0,0))
hoursArray = np.array((0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0))

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'PSK_2020-12-09.jpg'
#IMAGE_NAME =  '2020-11-03_PSK_20.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()
#IMAGE_PATH = "D:\\images\\jpg_2020_new_color2"
#IMAGE_PATH = "C:\\Users\\engel\\Box\\share\\images2020"
#IMAGE_PATH = "C:\\TEMP"
#IMAGE_PATH = "C:\\Users\\engel\\Box\share\\images2020"

IMAGE_PATH = "E:\\multisource_plots_EU_20m"
PATH_TO_OUTPUT = 'D:\\images\\files_analysis_20m_EU.csv'

#IMAGE_PATH = "D:\\images\\multisource_temp"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(IMAGE_PATH,IMAGE_NAME)
print("Path to image: ",PATH_TO_IMAGE)
# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

monthInProcess = 0
starting = True
fdate = ''

# proces files in this directory
for imgfile in os.listdir(IMAGE_PATH):  
    
    PATH_TO_IMAGE = os.path.join(IMAGE_PATH,imgfile)
    if PATH_TO_IMAGE.endswith('.jpg'):
        print("Process file: ", PATH_TO_IMAGE)
        fname = os.path.basename(PATH_TO_IMAGE)
        print("fname:",fname)
      #  print("file date: ",fname[6:16])
        fdate = fname[6:16]
        currentMonth = int(fdate[5:7])
        switchMonth = False
    #    print("current month:",currentMonth," monthInProcess:",monthInProcess)
        if starting:
            monthInProcess = currentMonth
            starting = False
        else:
            if currentMonth != monthInProcess:  # we hit next month
                switchMonth = True
                print("SWITCH MONTH")
   #     print("current month:",currentMonth," monthInProcess:",monthInProcess)
     #   df = open('D:\\images\\files_analysis_40m_NA.csv','a+')
        df = open(PATH_TO_OUTPUT,'a+')
        df.write(fdate + ",")
        df.close()
        proc_image(PATH_TO_IMAGE, switchMonth, PATH_TO_OUTPUT)


df = open(PATH_TO_OUTPUT,'a+')
for i in range(12):
    df.write(str(monthArray[i]) + ",")
df.write("\n")
for i in range(12):
    df.write(str(hoursArray[i]) + ",")
df.close()

print(monthArray)
print(hoursArray)



