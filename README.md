# TID_detection
Uses Tensorflow Object Detection to find Large Scale Traveling Ionospheric Disturbance signature in radio spot plots.
It is stock Tensorflow Object Detection EXCEPT that it uses a customized program Object_detection_image_auto_combobox.py
to go through a directory of spot plots to find and determine size of TID activity signatures; and to do this it calls
a customized version of vis_util.visualize_boxes_and_labels_on_image_array, which outputs a .csv file containing
(for each spot plot) the number of TIDs detected and the hours of TID activity (which is extent limits of all detected
TID boxes).
