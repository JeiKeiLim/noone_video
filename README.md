# Deep face detection
Finding faces from DNN and Haar face detection. <br />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/intro01.gif" >

## Introduction
This work was to experiment face detection using OpenCV for the video anonymization.

In this repository, the all included methods were exploited from OpenCV examples including 
1. Haar detector - Classic Haar cascade detector
2. DNN with no crop
3. DNN with crop
4. DNN with split videos by the ratio
    - Crop video by 1:1 ratio and slides the crop area to the rest
5. DNN with split videos by crop size
    - Split videos by less or more than 5x5 with overlapping area and apply method 4.

I suggest you to take a look at [Video anonymization by FaceDetection-DSFD](https://github.com/JeiKeiLim/FaceDetection-DSFD) for the best performance and [Video anonymization by lightDSFD](https://github.com/JeiKeiLim/lightDSFD) for faster detection but less accurate.  

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_02.gif" />

## Usage
    usage: noone_video.py [-h] [--file FILE] [--out OUT] [--proto_file PROTO_FILE]
                          [--model_file MODEL_FILE] [--haar_file HAAR_FILE]
                          [--vertical VERTICAL] [--show SHOW]
                          [--debug_rect DEBUG_RECT]
                          [--debug_fill_color DEBUG_FILL_COLOR]
                          [--debug_text DEBUG_TEXT]
                          [--detector_dnn_nocrop DETECTOR_DNN_NOCROP]
                          [--detector_dnn_crop DETECTOR_DNN_CROP]
                          [--detector_dnn_scan DETECTOR_DNN_SCAN]
                          [--detector_dnn_crop_scan DETECTOR_DNN_CROP_SCAN]
                          [--detector_haar DETECTOR_HAAR]
                          [--detector_combined DETECTOR_COMBINED]
                          [--blur_faces BLUR_FACES] [--reduce_scale REDUCE_SCALE]
                          [--verbose VERBOSE] [--blur_level BLUR_LEVEL]
                          [cam_number]

## Blurring faces on videos <br />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/intro02.gif">

## Detailed arguments!
    
    positional arguments:
      cam_number            Index number of the camera
    
    optional arguments:
      -h, --help            show this help message and exit
      --file FILE           Video file path
      --out OUT             Output video path
      --proto_file PROTO_FILE
                            DNN proto file path
      --model_file MODEL_FILE
                            DNN model file path
      --haar_file HAAR_FILE
                            Haar cascade xml file path
      --vertical VERTICAL   0 : horizontal video(default), 1 : vertical video
      --show SHOW           0 : Hide figure, 1 : show figure
      --debug_rect DEBUG_RECT
                            Show rectangular around the detected faces. Default :
                            0
      --debug_fill_color DEBUG_FILL_COLOR
                            Fill colors on the candidate area of the faces,
                            Default : 0
      --debug_text DEBUG_TEXT
                            Show text information of the detectors, Default : 0
      --detector_dnn_nocrop DETECTOR_DNN_NOCROP
                            Use DNN without crop. Default : 0
      --detector_dnn_crop DETECTOR_DNN_CROP
                            Use DNN with crop. Default : 0
      --detector_dnn_scan DETECTOR_DNN_SCAN
                            Use DNN while scanning the frame. Default : 0
      --detector_dnn_crop_scan DETECTOR_DNN_CROP_SCAN
                            Use DNN while scanning the cropped frame. Default : 1
      --detector_haar DETECTOR_HAAR
                            Use Haar detector. Default : 0
      --detector_combined DETECTOR_COMBINED
                            Use combined detection result. Default : 0
      --blur_faces BLUR_FACES
                            Blur detected faces
      --reduce_scale REDUCE_SCALE
                            Reduce scale ratio. ex) 2 = half size of the input.
                            Default : 2
      --verbose VERBOSE     Show current progress and remaining time
      --blur_level BLUR_LEVEL
                            Blurriness of the detected face

