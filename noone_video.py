from deep_face_detector import DeepFaceDetector

import cv2
import argparse


# haarcascade file from https://github.com/opencv/opencv/tree/master/data/haarcascades
default_haar_file = 'model/haarcascade_frontalface_default.xml'

# dnn model from https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
default_proto_file = 'model/opencv_face_detector.prototxt'
default_model_file = 'model/res10_300x300_ssd_iter_140000.caffemodel'

parser = argparse.ArgumentParser(description="Deep face detection!")

parser.add_argument('cam_number', type=int, nargs='?', default=0,
                    help='Index number of the camera')
parser.add_argument('--file', type=str,
                    help="Video file path")
parser.add_argument('--out', type=str,
                    help="Output video path")
parser.add_argument('--proto_file', type=str, default=default_proto_file,
                    help="DNN proto file path")
parser.add_argument('--model_file', type=str, default=default_model_file,
                    help="DNN model file path")
parser.add_argument('--haar_file', type=str, default=default_haar_file,
                    help="Haar cascade xml file path")
parser.add_argument('--vertical', type=int, default=0,
                    help='0 : horizontal video(default), 1 : vertical video')

parser.add_argument('--show', type=int, default=1,
                    help="0 : Hide figure, 1 : show figure")

parser.add_argument('--debug_rect', type=int, default=0,
                    help='Show rectangular around the detected faces. Default : 0')
parser.add_argument('--debug_fill_color', type=int, default=0,
                    help='Fill colors on the candidate area of the faces, Default : 0')
parser.add_argument('--debug_text', type=int, default=0,
                    help='Show text information of the detectors, Default : 0')

parser.add_argument('--detector_dnn_nocrop', type=int, default=0,
                    help='Use DNN without crop. Default : 0')
parser.add_argument('--detector_dnn_crop', type=int, default=0,
                    help='Use DNN with crop. Default : 0')
parser.add_argument('--detector_dnn_scan', type=int, default=0,
                    help='Use DNN while scanning the frame. Default : 0')
parser.add_argument('--detector_dnn_crop_scan', type=int, default=1,
                    help='Use DNN while scanning the cropped frame. Default : 1')
parser.add_argument('--detector_haar', type=int, default=0,
                    help='Use Haar detector. Default : 0')
parser.add_argument('--detector_combined', type=int, default=0,
                    help='Use combined detection result. Default : 0')
parser.add_argument('--blur_faces', type=int, default=1,
                    help='Blur detected faces')

parser.add_argument('--reduce_scale', type=float, default=2,
                    help='Reduce scale ratio. ex) 2 = half size of the input. Default : 2')

args = parser.parse_args()

if args.file is None:
    cap = cv2.VideoCapture(args.cam_number)
else:
    cap = cv2.VideoCapture(args.file)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

detector = DeepFaceDetector(args.proto_file, args.model_file, frame_size=frame_size, haar_file=args.haar_file, is_vertical=args.vertical == 1)
detector_param = [
    args.detector_dnn_nocrop == 1,
    args.detector_dnn_crop == 1,
    args.detector_dnn_scan == 1,
    args.detector_dnn_crop_scan == 1,
    args.detector_haar == 1,
    args.detector_combined == 1,
]
detector.detector_param = detector_param

if args.out is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if args.vertical == 1:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale))
    else:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale))

    out = cv2.VideoWriter(args.out,
                          fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          out_size
                          )

is_draw_rect = args.debug_rect == 1
is_fill_color = args.debug_fill_color == 1
is_text_on = args.debug_text == 1
is_blur_faces = args.blur_faces == 1

while cap.isOpened():
    ret, frame = cap.read()

    if args.vertical == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    dst_frame = frame.copy()

    faces, face_maps, face_points = detector.find_face(frame)

    if is_draw_rect:
        detector.draw_square_on_faces(dst_frame)

    if is_blur_faces:
        detector.blur_faces(dst_frame)

    resized_img = cv2.resize(dst_frame, (int(dst_frame.shape[1] // args.reduce_scale), int(dst_frame.shape[0]//args.reduce_scale)))

    if is_fill_color:
        detector.fill_color_on_faces(resized_img)
    if is_text_on:
        detector.draw_classifier_list_text(resized_img)

    if args.out is not None:
        out.write(resized_img)

    if args.show == 1:
        cv2.imshow('frame', resized_img)

    key = cv2.waitKey(1)

    if key > 0:
        if key == ord('q'):
            break
        for i in range(6):
            if key == ord(str(i+1)):
                detector.detector_param[i] = not detector.detector_param[i]

        if key == ord('r'):
            is_draw_rect = not is_draw_rect
        elif key == ord('f'):
            is_fill_color = not is_fill_color
        elif key == ord('t'):
            is_text_on = not is_text_on
        elif key == ord('b'):
            is_blur_faces = not is_blur_faces


cap.release()
if args.out is not None:
    out.release()
if args.show == 1:
    cv2.destroyAllWindows()
