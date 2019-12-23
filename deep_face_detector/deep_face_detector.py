import cv2
import numpy as np


class DeepFaceDetector:

    def __init__(self, prototxt, model_file, frame_size, haar_file=None, is_vertical=False):
        self.prototxt = prototxt
        self.model_file = model_file
        self.haar_file = haar_file

        if self.haar_file is not None:
            self.cascade = cv2.CascadeClassifier(self.haar_file)
        else:
            self.cascade = None

        self.net = cv2.dnn.readNet(self.prototxt, self.model_file)
        self.is_vertical = is_vertical

        if self.is_vertical:
            self.frame_size = (frame_size[1], frame_size[0])
        else:
            self.frame_size = frame_size

        self.latest_faces = None
        self.latest_face_maps = None
        self.latest_face_points = None

        # find face deep -- no crop
        # find face deep -- crop
        # scan_face_deep
        # cropped_scan_face_deep
        # haar
        self.detector_param = [True, False, True, True, False, True]

        self.c_colors = ((255, 100, 100),
                      (100, 100, 255),
                      (0, 0, 255),
                      (255, 0, 255),
                      (0, 255, 0),
                      (255, 255, 0))
        # self.c_colors = np.array(self.c_colors)

    def find_face(self, frame, accept_confidence=(0.2, 0.2, 0.2, 0.2)):
        faces = [None] * 5

        if self.detector_param[0]:
            faces[0] = self.find_face_deep(frame, accept_confidence=accept_confidence[0], crop=False)
        if self.detector_param[1]:
            faces[1] = self.find_face_deep(frame, accept_confidence=accept_confidence[1], crop=True)
        if self.detector_param[2]:
            faces[2] = self.scan_face_deep(frame, accept_confidence=accept_confidence[2])
        if self.detector_param[3]:
            faces[3] = self.cropped_scan_face_deep(frame, accept_confidence=accept_confidence[3])
        if self.detector_param[4]:
            faces[4] = self.find_face_haar(frame)

        face_maps = np.array([self.face_list_to_map(faces[i]) for i in range(len(faces))])
        sum_face_map = np.sum(face_maps, axis=0).reshape((1, face_maps.shape[1], face_maps.shape[2]))
        face_maps = np.concatenate([face_maps, sum_face_map], axis=0)

        max_vs = [0.5, 0.5, 1, 1, 0.5]
        max_vs.append(np.sum(max_vs * np.array(self.detector_param)[:-1]))

        detected_face_points = [self.detect_rectanglulars(face_maps[i], max_v=max_vs[i]) for i in range(face_maps.shape[0])]

        self.latest_faces = faces
        self.latest_face_maps = face_maps
        self.latest_face_points = detected_face_points

        return faces, face_maps, detected_face_points

    def draw_square_on_faces(self, dst_frame):

        for i in range(len(self.detector_param)):
            if self.detector_param[i]:
                self.draw_sqaure_on_face(self.latest_face_points[i], dst_frame,
                                         box_color=self.c_colors[i], max_line_width=5)

    def fill_color_on_faces(self, dst_frame):
        fill_colors = np.array(self.c_colors) / 255

        n_classifier = np.sum(self.detector_param)

        if n_classifier == 0:
            return

        result = np.zeros((dst_frame.shape[0], dst_frame.shape[1], 3))

        for i in range(len(self.detector_param)):
            if self.detector_param:
                result += self.face_map_to_color_map(self.latest_face_maps[i],
                                                     s_color=(0, 0, 0), e_color=fill_colors[i],
                                                     size=(dst_frame.shape[1], dst_frame.shape[0])) * (1 / n_classifier)

        result[result > 255] = 255
        result = result.astype('uint8')

        cv2.addWeighted(result, 0.5, dst_frame, 0.5, 0, dst_frame)

    def blur_faces(self, dst_frame, detector_idx=5):
        if  len(self.latest_face_points[detector_idx]) == 0:
            return

        for i in range(len(self.latest_face_points[detector_idx])):
            left = self.latest_face_points[detector_idx][i][0].astype('int')
            top = self.latest_face_points[detector_idx][i][1].astype('int')
            right = self.latest_face_points[detector_idx][i][2].astype('int')
            bottom = self.latest_face_points[detector_idx][i][3].astype('int')

            dst_frame[top:bottom, left:right] = cv2.blur(dst_frame[top:bottom, left:right], (31, 31))

    def draw_classifier_list_text(self, dst_frame):
        c_names = ['(1) DNN - no crop',
                   '(2) DNN - crop',
                   '(3) DNN - scan',
                   '(4) DNN - cropped scan',
                   '(5) Haar',
                   '(6) Combined']

        for i in range(len(c_names)):
            if self.detector_param[i]:
                color = self.c_colors[i]
            else:
                color = np.array(self.c_colors[i]) * 0.2

            cv2.putText(dst_frame, c_names[i], (10, 30+(i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def find_face_haar(self, frame, resize=2.0, weight=0.75):

        if resize == 1:
            resized_frame = frame
        else:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] // resize), int(frame.shape[0] // resize)))

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(gray, 1.1, 3)

        face_list = []
        for (x, y, w, h) in faces:
            x = np.round(x * resize).astype('int')
            y = np.round(y * resize).astype('int')
            w = np.round(w * resize).astype('int')
            h = np.round(h * resize).astype('int')

            face_list.append([x, y, x+w, y+h, weight])

        return np.array(face_list)

    def find_face_deep(self, frame, accept_confidence=0.2, crop=False):

        blob_img = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, crop)

        self.net.setInput(blob_img)
        out = self.net.forward()

        detection_list = []

        for i in range(out.shape[2]):
            confidence = out[0, 0, i, 2]
            left = int(out[0, 0, i, 3] * frame.shape[1])
            top = int(out[0, 0, i, 4] * frame.shape[0])
            right = int(out[0, 0, i, 5] * frame.shape[1])
            bottom = int(out[0, 0, i, 6] * frame.shape[0])

            left = max(0, min(left, frame.shape[1] - 1))
            top = max(0, min(top, frame.shape[0] - 1))
            right = max(0, min(right + 1, frame.shape[1]))
            bottom = max(0, min(bottom + 1, frame.shape[0]))

            if confidence > accept_confidence:
                detection_list.append([left, top, right, bottom, confidence])

        if len(detection_list) > 0:
            return np.array(detection_list)

    def scan_face_deep(self, frame, accept_confidence=0.2, verbose=0):
        dir = int((frame.shape[1] - frame.shape[0]) > 0)

        n_scan = np.ceil( ((frame.shape[dir] / frame.shape[1-dir])-.5) * 2 ).astype('int')
        scan_step = np.ceil( (frame.shape[dir]-frame.shape[1-dir]) / (n_scan-1) ).astype('int')

        face_list = []

        for i in range(n_scan):
            s = i*scan_step
            e = min(s + frame.shape[1-dir], frame.shape[dir])

            if dir > 0:
                scan_frame = frame[:, s:e]
            else:
                scan_frame = frame[s:e, :]

            detected_face = self.find_face_deep(scan_frame, accept_confidence=accept_confidence, crop=True)

            if verbose > 1:
                cv2.imshow('frame %0d'%(i), cv2.resize(scan_frame, (500, 500)))

            if detected_face is not None:
                detected_face[:, 1-dir] += s
                detected_face[:, 3-dir] += s

                face_list.append(detected_face)

        if len(face_list) > 0:
            return np.concatenate(face_list, axis=0)

    def cropped_scan_face_deep(self, frame, accept_confidence=0.2, scale_factor=0.5, verbose=0):
        w_size = int(frame.shape[1] * scale_factor)
        h_size = int(frame.shape[0] * scale_factor)

        n_move = np.ceil((1/scale_factor) * 1.25).astype('int')
        n_step_w = np.ceil((frame.shape[1]-w_size) / n_move).astype('int')
        n_step_h = np.ceil((frame.shape[0] - h_size) / n_move).astype('int')

        face_lists = []
        for i in range(n_move):
            for j in range(n_move):
                sw = (i*n_step_w)
                ew = min((i*n_step_w)+w_size, frame.shape[1])
                sh = (j*n_step_h)
                eh = min((j*n_step_h)+h_size, frame.shape[0])

                cropped_frame = frame[sh:eh, sw:ew, :]

                face_list = self.scan_face_deep(cropped_frame, accept_confidence=accept_confidence, verbose=verbose)

                if face_list is not None:
                    # face_list *= (1/scale_factor)

                    face_list[:, 0] += sw
                    face_list[:, 2] += sw
                    face_list[:, 1] += sh
                    face_list[:, 3] += sh

                    face_lists.append(face_list)

        if len(face_lists) > 0:
            return np.concatenate(face_lists, axis=0)

    def face_list_to_map(self, face_list):
        map = np.zeros((self.frame_size[0], self.frame_size[1]))

        if face_list is None:
            return map

        for face in face_list:
            top = face[1].astype('int')
            bottom = face[3].astype('int')
            left = face[0].astype('int')
            right = face[2].astype('int')

            map[top:bottom, left:right] += face[4]

        return map

    def face_map_to_color_map(self, face_map, s_color=(0.2, 0.2, 0.2), e_color=(0, 0, 1), max_v=2, size=(-1, -1)):
        result = np.zeros((face_map.shape[0], face_map.shape[1], 3))

        f_map = face_map.copy()
        f_map[f_map > max_v] = max_v
        f_map = f_map / max_v

        for i in range(3):
            result[:, :, i] = (((1 - f_map) * s_color[i]) + (f_map * e_color[i])) * 255

        if size[0] > 0 and size[1] > 0:
            result = cv2.resize(result, size)

        return result.astype('uint8')

    def detect_rectanglulars(self, face_map, max_v=3):
        f_map = face_map.copy()
        f_map[f_map > max_v] = max_v
        f_map[f_map < (max_v/2)] = 0
        f_map = ((f_map / max_v) * 255)
        f_map[f_map > 255] = 255
        f_map = (255 - f_map).astype('uint8')

        ret, thresh = cv2.threshold(f_map, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result = []
        for i in range(0, len(contours)):
            if i == 0:
                remove_v = [[0, 0],
                            [0, self.frame_size[0]-1],
                            [self.frame_size[1]-1, self.frame_size[0]-1],
                            [self.frame_size[1]-1, 0]]
                remove_candidates = np.sum(np.sum((contours[i] == remove_v), axis=2) == 2, axis=1)
                remove_idx = np.argwhere(remove_candidates == 1)
                contour = np.delete(contours[i], remove_idx, 0)

                if contour.shape[0] < 4:
                    continue
            else:
                contour = contours[i]

            x,y,w,h = cv2.boundingRect(contour)

            result.append([x, y, x+w, y+h, 1.0])

        return np.array(result)

    def draw_sqaure_on_face(self, face_list, dst_frame, box_color=(0, 0, 255), min_line_width=4, max_line_width=30):
        for face in face_list:
            cv2.rectangle(dst_frame,
                          (face[0].astype('int'), face[1].astype('int')),
                          (face[2].astype('int'), face[3].astype('int')),
                          box_color,
                          int(np.round(face[4]*max_line_width).astype('int')+min_line_width))
