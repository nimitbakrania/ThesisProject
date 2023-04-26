import os
import numpy as np
import torch
import csv
import cv2
import dlib


class Dection(object):
    def __init__(self, model_name, face_max=True):
        self.model_name = model_name
        self.face_max = face_max

        if model_name == "CAFFE":
            modelFile = "../lib/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "../lib/deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
            self.model = net
        elif model_name == "TF":
            modelFile = "../lib/opencv_face_detector_uint8.pb"
            configFile = "../lib/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            self.model = net
        else:
            self.face_cascade = cv2.CascadeClassifier('../lib/haarcascade_frontalface_default.xml')

        self.conf_threshold = 0.7

    def face_detect(self, img, display=False):
        if self.model_name == "cv2":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
            if len(faces) == 0:
                return None
            else:
                face_max = [0, 0, 0, 0]
                for i in range(len(faces)):
                    face = faces[i]
                    if face[2] > face_max[2]:
                        face_max = face
                left = face_max[0]
                top = face_max[1]
                right = left + face_max[2]
                bottom = top + face_max[2]
                face_img = img[top:bottom, left:right]
        else:

            frameOpencvDnn = img.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            img_mean = np.mean(frameOpencvDnn, (0, 1))
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), img_mean, False, False)

            self.model.setInput(blob)
            detections = self.model.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            if display:
                cv2.imshow("frame", frameOpencvDnn)
                cv2.waitKey(0)
            if len(bboxes) > 1:
                bbox_max = [0, 0, 0, 0]
                bbox_max_len = bbox_max[2] - bbox_max[0]
                for bbox in bboxes:
                    if (bbox[2] - bbox[0]) > bbox_max_len:
                        bbox_max = bbox
                        bbox_max_len = bbox_max[2] - bbox_max[0]

                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]
            elif len(bboxes) == 0:
                return None
            else:
                bbox_max = bboxes[0]
                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]

        return face_img


class LandmarksDetection(object):
    def __init__(self):
        PREDICTOR_PATH = '../lib/shape_predictor_68_face_landmarks.dat'
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def landmarks_detect(self, img_rgb, display):
        img_shape = img_rgb.shape
        x1 = 0
        y1 = 0
        x2 = x1 + img_shape[0]
        y2 = y1 + img_shape[1]
        rect = dlib.rectangle(x1, y1, x2, y2)

        img_key = img_rgb.copy()
        predictor = self.a_predictor
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        points_keys = []

        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            points_keys.append([point[0, 0], point[0, 1]])
            cv2.circle(img_key, pos, 2, (255, 0, 0), -1)

        if display:
            cv2.imshow("landmark", img_key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("landmark display")
        self.a_landmark = points_keys
        return points_keys


def get_file_list(read_path):
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):
        floder = dirs[i]
        floder_path = path + "/" + floder

        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        else:
            file_name_list.append(floder_path)

    return file_name_list


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def save_csv(csv_path, data):
    with open(csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close()


def read_csv(csv_path):
    data_list = []
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        data_list.append(item)

    return data_list


def read_txt(txt_path):
    f = open(txt_path)
    data = f.readlines()
    for index in range(len(data)):
        data[index] = data[index][:-1]
    return data

def video_to_frames(pathIn='',
                    pathOut='',
                    extract_time_interval=-1,
                    only_output_video_info=False,
                    extract_time_points=None,
                    initial_extract_time=0,
                    end_extract_time=None,
                    output_prefix='frame',
                    jpg_quality=100,
                    isColor=True):

    cap = cv2.VideoCapture(pathIn)  
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    dur = n_frames / fps  

    if only_output_video_info:
        print('only output the video information (without extract frames)::::::')
        print("Duration of the video: {} seconds".format(dur))
        print("Number of frames: {}".format(n_frames))
        print("Frames per second (FPS): {}".format(fps))

    elif extract_time_points is not None:
        if max(extract_time_points) > dur:  
            raise NameError('the max time point is larger than the video duration....')
        try:
            os.mkdir(pathOut)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not isColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                print('Write a new frame: {}, {}th'.format(success, count + 1))
                cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                count = count + 1

    else:
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

        elif extract_time_interval > 0 and extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        else:
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1


def frame_to_face(frame_dir, face_dir, model_name, normal_size=None, save_mode='.jpg'):
    face_detector = FaceDection(model_name=model_name)

    for root, dirs, files in os.walk(frame_dir):

        if dirs == []:

            for file_name in files:
                file_path = os.path.join(root, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    print("if img is None:")
                    continue

                if normal_size is not None:
                    img_normal = cv2.resize(img, normal_size)
                else:
                    img_normal = img
                face_img = face_detector.face_detect(img_normal)
                if face_img is None:
                    print("if face_img is None:")
                    continue

                frame_dir_split = frame_dir.split('/')
                file_path_split = file_path.split('/')
                file_path_split.pop() 
                sub_split = [item for item in file_path_split if item not in frame_dir_split]
                save_dir = face_dir
                for item in sub_split:
                    save_dir = os.path.join(save_dir, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, file_name.split('.')[0] + save_mode)
                cv2.imwrite(save_path, face_img)


def analysis(true_right, true_wrong, false_right, false_wrong):
    false_all = false_wrong + false_right
    true_all = true_wrong + true_right
    true_class = true_right + false_wrong
    false_class = true_wrong + false_right
    all = true_right + true_wrong + false_wrong + false_right

    accuracy = (true_right + false_right) / all
    APCER = false_wrong / false_all
    BPCER = true_wrong / true_all
    ACER = (APCER + BPCER) / 2
    FAR = false_wrong / true_class
    FRR = true_wrong / false_class

    print("accuracy", accuracy, "APCER", APCER, "BPCER", BPCER, "ACER", ACER, "FAR", FAR, "FRR", FRR)


def img_preview(img_dir):
    file_list = get_file_list(img_dir)
    for file_path in file_list:
        img = cv2.imread(file_path)
        if img is None:
            continue
        img_shape = img.shape
        print("img_shape", img_shape)
        cv2.namedWindow("img_show", 0)
        cv2.imshow("img_show", img)
        cv2.waitKey(0)


