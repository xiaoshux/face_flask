import os
import cv2
import face_recognition
import numpy as np


class FaceRecognition:
    known_face_names = []
    known_face_encodings = []

    # 构造函数，导入本地图片
    def __init__(self):
        path = "./static/image/"
        temp_file_name_list = os.listdir(path)
        file_name_list = []
        for imagename in temp_file_name_list:
            temp = imagename.split('.')
            if len(temp) > 1:
                file_name_list.append(imagename)
        for fileName in file_name_list:
            name = fileName.split('.')[0]
            fileName = path + fileName
            image = face_recognition.load_image_file(fileName)
            face_codeing = face_recognition.face_encodings(image)[0]
            self.known_face_names.append(name)
            self.known_face_encodings.append(face_codeing)

    # 加入人脸信息，初始化
    def add_face_token(self, img_rgb_array, name):
        face_code = face_recognition.face_encodings(img_rgb_array)[0]
        self.known_face_names.append(name)
        self.known_face_encodings.append(face_code)

    # 对比人脸返回人脸信息
    def compare_face_token(self, frame):
        # 初始化变量
        faces = []
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # 判断是否相似，tolerance=相似度，返回boolen类型列表
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.4)
            # 表示两个脸之间的相似度，返回一个数组
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            # 返回最小数
            best_match_index = np.argmin(face_distances)
            if matches.count(True) < 1:
                name = "unknown"
                face_names.append(name)
            elif matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            face = {"face_name": name, "face_location": {"top": top, "right": right, "bottom": bottom, "left": left}}
            faces.append(face)
        faces_location = {"face_num": len(face_locations), "faces": faces}
        return faces_location

    # 获取名字人脸信息二维数组
    def get_know_token(self):
        # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        know_token = [str(i) for i in zip(self.known_face_names, self.known_face_encodings)]
        return know_token

    # 通过姓名获取一张图片地址
    def getfile(self, name):
        path = "./static/image/"
        temp_file_name_list = os.listdir(path)
        file_name_list = []
        for imagename in temp_file_name_list:
            temp = imagename.split('.')
            if len(temp) > 1:
                file_name_list.append(imagename)
        for fileName in file_name_list:
            if name == fileName.split('.')[0]:
                filepath = path + fileName
                break
        return filepath

    # 通过姓名获取图片流
    def getfiles(self, name):
        path = "./static/image/" + name + '/'
        filenme = os.listdir(path)
        return filenme
