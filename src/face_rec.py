from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
import imageio
from sklearn.svm import SVC
import base64
from base64 import encodebytes
import re
import io
import requests
import json
from PIL import Image


# convert base64 string to image array python

def base64_to_image(base64_string):
    # import base64 library
    # convert base64 string to byte
    # convert byte to image
    img = Image.open(io.BytesIO(
        base64.decodebytes(bytes(base64_string, "utf-8"))))
    return np.array(img)


def checkin(userId):
    url = "https://dut-timetrackingapp.herokuapp.com/api/v1/checkin"

    payload = json.dumps({
        "groupId": "6150b5c637cef39b11366cc8",
        "userId": userId
    })
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = '../FaceRecog/Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = '../FaceRecog/Models/20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN phat hien khuon mat
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lay tensor input va output
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Cai dat cac mang con
            pnet, rnet, onet = align.detect_face.create_mtcnn(
                sess, "../FaceRecog/src/align")

            people_detected = set()
            person_detected = collections.Counter()

            f = open("../FaceRecog/image/image.txt", "r")
            frame = base64_to_image(f.read())

            # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
            bounding_boxes, _ = align.detect_face.detect_face(
                frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            faces_found = bounding_boxes.shape[0]
            try:
                # Neu co it nhat 1 khuon mat trong frame
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # Cat phan khuon mat tim duoc
                        cropped = frame[bb[i][1]:bb[i]
                                        [3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(
                            -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {
                            images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(
                            embeddings, feed_dict=feed_dict)

                        # Dua vao model de classifier
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = ""
                        if(best_class_probabilities < 0.5):
                            best_name = "Unknown"
                        else:
                            best_name = class_names[best_class_indices[0]]
                        # Lay ra ten va ty le % cua class co ty le cao nhat
                        print("Name: {}, Probability: {}".format(
                            best_name, best_class_probabilities))
                        if(best_name != "Unknown"):
                            checkin(best_name)
                        print(best_name)
            except:
                pass


main()
