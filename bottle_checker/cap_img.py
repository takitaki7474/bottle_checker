# -*- coding: utf-8 -*-

import time
import cv2
import os
import argparse
import numpy as np
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--img_name', '-im', default='0', type=str, help='image name to save')
parser.add_argument('--output_folder', '-o', default='./images', type=str, help='folder name to save')
args = parser.parse_args()

img_name = args.img_name
output_folder = args.output_folder

cap = cv2.VideoCapture(0)
cap.set(3,1260)
cap.set(4,945)
#cap.set(5,60)
image_count = 0

image_names = os.listdir(output_folder)
for image_name in image_names:
    image_count += 1

now_micro = datetime.datetime.now().microsecond
#cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)

    if k == 27:
        img_name = str(image_count) + "_" + str(now_micro) + ".png"
        img_path = os.path.join(output_folder, img_name)
        cv2.destroyAllWindows()
        cv2.imwrite(img_path, frame)
        print("保存先: {}".format(img_path))
        img = cv2.imread(img_path,1)
        cv2.imshow("frame", img)
        img = cv2.resize(img,(640,480))
        cv2.imwrite(img_path, img)
        cv2.waitKey(5000)
        break

cap.release()
