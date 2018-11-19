# -*- coding:utf-8 -*-

import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, training, optimizers, serializers
from chainer.datasets import tuple_dataset, split_dataset_random
from chainer.training import extensions
import cnn_mynet

gpu_id = -1

infer_net = cnn_mynet.MyNet_6(3)
infer_net = L.Classifier(infer_net)
serializers.load_npz("./learned_model/" + "model_v7.model", infer_net) #変更点

def detect(original_img, cascadefilename="./detect_model/cascade.xml"):
    cascade = cv2.CascadeClassifier(cascadefilename)
    img_list = []

    if cascade.empty():
        print('cannot load cascade file')
        sys.exit(-1)

    srcimg = original_img
    dstimg = srcimg.copy()

    if srcimg is None:
        print('cannot load image')

    objects = cascade.detectMultiScale(srcimg, 1.1, 3)
    objects_is = 0
    if len(objects) > 0:
        for rect in objects:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            if w<100 or h<100:
                continue

            img_list.append(srcimg[y:y+h, x:x+w])
            cv2.rectangle(dstimg, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), (0,0,225), thickness=2)
            objects_is = 1

    if objects_is == 0 :
        img_list = []
        dstimg = []

    return img_list, dstimg, objects_is

def preprocessing(img):

    img = cv2.resize(img,(28,28))
    img = img.astype(np.float32)
    img = img[None,...]
    return img

def evaluation_filter(ans_list, prob_list, cut_image_list):

    result = []
    reserve = []
    count = 0
    for ans, prob, cut_img in zip(ans_list, prob_list, cut_image_list):
        if count == 0:
            reserve.append((ans, prob, cut_img))
        sep = [i for i in prob if i >= 80]
        if sep != []:
            result.append((ans, prob, cut_img))

        count += 1

    if result == []:
        result = reserve

    return result

def evaluation(img_path):

    ans_list = []
    prob_list = []
    img = cv2.imread(img_path)
    cut_image_list, dstimg, objects_is = detect(img)

    if objects_is == 1:
        for cut_img in cut_image_list:
            cut_img = preprocessing(cut_img)
            y = infer_net.predictor(cut_img)
            y = y.data
            prob_list.append([round(i*100,2) for i in F.softmax(y)[0].data])
            ans = y.argmax(axis=1)[0]
            ans_list.append(ans)

        results = evaluation_filter(ans_list, prob_list, cut_image_list)

    if objects_is == 0:
        results = 0
        dstimg = []

    return results, dstimg
