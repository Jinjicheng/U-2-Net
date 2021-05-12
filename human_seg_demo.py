#!/usr/bin/envpython
#encoding=utf-8
'''
@author:金吉成
@license:海格星航
@contact:13037106168@168.com
@file: .py
@time: 2021/5/12
@desc: 人体分割
'''
import os
import cv2
import torch
from torchvision import transforms
import numpy as np

from data_loader import RescaleT
from data_loader import ToTensorLab
from model import U2NET # full size version 173.6 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def process_img(img):
    in_trans = {'imidx':np.array([0]), 'image':img, 'label':np.zeros(img.shape)}
    my_transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    inn = my_transform(in_trans)
    return inn['image']


def vis_output(ori_img, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    out_img = np.array(predict_np * 255, dtype=np.uint8)
    image = ori_img.copy()

    out_img = cv2.resize(out_img, (image.shape[1], image.shape[0]))
    # found contours of out_img
    th, dst = cv2.threshold(out_img, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    res = cv2.bitwise_and(ori_img, ori_img, mask=dst)

    cv2.imshow('seg', out_img)
    cv2.imshow('object', res)
    cv2.waitKey(0)


def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'

    image_dir = './test_data/testmy/girl.jpg'
    prediction_dir = './test_data/testmy/girl_human_seg.png'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')

    # --------- 2. load image ---------
    ori_img = cv2.imread(image_dir)
    input_test = process_img(ori_img)
    input_test = input_test.type(torch.FloatTensor).unsqueeze(0)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    if torch.cuda.is_available():
        input_test = input_test.cuda()
    else:
        input_test = input_test

    d1,d2,d3,d4,d5,d6,d7= net(input_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    vis_output(ori_img, pred, prediction_dir)
    del d1,d2,d3,d4,d5,d6,d7


if __name__ == "__main__":
    main()
