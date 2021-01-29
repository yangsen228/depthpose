import pyrealsense2 as rs
import numpy as np
import cv2
import os
import scipy.io as scio

W = 640
H = 480
save_path = 'data/depth_image_seq01'

import torch
import time
import sys
# Yolov3
sys.path.append('/home/yxs/Yang/object_detection/common/yolov3')
from models.experimental import attempt_load
from utils.general import check_img_size, set_logging
from utils.torch_utils import select_device
sys.path.append('/home/yxs/Yang/object_detection/person/yolov3_person')
from yolov3_person import person_detection
# A2J
sys.path.append('/home/yxs/Yang/pose_estimation/depth/common/A2J/src')
from model import A2J_model
sys.path.append('/home/yxs/Yang/pose_estimation/depth/body/a2j_body')
from a2j_body import pose_estimation

dataset = 'kinect'
if dataset == 'itop':
    data_len = 17991
elif dataset == 'kinect':
    data_len = 1500

def yolov3_init(img_size=640, device=''):
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    yolov3_model = attempt_load(weights='./yolov3.pt', map_location=device)  # load FP32 model
    imgsz = check_img_size(img_size, s=yolov3_model.stride.max())  # check img_size
    if half:
        yolov3_model.half()  # to FP16

    # Warm up
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = yolov3_model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    return yolov3_model, device, half, imgsz

def a2j_init(keypointsNumber=15, model_dir='/home/yxs/Yang/pose_estimation/depth/common/A2J/model/ITOP_side_ver5.pth'):
    a2j_model = A2J_model(num_classes = keypointsNumber)
    a2j_model.load_state_dict(torch.load(model_dir)) 
    a2j_model = a2j_model.cuda()
    a2j_model.eval()

    return a2j_model

def main():
    # Yolov3 initialization
    yolov3_model, device, half, imgsz = yolov3_init()

    # A2J initialization
    a2j_model = a2j_init()

    # Load data
    if dataset == 'kinect':
        depth_images = np.load('data/kinect/kinect_depth_images_030.npy')

    # B = scio.loadmat('/home/yxs/Yang/pose_estimation/depth/common/A2J/data/itop_side/itop_side_bndbox_train.mat' )['FRbndbox_train']
    # bbox_save = np.ones((data_len, 4)).astype(int) * -1

    # Start streaming
    idx = 0
    while True:
        # Load data
        if dataset == 'itop':       # (240, 320)
            img0 = scio.loadmat('/home/yxs/titech/datasets/ITOP/A2J/side_test/' + str(idx+1) + '.mat')['DepthNormal'][:,:,3]
        elif dataset == 'kinect':   # (424, 512)
            img0 = depth_images[idx*4].astype(float)
            img0 = 4.5 - img0 / 1000.0

        img = (img0-np.amin(img0))*255.0/(np.amax(img0)-np.amin(img0))
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

        # Person detection
        t1 = time.time()
        bbox_list = person_detection(yolov3_model, img, imgsz, device, half)  # shape = [num_people, 4]
        t2 = time.time()

        for xmin, ymin, xmax, ymax, conf in bbox_list:
            # Draw box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            # Pose estimation
            preds = pose_estimation(a2j_model, img0, [xmin, ymin, xmax, ymax])
            # Draw pose
            for joint_id in range(len(preds)):
                cv2.circle(img, tuple(preds[joint_id][:-1].astype(np.uint16)), 5, (0,0,255), -1)
        t3 = time.time()
        print('bbox: {}s; pos: {}s'.format(t2-t1, t3-t2)) # 0.063s, 0.063s

        # Draw index
        cv2.putText(img, str(idx), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

        cv2.imshow('test', img)

        idx += 1
        if idx >= data_len:
            break

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    
    # scio.savemat('/home/yxs/Yang/pose_estimation/depth/common/A2J/data/itop_side/itop_side_bndbox_train.mat', {'FRbndbox_train':bbox_save})


if __name__=='__main__':
    with torch.no_grad():
        main()