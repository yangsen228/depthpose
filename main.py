import pyrealsense2 as rs
import numpy as np
import cv2
import os

W = 640
H = 480
save_path = 'data/'

import torch
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

def a2j_init(keypointsNumber=15, model_dir='/home/yxs/Yang/pose_estimation/depth/common/A2J/model/ITOP_side.pth'):
    a2j_model = A2J_model(num_classes = keypointsNumber)
    a2j_model.load_state_dict(torch.load(model_dir)) 
    a2j_model = a2j_model.cuda()
    a2j_model.eval()

    return a2j_model

def main():
    seq_num = 2
    detected = False

    # Yolov3 initialization
    yolov3_model, device, half, imgsz = yolov3_init()

    # A2J initialization
    a2j_model = a2j_init()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    idx = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            img = (depth_image-np.amin(depth_image))*255.0/(np.amax(depth_image)-np.amin(depth_image))
            img = img.astype(np.uint8)
            img = cv2.equalizeHist(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            depth_colormap = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

            # Person detection
            bbox_list = person_detection(yolov3_model, depth_colormap, imgsz, device, half)  # shape = [num_people, 4]
            
            for xmin, ymin, xmax, ymax in bbox_list:
                detected = True
                # Draw bbox
                # cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                cv2.rectangle(depth_colormap, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                # Pose estimation
                preds = pose_estimation(a2j_model, depth_image, [xmin, ymin, xmax, ymax])
                # Draw pose
                for joint_id in range(len(preds)):
                    cv2.circle(depth_colormap, tuple(preds[joint_id][:-1].astype(np.uint16)), 10, (0,0,255), -1)                

            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images1)
            idx += 1

            # Record data
            if detected:
                cv2.imwrite(os.path.join(save_path, 'color_seq{:02d}/{:05d}.jpg'.format(seq_num, idx)), depth_colormap)
                np.save(os.path.join(save_path, 'depth_seq{:02d}/{:05d}.npy'.format(seq_num, idx)), depth_image)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()

if __name__=='__main__':
    with torch.no_grad():
        main()