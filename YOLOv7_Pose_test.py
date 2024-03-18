import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.torch_utils import select_device

poseweights="yolov7-w6-pose.pt"
source="football1.mp4"
device='cpu'
view_img=False
save_conf=False
line_thickness = 3
hide_labels=False
hide_conf=True

device = select_device('mpu')
