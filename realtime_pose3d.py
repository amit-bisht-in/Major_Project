import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from collections import deque
import time

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Import Model class (adjust path if different in your project)
try:
    sys.path.append(os.getcwd())
    from common.model_poseformer import PoseTransformerV2 as Model
except ModuleNotFoundError:
    print("Error: Could not find 'lib.models.poseformer'. Please verify the file exists.")
    print("Check 'lib/models/' directory or adjust the import path.")
    sys.exit(1)

# Set Matplotlib to use DejaVu Sans to avoid Fira Sans error
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Use non-interactive backend initially
matplotlib.use('Agg')
plt.switch_backend('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img

def show3Dpose(vals, ax):
    ax.clear()  # Clear previous plot
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

class RealtimePoseEstimator:
    def __init__(self, args):
        self.args = args
        self.frame_buffer = deque(maxlen=args.frames)
        self.keypoint_buffer = deque(maxlen=args.frames)
        self.debug_mode = getattr(args, 'debug', False)

        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure CUDA drivers and PyTorch CUDA are installed.")

        # Initialize model (CUDA)
        self.model = nn.DataParallel(Model(args=args)).cuda()
        model_path = sorted(glob.glob(os.path.join(args.previous_dir, '27_243_45.2.bin')))[0]
        pre_dict = torch.load(model_path)
        self.model.load_state_dict(pre_dict['model_pos'], strict=True)
        self.model.eval()

        # Initialize pose detector
        self.pose_detector = self._init_pose_detector()
        self.detector_type = None

        # Joint mappings
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]

        # Rotation for camera-to-world transformation
        self.rot = np.array([0.1407056450843811, -0.1500701755285263,
                             -0.755240797996521, 0.6223280429840088], dtype='float32')

        # Setup OpenCV window for display
        self.window_name = "Real-time 3D Pose Estimation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 640)

        # Setup matplotlib for 3D rendering (off-screen)
        matplotlib.use('Agg')  # Use non-interactive backend
        self.fig_3d = plt.figure(figsize=(6.4, 4.8), dpi=100)
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')

        # Debug statistics
        self.detection_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'low_confidence_detections': 0,
            'failed_detections': 0
        }

    def _init_pose_detector(self):
        """Initialize pose detector with priority: MediaPipe -> HRNet -> OpenPose"""
        try:
            import mediapipe as mp
            print(f"MediaPipe version: {mp.__version__}")
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.detector_type = "MediaPipe"
            if self.debug_mode:
                print("✓ Using MediaPipe for 2D pose detection")
            return pose
        except Exception as e:
            if self.debug_mode:
                print(f"⚠ MediaPipe initialization failed: {str(e)}")

        try:
            from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
            self.detector_type = "HRNet"
            if self.debug_mode:
                print("✓ Using HRNet for 2D pose detection")
            return hrnet_pose
        except ImportError as e:
            if self.debug_mode:
                print(f"⚠ HRNet import failed: {str(e)}")

        print("ERROR: No pose detection library available! Please install MediaPipe or HRNet.")
        self.detector_type = None
        return None

    def mediapipe_to_coco_format(self, mp_landmarks, img_width, img_height):
        """Convert MediaPipe landmarks to COCO format with improved mapping"""
        if mp_landmarks is None:
            return np.zeros((17, 2)), np.zeros(17)

        # Improved MediaPipe to COCO joint mapping
        mp_to_coco = {
            0: 0,   # nose
            2: 1,   # left_eye_inner -> left_eye
            5: 2,   # right_eye_inner -> right_eye
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16  # right_ankle
        }

        keypoints = np.zeros((17, 2))
        confidence = np.zeros(17)

        # Extract landmarks with better error handling
        landmarks = mp_landmarks.landmark

        for mp_idx, coco_idx in mp_to_coco.items():
            if mp_idx < len(landmarks):
                landmark = landmarks[mp_idx]
                # Convert normalized coordinates to pixel
