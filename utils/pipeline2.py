import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from batch_face import RetinaFace

from .general import prep_input_numpy
from .results import GazeResultContainer
#此行被修改了 from models.model import L2CS
from models.model_v3_50_EMABo import XModel
from utils.general import mean_gaze


class Pipeline:

    def __init__(
            self,
            weights: str,
            num_bins: int,
            device: str = 'cpu',
            backbone: str = 'resnet50',
            cfg: str = 'models/yaml/resnet50-GMSFF-EMABo_v3.yaml',
            include_detector: bool = True,  # 启用人脸检测
            confidence_threshold: float = 0.7
    ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        #此处被修改了self.model = L2CS(backbone)
        self.model = XModel(cfg)
        self.model.load_state_dict(torch.load(self.weights))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested   初始化 RetinaFace 检测器
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(
                    gpu_id=device.index)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)  # 检测图像中的人脸
        # 遍历检测到的人脸，提取边界框（box）、关键点（landmark）和置信度（score）：
            if faces is not None:
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0, 1))
                yaw = np.empty((0, 1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes),
            landmarks=np.stack(landmarks),
            scores=np.stack(scores)
        )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):

        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")

        # Predict 
        gaze_pred = self.model(img)  #到这为止都ok

        pitch_predicted = gaze_pred[:, 0, 0, 0, 0]  # 提取 pitch
        yaw_predicted = gaze_pred[:, 0, 0, 0, 1]  # 提取 yaw

        # 确保输出的形状为 (1,)
        pitch_predicted = pitch_predicted.view(-1)
        yaw_predicted = yaw_predicted.view(-1)

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted
