import clip
import cv2
from PIL import Image
import numpy as np
import torch

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

def xywh2tlwh(box):
    """Convert bounding box from x center, y center, width, height to x min, y min, width, height"""
    box = np.asarray(box)
    box[0] = box[0] - (box[2] / 2)
    box[1] = box[1] - (box[3] / 2)
    return box

class ZSTracker():
    """Zero shot object tracking using CLIP"""
    def __init__(self, modelName="ViT-B/16", maxCosineDistance=0.4, nnBudget=None, device="cuda", jit=False):
        self.device = device
        self.model, self.transform = clip.load(modelName, device=device, jit=jit)
        self.model.eval()

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", maxCosineDistance, nnBudget)
        self.tracker = Tracker(metric)

    def getPatches(self, img, boxes):
        """
        Get image patches using bounding boxes
        :param img: numpy array of image with shape (h, w, c)
        :param boxes: list of bounding boxes as (xcen, ycen, width, height)
        :return: list of patches as torch tensors
        """
        patches = []
        for box in boxes:
            # clip box to fit in bounds
            box = xywh2tlwh(box)
            box[2:] += box[:2]
            box[:2] = np.maximum(0, box[:2])
            box[2:] = np.minimum(np.asarray(img.shape[:2][::-1]) - 1, box[2:])
            if np.any(box[:2] >= box[2:]):
                continue
            xmin, ymin, xmax, ymax = box.astype(int)
            patch = self.transform(Image.fromarray(img[ymin:ymax, xmin:xmax])).to(self.device)
            patches.append(patch)
        return patches

    def encode(self, img, boxes):
        """
        Encodes object patch into features using CLIP
        :param img: numpy array of image with shape (h, w, c)
        :param boxes: list of bounding boxes as (xcen, ycen, width, height)
        :return: numpy array of features for each object patch with shape (# objects, # features)
        """
        patches = self.getPatches(img, boxes)
        features = self.model.encode_image(torch.stack(patches)).detach().cpu().numpy()
        for idx, i in enumerate(features):
            if np.isnan(i[0]):
                print("Got nan values :(")

        return features

    def update(self, img, detections):
        """
        Update and predict with tracker
        :param img: numpy array of image with shape (h, w, c)
        :param detections: list of detections in form of [class number, confidence, bounding box in xcen, ycen, width, height]
        :return: list of detections from tracker in form [class number, track id, bounding box in xmin, ymin, xmax, ymax]
        """
        self.tracker.predict()

        if len(detections) > 0:
            features = self.encode(img, [det[-1] for det in detections])
            dets = []
            for idx, (classNum, conf, bbox) in enumerate(detections):
                box = xywh2tlwh(bbox)
                dets.append(Detection(box, conf, classNum, features[idx]))

            self.tracker.update(dets)

        tracked = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            xyxy = track.to_tlbr()
            class_num = track.class_num
            bbox = xyxy
            tracked.append([class_num, track.track_id, bbox])

        return tracked