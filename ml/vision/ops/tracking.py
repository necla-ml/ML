"""
Adapted from https://github.com/ZQPei/deep_sort_pytorch.
"""

import torch as th
from torchvision.ops import nms
import numpy as np

from .deep_sort.nn_matching import NearestNeighborDistanceMetric
#from .deep_sort.preprocessing import non_max_suppression as nms
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker

class DeepSort(object):
    def __init__(self,  #min_confidence=0.3, 
                        #nms_max_overlap=1.0,
                        
                        max_dist=0.2,  
                        nn_budget=100, 

                        max_iou_distance=0.7,   # 0.7
                        max_age=30,             # 30 
                        n_init=3,               # 3
                        ):
        """DeepSort Tracker wrapper.
        Args:
            min_confidence(float): bbox confidence to admit
            nms_max_overlap(float): NMS threshold
            
            max_dist(float): max feature vector distance to considered a match
            nn_budget(int): if not None, fix samples per class to at most this number. 
                            Removes the oldest samples when the budget is reached.
            
            max_iou_distance(float): max IoU overlap
            max_age(int): maximum number of missed misses before a track is deleted
            n_init(int): number of consecutive detections before the track is confirmed. 
        """
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.tracker = Tracker(NearestNeighborDistanceMetric("cosine", max_dist, nn_budget), 
                                max_iou_distance=max_iou_distance, 
                                max_age=max_age, 
                                n_init=n_init))
    
    @property
    def hits(self):
        return self.tracker.n_init

    @property
    def tracks(self):
        return self.tracker.tracks

    def update(self, bbox_xyxy, confidences, features, frame):
        self.height, self.width = frame.shape[-2:]
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) 
                        for i, conf in enumerate(confidences) 
                        if conf > self.min_confidence]

        # run on non-maximum supression
        if False:
            # FIXME: not necessary suitable for class-agnostic NMS
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = nms(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        matches = self.tracker.update(detections)
        
        if True:
            return matches
        else:
            # output bbox identities
            outputs = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                outputs.append([x1, y1, x2, y2, track_id])

            if len(outputs) > 0:
                outputs = th.stack(outputs)

            return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """Convert bbox from xc_yc_w_h to xtl_ytl_w_h
        Args:
            bbox_xywh(Tensor or ndarray): bboxes in the format of [xc, yc, w, h]
        Returns:
            bbox_tlwh(Tensor or ndarray): bboxes in [x, y, w, h]
        """
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        else:
            raise ValueError(f"Unexpected type(bbox_xywh)={type(bbox_xywh)}")
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """Convert bbox from top_left_w_h to x1y1x2y2.
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h