import torch as th
import numpy as np

from .deep_sort.nn_matching import NearestNeighborDistanceMetric
#from .deep_sort.preprocessing import non_max_suppression as nms
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker


class DeepSort(object):
    def __init__(self, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, nn_budget=100):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
    
    @property
    def hits(self):
        return self.tracker.n_init

    @property
    def tracks(self):
        return self.tracker.tracks

    def update(self, bbox_xyxy, confidences, features, frame):
        self.height, self.width = frame.shape[-2:]
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]

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

    #@staticmethod
    def _xyxy_to_tlwh(self, bbox_xyxy):
        if len(bbox_xyxy) > 0:
            x1, y1 = bbox_xyxy[:, 0, np.newaxis], bbox_xyxy[:, 1, np.newaxis]
            x2, y2 = bbox_xyxy[:, 2, np.newaxis], bbox_xyxy[:, 3, np.newaxis]
            return np.concatenate((x1, y1, x2 - x1, y2 - y1), axis=1)
        else:
            return np.empty(0)

    def _xywh_to_tlwh(self, bbox_xywh):
        """
        XXX:
            Convert bbox from xc_yc_w_h to xtl_ytl_w_h
            Thanks JieChen91@github.com for reporting this bug!
        """
        x1 = (bbox_xywh[:,0] - bbox_xywh[:,2]/2.)[..., np.newaxis]
        y1 = (bbox_xywh[:,1] - bbox_xywh[:,3]/2.)[..., np.newaxis]
        return np.concatenate((x1, y1, bbox_xywh[:,2:4]), axis=1)

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
            Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2