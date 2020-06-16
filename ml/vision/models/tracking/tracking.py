import torch as th
from ml import logging

class Track(object):
    def __init__(self, tid, cls=None, length=5, **kwargs):
        """
        Args:
            kwargs:
                tid(int):
                length(int): max length of history
                history(List[Tensor[4]]):
        """
        self.tid = tid
        self.cls = cls
        self.score = 0
        self.count = 0
        self.length = length
        self.history = []
        self.predicted = False
        self.__dict__.update(kwargs)
    
    @property
    def last(self):
        return self.history[-1] if self.history else None

    def update(self, instance):
        """Update track statistics and history given a new observation or last tracked.
        instance(xyxysc or xyxy): up to date detection or predicted position
        """
        instance = instance.cpu()
        info = len(instance)
        score = self.score
        if info == 4+2:
            # detection
            self.predicted = False
            score, cls = instance[-2:]
            score, cls = score.item(), cls.item()
            xyxysc = instance
            if cls != self.cls:
                logging.warning(f"Reconcile with detection of inconsistent class from {self.cls} to {cls}")
                xyxysc = instance.clone()
                xyxysc[-2] = score = self.score
                xyxysc[-1] = self.cls
        elif info == 4:
            # prediction
            self.predicted = True
            logging.warning(f"track[{self.tid}] predicted={instance.round().int().tolist()}({self.score:.2f})")
            xyxysc = th.cat((instance, th.Tensor([self.score, self.cls])))
        count = self.count
        self.count += 1
        self.score = (count * self.score + score) / self.count
        self.history.append(xyxysc)
        if len(self.history) > self.length:
            self.history.pop(0)

class Tracker(object):
    def __init__(self, *args, **kwargs):
        self.tracks = {}

    def __contains__(self, tid):
        return tid in self.tracks

    def get(tid):
        return self.tracks.get(tid, None)

    def snapshot(self, fresh=True):
        """Return last RoIs on the tracks.
        Returns:
            snapshot(List[Tuple(tid, Tensor[N, 6])]): [(tid, xyxysc)]
        """
        return [(tid, trk.last) for tid, trk in self.tracks.items() if not fresh or not trk.predicted]
    
    def update(self, xyxysc, features):
        """Update with detection bboxes and features.
        Args:
            dets(Tensor[N, 6]): object detections in xyxysc
            features(Tensor[N, C, H, W]): pooled object features
        """
        pass