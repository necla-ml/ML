from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pathlib import Path

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator, TwoMLPHead
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign
import torch as th

from .... import nn, logging
from .. import backbone

def mask_rcnn(pretrained=False, num_classes=1+90, representation=1024, backbone=None, with_mask=True, **kwargs):
    if backbone is None:
        model = maskrcnn_resnet50_fpn(pretrained, pretrained_backbone=not pretrained, progress=True, **kwargs)
    else:
        model = maskrcnn_resnet50_fpn(pretrained, pretrained_backbone=False, progress=True, **kwargs)
        model.backbone = backbone

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    out_features = model.roi_heads.box_predictor.cls_score.out_features
    if representation != in_features:
        logging.info(f"Replaced box_head with representation size of {representation}")
        out_channels = model.backbone.out_channels
        resolution = model.roi_heads.box_roi_pool.output_size[0]
        model.roi_heads.box_head = TwoMLPHead(out_channels * resolution ** 2, representation)

    if representation != in_features or num_classes != out_features:
        logging.info(f"Replaced box_predictor with (representation, num_classes) = ({representation}, {num_classes})")
        model.roi_heads.box_predictor = FastRCNNPredictor(representation, num_classes)
        
    if not with_mask:
        model.roi_heads.mask_roi_pool = None
        model.roi_heads.mask_head = None
        model.roi_heads.mask_predictor = None
    
    return THDetector(model)

def mmdet_load(cfg, chkpt=None, with_mask=False, **kwargs):
    r"""Load an mmdet detection model from cfg and checkpoint.

    Args:
        cfg(str): config filename in configs/
        chkpt(str): optional path to checkpoint
        with_mask(bool): whether to perform semantic segmentation

    Kwargs:
        device(str): default to 'cuda:0' if not specified

    mmdet:
        htc:
            cfg: 'htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
            chkpt: 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
        cascade_rcnn:
            cfg: 'cascade_rcnn_r101_fpn_1x.py'
            chkpt: 'cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'
    """

    from mmdet.apis import init_detector
    import mmdet
    cfg = Path(mmdet.__file__).parents[1] / 'configs' / cfg
    chkpt = chkpt and str(chkpt) or chkpt
    model = init_detector(str(cfg), chkpt, device=kwargs.get('device', 'cuda:0'))
    if not with_mask:
        model.mask_roi_extractor = None
        model.mask_head = None
        model.semantic_roi_extractor = None
        model.semantic_head = None

    return MMDetector(model)

class Detector(nn.Module):
    #__metaclass__ = ABCMeta

    def __init__(self, model):
        super(Detector, self).__init__()
        self.module = model

    def __getattr__(self, name):
        try:
            return super(Detector, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    @property
    def __class__(self):
        return self.module.__class__

    @property
    def with_rpn(self):
        return False

    @property
    def with_det(self):
        return False

    @property
    def with_mask(self):
        return False
    
    @property
    def with_keypts(self):
        return False

    @abstractmethod
    def backbone(self, images, **kwargs):
        pass

    @abstractmethod
    def rpn(self, images, pooling=False, **kwargs):
        pass

    @abstractmethod
    def detect(self, images, pooling=False, **kwargs):
        pass
    
    @abstractmethod
    def forward(self, images, targets=None):
        pass

    @abstractmethod
    def show_result(self,
                    img,
                    result,
                    classes=None,
                    score_thr=0.3,
                    wait_time=0,
                    out_file=None):
        pass

class THDetector(Detector):
    COCO_CLASSES = ('bg',
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 
                    'fire_hydrant', 'X street sign', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'X hat', 'backpack', 'umbrella', 'X shoe', 'X eye glasses' , 
                    'handbag', 'tie', 'suitcase', 'frisbee',  'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 
                    'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'X plate', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 
                    'cake', 'chair', 'couch', 'potted_plant', 'bed', 'X mirror', 'dining_table', 'X window', 'X desk', 'toilet', 
                    'X door', 'tv', 'laptop',  'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 
                    'sink', 'refrigerator', 'X blender', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush',
                    'X hair brush')

    #__metaclass__ = Detector #ABCMeta

    def __init__(self, model):
        super(THDetector, self).__init__(model)

    @property
    def with_rpn(self):
        # self.model.roi_heads.[box_roi_pool | box_head | box_predictor]
        heads = self.module.roi_heads
        return hasattr(heads, 'box_head') and heads.box_head is not None

    @property
    def with_det(self):
        # self.model.roi_heads.box_predictor
        heads = self.module.roi_heads
        return hasattr(heads, 'box_predictor') and heads.box_predictor is not None

    @property
    def with_mask(self):
        return self.module.roi_heads.has_mask

    @property
    def with_keypts(self):
        return self.module.roi_heads.has_keypoint

    def backbone(self, images, **kwargs):
        r"""Returns backbone features and transformed input image list.

        Args:
            images(tensor | List[tensor | str]): a batch tensor of images, a list of image tensors, or image filenames
        
        Returns:
            images(ImageList): a transformed image list with scaled/padded image batch and shape meta
            features(tensor): backbone features in a batch
        """

        mode = self.training
        self.eval()
        model = self.module
        dev = next(model.parameters()).device

        if th.is_tensor(images):
            if images.dim() == 3:
                images = images.unsqueeze(0)
        elif not isinstance(images, list):
            images = [images]

        from ml import cv
        images = [
            image.to(dev) if th.is_tensor(image) else cv.toTorch(cv.imread(image), device=dev) 
            for image in images
        ]

        original_image_sizes = [img.shape[-2:] for img in images]
        with th.no_grad():
            images, _ = model.transform(images, targets=None)
            self.train(mode)
            return model.backbone(images.tensors), images, original_image_sizes
    
    def rpn(self, images, **kwargs):
        r"""Returns RPN proposals as well as backbone features and transformed input image list.

        Args:
            images(tensor): a batch tensor of images

        Kwargs:
            pooling(bool): whether to compute pooled and transformed RoI features and representations
            targets(dict): target descriptor of keys
                boxes(float): list of RoI box tensor of shape(N, 4)
                labels(int64):
                keypoints(float):
        """
        features, images, original_image_sizes = self.backbone(images, **kwargs)

        # Layered outputs or a last layer single batch tensor
        if isinstance(features, th.Tensor):
            features = OrderedDict([(0, features)])
        
        targets = kwargs.get('targets')
        if targets is not None:
            for t in targets:
                assert t['boxes'].dtype.is_floating_point, 'target boxes must be of float type'
                assert t['labels'].dtype == th.int64, 'target labels must be of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == th.float32, 'target keypoints must be of float type'

        pooling = kwargs.pop('pooling', False)
        mode = self.training
        self.eval()
        model = self.module
        with th.no_grad():
            proposals, _ = model.rpn(images, features, **kwargs)
            if self.training:
                proposals, matched_idxs, labels, regression_targets = model.select_training_samples(proposals, targets)
                
            if pooling:
                roi_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
                box_features = model.roi_heads.box_head(roi_features)
                self.train(mode)
                return (proposals, box_features, roi_features), (features, images, original_image_sizes)
            else:
                self.train(mode)
                return proposals, (features, images, original_image_sizes)

    def detect(self, images, **kwargs):
        r"""Returns detections as well as RPN proposals and backbone features.

        Args:
            images(tensor): a batch tensor of images

        Kwargs:
            score_thr(float): threshold to filter out low scored objects
            pooling(bool): whether to compute pooled and transformed RoI features and representations
            targets(dict): target descriptor of keys
                boxes(float):
                labels(int64):
                keypoints(float):
        Returns:
            results(list[tensor]): a list of sorted detection tensors per image tensor([[x1, y1, x2, y2, score, cls]*]+)

        Note:
            - clipped to image
            - bg removed
            - empty or too small filtering
            - scoring threshold: 0.05
            - per class NMS threshold: 0.5
        """
        mode = self.training
        self.eval()
        model = self.module
        (proposals, box_features, roi_featurs), (features, images, original_image_sizes) = self.rpn(images, pooling=True)
        with th.no_grad():
            class_logits, box_regression = model.roi_heads.box_predictor(box_features)
            boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            score_thr = kwargs.pop('score_thr', 0.3)
            num_images = len(boxes)
            results = []
            for i in range(num_images):
                selection = scores[i] > score_thr
                res = dict(
                    boxes=boxes[i][selection],
                    scores=scores[i][selection],
                    labels=labels[i][selection],
                )
                #print(f"images[{i}]: scores[{len(scores[i])}] {scores[i]}")
                results.append(res)

            if kwargs.get('pooling'):
                det_roi_features = model.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
                det_box_features = model.roi_heads.box_head(det_roi_features)
                results = model.transform.postprocess(results, images.image_sizes, original_image_sizes)
                results = [th.cat([res['boxes'], res['scores'].view(-1,1), res['labels'].view(-1,1).float()], dim=1) for res in results]
                self.train(mode)
                return ((results, det_box_features, det_roi_features), 
                        (proposals, box_features, roi_featurs), 
                        (features, images))
            else:
                results = model.transform.postprocess(results, images.image_sizes, original_image_sizes)
                results = [th.cat([res['boxes'], res['scores'].view(-1,1), res['labels'].view(-1,1).float()], dim=1) for res in results]
                self.train(mode)
                return results

    def forward(self, images, targets=None):
        return self.module(images, targets)

    def show_result(self,
                    img,
                    result,
                    classes=None,
                    score_thr=0.3,
                    wait_time=0,
                    out_file=None):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            wait_time (int): Value of waitKey param.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.
        """
        import mmcv
        img = mmcv.imread(img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        # draw bounding boxes
        bboxes = bbox_result[:, :-1].cpu().numpy()
        labels = bbox_result[:, -1].cpu().int().numpy()
        #print(bbox_result.shape, bboxes.shape, labels.shape)
        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            class_names=self.COCO_CLASSES if classes is None else classes,
            score_thr=score_thr,
            show=out_file is None,
            wait_time=wait_time,
            out_file=out_file)

class MMDetector(Detector):
    def __init__(self, model):
        super(MMDetector, self).__init__(model)

    @property
    def __class__(self):
        return self.module.__class__

    @property
    def with_rpn(self):
        return hasattr(self.module, 'rpn_head') and self.module.rpn_head is not None

    @property
    def with_det(self):
        return True

    @property
    def with_mask(self):
        return self.module.with_mask
    
    @property
    def with_keypts(self):
        raise NotImplementedError

    def backbone(self, images, **kwargs):
        r"""Returns list of backbone features and transformed images as well as meta info.
        """
        from mmdet.apis.inference import inference_detector, LoadImage
        from mmdet.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        model = self.module
        cfg = model.cfg
        device = next(model.parameters()).device
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)    
        results = []
        for img in images:
            data = dict(img=img)
            data = test_pipeline(data)
            data = scatter(collate([data], samples_per_gpu=1), [device])[0]
            img = data['img'][0]
            img_meta = data['img_meta'][0]
            data['img'] = img
            data['img_meta'] = img_meta
            data['feats'] = model.extract_feat(img)
            results.append(data)
            #print(img.shape, img_meta)
        
        #return model.backbone(images.tensors), images, original_image_sizes
        return results
        
    def rpn(self, images, **kwargs):
        r"""Returns a list of RPN proposals and RoI pooled features if necessary.
        """
        results = self.backbone(images, **kwargs)
        model = self.module
        pooling = kwargs.pop('pooling', False)
        for res in results:
            x = res['feats']
            img_meta = res['img_meta']
            proposals = model.simple_test_rpn(x, img_meta, model.test_cfg.rpn)[0]
            res['proposals'] = proposals
            #print(f"proposals: {len(proposals)}, {proposals[0]}")
            if pooling:
                from mmdet.core import bbox2roi
                rois = bbox2roi([proposals])
                stages = self.num_stages if hasattr(model, 'num_stages') else 2
                if stages <= 2:
                    # TODO
                    roi_feats = model.bbox_roi_extractor(x[:len(model.bbox_roi_extractor.featmap_strides)], rois)
                    if model.with_shared_head:
                        roi_feats = model.shared_head(roi_feats)

                    box_head = model.box_head
                    if box_head.with_avg_pool:
                        box_feats = box_head.avg_pool(roi_feats)                    
                    box_feats = box_feats.view(box_feats.shape[0], -1)
                    res['roi_feats'] = roi_feats
                    res['box_feats'] = box_feats
                    #print(roi_feats.shape, box_feats.shape)
                else:
                    ms_scores = []
                    ms_bbox_result = {}
                    for i in range(stages):
                        bbox_roi_extractor = model.bbox_roi_extractor[i]
                        bbox_head = model.bbox_head[i]
                        roi_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
                        if model.with_shared_head:
                            roi_feats = model.shared_head(roi_feats)
                        
                        cls_score, bbox_pred = bbox_head(roi_feats)
                        ms_scores.append(cls_score)
                        #print(f"[stage{i}] {rois.shape}, {roi_feats.shape}, {cls_score.shape}, {bbox_pred.shape}")
                        if i < stages - 1:
                            bbox_label = cls_score.argmax(dim=1)
                            rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])

                    cls_score = sum(ms_scores) / stages
                    res['proposals'] = th.cat([rois[:, 1:], cls_score.max(dim=1, keepdim=True)[0]], dim=1)
                    res['bbox_pred'] = bbox_pred
                    res['roi_feats'] = roi_feats
                    print(bbox_pred.shape, bbox_pred[0])
                    #res['box_feats'] = box_feats
                    #proposals = res['proposals']
                    #print(f"refined proposals: {len(proposals)}, {proposals[0]}")
        return results

    def detect(self, images, **kwargs):
        r"""Detect objects in one or more images.
        Args:
            images(str/ndarray | List[str/ndarray]): filename or a list of filenames

        Kwargs:
            score_thr(float): threshold to filter out low scored objects

        Returns:
            results: a list of sorted per image detection [[x1, y1, x2, y2, score, cls]*]
        """
        from mmdet.apis import inference_detector
        if not isinstance(images, list):
            images = [images]

        results = []
        score_thr = kwargs.pop('score_thr', 0.3)
        for i, img in enumerate(images):
            res = inference_detector(self.module, img)
            dets = []
            # det: (x1, y1, x2, y2, score, class)
            for c, det in enumerate(res): # no bg, 0:80
                det = th.from_numpy(det[det[:, -1] > score_thr])
                #print(f"{len(det)} detections of class {c}")
                if len(det) > 0:
                    cls = th.Tensor([c] * len(det)).view(-1, 1)
                    det = th.cat([det, cls], dim=1)
                    #print(det)
                    dets.append(det)

            if dets:
                dets = th.cat(dets)
                sorted = th.argsort(dets[:, 4+1], descending=True)
                dets = dets[sorted]
                results.append(dets)
            else:
                results.append(th.zeros(0, 5+1))
            #print(f"[{i}] {len(dets)} detections")

        return results
    
    def forward(self, images, targets=None):
        raise NotImplementedError
    
    def show_result(self,
                    img,
                    result,
                    classes=None,
                    score_thr=0.3,
                    wait_time=0,
                    out_file=None):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            wait_time (int): Value of waitKey param.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.
        """
        import mmcv
        img = mmcv.imread(img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        if False:
            # TODO: Show mask result
            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5

        # draw bounding boxes
        bboxes = bbox_result[:, :-1].numpy()
        labels = bbox_result[:, -1].int().numpy()
        #print(bbox_result.shape, bboxes.shape, labels.shape)
        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            class_names=self.CLASSES if classes is None else classes,
            score_thr=score_thr,
            show=out_file is None,
            wait_time=wait_time,
            out_file=out_file)