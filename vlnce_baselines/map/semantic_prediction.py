import attr
import time
from typing import Any, Union, List, Tuple
from abc import ABCMeta, abstractmethod

import cv2
import torch
import numpy as np

from habitat import Config

import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from vlnce_baselines.map.RepViTSAM.setup_repvit_sam import build_sam_repvit
from vlnce_baselines.common.utils import get_device


VisualObservation = Union[torch.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class Segment(metaclass=ABCMeta):
    config: Config
    device: torch.device
    
    def __attrs_post_init__(self):
        self._create_model(self.config, self.device)
    
    @abstractmethod
    def _create_model(self, config: Config, device: torch.device) -> None:
        pass
    
    @abstractmethod
    def segment(self, image: VisualObservation, **kwargs) -> Any:
        pass
    

@attr.s(auto_attribs=True)
class GroundedSAM(Segment):
    height: float = 480.
    width: float = 640.
    
    def _create_model(self, config: Config, device: torch.device) -> Any:
        GROUNDING_DINO_CONFIG_PATH = config.MAP.GROUNDING_DINO_CONFIG_PATH
        GROUNDING_DINO_CHECKPOINT_PATH = config.MAP.GROUNDING_DINO_CHECKPOINT_PATH
        SAM_CHECKPOINT_PATH = config.MAP.SAM_CHECKPOINT_PATH
        SAM_ENCODER_VERSION = config.MAP.SAM_ENCODER_VERSION
        RepViTSAM_CHECKPOINT_PATH = config.MAP.RepViTSAM_CHECKPOINT_PATH
        # device = torch.device("cuda", config.TORCH_GPU_ID if torch.cuda.is_available() else "cpu")
        
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=device
            )
        if config.MAP.REPVITSAM:
            sam = build_sam_repvit(checkpoint=RepViTSAM_CHECKPOINT_PATH)
            sam.to(device=device)
        else:
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.box_threshold = config.MAP.BOX_THRESHOLD
        self.text_threshold = config.MAP.TEXT_THRESHOLD
        self.grounding_dino_model.model.eval()
        
    def _segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def _process_detections(self, detections: sv.Detections) -> sv.Detections:
        box_areas = detections.box_area
        i = len(detections) - 1
        while i >= 0:
            if box_areas[i] / (self.width * self.height) < 0.95:
                i -= 1
                continue
            else:
                detections.xyxy = np.delete(detections.xyxy, i, axis=0)
                if detections.mask is not None:
                    detections.mask = np.delete(detections.mask, i, axis=0)
                if detections.confidence is not None:
                    detections.confidence = np.delete(detections.confidence, i)
                if detections.class_id is not None:
                    detections.class_id = np.delete(detections.class_id, i)
                if detections.tracker_id is not None:
                    detections.tracker_id = np.delete(detections.tracker_id, i)
            i -= 1
            
        return detections
    
    @torch.no_grad()
    def segment(self, image: VisualObservation, **kwargs) -> Tuple[np.ndarray, List[str], np.ndarray]:
        classes = kwargs.get("classes", [])
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = []
        # t1 = time.time()
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        # t2 = time.time()
        detections = self._process_detections(detections)
        
        # 兼容不同版本的 supervision：使用属性而不是迭代
        for i in range(len(detections.xyxy)):
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            class_id = detections.class_id[i] if detections.class_id is not None else None
            
            if class_id is not None:
                labels.append(f"{classes[class_id]} {confidence:0.2f}")
            else:
                labels.append(f"unknown {confidence:0.2f}")
        # t3 = time.time()
        detections.mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        # t4 = time.time()
        # print("grounding dino: ", t2 - t1)
        # print("process detections: ", t3 - t2)
        # print("sam: ", t4 - t3)
        # annotated_image.shape=(h,w,3)
        if mask_annotator is not None:
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        else:
            # 旧版本 supervision：手动绘制 mask
            annotated_image = image.copy()
            if detections.mask is not None:
                for mask in detections.mask:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask_bool = mask.astype(bool)
                    annotated_image[mask_bool] = annotated_image[mask_bool] * 0.5 + color_mask * 0.5
        
        # 兼容不同版本的 BoxAnnotator.annotate() API
        try:
            # 新版本：labels 参数
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        except TypeError:
            # 旧版本：没有 labels 参数，手动绘制文本
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            # 手动添加标签（如果需要的话）
            import cv2
            for i, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # detectins.mask.shape=[num_detected_classes, h, w]
        # attention: sometimes the model can't detect all classes, so num_detected_classes <= len(classes)
        return (detections.mask.astype(np.float32), labels, annotated_image, detections)
    

class BatchWrapper:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    """
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, images: List[VisualObservation]) -> List:
        return [self.model(image) for image in images]