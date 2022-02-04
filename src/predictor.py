from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import cv2
import numpy as np
import os


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # Should I let the user specify these?
        self.mask_path = "./mask.json"
        self.config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        self.device = "cpu"
        self.dataset = "climb_dataset"
        # Set after configuration
        self.predictor = None
        self.train_metadata = None

    def configure(self):
        if self.predictor is None and self.train_metadata is None:
            return

        register_coco_instances(self.dataset, {}, self.mask_path, "")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.DATALOADER.NUM_WORKERS = 1
        # 3 classes (hold, volume, downclimb)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = os.path.join(self.model_path)
        cfg.MODEL.DEVICE = self.device
        cfg.DATASETS.TEST = (self.dataset,)
        # set the testing threshold for this model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

        # setup inference
        predictor = DefaultPredictor(cfg)
        train_metadata = MetadataCatalog.get(self.dataset)

        # dataset catalog needs to exist so the polygon classes show up correctly
        DatasetCatalog.get(self.dataset)

        self.predictor = predictor
        self.train_metadata = train_metadata

        return predictor, train_metadata

    def predict(self, save_result=True, show_result=False):
        if self._tiles is None:
            return None
        masks = dict()
        for tile in self._tiles:
            # im = cv2.imread(image_path)
            im = cv2.cvtColor(np.array(self._tiles[tile]), cv2.COLOR_RGB2BGR)
            outputs = self.predictor(im)

            # get hold masks
            mask = outputs['instances'].get('pred_masks')
            mask = mask.to(self.device)
            _, h, w = mask.shape
            bin_mask = np.zeros((h, w))

            # build binary mask
            for m in mask:
                bin_mask = np.add(bin_mask, np.array(m.long()))

            masks[tile] = bin_mask

            if save_result:
                pass
                # p_name = get_processed_name(tile)
                # cv2.imwrite(p_name, np.array(bin_mask), [
                #             cv2.IMWRITE_PNG_BILEVEL, 1])
                # cv2.imshow(p_name, np.array(bin_mask))

            if show_result:
                v = Visualizer(im[:, :, ::-1],
                               metadata=self.train_metadata,
                               #    scale=0.75,
                               scale=0.3,
                               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                               )
                v = v.draw_instance_predictions(
                    outputs["instances"].to(self.device))
                # cv2.imshow('climbnet', v.get_image()[:, :, ::-1])
                # cv2.waitKey(0)
        return masks
