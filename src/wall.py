import cv2
import os
from itertools import product


from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from PIL import Image


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

        return predictor, train_metadata,

    def predict(self):
        pass


class Wall:
    def __init__(self, image_path, predictor, output_dir='.'):
        self.predictor = predictor
        self.image_path = image_path
        self.output_dir = output_dir
        # internal values
        self._name, self._ext = self._process_image_path(self.image_path)
        self._tiles = None # ?
        self._tile_size = None # ?
        self._img_size = None # ?

    def _process_image_path(self, image_path):
        img_path = os.path.normpath(image_path).split(os.path.sep)
        return os.path.splitext(img_path[-1])

    def segment_image(self, tile_width, tile_height, store_result=True):
        tiles = dict()
        img = Image.open(self.image_path)
        img_w, img_h = img.size
        # set image size
        self._img_size = (img_w, img_h)
        tile_w = tile_width if tile_width is not None else img_w
        tile_h = tile_height if tile_height is not None else img_h
        grid = product(range(0, img_h-img_h % tile_h, tile_h),
                       range(0, img_w-img_w % tile_w, tile_w))
        for i, j in grid:
            box = (j, i, j+tile_w, i+tile_h)
            tile = img.crop(box)
            tile_name = f'{self._name}_{i}_{j}'
            out_file = os.path.join(self.output_dir, f'{tile_name}{self._ext}')
            if store_result:
                tile.save(out_file)
            tiles[tile_name] = tile
        # set tile data
        self._tiles = tiles
        self._tile_size = (tile_w, tile_h)
        return tiles

    def merge_tiles(self, store_result=True, output_file_name="merged"):
        if self._tiles is None:
            return None
        tile_w, tile_h = self._tile_size
        img_w, img_h = self._img_size
        if tile_w is None or tile_h is None:
            return None
        grid = product(range(0, img_h-img_h % tile_h, tile_h),
                       range(0, img_w-img_w % tile_w, tile_w))
        dst_img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        for i, j in grid:
            box = (j, i)
            tile_name = f'{self._name}_{i}_{j}'
            dst_img.paste(self._tiles[tile_name], box)
        if store_result:
            dst_img.save(os.path.join(self.output_dir, f'{output_file_name}{self._ext}'))
        return dst_img

