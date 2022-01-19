import os
import json

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np


# 1. split input image into subimages
# 2. detect holds for each image
# 3. approx holds by polygons
# 4. compute polygon "world" coordinates
# 5. store all data in a json file

def get_image_segments(image_file, n_v, n_w):
    """[summary]

    :param image_file: [description]
    :type image_file: [type]
    """
    # open image
    img = np.asarray(Image.open(image_file))
    w, h, channels = img.shape
    tiles = img.reshape(
        n_w, w // n_w, 
        n_v, h // n_v, 
        channels)
    tiles = tiles.swapaxes(1, 2).reshape(n_w * n_v, w // n_w, h // n_v, channels)
    for idx, tile in enumerate(tiles):
        print(tile)
        tile_image = Image.fromarray(tile)
        tile_image.save(f'{idx}_split.png')
    return tiles


def tile(filename, dir_in = "", dir_out = "", d = (None, 800) ):
    from itertools import product
    tile_names = []
    name, ext = os.path.splitext(filename)
    ext = '.png'
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    d_zero = d[0] if d[0] is not None else w
    d_one = d[1] if d[1] is not None else h
    grid = product(range(0, h-h%d_one, d_one), range(0, w-w%d_zero, d_zero))
    for i, j in grid:
        box = (j, i, j+d_zero, i+d_one)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        tile_names.append(out)
        img.crop(box).save(out)
    return tile_names


def run_inference(image_paths, model_path, output_path_prefix):
    register_coco_instances("climb_dataset", {}, "./mask.json", "")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (hold, volume, downclimb)
    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.DATASETS.TEST = ("climb_dataset",)
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

    # setup inference
    predictor = DefaultPredictor(cfg)
    train_metadata = MetadataCatalog.get("climb_dataset")

    # dataset catalog needs to exist so the polygon classes show up correctly
    DatasetCatalog.get("climb_dataset")
    for image_path in image_paths:
        im = cv2.imread(image_path)
        outputs = predictor(im)

        # import pdb; pdb.set_trace()

        # get hold masks
        mask = outputs['instances'].get('pred_masks')
        mask = mask.to('cpu')
        _, h, w = mask.shape
        bin_mask = np.zeros((h, w))

        # build binary mask
        for m in mask:
            bin_mask = np.add(bin_mask, np.array(m.long()))
        
        print(output_path_prefix + image_path)
        cv2.imwrite(output_path_prefix + image_path, np.array(bin_mask), [cv2.IMWRITE_PNG_BILEVEL, 1])
        # cv2.imshow(output_path_prefix + image_path, np.array(bin_mask))

        v = Visualizer(im[:, :, ::-1],
                    metadata=train_metadata,
                    #    scale=0.75,
                    scale=0.2,
                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                    )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('climbnet', v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

def concat_images_v(images_list):
    images = [Image.open(i) for i in images_list]
    overall_height = sum([i.height for i in images])
    dst = Image.new('RGB', (max([i.width for i in images]), overall_height), (255, 255, 255))
    h = 0
    for im in images:
        dst.paste(im, (0, h))
        h += im.height
    return dst

def poly_approx(image):
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour approximate the curve and
    # detect the shapes.
    approximations = []
    for cnt in contours[1::]:
        # epsilon = 0.01*cv2.arcLength(cnt, True)
        epsilon = 0.005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approximations.append([a[0] for a in approx.tolist()])
        # print(approx)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
    with open('test_pol.json', 'w') as f:
        f.write(json.dumps(dict(holds=approximations)))
    cv2.imshow("final", img)
    cv2.imwrite('contours.png', img)
    cv2.waitKey(0)

def point_in_polygon(point, polygon_file):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    with open(polygon_file, 'r') as f:
        polygons = json.loads(f.read()).get('holds')

    Ppolygons = []
    for p in polygons:
        if len(p) >= 3:
            Ppolygons.append(Polygon(tuple(tuple(c) for c in p)))
    return [polygon.contains(point) for polygon in Ppolygons]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Climbnet demo')
    parser.add_argument('image_path', type=str,
                        help='image file')
    parser.add_argument('model_path', type=str,
                        help='climbnet model weights')
    parser.add_argument('output_image_path', type=str,
                        help='output image path')

    args = parser.parse_args()

    # run_inference(args.image_path, args.model_path, args.output_image_path)
    # print(point_in_polygon('test_pol.json'))
    # get_image_segments('test.jpg', 1, 5)

    # tiles = tile('test.jpg')
    # run_inference(tiles, args.model_path, 'out_')
    # concat_images_v(['out_' + name for name in tiles]).save('all_test.png')
    poly_approx('all_test_mod.png')
