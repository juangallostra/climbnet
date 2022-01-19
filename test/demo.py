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

def run_inference(image_path, model_path, output_path):
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
    cv2.imwrite(output_path, np.array(bin_mask), [cv2.IMWRITE_PNG_BILEVEL, 1])
    cv2.imshow(output_path, np.array(bin_mask))

    v = Visualizer(im[:, :, ::-1],
                   metadata=train_metadata,
                   #    scale=0.75,
                   scale=0.2,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('climbnet', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

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

def point_in_polygon(polygon_file):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    with open(polygon_file, 'r') as f:
        polygons = json.loads(f.read()).get('holds')

    point = Point(545, 150)
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

    run_inference(args.image_path, args.model_path, args.output_image_path)
    concat_images_v(['2.png', '3.png', '4.png']).save('all_test.png')
    poly_approx('all_test.png')
    print(point_in_polygon('test_pol.json'))