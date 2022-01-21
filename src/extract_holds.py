import os
import json
from itertools import product
import numpy as np
from PIL import Image
from os import walk


import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

IMAGE_EXTENSION = '.png'
INPUT_DIR = 'raw_images'
OUTPUT_DIR = 'processed_images'

# All this should better be refactored into a class

def segment_image_into_tiles(filename, tile_dimensions = (None, None), dir_in = "", dir_out = "", ext = IMAGE_EXTENSION, save_tiles=True):
    """
    Segments an image into tiles.

    :param filename: The name of the image file to segment.
    :type filename: str
    :param dir_in: The directory containing the image file.
    :type dir_in: str
    :param dir_out: The directory to save the tiles to.
    :type dir_out: str
    :param ext: The extension of the image file.
    :type ext: str
    :param tile_dimensions: The dimensions of the tiles.
    :type tile_dimensions: tuple
    :param save_tiles: Whether or not to save the tiles.
    :type save_tiles: bool
    """
    tiles = dict()
    name, _ = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    img_w, img_h = img.size
    tile_w = tile_dimensions[0] if tile_dimensions[0] is not None else img_w
    tile_h = tile_dimensions[1] if tile_dimensions[1] is not None else img_h
    grid = product(range(0, img_h-img_h%tile_h, tile_h), range(0, img_w-img_w%tile_w, tile_w))
    for i, j in grid:
        box = (j, i, j+tile_w, i+tile_h)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        tile = img.crop(box)
        if save_tiles:
            tile.save(out)
        tiles[out] = tile 
    return tiles

def config_detector(model_path):
    # config values
    mask_path = "./mask.json"
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    device = "cpu"
    dataset = "climb_dataset"
    # General detector setup
    register_coco_instances(dataset, {}, mask_path, "")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (hold, volume, downclimb)
    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = (dataset,)
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

    # setup inference
    predictor = DefaultPredictor(cfg)
    train_metadata = MetadataCatalog.get(dataset)

    # dataset catalog needs to exist so the polygon classes show up correctly
    DatasetCatalog.get(dataset)

    return predictor, train_metadata, device

def find_holds(tiles, predictor, train_metadata, device, save_tiles=True, show_results=False):
    masks = dict()
    # if tiles is a list of images, load them in memory
    if type(tiles) is not dict:
        tiles = {tile: Image.open(tile) for tile in tiles}

    for tile in tiles:
        # im = cv2.imread(image_path)
        im = cv2.cvtColor(np.array(tiles[tile]), cv2.COLOR_RGB2BGR)
        outputs = predictor(im)

        # get hold masks
        mask = outputs['instances'].get('pred_masks')
        mask = mask.to(device)
        _, h, w = mask.shape
        bin_mask = np.zeros((h, w))

        # build binary mask
        for m in mask:
            bin_mask = np.add(bin_mask, np.array(m.long()))
        
        masks[tile] = bin_mask

        if save_tiles:
            p_name = get_processed_name(tile)
            cv2.imwrite(p_name, np.array(bin_mask), [cv2.IMWRITE_PNG_BILEVEL, 1])
            # cv2.imshow(p_name, np.array(bin_mask))

        if show_results:
            v = Visualizer(im[:, :, ::-1],
                    metadata=train_metadata,
                    #    scale=0.75,
                    scale=0.3,
                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                    )
            v = v.draw_instance_predictions(outputs["instances"].to(device))
            # cv2.imshow('climbnet', v.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
    return masks

def get_processed_name(tile):
    name, _ = os.path.splitext(tile)
    name = name.replace('raw', 'processed')
    return name + '_processed' + IMAGE_EXTENSION

def concat_images_v(images_list):
    images = [Image.open(i) for i in images_list]
    overall_height = sum([i.height for i in images])
    dst = Image.new('RGB', (max([i.width for i in images]), overall_height), (255, 255, 255))
    h = 0
    for im in images:
        dst.paste(im, (0, h))
        h += im.height
    return dst

def main(image_paths, model_path, tile_width, tile_height, combine_results):
    if tile_width == 0:
        tile_width = None
    if tile_height == 0:
        tile_height = None
    predictor, train_metadata, device = config_detector(model_path)
    for image_path in image_paths:
        tiles = segment_image_into_tiles(image_path, (tile_width, tile_height), INPUT_DIR, OUTPUT_DIR)
        tile_masks = find_holds(tiles, predictor, train_metadata, device, OUTPUT_DIR)
        dst = concat_images_v([get_processed_name(tile) for tile in tile_masks])
        dst.save(os.path.join(OUTPUT_DIR, os.path.splitext(image_path)[0] + '_hold_masks.png'))
        if combine_results:
            all_mask = find_holds([INPUT_DIR + '/' + image_path], predictor, train_metadata, device, OUTPUT_DIR)
            background = Image.open(os.path.join(OUTPUT_DIR,  os.path.splitext(image_path)[0] + '_hold_masks.png'))
            img = Image.open(os.path.join(OUTPUT_DIR, os.path.splitext(image_path)[0] + '_processed.png'))
            # make black pixels transparent
            img = img.convert("RGBA")
            datas = img.getdata()

            new_data = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            img.putdata(new_data)
            background.paste(img, (0, 0), img)
            background.save(os.path.join(OUTPUT_DIR, os.path.splitext(image_path)[0] + '_superimposed.png'),"PNG")
            poly_approx(os.path.join(OUTPUT_DIR, os.path.splitext(image_path)[0] + '_superimposed.png'), os.path.splitext(image_path)[0])
        else:
            poly_approx(os.path.join(OUTPUT_DIR, os.path.splitext(image_path)[0] + '_hold_masks.png'), os.path.splitext(image_path)[0])

def poly_approx(image, name):
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
    with open('extracted_data/' + name + '_holds.json', 'w') as f:
        f.write(json.dumps(dict(holds=approximations)))
    # cv2.imshow("final", img)
    cv2.imwrite(os.path.splitext(image)[0] + '_contours.png', img)
    # cv2.waitKey(0)
    return dict(holds=approximations)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hold detection')
    
    parser.add_argument('images_path', type=str, default='', help='The input wall image file')
    parser.add_argument('model_path', type=str, default='', help='climbnet model weights')
    parser.add_argument('tile_width', type=int, default=0, help='Tile width')
    parser.add_argument('tile_height', type=int, default=0, help='Tile height')
    parser.add_argument('combine_results', type=bool, default=False, help='Mix results of split detection and overall detection')
    
    args = parser.parse_args()


    f = []
    for (dirpath, dirnames, filenames) in walk(args.images_path):
        f.extend(filenames)
        break

    main(f, args.model_path, args.tile_width, args.tile_height, args.combine_results)

# sample run:
# > python .\extract_holds.py s1.jpg ..\model_weights\model_d2_R_50_FPN_3x.pth 0 600 False