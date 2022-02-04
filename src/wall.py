import os
from itertools import product
from PIL import Image
import cv2
import json


class Wall:
    def __init__(self, image_path, predictor, output_dir='.'):
        self.predictor = predictor
        self.image_path = image_path
        self.output_dir = output_dir
        # internal values - not a fan of maintaining partial state
        self._name, self._ext = self._process_image_path(self.image_path)
        self._tiles = None  # ?
        self._tile_size = None  # ?
        self._img_size = None  # ?

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
        # grid = product(range(0, img_h-img_h % tile_h, tile_h),
        #                range(0, img_w-img_w % tile_w, tile_w))
        grid = product(range(0, img_h, tile_h),
                       range(0, img_w, tile_w))
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
        # TODO: make this method generic
        if self._tiles is None:
            return None
        tile_w, tile_h = self._tile_size
        img_w, img_h = self._img_size
        if tile_w is None or tile_h is None:
            return None
        # grid = product(range(0, img_h-img_h % tile_h, tile_h),
        #                range(0, img_w-img_w % tile_w, tile_w))
        grid = product(range(0, img_h, tile_h),
                       range(0, img_w, tile_w))
        dst_img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        for i, j in grid:
            box = (j, i)
            tile_name = f'{self._name}_{i}_{j}'
            dst_img.paste(self._tiles[tile_name], box)
        if store_result:
            dst_img.save(os.path.join(self.output_dir,
                         f'{output_file_name}{self._ext}'))
        return dst_img

    def find_holds(self, save_result=True, show_result=False):
        pass

    def extract_polygons(self, save_result=True, show_result=False):
        img = cv2.imread(self.image_path)
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
        if save_result:    
            cv2.imwrite('contours.png', img)
        if show_result:
            cv2.imshow('final', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    # some tests
    w = Wall('./raw_images/monkey_rock_V3.JPG', None)
    w.segment_image(tile_width=400, tile_height=400, store_result=True)
    w.merge_tiles(store_result=True, output_file_name="merged")
