import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import utils

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class InpaintDataset(Dataset):
    def __init__(self, config):
        assert config.mask_type in ALLMASKTYPES
        self.config = config
        self.imglist = utils.get_files(config.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.imgsize, self.config.imgsize))

        # mask
        if self.config.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.config.imgsize, margin = self.config.margin, bbox_shape = self.config.bbox_shape, times = 1)
        if self.config.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.config.imgsize, margin = self.config.margin, bbox_shape = self.config.bbox_shape, times = self.config.mask_num)
        if self.config.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.config.imgsize, max_angle = self.config.max_angle, max_len = self.config.max_len, max_width = self.config.max_width, times = self.config.mask_num)
        
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
        
class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self, config):
        self.config = config
        self.namelist = utils.get_names(config.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        imgpath = os.path.join(self.config.baseroot, imgname)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.imgsize, self.config.imgsize))
        # mask
        maskpath = os.path.join(self.config.maskroot, imgname)
        img = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return img, mask, imgname
    
import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch 
import cv2
import json
import numpy as np
import json
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)
def is_json_file(fname):
    return fname.lower().endswith('json')



class ImageDataset(Dataset):
    def __init__(self,  config
                        ):
        super().__init__()
        self.config = config
        
        imags_path=self.config.imags_path,
        labels_path=self.config.labels_path,
        scan_subdirs=self.config.scan_subdirs, 
        transforms=None
        
        self.img_shape = self.config.img_shapes  # [W, H, C]
        self.random_crop = self.config.random_crop

        self.mode = 'RGB'
        if self.img_shape[2] == 1:
            self.mode = 'L' # convert to greyscale
        if scan_subdirs:
            self.samples_img,self.samples_label = self.make_dataset_from_subdirs()
        else:
            self.samples_img = [entry.path for entry in os.scandir(imags_path) 
                                                if is_image_file(entry.name)]
            self.samples_label = [entry.path for entry in os.scandir(labels_path) 
                                                if is_json_file(entry.name)]
        
        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])
    def __len__(self):
        return len(self.samples_img)

    def __getitem__(self, index):
        img = pil_loader(self.samples_img[index], self.mode)
        labels_mask = self.draw_mask_from_annotation(self.samples_label[index])

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape[:2]))(img)
                labels_mask = T.Resize(max(self.img_shape[:2]))(labels_mask)
            img = T.RandomCrop(self.img_shape[:2])(img)
            labels_mask = T.RandomCrop(self.img_shape[:2])(labels_mask)
        else:
            img = T.Resize(self.img_shape[:2])(img)
            labels_mask = T.Resize(self.img_shape[:2])(labels_mask)
        # mask
        if self.config.mask_type == 'single_bbox':
            mask_sc = self.bbox2mask(shape = self.config.imgsize, margin = self.config.margin, bbox_shape = self.config.bbox_shape, times = 1)
        if self.config.mask_type == 'bbox':
            mask_sc = self.bbox2mask(shape = self.config.imgsize, margin = self.config.margin, bbox_shape = self.config.bbox_shape, times = self.config.mask_num)
        if self.config.mask_type == 'free_form':
            mask_sc = self.random_ff_mask(shape = self.config.imgsize, max_angle = self.config.max_angle, max_len = self.config.max_len, max_width = self.config.max_width, times = self.config.mask_num)
        

        mask_sc = torch.from_numpy(mask_sc.astype(np.float32))
        mask = torch.logical_or(labels_mask, mask_sc).to(torch.float32).contiguous()
        img = self.transforms(img)
        img.mul_(2).sub_(1).to(torch.float32).contiguous() # [0, 1] -> [-1, 1]

        return img, mask

    def make_dataset_from_subdirs(self):
        samples_img = []
        samples_img = utils.get_files(self.config.imags_path)
        samples_label = utils.get_files(self.config.labels_path)
        samples_img = [img for img in samples_img if is_image_file(img)]
        samples_label = [lb for lb in samples_label if is_json_file(lb)]
        return samples_img,samples_label
    def draw_mask_from_annotation(self,json_path, labels = []):
        with open(json_path, 'rb') as f:
            labelme_data = json.load(f)
        imageHeight = labelme_data['imageHeight']
        imageWidth = labelme_data['imageWidth']
        polygon_points = []
        for obj in labelme_data['shapes']:
            label = obj['label']
            points = obj['points']
            if labels:
                for lab in labels:
                    if label == lab:
                        polygon_points.append([(x, y) for x, y in points])
            else:
                polygon_points.append([(x, y) for x, y in points])
        mask = np.zeros((imageHeight,imageWidth),dtype=np.uint8)
        for point in polygon_points:
            point = np.array(point,dtype=np.int32)
            mask = cv2.fillPoly(mask, pts=[point], color=1)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int32)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int32)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask
    
    
    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    
    
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class ImageDataset_best(Dataset):
    def __init__(self, folder_path, 
                       img_shape, # [W, H, C]
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L' # convert to greyscale

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path) 
                                              if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = pil_loader(self.data[index], self.mode)

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape[:2]))(img)
            img = T.RandomCrop(self.img_shape[:2])(img)
        else:
            img = T.Resize(self.img_shape[:2])(img)

        img = self.transforms(img)
        img.mul_(2).sub_(1) # [0, 1] -> [-1, 1]

        return img