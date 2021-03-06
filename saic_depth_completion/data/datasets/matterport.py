import os
import torch

import numpy as np
from PIL import Image
import json
import random

# ROOT = '/Vol1/dbstore/datasets/depth_completion/Matterport3D/'
ROOT = "/Vol0/user/d.senushkin/datasets/matterport3d"

class Matterport:
    def __init__(
            self, root=ROOT, coco_path="",split="train", refined_depth=False, args=None, transforms=None, 
    ):
        self.transforms = transforms
        self.split = split
        self.data_root = root
        self.split_file = os.path.join(root, "splits", split + ".txt")
        # if not os.path.exists(coco_path) and not os.path.exists(root):
        #     self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.depth_name, self.render_name, self.rawD_path_list, self.mask_path_list = [], [], [], [], []
        self.normal_name = []
        self.coco_path = coco_path
        self.refined_depth = refined_depth
        self._load_data()
        self.args = args

    def _load_data(self):

        with open(self.coco_path, 'r') as j:
            coco_annotation = json.loads(j.read())
        
        # random.seed(30)
        # random.shuffle(coco_annotation["images"])
        for item in coco_annotation["images"]: # TODO 
            self.depth_name.append(os.path.join(self.data_root, item["raw_meshD_path"]))
            self.color_name.append(os.path.join(self.data_root, item["mirror_color_image_path"]))
            self.rawD_path_list.append(os.path.join(self.data_root, item["raw_meshD_path"]))
            self.mask_path_list.append(os.path.join(self.data_root, item["mirror_instance_mask_path"]))
            
            if self.refined_depth:
                self.render_name.append(os.path.join(self.data_root, item["refined_meshD_path"])) # completed + mirrored depth [in m3d folder]
            else:
                self.render_name.append(os.path.join(self.data_root, item["raw_meshD_path"]))  # completed depth

    def _get_data_list(self, filename):
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        data_list = []
        for ele in content:
            left, _, right = ele.split('/')
            valid, resize_count, one_scene_name, num_1, num_2, png = self._split_matterport_path(right)
            if valid == False:
                print(f'Invalid data_id in datalist: {ele}')
            data_list.append((left, one_scene_name, num_1, num_2))
        return set(data_list)

    def _split_matterport_path(self, path):
        try:
            left, png = path.split('.')
            lefts = left.split('_')
            resize_count = left.count('resize')
            one_scene_name = lefts[resize_count]
            num_1 = lefts[resize_count+1][-1]
            num_2 = lefts[resize_count+2]
            return True, resize_count, one_scene_name, num_1, num_2, png
        except Exception as e:
            print(e)
            return False, None, None, None, None, None

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color           = np.array(Image.open(self.color_name[index]).resize((self.args.input_width,self.args.input_height), Image.NEAREST)).transpose([2, 0, 1]) / 255.
        # mesh depth
        render_depth    = np.array(Image.open(self.render_name[index]).resize((self.args.input_width,self.args.input_height), Image.NEAREST)) / float(self.args.depth_shift)
        # raw depth
        depth           = np.array(Image.open(self.depth_name[index]).resize((self.args.input_width,self.args.input_height), Image.NEAREST)) / float(self.args.depth_shift)
        
        # normals = np.array(Image.open(self.normal_name[index]).resize((160,128), Image.NEAREST)).transpose([2, 0, 1])
        # normals = ((normals / 255) - 0.5)*2 #(normals - 90.) / 180.

        mask = np.zeros_like(depth)
        mask[np.where(depth > 0)] = 1

        if self.split != "train":
            return  {
                'color':        torch.tensor(color, dtype=torch.float32),
                'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
                'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
                'gt_depth':     torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
                'gt_depth_path':self.render_name[index],
                'color_img_path':self.color_name[index],
                'rawD_path':self.rawD_path_list[index],
                'mask_path':self.mask_path_list[index]
                
            }
        else:
            return  {
                'color':        torch.tensor(color, dtype=torch.float32),
                'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
                'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
                'gt_depth':     torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
                'gt_depth_path':self.render_name[index],
                'color_img_path':self.color_name[index],
                'rawD_path':self.rawD_path_list[index],
                'mask_path':self.mask_path_list[index]
            }