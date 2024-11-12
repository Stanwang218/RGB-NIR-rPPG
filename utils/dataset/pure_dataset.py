from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
from PIL import Image
import numpy as np
import ast
import torch
import torchvision.transforms as transforms
from sig_util import *
import cv2 as cv
from torch.utils.data._utils.collate import default_collate
import glob
  
class MSTmap_PURE_cut(Dataset):
    @staticmethod
    def split_dataset(config):
        img_size = 224
        train_subject = config['train_subject']
        test_subject = config['test_subject']
        valid_subject = config['valid_subject']
        map_type = config['map_type']
        selected_topics = config['selected_topic']
        # dataset_path = config['path'] # label path
        dataset_path = '/data/PreprocessedData/PURE_MSTmap'
        csv_file = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
        
        train_bvp_list, test_bvp_list, valid_bvp_list = [], [], []
        print(dataset_path)
        bvp_list = csv_file['project_name'].tolist()
        
        for bvp_name in bvp_list:
            subject_name = int(os.path.basename(bvp_name).split('_')[0])
            subject_no = subject_name // 100
            if subject_no in train_subject:
                train_bvp_list.append(bvp_name + ".npy")
            if subject_no in test_subject:
                test_bvp_list.append(bvp_name + ".npy")
            if subject_no in valid_subject:
                valid_bvp_list.append(bvp_name+ ".npy")
            
            
        train_dataset, test_dataset, valid_dataset = \
                MSTmap_PURE_cut(dataset_path, train_bvp_list, map_type), MSTmap_PURE_cut(dataset_path, test_bvp_list, map_type), MSTmap_PURE_cut(dataset_path, valid_bvp_list, map_type)
        # exit()
        return train_dataset, test_dataset, valid_dataset
    
    def __init__(self, root='/data/PreprocessedData/PURE_MSTmap',
                 bvp_list = None, map_list=["CHROM_POS_G", "YUV"],  img_size = 224, img_shape = (224, 224)):
        super().__init__()
        print(f"Using map {map_list}")
        self.img_size = img_size
        self.root = root
        self.bvp_path = os.path.join(root, 'bvp')
        self.maps = map_list
        self.h, self.w = img_shape
        
        
        if bvp_list is None:
            self.bvp_list = os.listdir(self.bvp_path)
        else:
            self.bvp_list = bvp_list
        # f = open()
        self.chrom_path = os.path.join(self.root, "CHROM")
        self.pos_path = os.path.join(self.root, "POS")
        self.yuv_path = os.path.join(self.root, "YUV")
        self.rgb_path = os.path.join(self.root, "RGB")
        self.nir_path = os.path.join(self.root, "NIR")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.h,self.w)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        hr_df = pd.read_csv(os.path.join(self.root, 'data.csv'))
        self.dict = dict()
        for i, row in hr_df.iterrows():
            self.dict[row['project_name']] = row['label']
    
    def __len__(self):
        return len(self.bvp_list)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.bvp_path, self.bvp_list[index])
        bvp = np.cumsum(np.load(label_path))
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        
        map_name = self.bvp_list[index].replace('label', 'input').replace('npy', 'png')
        # print(map_name)
        # chrom_path, pos_path = os.path.join(self.chrom_path, map_name), os.path.join(self.pos_path, map_name)
        
        # print(chrom_path)
        map_list = []
        for _map in self.maps:
            if _map == "CHROM":
                chrom_path = os.path.join(self.chrom_path, map_name)
                map_list.append(cv.imread(chrom_path))
            elif _map == "POS":
                pos_path = os.path.join(self.pos_path, map_name)
                map_list.append(cv.imread(pos_path))
            elif _map == "YUV":
                yuv_path = os.path.join(self.yuv_path, map_name)
                map_list.append(cv.imread(yuv_path))
            elif _map == "RGB":
                rgb_path = os.path.join(self.rgb_path, map_name)
                map_list.append(cv.imread(rgb_path))
            elif _map == "NIR":
                nir_path = os.path.join(self.nir_path, map_name)
                map_list.append(cv.imread(nir_path))
            elif _map == "CHROM_POS_G":
                chrom_path = os.path.join(self.chrom_path, map_name)
                pos_path = os.path.join(self.pos_path, map_name)
                rgb_path = os.path.join(self.rgb_path, map_name)
                map_list.extend([cv.imread(chrom_path)[:, :, 2, np.newaxis], cv.imread(pos_path)[:, :, 2, np.newaxis], cv.imread(rgb_path)[:, :, 1, np.newaxis]])
                
        # feature_map1 = map_list[0]
        # feature_map2 = map_list[1]
        feature_map = np.concatenate(map_list, axis=2)
        # print(feature_map.shape)
        # print(bvp.shape)
        for c in range(feature_map.shape[2]):
            for r in range(feature_map.shape[0]):
                feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) \
                        / (0.00001 + np.max(feature_map[r, :,c]) - np.min(feature_map[r, :, c])))
        
        feature_map_list = []
        for i in range(0, feature_map.shape[2], 3):
            temp_feature_map = Image.fromarray(np.uint8(feature_map[:,:,i:i+3]))
            feature_map_list.append(temp_feature_map)
            
        # feature_map1 = Image.fromarray(np.uint8(feature_map[:,:,0:3]))
        # feature_map2 = Image.fromarray(np.uint8(feature_map[:,:,3:6]))

        if self.transform:
            for i, feature_map in enumerate(feature_map_list):
                feature_map_list[i] = self.transform(feature_map)
            # feature_map1 = self.transform(feature_map1)
            # feature_map2 = self.transform(feature_map2)
            feature_map = np.concatenate(feature_map_list, axis = 0)
        hr = torch.tensor(self.dict[os.path.basename(label_path)[:-4]]) 
        return feature_map, bvp, hr
    
if __name__ == '__main__':
    dataset = MSTmap_PURE_cut()
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    print(len(dataset))
    for img, bvp, hr in dataloader:
        print(img.shape)
        print(bvp.shape)
        exit()
    print(dataset[0][0].shape)
    print(dataset[0][2])
    exit()
    import cv2 as cv
    path = '/Users/ziyuanwang/Documents/DATASET/MSTMap/data.csv'
    dataset = driver_dataset(path)
    print(dataset[0][1].shape)
    print(dataset[0][2])
    exit(0)
    dataloader = DataLoader(dataset, batch_size = 1)
    with torch.no_grad():
        for pic, label, hr in dataloader:
            print(pic.shape)
            loss, BVP_map, mask = model(pic[:, :-1])
            # rgb = BVP
            print(mask.shape)
            BVP_map = BVP_map.squeeze()
            rgb, yuv = BVP_map[:3], BVP_map[3:] # prediction
            mask = mask.repeat(3, 1, 1)
            rgb_original = pic[0, :3]
            yuv_original = pic[0, 3:6]
            # BVP_map = BVP_map * mask + (1 - mask) * rgb_original
            cv.imshow('win1', np.vstack([rgb.numpy().transpose(2, 1, 0), rgb_original.numpy().transpose(2, 1, 0)]))
            cv.imshow('win', np.vstack([yuv.numpy().transpose(2, 1, 0), yuv_original.numpy().transpose(2, 1, 0)]))
            cv.waitKey(0)
            # print(pic.shape)
            # print(label.shape)
            # print(hr.shape)
            break
    # print(dataset.csv_file)
    # print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2])