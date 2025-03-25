from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
from PIL import Image
import numpy as np
import ast
import torch
import torchvision.transforms as transforms
from utils.dataset.sig_util import *
import cv2 as cv
from torch.utils.data._utils.collate import default_collate
import glob

class driver_dataset(Dataset):
    @staticmethod
    def str_to_list( s):
        return ast.literal_eval(s)
    
    @staticmethod
    def split_dataset(config):
        video_len = 900
        img_size = 224
        train_subject = config['train_subject']
        test_subject = config['test_subject']
        valid_subject = config['valid_subject']
        dataset_path = config['path']
        exclude_list = config['exclude_list']
        step = config['step']
        train_dict_list, test_dict_list, valid_dict_list = [], [], []
        dataframe = pd.read_csv(dataset_path)
        hr_column = [] # if the step changes
        for i, row in dataframe.iterrows():
            map_path, label_path , hr = row['map_path'], row['label_path'], row['heart_rate']
            hr = driver_dataset.str_to_list(hr)
            if len(hr) != (1 + (video_len - img_size) // step):
                if (i + 1) % 10 == 0:
                    print(f"Finish extract heart rate from {i + 1} samples")
                temp_hr = hr_extract(label_path, step = step)
                hr_column.append(temp_hr)
                hr = temp_hr
            subject_name = os.path.basename(map_path).split('_')[0]
            subject_full_name = os.path.basename(map_path).split('input')[0]
            print(subject_full_name)
            if subject_full_name in exclude_list:
                continue
            if subject_name in train_subject:
                train_dict_list.append({'map_path': map_path,
                                   'rppg_path': label_path,
                                   'hr': hr})
            elif subject_name in test_subject:
                test_dict_list.append({'map_path': map_path,
                                   'rppg_path': label_path,
                                   'hr': hr})
            else:
                valid_dict_list.append({'map_path': map_path,
                        'rppg_path': label_path,
                        'hr': hr})
        if hr_column:
            print(f"A new step is specified, update the csv file !")
            dataframe['heart_rate'] = hr_column
            dataframe.to_csv(dataset_path)
            
        train_dataset, test_dataset, valid_dataset = \
                driver_dataset(train_dict_list, step = step), driver_dataset(test_dict_list, step = step), driver_dataset(valid_dict_list, step = step)
        return train_dataset, test_dataset, valid_dataset
        
    
    def __init__(self, dict_list, img_size = 224, step = 10) -> None:
        super().__init__()
        self.step = step
        self.img_size = img_size
        self.data_dict_list = dict_list
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        self.three_chan = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.single_chan = transforms.Normalize(mean=[0.1307], std=[0.3081])
    
    def __len__(self):
        return len(self.data_dict_list)

    def __getitem__(self, index) : # N, c, T, ROI
        # 196, 900, 7
        pic_dict :dict = self.data_dict_list[index]
        hr = torch.round(torch.tensor(pic_dict['hr'])).type(torch.int)
        ppg = torch.from_numpy(np.cumsum(np.load(pic_dict['rppg_path'])))
        t = ppg.shape[0]
        # print(t)
        ppg = [ppg[i : i + self.img_size].unsqueeze(0) 
               for i in range(0, t - self.img_size, self.step)]
        ppg = torch.cat(ppg, dim=0)
        pic = np.uint8(np.load(pic_dict['map_path'])).transpose(1, 0, 2) # time first
        # print(pic.shape)
        
        map1_list = [self.three_chan(self.transforms(pic[i: i+self.img_size, :, :3])).unsqueeze(0)
                    for i in range(0, t - self.img_size, self.step)]
        
        map2_list = [self.three_chan(self.transforms(pic[i: i+self.img_size, :, 3:6])).unsqueeze(0)
                    for i in range(0, t - self.img_size, self.step)]
        
        map3_list = [self.single_chan(self.transforms(pic[i: i+self.img_size, :, 6:])).unsqueeze(0)
                    for i in range(0, t - self.img_size, self.step)]
        
        map1, map2, map3 = torch.cat(map1_list, dim=0), torch.cat(map2_list, dim=0), torch.cat(map3_list, dim =0)
        
        pic = torch.cat([map1, map2, map3], dim=1)
        return pic, ppg, hr

class MSTmap_dataset(Dataset):
    @staticmethod
    def customized_collate_fn(batch):
        batch = default_collate(batch)
        batch[0] = batch[0].view(-1, 6, 224, 224)
        batch[1] = batch[1].view(-1, 224)
        batch[2] = batch[2].view(-1)
        return batch
        
    
    @staticmethod
    def str_to_list( s):
        return ast.literal_eval(s)
    
    @staticmethod
    def split_dataset(config):
        video_len = 900
        img_size = 224
        train_subject = config['train_subject']
        test_subject = config['test_subject']
        valid_subject = config['valid_subject']
        dataset_path = config['path'] # label path
        step = config['step']
        train_bvp_list, test_bvp_list, valid_bvp_list = [], [], []
        
        print(dataset_path)
        bvp_list = [os.path.basename(i) for i in glob.glob(f"{dataset_path}/*label*.npy")]
        
        hr_column = [] # if the step changes
        for bvp_name in bvp_list:
            subject_name = os.path.basename(bvp_name).split('_')[0]
            if subject_name in train_subject:
                train_bvp_list.append(bvp_name)
            elif subject_name in test_subject:
                test_bvp_list.append(bvp_name)
            else:
                valid_bvp_list.append(bvp_name)
            
            
        train_dataset, test_dataset, valid_dataset = \
                MSTmap_dataset(train_bvp_list, label_path=dataset_path), MSTmap_dataset(test_bvp_list, label_path=dataset_path), MSTmap_dataset(valid_bvp_list, label_path=dataset_path)
        return train_dataset, test_dataset, valid_dataset
    
    def __init__(self, bvp_list = None, img_size = 224, label_path = None, pic_path = None):
        super().__init__()
        self.img_size = img_size
        self.bvp_path = label_path
        if pic_path is None:
            self.path = '/data/PreprocessedData/CHROM_POS'
        else:
            self.path = pic_path
        if bvp_list is None:
            self.bvp_list = os.listdir(self.bvp_path)
        else:
            self.bvp_list = bvp_list
        # f = open()
        self.chrom_path = os.path.join(self.path, "CHROM")
        self.pos_path = os.path.join(self.path, "POS")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,900)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        hr_df = pd.read_csv('/data/PreprocessedData/raw/hr.csv')
        self.dict = dict()
        for i, row in hr_df.iterrows():
            self.dict[row['label_path']] = row['heart_rate']
    
    def __len__(self):
        return len(self.bvp_list)
    
    def __getitem__(self, index):
        # print(self.chrom_path)
        label_path = os.path.join(self.bvp_path, self.bvp_list[index])
        bvp = np.cumsum(np.load(label_path))
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        subject_name = self.bvp_list[index].split('_')[0]
        map_name = self.bvp_list[index].replace('label', 'input').replace('npy', 'png')
        # print(map_name)
        chrom_path, pos_path = os.path.join(self.chrom_path, map_name), os.path.join(self.pos_path, map_name)
        # print(chrom_path)
        feature_map1 = cv.imread(chrom_path)
        feature_map2 = cv.imread(pos_path)
        feature_map = np.concatenate((feature_map1, feature_map2), axis=2)
        # print(feature_map.shape)
        # print(bvp.shape)
        for c in range(feature_map.shape[2]):
            for r in range(feature_map.shape[0]):
                feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) \
                        / (0.00001 + np.max(feature_map[r, :,c]) - np.min(feature_map[r, :, c])))
            feature_map1 = Image.fromarray(np.uint8(feature_map[:,:,0:3]))
            feature_map2 = Image.fromarray(np.uint8(feature_map[:,:,3:6]))
        timestep = feature_map.shape[1]
        num_slices = timestep // self.img_size  
        # for i in range(num_slices):
        if self.transform:
                feature_map1 = self.transform(feature_map1)
                feature_map2 = self.transform(feature_map2)
                feature_map = np.concatenate((feature_map1, feature_map2), axis = 0)
        hr = torch.tensor(self.str_to_list(self.dict[label_path]))
        feature_map = feature_map[:, :, :num_slices * self.img_size].reshape(-1, self.img_size, self.img_size, num_slices)
        feature_map = feature_map.transpose(3, 0, 1, 2)
        bvp = bvp[: num_slices * self.img_size].reshape(num_slices, self.img_size)
        return feature_map, bvp, hr
        
class MSTmap_dataset_cut(Dataset):
    @staticmethod
    def customized_collate_fn(batch):
        batch = default_collate(batch)
        batch[0] = batch[0].view(-1, 6, 224, 224)
        batch[1] = batch[1].view(-1, 224)
        batch[2] = batch[2].view(-1)
        return batch
    
    @staticmethod
    def rhy_collate_fn(batch):
        batch = default_collate(batch)
        bz = batch[0].shape[0]
        # print(batch[0].shape)
        batch[0] = batch[0].permute(0, 3, 1, 2).reshape(bz, 224, 3, 14, 14)
        batch[1] = batch[1].view(-1, 224)
        batch[2] = batch[2].view(-1)
        return batch
        
    
    @staticmethod
    def str_to_list( s):
        return ast.literal_eval(s)
    
    @staticmethod
    def split_dataset(config, img_size = (224, 224), pretrained=False):
        train_subject = config['train_subject']
        test_subject = config['test_subject']
        valid_subject = config['valid_subject']
        exclude_list = config['exclude_list']
        map_type = config['map_type']
        selected_topics = config['selected_topic']
        dataset_path = config['path'] # label path
        # dataset_path = '/data/PreprocessedData/MSTMap_new'
        csv_file = pd.read_csv(os.path.join(dataset_path, 'old_data.csv'))
        
        train_bvp_list, test_bvp_list, valid_bvp_list = [], [], []
        
        print(dataset_path)
        bvp_list = csv_file['project_name'].tolist()
        
        for bvp_name in bvp_list:
            subject_name = os.path.basename(bvp_name).split('_')[0]
            subject_full_name = os.path.basename(bvp_name).split('input')[0][:-1]
            full_name_split_list = subject_full_name.split('_')[1:]
            topic_name = "_".join(full_name_split_list)
            if len(selected_topics) != 0:
                if topic_name not in selected_topics:
                    # print("filtering", subject_full_name)
                    continue
            # print(subject_full_name)
            if subject_full_name in exclude_list:
                # print("filtering", subject_full_name)
                continue
            if subject_name in train_subject:
                train_bvp_list.append(bvp_name + ".npy")
            if subject_name in test_subject:
                test_bvp_list.append(bvp_name + ".npy")
            if subject_name in valid_subject:
                valid_bvp_list.append(bvp_name+ ".npy")
            
            
        train_dataset, test_dataset, valid_dataset = \
                MSTmap_dataset_cut(dataset_path, train_bvp_list, map_type, img_size, pretrained), MSTmap_dataset_cut(dataset_path, test_bvp_list, map_type, img_size, pretrained), MSTmap_dataset_cut(dataset_path, valid_bvp_list, map_type, img_size, pretrained)
        # exit()
        return train_dataset, test_dataset, valid_dataset
    
    def __init__(self, root, bvp_list = None, map_list=["CHROM", "POS"],  img_size = (224, 224), pretrained = False):
        super().__init__()
        print(f"Using map {map_list}")
        self.h, self.w = img_size
        # self.img_size = img_size
        self.root = root
        self.bvp_path = os.path.join(root, 'bvp')
        self.maps = map_list
        self.pretrained = pretrained
        self.maps_type = ['CHROM', 'POS', 'YUV', 'RGB', 'NIR', 'CHROM_POS_G']
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
        hr_df = pd.read_csv(os.path.join(self.root, 'old_data.csv'))
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
        maps = []
        if self.pretrained:
            idx = np.random.choice(np.arange(0, len(self.maps_type)), size=2, replace=False)
            maps = self.maps_type[idx]
        else:
            maps = self.maps
        map_name = self.bvp_list[index].replace('label', 'input').replace('npy', 'png')

        map_list = []
        for _map in maps:
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
        return feature_map, bvp, hr, self.bvp_list[index]
