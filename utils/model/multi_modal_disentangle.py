import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import shutil
import numpy as np
import scipy.io as sio

# sys.path.append('..')

from .resnet import resnet18, resnet18_part
import time

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=2, num_channel = 6, up_time = 3):
        super(Generator, self).__init__()

        curr_dim = conv_dim

        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(up_time):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding = 1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, num_channel, kernel_size=7, stride=1, padding=3, bias=False))

        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        x = self.img_reg(features)

        return x


class multi_modal_extractor(nn.Module):
    def __init__(self, video_length = 224, num_channel = 6, num_output=34):
        super().__init__()
        self.extractor = resnet18(pretrained=False, num_classes=1, num_output=num_output)
        # self.extractor.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.extractor.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.extractor(x)
        hr_feat, feat_out, feat = self.extractor(x) # feat_out: output of encoder(latent variable)
        return hr_feat, feat_out, feat

class multi_modal_mutlti_task_disentangle(nn.Module):
    def __init__(self, video_length = 224, num_channel = [6, 3], decov_num=1, fusion="add"):
        super().__init__()
        self.video_length = video_length
        self.modal1_channel = num_channel[0]
        self.modal2_channel = num_channel[1]
        self.extractor1 = multi_modal_extractor(video_length, self.modal1_channel)
        # ecg extractor
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 10))
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[1, 3], stride=[1, 3],
                            padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=[1, 5], stride=[1, 5],
                               padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        self.video_length = video_length
        self.poolspa = nn.AdaptiveAvgPool2d((1, int(self.video_length)))
        self.ecg_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
        self.Noise_encoder1 = resnet18_part()
        self.Noise_encoder1.conv1 = nn.Conv2d(self.modal1_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder1 = Generator(conv_dim=128, repeat_num=decov_num, num_channel= self.modal1_channel)
        
        self.Noise_encoder2 = resnet18_part()
        self.Noise_encoder2.conv1 = nn.Conv2d(self.modal1_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder2 = Generator(conv_dim=128, repeat_num=decov_num, num_channel= self.modal2_channel)
    
    def forward(self, img):
        hr, rgb_feat, ecg_feat = self.extractor1(img)
        # heart rate predictions, heart rate features and ecg features
        
        # nir_feat = self.extractor2(img) # modal2

        # feat_out: 128, 28, 28 
        # feat: 256, 14, 14 
        x = self.feature_pool(ecg_feat)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ecg_conv(x)

        ecg = x.view(-1, int(self.video_length))
        
        # encode noise
        rgb_n = self.Noise_encoder1(img)
        # nir_n = self.Noise_encoder2(img)
        # print(nir_n.shape)
        # print(nir_feat.shape)
        rgb_total_feat = rgb_feat + rgb_n
        # nir_total_feat = nir_feat + nir_n
        recon_rgb, recon_nir = self.decoder1(rgb_total_feat), self.decoder2(rgb_total_feat)
        
        return hr, ecg, \
                rgb_feat, \
                rgb_n, \
                recon_rgb, recon_nir
        #  hr features for modal1, hr features for modal2
        #  noise features for modal1, noise features for modal2
        #  reconstructed imgs for modal1 and modal2

# ONE ENCODER, TWO DECODER
   
class double_disentangle_cross(nn.Module):
    def __init__(self, video_length = 224, num_channel=[6, 3], fusion='add') -> None:
        super().__init__()
        self.num_modal1 = num_channel[0]
        self.num_modal2 = num_channel[1]
        self.test = False
        self.multi_modal_encoder_decoder = multi_modal_mutlti_task_disentangle(video_length, num_channel, fusion=fusion)
        
    def forward(self, img):
        batch_size = img.size(0)
        hr_list, feat_hr_modal_list, feat_n_modal_list = [], [], []
        
        idx1 = torch.randint(batch_size, (batch_size,))
        idx2 = torch.randint(batch_size, (batch_size,))

        idx1 = idx1.long()
        idx2 = idx2.long()
        
        hr, ecg, rgb_feat, rgb_n, recon_rgb, recon_nir \
                                                = self.multi_modal_encoder_decoder(img)
        if self.test:
            return hr_list, ecg, feat_hr_modal_list, feat_n_modal_list, recon_rgb, recon_nir, idx1, idx2                     
        
        # --------------- RGB ---------------
        rgb_feat_group1, rgb_feat_group2 = rgb_feat[idx1], rgb_feat[idx2]
        rgb_n_group1, rgb_n_group2 = rgb_n[idx1], rgb_n[idx2]
        # cross addition
        fake_feat1_modal1 = rgb_feat_group1 + rgb_n_group2
        fake_feat2_modal1 = rgb_feat_group2 + rgb_n_group1
        
        # decode features
        rgb_imgf1 = self.multi_modal_encoder_decoder.decoder1(fake_feat1_modal1) # fake pictures
        rgb_imgf2 = self.multi_modal_encoder_decoder.decoder1(fake_feat2_modal1) # fake pictures
        
        # encode fake imgs
        
        # --------------- NIR ---------------
        # nir_feat_group1, nir_feat_group2 = nir_feat[idx1], nir_feat[idx2]
        # nir_n_group1, nir_n_group2 = nir_n[idx1], nir_n[idx2]
        
        # cross addition
        # fake_feat1_modal2 = nir_feat_group1 + nir_n_group2
        # fake_feat2_modal2 = nir_feat_group2 + nir_n_group1
        
        # # decode features
        # nir_imgf1 = self.multi_modal_encoder_decoder.decoder2(fake_feat1_modal2) # fake nir
        # nir_imgf2 = self.multi_modal_encoder_decoder.decoder2(fake_feat2_modal2) # fake nir
        
        # encode and decode fake maps
        fake_hr_1, fake_ecg, \
        f_rgb_feat_group1, \
        f_rgb_n_group2, \
        fake_img1_group1, fake_img2_group1 = self.multi_modal_encoder_decoder(rgb_imgf1)
        
        fake_hr_2, fake_ecg, \
        f_rgb_feat_group2, \
        f_rgb_n_group1, \
        fake_img1_group2, fake_img2_group2 = self.multi_modal_encoder_decoder(rgb_imgf2)
        
        hr_list = [hr, fake_hr_1, fake_hr_2] # true hr, fake_hr_1, fake_hr_2
        
        feat_hr_modal_list = [
            rgb_feat[idx1], rgb_feat[idx2],
            f_rgb_feat_group1, f_rgb_feat_group2,         
        ]
        
        feat_n_modal_list = [
            rgb_n[idx1],rgb_n[idx2],
            f_rgb_n_group1, f_rgb_n_group2, # feature map
        ]
        
        return hr_list, ecg, feat_hr_modal_list, feat_n_modal_list, recon_rgb, recon_nir, idx1, idx2

                                    
                        