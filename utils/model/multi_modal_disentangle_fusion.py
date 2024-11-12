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
    def __init__(self, video_length = 224, num_channel = 6):
        super().__init__()
        self.extractor = resnet18(pretrained=False, num_classes=1, num_output=345)
        # self.extractor.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.extractor.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        hr_feat, feat_out, feat = self.extractor(x) # feat_out: output of encoder(latent variable)
        return hr_feat, feat_out, feat

class multi_modal_mutlti_task_disentangle(nn.Module):
    def __init__(self, video_length = 224, num_channel = [6, 3], decov_num=1, fusion="add"):
        super().__init__()
        self.video_length = video_length
        self.modal1_channel = num_channel[0]
        self.modal2_channel = num_channel[1]
        self.extractor1 = multi_modal_extractor(video_length, self.modal1_channel)
        self.extractor2 = multi_modal_extractor(video_length, self.modal2_channel)
        self.fusion = fusion
        if fusion == "add":
            self.hr_predictor = nn.Linear(512, 1)
        elif fusion == "concat":
            self.hr_predictor = nn.Linear(1024, 1)
        # ecg extractor
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 10))
        if self.fusion == 'add':
            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[1, 3], stride=[1, 3],
                                padding=[0, 0]),  # [1, 128, 32]
                nn.BatchNorm2d(64),
                nn.ELU(),
            )
        elif self.fusion == 'concat':            
            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=[1, 3], stride=[1, 3],
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
        self.Noise_encoder2.conv1 = nn.Conv2d(self.modal2_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder2 = Generator(conv_dim=128, repeat_num=decov_num, num_channel= self.modal2_channel)
    
    def forward(self, img1, img2):
        hr1_feat, feat_hr1, ecg_feat1 = self.extractor1(img1) # modal1
        # features for heart rate predictions, heart rate features and ecg features
        hr2_feat, feat_hr2, ecg_feat2 = self.extractor2(img2) # modal2
        if self.fusion == 'add':
            hr = self.hr_predictor(hr1_feat + hr2_feat)
            feat = ecg_feat1 + ecg_feat2
        elif self.fusion == 'concat':
            temp_hr_feat = torch.cat([hr1_feat, hr2_feat], dim = 1)
            hr = self.hr_predictor(temp_hr_feat)
            feat = torch.cat([ecg_feat1, ecg_feat2], dim = 1)
        # feat_out: 128, 28, 28 
        # feat: 256, 14, 14 
        x = self.feature_pool(feat)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ecg_conv(x)

        ecg = x.view(-1, int(self.video_length))
        
        # encode noise
        feat_n1, feat_n2 = self.Noise_encoder1(img1), self.Noise_encoder2(img2)
        
        feat1, feat2 = feat_hr1 + feat_n1, feat_hr2 + feat_n2
        recon_img1, recon_img2 = self.decoder1(feat1), self.decoder2(feat2)
        
        return hr, ecg, \
                feat_hr1, feat_hr2, \
                feat_n1, feat_n2, \
                recon_img1, recon_img2
        #  hr features for modal1, hr features for modal2
        #  noise features for modal1, noise features for modal2
        #  reconstructed imgs for modal1 and modal2
        
class multi_modal_cross(nn.Module):
    def __init__(self, video_length = 224, num_channel=[6, 3], fusion='add') -> None:
        super().__init__()
        self.num_modal1 = num_channel[0]
        self.num_modal2 = num_channel[1]
        self.multi_modal_encoder_decoder = multi_modal_mutlti_task_disentangle(video_length, num_channel, fusion=fusion)
        
    def forward(self, img):
        batch_size = img.size(0)
        img1, img2 = img[:, :self.num_modal1], img[:, self.num_modal1:]
        
        idx1 = torch.randint(batch_size, (batch_size,))
        idx2 = torch.randint(batch_size, (batch_size,))

        idx1 = idx1.long()
        idx2 = idx2.long()
        
        hr, ecg, feat_hr_modal1, feat_hr_modal2, feat_n_modal1, feat_n_modal2, recon_img_modal1, recon_img_modal2 \
                                                = self.multi_modal_encoder_decoder(img1, img2)
                                                
        
        # --------------- modal 1 ---------------
        feat_hr_modal1_group1, feat_hr_modal1_group2 = feat_hr_modal1[idx1], feat_hr_modal1[idx2]
        feat_n_modal1_group1, feat_n_modal1_group2 = feat_n_modal1[idx1], feat_n_modal1[idx2]
        # cross addition
        fake_feat1_modal1 = feat_hr_modal1_group1 + feat_n_modal1_group2
        fake_feat2_modal1 = feat_hr_modal1_group2 + feat_n_modal1_group1
        
        # decode features
        imgf1_modal1 = self.multi_modal_encoder_decoder.decoder1(fake_feat1_modal1) # fake pictures
        imgf2_modal1 = self.multi_modal_encoder_decoder.decoder1(fake_feat2_modal1) # fake pictures
        
        # encode fake imgs
        
        # --------------- modal 2 ---------------
        feat_hr_modal2_group1, feat_hr_modal2_group2 = feat_hr_modal2[idx1], feat_hr_modal2[idx2]
        feat_n_modal2_group1, feat_n_modal2_group2 = feat_n_modal2[idx1], feat_n_modal2[idx2]
        # cross addition
        fake_feat1_modal2 = feat_hr_modal2_group1 + feat_n_modal2_group2
        fake_feat2_modal2 = feat_hr_modal2_group2 + feat_n_modal2_group1
        
        # decode features
        imgf1_modal2 = self.multi_modal_encoder_decoder.decoder2(fake_feat1_modal2) # fake feature1
        imgf2_modal2 = self.multi_modal_encoder_decoder.decoder2(fake_feat2_modal2) # fake feature2
        
        # encode and decode fake maps
        fake_hr_1, fake_ecg, \
        ffeat_hr_modal1_group1, ffeat_hr_modal2_group1, \
        ffeat_n_modal1_group2, ffeat_n_modal2_group2, \
        fake_img1_group1, fake_img2_group1 = self.multi_modal_encoder_decoder(imgf1_modal1, imgf1_modal2)
        
        fake_hr_2, fake_ecg, \
        ffeat_hr_modal1_group2, ffeat_hr_modal2_group2, \
        ffeat_n_modal1_group1, ffeat_n_modal2_group1, \
        fake_img1_group2, fake_img2_group2 = self.multi_modal_encoder_decoder(imgf2_modal1, imgf2_modal2)
        
        hr_list = [hr, fake_hr_1, fake_hr_2] # true hr, fake_hr_1, fake_hr_2
        
        feat_hr_modal_list = [
            feat_hr_modal1[idx1], feat_hr_modal1[idx2],
            feat_hr_modal2[idx1], feat_hr_modal2[idx2],
            ffeat_hr_modal1_group1,ffeat_hr_modal1_group2,
            ffeat_hr_modal2_group1, ffeat_hr_modal2_group2,

        ]
        
        feat_n_modal_list = [
            feat_n_modal1[idx1], feat_n_modal1[idx2],
            feat_n_modal2[idx1], feat_n_modal2[idx2],
            ffeat_n_modal1_group1, ffeat_n_modal1_group2,
            ffeat_n_modal2_group1, ffeat_n_modal2_group2
        ]
        
        return hr_list, ecg, feat_hr_modal_list, feat_n_modal_list, recon_img_modal1, recon_img_modal2, idx1, idx2

                                    
                            
                    

if __name__ == '__main__':
    test_tensor = torch.rand([1, ])
    from torchinfo import summary
    
    # net = HR_disentangle()
    c = 11
    test_tensor = torch.randn([2, c, 224, 224])
    net = multi_modal_extractor(224, 8)
    net = multi_modal_cross()
    print(net(test_tensor)[1].shape)
    # summary(net, input_size=[1, c, 224, 224])
    # torch.load