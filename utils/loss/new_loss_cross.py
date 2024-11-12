# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable, Function
# import os
# import shutil
# import numpy as np
# import scipy.io as sio
# from scipy.stats import norm

# class Cross_loss(nn.Module):
#     def __init__(self, lambda_cross_fhr = 0.000005, lambda_cross_fn = 0.000005, lambda_cross_hr = 1):
#         super(Cross_loss, self).__init__()

#         self.lossfunc_HR = nn.L1Loss();
#         self.lossfunc_feat = nn.L1Loss();

#         self.lambda_fhr = lambda_cross_fhr;
#         self.lambda_fn = lambda_cross_fn;
#         self.lambda_hr = lambda_cross_hr;

#     def forward(self, hr_list, feat_hr_list, feat_n_list, idx1, idx2):
#         loss_hr1 = self.lossfunc_HR(hr_list[1], hr_list[0][idx1, :])
#         loss_hr2 = self.lossfunc_HR(hr_list[2], hr_list[0][idx2, :])
        
#         # feat_hr_list:
#         # 0, features for group 1, modal 1
#         # 1, features for group 2, modal 1
#         # 2, features for group 1, modal 2
#         # 3, features for group 2, modal 2
        
#         # 4, fake features for group 1, modal 1
#         # 5, fake features for group 2, modal 1
#         # 6, fake features for group 1, modal 2
#         # 7, fake features for group 2, modal 2
        
#         # heart rate features
#         loss_fhr1_modal1 = self.lossfunc_feat(feat_hr_list[0], feat_hr_list[4])
#         loss_fhr2_modal1 = self.lossfunc_feat(feat_hr_list[1], feat_hr_list[5])
#         loss_fhr1_modal2 = self.lossfunc_feat(feat_hr_list[2], feat_hr_list[6])
#         loss_fhr2_modal2 = self.lossfunc_feat(feat_hr_list[3], feat_hr_list[7])
        
#         # noise features
#         loss_fn1_modal1 = self.lossfunc_feat(feat_n_list[0], feat_n_list[4])
#         loss_fn2_modal1 = self.lossfunc_feat(feat_n_list[1], feat_n_list[5])
#         loss_fn1_modal2 = self.lossfunc_feat(feat_n_list[2], feat_n_list[6])
#         loss_fn2_modal2 = self.lossfunc_feat(feat_n_list[3], feat_n_list[7])
        
#         loss = self.lambda_hr * (loss_hr1 + loss_hr2) / 2 \
#             + self.lambda_fhr * (loss_fhr1_modal1 + loss_fhr2_modal1 + loss_fhr1_modal2 + loss_fhr2_modal2) / 4 \
#                 + self.lambda_fn * (loss_fn1_modal1 + loss_fn2_modal1 + loss_fn1_modal2 + loss_fn2_modal2) / 4
#         return loss # * 10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import os
import shutil
import numpy as np
import scipy.io as sio
from scipy.stats import norm

class Cross_loss(nn.Module):
    def __init__(self, lambda_cross_fhr = 0.000005, lambda_cross_fn = 0.000005, lambda_cross_hr = 1):
        super(Cross_loss, self).__init__()

        self.lossfunc_HR = nn.L1Loss();
        self.lossfunc_feat = nn.L1Loss();

        self.lambda_fhr = lambda_cross_fhr;
        self.lambda_fn = lambda_cross_fn;
        self.lambda_hr = lambda_cross_hr;

    def forward(self, hr_list, feat_hr_list, feat_n_list, idx1, idx2):
        loss_hr1 = self.lossfunc_HR(hr_list[1], hr_list[0][idx1, :])
        loss_hr2 = self.lossfunc_HR(hr_list[2], hr_list[0][idx2, :])
        
        # feat_hr_list:
        # 0, features for group 1, modal 1
        # 1, features for group 2, modal 1
        
        # 4, fake features for group 1, modal 1
        # 5, fake features for group 2, modal 1
        
        # heart rate features
        loss_fhr1_modal1 = self.lossfunc_feat(feat_hr_list[0], feat_hr_list[2])
        loss_fhr2_modal1 = self.lossfunc_feat(feat_hr_list[1], feat_hr_list[3])
        
        # noise features
        loss_fn1_modal1 = self.lossfunc_feat(feat_n_list[0], feat_n_list[2])
        loss_fn2_modal1 = self.lossfunc_feat(feat_n_list[1], feat_n_list[3])

        
        loss = self.lambda_hr * (loss_hr1 + loss_hr2) / 2 \
            + self.lambda_fhr * (loss_fhr1_modal1 + loss_fhr2_modal1) / 2 \
                + self.lambda_fn * (loss_fn1_modal1 + loss_fn2_modal1) / 2
        return loss