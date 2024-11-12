import torch.nn as nn

class Img_loss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, lambda_img, channel_list = [6, 3]):
        super(Img_loss, self).__init__()
        self.lambda_img = lambda_img / 2;
        self.channel_modal1 = channel_list[0]
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.L1Loss()
    
    def forward(self, img1, img2, org_img):
        org_img1, org_img2 = org_img[:, :self.channel_modal1], org_img[:, self.channel_modal1:]
        loss1 = self.loss1(org_img1, img1) * self.lambda_img
        loss2 = self.loss2(org_img2, img2) * self.lambda_img
        return loss1 + loss2
    

class Double_Img_loss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, lambda_img, channel_list = [6, 3]):
        super(Double_Img_loss, self).__init__()
        self.lambda_img = lambda_img / 2;
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.L1Loss()
    
    def forward(self, img1, img2, org_img1, org_img2):
        loss1 = self.loss1(org_img1, img1) * self.lambda_img
        loss2 = self.loss2(org_img2, img2) * self.lambda_img
        return loss1, loss2