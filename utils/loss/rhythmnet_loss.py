import torch.nn as nn
import numpy as np
import torch


class MyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        output = torch.zeros(1).to("cuda:0")

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t

        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1/ctx.T)*torch.sign(ctx.hr_mean - hr)

        output = (1/ctx.T - 1)*torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None

class mySmooth_loss(nn.Module):
    def __init__(self):
        super(mySmooth_loss, self).__init__()
    
    def forward(self, outputs):
        outputs_mean = outputs.mean(1, keepdim = True)
        loss = torch.abs(outputs - outputs_mean).mean()
        return loss

class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        # self.custom_loss = MyLoss()
        self.smooth_loss_fn = mySmooth_loss()
        self.device = 'cuda:0'

    def forward(self, resnet_outputs, gru_outputs, target):
        frame_rate = 30
        time = gru_outputs.shape[1]
        # resnet_outputs, gru_outputs, _ = outputs
        # target_array = target.repeat(1, resnet_outputs.shape[1])
        l1_loss = self.l1_loss(resnet_outputs, target)
        # print(gru_outputs.shape)
        # print(target.shape)
        # print(target.reshape(-1, 1).repeat(1, time).shape)
        # print(target.shape)
        smooth_loss_component = self.smooth_loss_fn(gru_outputs)
        # smooth_loss_component = self.smooth_loss(gru_outputs)

        loss = l1_loss + self.lambd*smooth_loss_component
        print(loss)
        return l1_loss, self.lambd*smooth_loss_component

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        # hr_mean = self.gru_outputs_considered.mean()
        for hr_t in self.gru_outputs_considered:
            # custom_fn = MyLoss.apply
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]
