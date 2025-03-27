# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import os

import timm.models.vision_transformer
from timm.models.vision_transformer import Block
import torch.nn.functional as F

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # self.cross_block = cross_Block(dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'],
        #  mlp_ratio=kwargs['mlp_ratio'], qkv_bias=True, qk_scale=None, norm_layer=kwargs['norm_layer'])
        self.global_pool = global_pool
        self.hr = nn.Linear(self.embed_dim, 1)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        ppg = F.relu(self.head(x))
        hr = self.hr(x)
        return ppg, hr

class VisionTransformer_double(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, embed_dim =768, norm_layer = partial(nn.LayerNorm, eps=1e-6), fusion = 'add'):
        super().__init__()
        assert fusion in ['add', 'multiply']
        self.fusion = fusion
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        d = {
            'patch_size': 16, 'embed_dim': self.embed_dim,
            'depth': 12, 'num_heads': 12, 'mlp_ratio': 4,
            'qkv_bias': True, 'in_chans': 6,
            'norm_layer': norm_layer, 'qkv_bias': True
        }
        self.encoder1 = VisionTransformer(self.global_pool, **d) # for CPG + YUV
        d['in_chans'] = 3
        self.encoder2 = VisionTransformer(self.global_pool, **d) # for CPG + YUV # for NIR
        del self.encoder1.hr, self.encoder1.head
        del self.encoder2.hr, self.encoder2.head
        
        # self.cross_block = cross_Block(dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'],
        #  mlp_ratio=kwargs['mlp_ratio'], qkv_bias=True, qk_scale=None, norm_layer=kwargs['norm_layer'])
        self.global_pool = global_pool
        self.hr = nn.Linear(self.embed_dim, 1)
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.encoder1.norm, self.encoder2.norm  # remove the original norm
        self.head = nn.Linear(embed_dim, 224)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def forward(self, x, y):
        x = self.encoder1.forward_features(x)
        y = self.encoder2.forward_features(y)
        if self.fusion == 'add':
            feat = x + y
        else:
            feat = x * y
        ppg = self.head(feat)
        hr = self.hr(feat)
        return ppg, hr


def vit_base_patch16(pretrain = False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    # pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if pretrain:
        pretrained_dict = torch.load(os.path.join(os.getcwd(), 'ckpt', 'jx_vit_base.pth'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("Load pretrained-ckpt from ImageNet")
    return model



def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# test code
# from torchinfo import summary
# # model = vit_base_patch16(in_chans=6, num_classes = 224)
# model = VisionTransformer_double()
# tensor = torch.randn((1,6, 224, 224))
# tensor_nir = torch.randn((1,3, 224, 224))
# ppg, hr = model(tensor, tensor_nir)
# # summary(model, input_size=(1, 6, 224, 224))
# print(ppg.shape)