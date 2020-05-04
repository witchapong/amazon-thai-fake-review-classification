import numpy as np
import pandas as pd
from pathlib import Path
import PIL, mimetypes, os
import matplotlib.pyplot as plt
from typing import *
import torch
from torch import nn
from fastai.vision import *
from fastai.callbacks.hooks import *

def show_random_img(path:Path):
    rand_idx = np.random.randint(len((path).ls()))
    img = PIL.Image.open((path).ls()[rand_idx])
    _ = plt.imshow(img)

def get_image_extensions(): return set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_img_dir_df(path: Path, extensions: Collection):
    img_dirs = [f for f in list(os.scandir(path)) if not f.name.startswith('.')
            and ((not extensions) or f'.{f.name.split(".")[-1].lower()}' in extensions)]
    return pd.DataFrame({'dir':[dir.path for dir in img_dirs],
                         'image_name':[dir.name.split('.')[0] for dir in img_dirs]})

def get_custom_head(class_num:int,fc_sizes:Collection=[1024,512],emb_size:int=300):
    '''
    fc_sizes[0] should equal to 2*output_dim of last layer of the main model (because of the AdaptiveConcatPool2d)
    '''
    return nn.Sequential(AdaptiveConcatPool2d(),
                            Flatten(),
                            nn.BatchNorm1d(fc_sizes[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.Dropout(p=0.25),
                            nn.Linear(in_features=fc_sizes[0], out_features=fc_sizes[1], bias=True),
                            nn.ReLU(True),
                            nn.BatchNorm1d(fc_sizes[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.Dropout(p=0.25),
                            nn.Linear(in_features=fc_sizes[1], out_features=emb_size, bias=True),
                            nn.Linear(in_features=emb_size, out_features=class_num, bias=True))

def get_bottle_out(model,input,layer):
    with hook_output(getattr(list(model.modules())[0],layer)) as hook_out: 
        _ = model(input)
    return hook_out.stored.cpu()

def get_image_embs(img_dir:Collection, image_transformer:object , model:nn.Module, layer:str, bs:int=64):
    
    # make sure model is in no gradient mode + is in device
    _ = model.eval()
    _ = model.to(model.device)
    
    # create dataset & dataloader
    sku_img_ds_all = SKUImageDS(img_dir,image_transformer)
    all_dl = DataLoader(sku_img_ds_all,batch_size=bs,shuffle=False)
    
    # extract image feature
    img_feat = []
    for xb, _ in tqdm(all_dl):
        xb = xb.to(device)
        img_feat.append(get_bottle_out(model,xb,layer))
        
    img_feat = torch.stack(img_feat).numpy()
    
    img_feat_df = pd.DataFrame(data=img_feat,index=img_dir)
    
    return img_feat_df


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_size, conv_stride, conv_padding, max_pool_size, max_pool_stride):
        super().__init__()
        self.conv2d_block = nn.Sequential(nn.Conv2d(in_ch, out_ch, conv_size, stride=conv_stride, padding=conv_padding),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(max_pool_size,stride=max_pool_stride))
        
    def forward(self, input): return self.conv2d_block(input)
        
def get_encoder(conv_nums, conv_sizes, conv_strides, conv_paddings, max_pool_sizes, max_pool_strides, in_channel=3):
    in_channels = [in_channel] + conv_nums[:-1]
    conv2d_block = [Conv2dBlock(in_ch, out_ch, conv_size, conv_stride, conv_padding, max_pool_size, max_pool_stride)
        for in_ch, out_ch, conv_size, conv_stride, conv_padding, max_pool_size, max_pool_stride 
        in zip(in_channels, conv_nums, conv_sizes, conv_strides, conv_paddings, max_pool_sizes, max_pool_strides)]

    # swap last block
    conv2d_block = conv2d_block[:-1] + [nn.Conv2d(in_channels[-1], conv_nums[-1], conv_sizes[-1], stride=conv_strides[-1], padding=conv_paddings[-1])]
    
    return nn.Sequential(*conv2d_block)

def get_conv2d_flatten_dim(input_size, conv_nums, conv_sizes, conv_strides, conv_paddings, max_pool_sizes,
    max_pool_strides, in_channel=3, return_intermediate=False):
    input_size = input_size
    out_sizes=[]
    
    i=0
    for conv_num, conv_size, conv_stride, conv_padding, max_pool_size, max_pool_stride in zip(conv_nums,
        conv_sizes, conv_strides, conv_paddings, max_pool_sizes, max_pool_strides):
        
        conv_out_size = (input_size + 2*conv_padding - (conv_size-1) - 1 + conv_stride) // conv_stride
        
        # condition for last conv layer with only linear block
        if i==len(conv_nums)-1:
            out_sizes.append((conv_num,conv_out_size,conv_out_size))
        else:
            max_pool_out_size = (conv_out_size - (max_pool_size-1) - 1 + max_pool_stride) // max_pool_stride
            input_size = max_pool_out_size
            out_sizes.append((conv_num,max_pool_out_size,max_pool_out_size))
        i+=1
        
    if not return_intermediate:
        return conv_nums[-1] * conv_out_size**2
    return conv_nums[-1] * conv_out_size**2, out_sizes

class TConv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_size, conv_stride, conv_padding, conv_out_padding):
        super().__init__()
        self.transposed_conv2d_block = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, conv_size,
                                                        stride=conv_stride, padding=conv_padding,
                                                        output_padding=conv_out_padding),
                                                     nn.BatchNorm2d(out_ch),
                                                     nn.ReLU(True))
        
    def forward(self, input): return self.transposed_conv2d_block(input)
        
def get_decoder(in_channels, out_channels, conv_sizes, conv_strides, conv_paddings, conv_out_paddings):
    transposed_conv2d_block = [TConv2dBlock(in_ch, out_ch, conv_size, conv_stride, conv_padding, conv_out_padding)
        for in_ch, out_ch, conv_size, conv_stride, conv_padding, conv_out_padding
        in zip(in_channels, out_channels, conv_sizes, conv_strides, conv_paddings, conv_out_paddings)]
    
    # swap last block
    transposed_conv2d_block = transposed_conv2d_block[:-1] + [nn.ConvTranspose2d(in_channels=in_channels[-1],
        out_channels=out_channels[-1], kernel_size=conv_sizes[-1], stride=conv_strides[-1],
        padding=conv_paddings[-1], output_padding=conv_out_paddings[-1])] + [nn.Sigmoid()]
    return nn.Sequential(*transposed_conv2d_block)

class ConvAE(nn.Module):
    def __init__(self,input_size=224,in_channel=3,conv_nums=[8,16,32],conv_sizes=[3,3,3],conv_strides=[3,2,1],
        conv_paddings=[2,1,1],
        max_pool_sizes=[2,2,2],max_pool_strides=[2,2,2],
        bottle_dim=512,
        dconv_strides=[2,4,4], dconv_paddings=[2,1,2], dconv_out_paddings=[0,0,1]):
        
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = get_encoder(conv_nums=conv_nums, conv_sizes=conv_sizes, conv_strides=conv_strides,
            conv_paddings=conv_paddings, max_pool_sizes=max_pool_sizes, max_pool_strides=max_pool_strides)
        
        self.flatten = nn.Flatten()
        
        conv_flatten_sz = get_conv2d_flatten_dim(input_size=input_size, conv_nums=conv_nums,
            conv_sizes=conv_sizes, conv_strides=conv_strides, conv_paddings=conv_paddings,
            max_pool_sizes=max_pool_sizes, max_pool_strides=max_pool_strides, in_channel=in_channel)
        
        self.fc_e = nn.Linear(conv_flatten_sz,bottle_dim)
        
        self.fc_d = nn.Linear(bottle_dim,conv_flatten_sz)
        
        dconv_in_channels = conv_nums[::-1]
        dconv_out_channels = dconv_in_channels[1:] + [in_channel]
        
        self.decoder = get_decoder(in_channels=dconv_in_channels, out_channels=dconv_out_channels,
            conv_sizes=conv_sizes, conv_strides=dconv_strides, conv_paddings=dconv_paddings,
            conv_out_paddings=dconv_out_paddings)

    def forward(self, input):
        # thru encoder
        output = self.encoder(input)
        # record shape
        _, n_f, h, w = output.shape
        # flatten
        output = self.flatten(output)
        # pass through bottle neck
        output = torch.relu(self.fc_d(self.fc_e(output)))
        # reshape
        output = output.view(-1, n_f, h, w)
        output = self.decoder(output)
        return output