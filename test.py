#%%
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import UNet
from util import load

warnings.filterwarnings(action='ignore')

#%%
# functions

def load_image_resize_withoutalpha(imfile, new_w, new_h):
    aaa=Image.open(imfile).convert('RGB').resize((new_w, new_h), Image.BILINEAR)
    img = np.array(aaa).astype(np.uint8)
    return img

fn_totensor=lambda x: torch.from_numpy(x.transpose((2,0,1)).astype(np.float32))
fn_tonumpy=lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_norm = lambda x, mean, std: (x-mean)/std
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class = lambda x: 1.0 * (x > 0.5)

#%%
lr = 1e-3
num_epoch = 100

########################### path ############################
test_file='test/input5-2_059.JPG'
ckpt_dir = 'RGB_batch4_lr1e-3_train/ckpt'
result_dir ='./testresult'
#############################################################

print("learning rate: %.4e" % lr)
print("number of epoch: %d" % num_epoch)
print("test file: %s" % test_file)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)

# make folder if doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
#%%
origin_size=Image.open(test_file).size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
# 255로 나누는 이유 : 이미지 값의 범위를 0~255에서 0~1 값의 범위를 갖도록 하기 위함
input=load_image_resize_withoutalpha(test_file, 512, 512)/255.0#(512, 512, 3), ndim=3
if input.ndim == 2:
    input = input[:, :, np.newaxis]

input=fn_totensor(fn_norm(input, 0.5, 0.5)).to(device)#torch.Size([3, 512, 512])
input=input.unsqueeze(0)#torch.Size([1, 3, 512, 512])

#%%
net = UNet().to(device)
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)

#%%
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad(): # no backward pass 
    net.eval()
    output=net(input)# torch.Size([1, 3, 512, 512])
    
    output=fn_tonumpy(fn_class(output))# (1, 512, 512, 1)
    input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))# (4, 512, 512, 3)
    
    name='output_'+test_file.split('_')[-1]
    nameee='comp_'+test_file.split('_')[-1]
    
    print(output.shape)
    print(input.shape)
    
    compout=np.zeros((output.shape[0], output.shape[1], output.shape[2], 3))
    for each_channel in range(3):
        compout[:,:,:,each_channel] = output[:,:,:,0]
    comparray=np.hstack((compout,input))
    # plt.imshow(comparray[:,:,::-1]) 
    # plt.show()

    comp=Image.fromarray(np.uint8(comparray.squeeze()*225))
    comp.resize((origin_size[0], 2*origin_size[1])).save(os.path.join(result_dir, nameee))
    
    output=Image.fromarray(np.uint8(output.squeeze()*225))
    output.resize(origin_size).save(os.path.join(result_dir, name))
# %%
