#%%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from dataset import *
from model import UNet
from util import *

# parser
parser = argparse.ArgumentParser(description="Test the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3,  type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./train", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

args = parser.parse_args()
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir =args.result_dir

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)

# make folder if doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

#%%
# network train
transform=transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

dataset_test=Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
# dataset_test=Dataset(data_dir='teeth_all/test', transform=transform)

loader_test=DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

# network generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

# loss function, optimizer
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)

# variables
num_data_test=len(dataset_test)

num_batch_test=np.ceil(num_data_test/batch_size)

# label size
# label_lst=os.listdir(os.path.join(data_dir, 'label'))
# label_path=os.path.join(data_dir, 'label', label_lst[0])
origin_size=(512, 512)

# functions
fn_tonumpy=lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class=lambda x:1.0*(x>0.5) # network output image->binary class로 분류

#%%
# test network
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad(): # no backward pass 
    net.eval()
    loss_arr=[]

    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        output=net(input)# torch.Size([4, 1, 512, 512])
        
        # loss function
        loss = fn_loss(output, label)
        loss_arr+=[loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                      (batch, num_batch_test, np.mean(loss_arr)))

        label=fn_tonumpy(label)
        input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))# (4, 512, 512, 3)
        output=fn_tonumpy(fn_class(output))# (4, 512, 512, 1)
        
        # save result
        for i in range(label.shape[0]):
            id = batch_size * (batch - 1) + i
            
            outputpath=os.path.join(result_dir, 'output_%04d.png' % id)
            # inputpath=os.path.join(result_dir, 'input_%04d.png' % id)
            # labelpath=os.path.join(result_dir, 'label_%04d.png' % id)
            
            outputimg=Image.fromarray(np.uint8(output[i].squeeze() *225))
            inputimg=Image.fromarray(np.uint8(input[i].squeeze()*255))
            
            compimg=Image.new('RGB',(1024, 512))
            compimg.paste(inputimg,(0,0))
            compimg.paste(outputimg,(512,0))
            compimg.save(outputpath)
            
            # inputimg.resize(origin_size).save(inputpath)
            # labelimg=Image.fromarray(np.uint8(label[i].squeeze()*255))
            # labelimg.resize(origin_size).save(labelpath)
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %(batch, num_batch_test, np.mean(loss_arr)))
