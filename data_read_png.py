#%%
import argparse
import os
import random
import shutil
from importlib.resources import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image

# define function
def load_image_resize_withoutalpha(imfile, new_w, new_h):
    # L: 8bit grayscale
	# L: model에서 맨 처음 in_channels 1
	# RGB: model에서 맨 처음 in_channels 3
	# model에서 맨 마지막 out_channels : 어디에 속할지 판단하는것(분류 종류?)
	# out_channels는 label과 채널 수가 맞아야 함
  
    # aaa=Image.open(imfile).convert('L').resize((new_w, new_h), Image.BILINEAR)
    aaa=Image.open(imfile).convert('RGB').resize((new_w, new_h))#, Image.Resampling.BILINEAR)
    img = np.array(aaa).astype(np.uint8) 
    # print(img.shape)
    return img

def load_image_resize_binary(imfile, new_w, new_h):    
    aaa=Image.open(imfile).resize((new_w, new_h))#, Image.Resampling.BILINEAR)
    img = np.array(aaa)[:,:,3].astype(np.uint8)
    return img

# set parser
parser = argparse.ArgumentParser(description="read png",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_dir", type=str)
parser.add_argument("--label_dir",type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--mode", default=1, type=int)
# 1. train/val/test
# 2. train/val

args = parser.parse_args()
dir_label=args.label_dir
dir_input=args.input_dir
lst_label=sorted(os.listdir(dir_label))
lst_input=sorted(os.listdir(dir_input))
img_cnts = len(os.listdir(dir_label))

dir_save_train=os.path.join(args.output_dir, 'train')
dir_save_val=os.path.join(args.output_dir, 'val')
dir_save_test=os.path.join(args.output_dir, 'test')

if os.path.exists(dir_save_train):
    shutil.rmtree(dir_save_train)

if os.path.exists(dir_save_val):
    shutil.rmtree(dir_save_val)

if os.path.exists(dir_save_test):
    shutil.rmtree(dir_save_test)

os.makedirs(dir_save_train)
os.makedirs(dir_save_val)
os.makedirs(dir_save_test)
# set train/val/test set
if args.mode==1:
    
    nframe_train=int(img_cnts*0.8)
    nframe_val=int((img_cnts-nframe_train)/2)
    nframe_test=img_cnts-nframe_train-nframe_val
elif args.mode==2:
    nframe_train=int(img_cnts*0.9)
    nframe_val=int((img_cnts-nframe_train))
    nframe_test=0
    
    # make val/testset (random)
lst_random=random.sample(lst_input,nframe_val+nframe_test)
lst_val=[]
lst_test=[]

for i in range(nframe_val+nframe_test):
    name=lst_random[i]
    num=name.strip('input''.jpg')
    if i<nframe_val:
        lst_val.append(num)
    else: lst_test.append(num)
del lst_random

    # make numpy file and save
for i in range(img_cnts):
    input=lst_input[i]
    label=lst_label[i]
    img_input=load_image_resize_withoutalpha(os.path.join(dir_input, input),512,512)
    img_label=load_image_resize_binary(os.path.join(dir_label, label),512,512)
        
    if input.strip('input''.jpg') in lst_val:
        np.save(os.path.join(dir_save_val,'label_%03d.npy' % i),img_label)
        np.save(os.path.join(dir_save_val,'input_%03d.npy' % i),img_input)
        label_a=load_image_resize_withoutalpha(os.path.join(dir_label, label),512,512)
       
    elif input.strip('input''.jpg') in lst_test:
        np.save(os.path.join(dir_save_test,'label_%03d.npy' % i),img_label)
        np.save(os.path.join(dir_save_test,'input_%03d.npy' % i),img_input)
    else:
        np.save(os.path.join(dir_save_train,'label_%03d.npy' % i),img_label)
        np.save(os.path.join(dir_save_train,'input_%03d.npy' % i),img_input)
