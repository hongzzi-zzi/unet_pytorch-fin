#%%
import os
from importlib.resources import path
import shutil
import random

dir_input='/home/h/Desktop/dataaaaa/input_all'
dir_label='/home/h/Desktop/dataaaaa/label_all'

lst_input=sorted(os.listdir(dir_input))
lst_label=sorted(os.listdir(dir_label))

img_cnts = len(os.listdir(dir_label))
#%% 1개씩만 빼고 총 8개의 testset

# lst_dir=['except_2','except_3',  'except_4', 'except_4-1', 'except_4-2', 'except_5-1', 'except_5-2']
# # lst_dir=['except_capture','except_face2','except_face3',  'except_4', 'except_4-1', 'except_4-2', 'except_5-1', 'except_5-2']


# for i in range(len(lst_dir)):
#     inputpath=os.path.join(lst_dir[i],'input')
#     labelpath=os.path.join(lst_dir[i],'label')
    
#     print(inputpath)
#     shutil.copytree(dir_input, inputpath)
#     shutil.copytree(dir_label, labelpath)
    
#     lst_inputdir=sorted(os.listdir(inputpath))
#     lst_labeldir=sorted(os.listdir(labelpath))
    
#     a='input'+lst_dir[i].split('_')[-1]
#     print(a)
#     for j in range(len(lst_inputdir)):
#         if a in lst_inputdir[j]:
#             os.remove(os.path.join(inputpath, lst_inputdir[j]))
#             os.remove(os.path.join(labelpath, lst_labeldir[j]))
            

# %% 전체에서 몇 개씩
l1=[]
l2=[]
l3=[]
l4=[]
l4_1=[]
l4_2=[]
l5_1=[]
l5_2=[]

lst=[]
for i in range(len(lst_input)):
    num=lst_input[i].strip('input''.JPG')
    
    if num.startswith('1'):
        l1.append(num)
    elif num.startswith('2'):
        l2.append(num)
    elif num.startswith('3'):
        l3.append(num)
    elif num.startswith('4_'):
        l4.append(num)
    elif num.startswith('4-1'):
        l4_1.append(num)
    elif num.startswith('4-2'):
        l4_2.append(num)
    elif num.startswith('5-1'):
        l5_1.append(num)
    elif num.startswith('5-2'):
        l5_2.append(num)
#%%
l1=random.sample(l1,30)
l2=random.sample(l2,30)
l3=random.sample(l3,30)
l4=random.sample(l4,30)
l4_1=random.sample(l4_1,30)
l4_2=random.sample(l4_2,30)
l5_1=random.sample(l5_1,30)
l5_2=random.sample(l5_2,30)

trainset=l1+l2+l3+l4+l4_1+l4_2+l5_1+l5_2

settt=set(trainset)
del l1, l2, l3, l4, l4_1, l4_2, l5_1, l5_2
print(len(trainset))
print(len(settt))

#%%
if os.path.exists('/home/h/UNet_jupyter/train30/input'):
        shutil.rmtree('/home/h/UNet_jupyter/train30/input')

if os.path.exists("/home/h/UNet_jupyter/train30/label"):
        shutil.rmtree('/home/h/UNet_jupyter/train30/label')
        
os.makedirs('/home/h/UNet_jupyter/train30/input')
os.makedirs('/home/h/UNet_jupyter/train30/label')
for num in trainset:
    input='input'+num+'.JPG'
    label='label'+num+'.png'
    
    from_input=os.path.join(dir_input,input)
    from_label=os.path.join(dir_label,label)
    
    to_input=os.path.join('/home/h/UNet_jupyter/train30/input', input)
    to_label=os.path.join('/home/h/UNet_jupyter/train30/label', label)
    shutil.copyfile(from_input,to_input)
    shutil.copyfile(from_label,to_label)
# %%
