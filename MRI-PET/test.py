import os,sys
sys.path.append('..')
import time
import numpy as np
import torch
from utils import L1_Norm, LEM, LEGM
from net import Net
import cv2
import argparse
flag = 'LEGM'


def hyper_parametrs():
    parse = argparse.ArgumentParser(description='Created by liam Zhang')
    parse.add_argument('--model_path',type=str,default='',\
                       help='the pretrained model path')
    parse.add_argument('--mri_file',type=str,default=r'',\
                       help='the test mri dataset path')
    parse.add_argument('--pet_file',type=str,default=r'', \
                       help='the test pet dataset path')
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = hyper_parametrs()
    print('==>Loading the model')
    model = Net()
    if not os.path.exists(args.model_path):
        raise Exception('No pretrained model could be found!')
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    print('==>Loading the test datasets')
    mri_file = args.mri_file
    pet_file= args.pet_file
    mri=[]
    pet=[]
    pet_tensor=[]
    if os.path.exists(mri_file) and os.path.isdir(mri_file):
        mri_list=sorted(os.listdir(mri_file))
        mri_list.sort(key=lambda x:int(x.split('.')[0]))
        for mri_name in mri_list:
            if mri_name.split('.')[-1]=='jpg' or mri_name.split('.')[-1]=='png' or mri_name.split('.')[-1]=='tif' or mri_name.split('.')[-1]=='svg':
                mri_name=os.path.join(mri_file,mri_name)
                mri_arr=cv2.imread(mri_name,0)
                mri_arr=mri_arr.astype(np.float32)/255.
                mri.append(torch.from_numpy(np.expand_dims(mri_arr,axis=0)).view(1,1,256,256))

    if os.path.exists(pet_file) and os.path.isdir(pet_file):
        pet_list=sorted(os.listdir(pet_file))
        pet_list.sort(key=lambda x:int(x.split('.')[0]))
        for pet_name in pet_list:
            if pet_name.split('.')[-1]=='jpg' or pet_name.split('.')[-1]=='png' or pet_name.split('.')[-1]=='tif' or pet_name.split('.')[-1]=='svg':
                pet_name=os.path.join(pet_file,pet_name)
                pet_arr=cv2.imread(pet_name)
                pet_ycrcb=cv2.cvtColor(pet_arr,cv2.COLOR_BGR2YCrCb)
                pet_ycrcb=pet_ycrcb.astype(np.float32)/255.
                pet_tensor.append(torch.from_numpy(np.expand_dims(pet_ycrcb[:,:,0],axis=0)).view(1,1,256,256))
                pet.append(pet_ycrcb)

    start=time.time()
    with torch.no_grad():
        model.eval()
        for i in range(len(mri)):
            L1=model.Transform_Encoder(mri[i])
            H1=model.Transform_Encoder(pet_tensor[i])
            if flag == 'Addition':
                f = torch.add(L1,H1)
            elif flag == 'Average':
                f = torch.add(L1,H1)/2.
            elif flag == 'Maximum':
                f = torch.maximum(L1,H1)
            elif flag == 'L1_Norm':
                f = L1_Norm(L1,H1)
            elif flag == 'LEM':
                f = LEM(L1,H1)
            elif flag == 'LEGM':
                f = LEGM(L1,H1)
            res = model.Transform_Decoder(f)
            res = res[0][0].detach().numpy()
            res = (res - res.min()) / (res.max() - res.min())
            pet[i][:,:,0] = res
            pet[i] *= 255.
            pet[i] = pet[i].astype(np.uint8)
            result = cv2.cvtColor(pet[i],cv2.COLOR_YCrCb2BGR)
            cv2.imwrite('F:/python_projects/DFENet/MRI-PET/results'+'/'+str(i+1)+'.jpg',result)
    end=time.time()
    print('the total time: %.4fs'%(end-start))






