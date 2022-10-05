import os,sys
sys.path.append('..')
import time
import cv2
import numpy as np
import torch
from utils import L1_Norm, LEM, LEGM
from net import Net

flag = 'LEM'
if __name__ == '__main__':
    print('==>Loading the model')
    model = Net()
    model_path = r'../Checkpoints/ckpt_10_0.5.pth'

    if not os.path.exists(model_path):
        raise Exception('No pretrained model could be found!')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print('==>Loading the test dataset')
    mri_file = '../Data/MRI-T1_T2'
    ct_file = '../Data/CT'
    mri = []
    ct = []

    if os.path.exists(mri_file) and os.path.isdir(mri_file):
        mri_list=sorted(os.listdir(mri_file))
        mri_list.sort(key=lambda x:int(x.split('.')[0]))
        for mri_name in mri_list:
            if mri_name.split('.')[-1]=='jpg' or mri_name.split('.')[-1]=='png' or mri_name.split('.')[-1]=='tif' or mri_name.split('.')[-1]=='svg':
                mri_name = os.path.join(mri_file,mri_name)
                mri_arr = cv2.imread(mri_name,cv2.IMREAD_GRAYSCALE)
                mri_arr = mri_arr.astype(np.float32)/255.
                mri.append(torch.from_numpy(np.expand_dims(mri_arr,axis=0)).view(1,1,256,256))

    if os.path.exists(ct_file) and os.path.isdir(ct_file):
        ct_list=sorted(os.listdir(ct_file))
        ct_list.sort(key=lambda x:int(x.split('.')[0]))
        for ct_name in ct_list:
            if ct_name.split('.')[-1]=='jpg' or ct_name.split('.')[-1]=='png' or ct_name.split('.')[-1]=='tif' or t_name.split('.')[-1]=='svg':
                ct_name=os.path.join(ct_file,ct_name)
                ct_arr=cv2.imread(ct_name,cv2.IMREAD_GRAYSCALE)
                ct_arr=ct_arr.astype(np.float32)/255.
                ct.append(torch.from_numpy(np.expand_dims(ct_arr,axis=0)).view(1,1,256,256))

    start=time.time()
    with torch.no_grad():
        model.eval()
        for i in range(len(mri)):
            L1=model.Transform_Encoder(mri[i])
            H1=model.Transform_Encoder(ct[i])
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
            res *= 255.
            result=np.uint8(res)
            cv2.imwrite('../results'+'/'+str(i+1)+'.jpg',result)

    end=time.time()
    print('the total time: %.4fs'%(end-start))






