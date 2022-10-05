import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imwrite
from numpy.lib.stride_tricks import as_strided
from torch import nn
from utils import L1_Norm, LEM, LEGM, LGE1, new_fusion_rule, Fusion_SPA, Fusion_Nuclear, LEM1

# from My_Net.model4 import Net
from model7 import Net
from PIL import Image
def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
def process(output_tensor,Cb,Cr):  # 单通道数据
    if output_tensor.is_cuda:
        output = output_tensor[0][0].cpu().numpy()
    else:
        output = output_tensor[0][0].numpy()
    output = (output - np.min(output)) / (np.max(output) - np.min(output))
    output *= 255
    res = Image.fromarray(np.uint8(output), mode='L')
    res=Image.merge('YCbCr',[res,Cb,Cr]).convert('RGB')
    return res

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
#CBAM(Convlution block attention module)卷积块注意力模块
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
def weight_init(m):
    if(isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear)):
        m.weight=nn.init.normal_(m.weight)
        if m.bias is not None:
            m.bias=nn.init.ones_(m.bias)
    if(isinstance(m,nn.BatchNorm2d)):
        m.weight=nn.init.normal_(m.weight)
        if m.bias is not None:
            m.bias=nn.init.ones_(m.bias)

L1_NORM = lambda b: np.sum(np.abs(b))
def weightedFusion(cr1, cr2, cb1, cb2):
    """
        Perform the weighted fusing for Cb and Cr channel (paper equation 6)
        Arg:    cr1     (torch.Tensor)  - The Cr slice of 1st image
                cr2     (torch.Tensor)  - The Cr slice of 2nd image
                cb1     (torch.Tensor)  - The Cb slice of 1st image
                cb2     (torch.Tensor)  - The Cb slice of 2nd image
        Ret:    The fused Cr slice and Cb slice
    """
    # Fuse Cr channel
    cr1,cr2,cb1,cb2=np.asarray(cr1),np.asarray(cr2),np.asarray(cb1),np.asarray(cb2)
    cr_up = (cr1 * L1_NORM(cr1 - 127.5) + cr2 * L1_NORM(cr2 - 127.5))
    cr_down = L1_NORM(cr1 - 127.5) + L1_NORM(cr2 - 127.5)
    cr_fuse = cr_up / cr_down

    # Fuse Cb channel
    cb_up = (cb1 * L1_NORM(cb1 - 127.5) + cb2 * L1_NORM(cb2 - 127.5))
    cb_down = L1_NORM(cb1 - 127.5) + L1_NORM(cb2 - 127.5)
    cb_fuse = cb_up / cb_down

    return cr_fuse, cb_fuse
#融合彩色图像通道的其它通道
def other_channels_fusion(Cb,Cb1,Cr,Cr1):
    Cb,Cb1,Cr,Cr1=np.asarray(Cb),np.asarray(Cb1),np.asarray(Cr),np.asarray(Cr1)
    h,w=Cb.shape[:2]
    Cb_f,Cr_f=np.zeros_like(Cb),np.zeros_like(Cr)
    for row in range(h):
        for col in range(w):
            if np.abs(Cb[row,col]-128)==0 and np.abs(Cb1[row,col]-128)==0:
                Cb_f[row,col]=128
            else:
                middle1=Cb[row,col]*np.abs(Cb[row,col]-128)+Cb1[row,col]*np.abs(Cb1[row,col]-128)
                middle2=np.abs(Cb[row,col]-128)+np.abs(Cb1[row][col]-128)
                Cb_f[row,col]=middle1/middle2

            if np.abs(Cr[row,col]-128)==0 and np.abs(Cr1[row,col]-128)==0:
                Cr_f[row,col]=128
            else:
                middle1=Cr[row,col]*np.abs(Cr[row,col]-128)+Cr1[row,col]*np.abs(Cr1[row,col]-128)
                middle2=np.abs(Cr[row,col]-128)+np.abs(Cr1[row][col]-128)
                Cr_f[row,col]=middle1/middle2
    Cb_f,Cr_f=np.clip(Cb_f,0,255).astype(np.int8),np.clip(Cr_f,0,255).astype(np.uint8)
    Cb_f,Cr_f=Image.fromarray(Cb_f,mode='L'),Image.fromarray(Cr_f,mode='L')
    return Cb_f,Cr_f
import cv2
flag = 'Addition'
if __name__ == '__main__':
    # MRI-T1_T2=cv2.imread(r'F:\noise_data_revision\mri_pet\mri_t1_t2\1.jpg',0)
    # pet=cv2.imread(r'F:\noise_data_revision\mri_pet\pet\1.jpg')
    # device=torch.device('cpu')
    # # response_map=new_fusion_rule(r'E:\RFN_Nest_network-trainDatas\test_images\MR_PET\MRI\15.jpg',r'E:\RFN_Nest_network-trainDatas\test_images\MR_PET\FDG_PET\15.jpg',device)
    # if len(MRI-T1_T2.shape)==3 and not ((MRI-T1_T2[:,:,0]==MRI-T1_T2[:,:,1]).all() and (MRI-T1_T2[:,:,0]==MRI-T1_T2[:,:,2]).all()):
    #     mri_ycrcb=cv2.cvtColor(MRI-T1_T2,cv2.COLOR_BGR2YCrCb)
    # if len(pet.shape) == 3 and not ((pet[:, :, 0] == pet[:, :, 1]).all() and (pet[:, :, 0] == pet[:, :, 2]).all()):
    #     pet_ycrcb=cv2.cvtColor(pet,cv2.COLOR_BGR2YCrCb)
    # MRI-T1_T2=MRI-T1_T2.astype(np.float32)
    # # pet=pet.astype(np.float32)
    # pet_ycrcb=pet_ycrcb.astype(np.float32)
    # MRI-T1_T2/=255.
    # pet_ycrcb/=255.
    # LR=MRI-T1_T2.copy()
    # HR=pet_ycrcb[:,:,0]
    # LR=torch.from_numpy(np.expand_dims(LR,axis=0)).view(1,1,256,256)
    # HR=torch.from_numpy(np.expand_dims(HR,axis=0)).view(1,1,256,256)
    model=Net()
    model_path = r'F:\python projects\My_Net\model4_gradient_weights\_21_0.5_new.pth'
    model1_path=r'F:\python projects\My_Net\model4_gradient_weights\_14_0.5_new_with_no_weight.pth'
    model2_path = r'F:\python projects\My_Net\model7_weights\_1_0.5_new_.pth'
    model3_path = r'F:\python projects\My_Net\ablation\1_res2net_block\_10_0.5_new_.pth'
    model4_path = r'F:\python projects\My_Net\ablation\transformer\_10_0.5_new_.pth'
    model5_path = r'F:\python projects\My_Net\mri_ct\cnn_branch\_21_0.5_new_.pth'
    mri_file=r'F:/noise_data_revision/mri_spect/mri_t1_t2'
    spect_file=r'F:/noise_data_revision/mri_spect/spect'
    mri=[]
    spect=[]
    spect_tensor=[]
    if os.path.exists(mri_file) and os.path.isdir(mri_file):
        mri_list=sorted(os.listdir(mri_file))
        mri_list.sort(key=lambda x:int(x.split('.')[0]))
        for mri_name in mri_list:
            if mri_name.split('.')[-1]=='jpg' or mri_name.split('.')[-1]=='png' or mri_name.split('.')[-1]=='tif' or mri_name.split('.')[-1]=='svg':
                mri_name=os.path.join(mri_file,mri_name)
                mri_arr=cv2.imread(mri_name,0)
                mri_arr=mri_arr.astype(np.float32)/255.
                mri.append(torch.from_numpy(np.expand_dims(mri_arr,axis=0)).view(1,1,256,256))

    if os.path.exists(spect_file) and os.path.isdir(spect_file):
        pet_list=sorted(os.listdir(spect_file))
        pet_list.sort(key=lambda x:int(x.split('.')[0]))
        for pet_name in pet_list:
            if pet_name.split('.')[-1]=='jpg' or pet_name.split('.')[-1]=='png' or pet_name.split('.')[-1]=='tif' or pet_name.split('.')[-1]=='svg':
                pet_name=os.path.join(spect_file,pet_name)
                pet_arr=cv2.imread(pet_name)
                pet_ycrcb=cv2.cvtColor(pet_arr,cv2.COLOR_BGR2YCrCb)
                pet_ycrcb=pet_ycrcb.astype(np.float32)/255.
                spect_tensor.append(torch.from_numpy(np.expand_dims(pet_ycrcb[:,:,0],axis=0)).view(1,1,256,256))
                spect.append(pet_ycrcb)

    start=time.time()
    with torch.no_grad():
        model.load_state_dict(torch.load(model3_path,map_location=torch.device('cpu')))
        model.eval()
        # L1=model.Transform_Encoder(LR)
        # H1=model.Transform_Encoder(HR)
        # # reconstruct_mri=model.Transform_Decoder(L1)
        # # reconstruct_mri=(reconstruct_mri-reconstruct_mri.min())/(reconstruct_mri.max()-reconstruct_mri.min())
        # # reconstruct_mri=reconstruct_mri[0][0].detach().numpy()
        # # cv2.imwrite('./results/reconstuct_mri_both.jpg',np.uint8(reconstruct_mri*255))
        # f=LGE(L1,H1)
        # net3=model.encoder.net3
        # net2=model.encoder.net2
        # s1=net2(LR)
        # s2=net3(LR)
        # s1=nn.AdaptiveAvgPool3d((1,256,256))(s1)
        # s2=nn.AdaptiveAvgPool3d((1,256,256))(s2)
        # plt.imshow(s1[0][0].detach().numpy(),'gray')
        # plt.show()
        # plt.clf()
        # plt.imshow(s2[0][0].detach().numpy(),'gray')
        # plt.show()
        # plt.clf()
        # s2=s2[0][0].detach().numpy()
        # s2=(s2-s2.min())/(s2.max()-s2.min())
        # detail=MRI-T1_T2-s2
        # plt.imshow(detail,'gray')
        # plt.show()
        # plt.show()
        # plt.clf()
        # res=model.Transform_Decoder(f)
        # plt.imshow(res[0][0].detach().numpy(),'gray')
        # plt.axis('off')
        # plt.show()
        # plt.clf()
        # res=res[0][0].detach().numpy()
        # res = (res - res.min()) / (res.max() - res.min())
        # pet_ycrcb[:,:,0]=res
        # pet_ycrcb*=255
        # pet_ycrcb=pet_ycrcb.astype(np.uint8)
        # result=cv2.cvtColor(pet_ycrcb,cv2.COLOR_YCrCb2BGR)
        # # res*=255
        # # result=np.uint8(res)
        # cv2.imshow('this',result)
        # cv2.imwrite('./results/LGE_Revision.jpg',result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        #
        for i in range(len(mri)):
            L1=model.Transform_Encoder(mri[i])
            H1=model.Transform_Encoder(spect_tensor[i])
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
            # res *= 255.
            # result=np.uint8(res)
            spect[i][:,:,0] = res
            spect[i] *= 255.
            spect[i] = spect[i].astype(np.uint8)
            result = cv2.cvtColor(spect[i],cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(r'F:/noise_data_revision/mri_spect/fusion_results/Addition'+'/'+str(i+1)+'.jpg',result)
    end=time.time()
    print('the total time: %.4fs'%(end-start))






