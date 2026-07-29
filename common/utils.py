
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import torch.nn as nn
from collections import OrderedDict

from torchvision.models import vgg19

import numpy as np

import urllib.request
import urllib
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import URLError, HTTPError
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}

def L1_Norm(source_imgs1,source_imgs2)->torch.FloatTensor:
    dimension=source_imgs1.shape
    res=[]
    temp1=torch.abs(source_imgs1)
    temp2=torch.abs(source_imgs2)
    l1_a=torch.sum(temp1,dim=1)
    l1_b=torch.sum(temp2,dim=1)
    l1_a_=torch.sum(l1_a,dim=0)
    l1_b_=torch.sum(l1_b,dim=0)
    #create the map for source images
    mask_value=l1_b_+l1_a_
    mask_sign_a=torch.divide(l1_a_,mask_value)
    mask_sign_b=torch.divide(l1_b_,mask_value)
    for i in range(dimension[1]):
        temp_matrix=torch.mul(mask_sign_a,source_imgs1[0,i,:,:])+torch.mul(mask_sign_b,source_imgs2[0,i,:,:])
        res.append(temp_matrix)
    result=torch.stack(res,dim=-1)
    result=result.permute([2,0,1])
    result_tf=result.unsqueeze(dim=0)
    return result_tf

def LEM(source_imgs1,source_imgs2)->torch.FloatTensor():  #(局部能量值最大)
    dimension=source_imgs1.shape
    device=source_imgs1.device
    s1=source_imgs1**2
    s2=source_imgs2**2
    weight=torch.ones(size=(dimension[1],1,3,3),device=device,dtype=s1.dtype)
    E_1=F.conv2d(s1,weight,stride=1,padding=1,groups=dimension[1])
    E_2=F.conv2d(s2,weight,stride=1,padding=1,groups=dimension[1])
    fusion_1=F.max_pool2d(E_1,3,1,1)
    fusion_2=F.max_pool2d(E_2,3,1,1)
    m=(fusion_1>fusion_2)*1
    res=torch.multiply(source_imgs1,m)+torch.multiply(source_imgs2,1-m)
    return res

def LEGM(source_imgs1,source_imgs2)->torch.FloatTensor():
    dimension=source_imgs1.shape
    device=source_imgs1.device
    s1 = source_imgs1 ** 2
    s2 = source_imgs2 ** 2
    weight = torch.ones(size=(dimension[1], 1, 3, 3), device=device, dtype=s1.dtype)
    E_1 = F.conv2d(s1, weight, stride=1, padding=1, groups=dimension[1])
    E_2 = F.conv2d(s2, weight, stride=1, padding=1, groups=dimension[1])
    h_sobel=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    v_sobel=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    h_sobel=torch.cat([h_sobel]*dimension[1],dim=0)
    v_sobel=torch.cat([v_sobel]*dimension[1],dim=0)
    grad_img1=torch.conv2d(source_imgs1,h_sobel,stride=1,padding=1,groups=dimension[1])**2+torch.conv2d(source_imgs1,v_sobel,stride=1,padding=1,groups=dimension[1])**2
    grad_img2 = torch.conv2d(source_imgs2, h_sobel, stride=1, padding=1, groups=dimension[1]) ** 2 + torch.conv2d(
        source_imgs2, v_sobel, stride=1, padding=1, groups=dimension[1]) ** 2
    five_adjacent=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    five_adjacent=torch.cat([five_adjacent]*dimension[1],dim=0)
    LE_img1=torch.conv2d(grad_img1,five_adjacent,stride=1,padding=1,groups=dimension[1])
    LE_img2=torch.conv2d(grad_img2,five_adjacent,stride=1,padding=1,groups=dimension[1])
    LE_img1=LE_img1/torch.max(LE_img1)
    LE_img2=LE_img2/(torch.max(LE_img2))
    E_1=E_1/torch.max(E_1)
    E_2=E_2/torch.max(E_2)
    m1=0.4*E_1+LE_img1
    m2=0.4*E_2+LE_img2
    m1=torch.exp(m1)/(torch.exp(m1)+torch.exp(m2))
    m2=torch.exp(m2)/(torch.exp(m1)+torch.exp(m2))
    fusion_1 = F.max_pool2d(m1, 3, 1, 1)
    fusion_2 = F.max_pool2d(m2, 3, 1, 1)
    m = (fusion_1> fusion_2) * 1
    res = torch.multiply(source_imgs1, m) + torch.multiply(source_imgs2, 1 - m)
    return res


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial
class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + 1e-10)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + 1e-10)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f

# fuison strategy based on nuclear-norm (channel attention form NestFuse)
class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_ir)
        global_p2 = nuclear_pooling(en_vi)
        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + 1e-10)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + 1e-10)
        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
        tensor_f = global_p_w1 * en_ir + global_p_w2 * en_vi
        return tensor_f
# sum of S V for each chanel
def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1)
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + 1e-10)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors
class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()
    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 3:
                feature_maps.append(x)
        return feature_maps

def isGray(img:np.ndarray)->bool:
    if len(img.shape)<3:
        return True
    if img.shape[2]==1:
        return True
    b,g,r=img[:,:,0],img[:,:,1],img[:,:,2]
    if (b==g).all() and (b==r).all():
        return True
    return False
def new_fusion_rule(source_img1_name,source_img2_name,device)->torch.FloatTensor:
    vgg_model=VGG19(device=device)
    src1=cv.imread(source_img1_name)
    src2=cv.imread(source_img2_name)
    if not isGray(src1):
        src1=cv.cvtColor(src1,cv.COLOR_BGR2YCrCb)
        src1_new=src1[:,:,0]
    else:
        src1_new=src1
    if not isGray(src2):
        src2=cv.cvtColor(src2,cv.COLOR_BGR2YCrCb)
        src2_new=src2[:,:,0]
    else:
        src2_new=src2
    #转为torch tensor
    if src1_new.ndim==2:
        src1_tensor=np.repeat(src1_new[None,None],3,axis=1).astype(np.float32)
    else:
        src1_tensor=np.transpose(src1_new,(2,0,1))[None].astype(np.float32)

    if src2_new.ndim==2:
        src2_tensor=np.repeat(src2_new[None,None],3,axis=1).astype(np.float32)
    else:
        src2_tensor=np.transpose(src2_new,(2,0,1))[None].astype(np.float32)
    src1_tensor/=255.
    src2_tensor/=255.
    src1_tensor,src2_tensor=torch.from_numpy(src1_tensor).to(device),torch.from_numpy(src2_tensor).to(device)
    feature_map1=vgg_model(src1_tensor)
    feature_map2=vgg_model(src2_tensor)
    feature_map1=F.interpolate(feature_map1[0],size=src1_tensor.shape[2:])
    feature_map2=F.interpolate(feature_map2[0],size=src2_tensor.shape[2:])
    feature_map1=torch.sum(feature_map1,dim=1,keepdim=True)
    feature_map2=torch.sum(feature_map2,dim=1,keepdim=True)
    total_map=torch.cat([feature_map1,feature_map2],dim=1)
    total_map=torch.exp(total_map)/(torch.exp(total_map).sum(dim=1,keepdim=True))
    return total_map

def new_fusion_rule1(feature_maps1,feature_maps2)->torch.FloatTensor:
    m1=(feature_maps1-feature_maps1.min())/(feature_maps1.max()-feature_maps1.min())
    m2=(feature_maps2-feature_maps2.min())/(feature_maps2.max()-feature_maps2.min())
    Feature_maps1=torch.sum(m1,dim=1,keepdim=True)
    Feature_maps2=torch.sum(m2,dim=1,keepdim=True)
    total_map=torch.cat([Feature_maps1,Feature_maps2],dim=1)
    total_map=torch.exp(total_map)/(torch.exp(total_map).sum(dim=1,keepdim=True))
    Map=feature_maps1*total_map[:,0:1]+feature_maps2*total_map[:,1:2]
    return Map

def requestImg(url, name, num_retries=3):
    img_src = url
    # print(img_src)
    header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
      Chrome/35.0.1916.114 Safari/537.36',
    'Cookie': 'AspxAutoDetectCookieSupport=1'
    }
    # Request类可以使用给定的header访问URL
    req = urllib.request.Request(url=img_src, headers=header)
    try:
        response = urllib.request.urlopen(req) # 得到访问的网址
        filename = name + '.png'
        with open(filename, "wb") as f:
            content = response.read() # 获得图片
            f.write(content) # 保存图片
            response.close()
    except HTTPError as e: # HTTP响应异常处理
        print(e.reason)
    except URLError as e: # 一定要放到HTTPError之后，因为它包含了前者
        print(e.reason)
    except IncompleteRead or RemoteDisconnected as e:
        if num_retries == 0: # 重连机制
            return
        else:
            requestImg(url, name, num_retries-1)


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()

def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    forward = type(module).forward
    if isinstance(module, nn.Conv2d):
        setattr(self, "conv2d_forward", module.conv2d_forward)
    if isinstance(module, nn.BatchNorm2d):
        setattr(self, "_check_input_dim", module._check_input_dim)
        setattr(self, "num_batches_tracked", module.num_batches_tracked)
        setattr(self, "running_mean", module.running_mean)
        setattr(self, "running_var", module.running_var)

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(self, name, attr)

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild
        setattr(self, name, fchild)

    def fmodule(*args, **kwargs):
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):
            setattr(self, name, param)
        return forward(self, *args, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params')
        return fmodule_internal(*args, **kwargs)
    return fmodule

#pcnn网络
def PA_PCNN(matrix,args)->np.ndarray:
    iteraion_times=args.iter_times
    alpha_f=args.alpha_f
    alpha_e=args.alpha__e
    lambda_1=args.lambda_1
    V_E=args.v_e
    h,w=matrix.shape[:2]
    F=np.abs(matrix)
    U=np.zeros(shape=(h,w))
    Y=np.zeros(shape=(h,w))
    T=np.zeros(shape=(h,w))
    E=np.ones(shape=(h,w))
    #synaptic weights
    W=np.array([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]],dtype=np.float)
    for i in range(iteraion_times):
        K=cv.filter2D(np.uint8(Y),ddepth=cv.CV_32F,kernel=W,borderType=cv.BORDER_REFLECT)
        U=np.exp(-alpha_f)*U+np.multiply(F,1+lambda_1*K)
        Y=(U>E)*1
        E=np.exp(-alpha_e)*E+V_E*Y
        T=T+Y
    return T

from PIL import Image
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

#融合彩色图像通道的其它通道
def other_channels_fusion(Cb,Cb1,Cr,Cr1):
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
    return Cb_f,Cr_f
# import cv2 as cv
# if __name__=='__main__':
#     IR=cv.imread(r'E:\LytroDataset\LytroDataset\lytro-03-A.jpg')
#     VIS=cv.imread(r'E:\LytroDataset\LytroDataset\lytro-03-B.jpg')
#     IR=cv.resize(IR,dsize=(256,256),interpolation=cv.INTER_CUBIC)
#     VIS=cv.resize(VIS,dsize=(256,256),interpolation=cv.INTER_CUBIC)
#     IR_ycrcb=cv.cvtColor(IR,cv.COLOR_BGR2YCrCb)
#     result = np.zeros_like(IR_ycrcb)
#     VIS_ycrcb=cv.cvtColor(VIS,cv.COLOR_BGR2YCrCb)
#     IR_ycrcb=IR_ycrcb.astype(np.float32)
#     VIS_ycrcb=VIS_ycrcb.astype(np.float32)
#     LR=IR_ycrcb[:,:,0]/255.
#     HR=VIS_ycrcb[:,:,0]/255.
#     LR = torch.from_numpy(np.expand_dims(LR,axis=0)).view(1, 1, 256, 256)
#     HR = torch.from_numpy(np.expand_dims(HR,axis=0)).view(1, 1, 256, 256)
#     model = Net()
#     model_path = r'E:\python projects\My_Net\weights\pretrain\_15_0.5_new.pth'
#     model1_path = r'E:\python projects\My_Net\model4_gradient_weights\_21_0.5_new.pth'
#     with torch.no_grad():
#         model.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
#         model.eval()
#         L1 = model.Transform_Encoder(LR)
#         H1 = model.Transform_Encoder(HR)
#         f =torch.maximum(L1,H1)
#         res = model.Transform_Decoder(f)
#         plt.imshow(res[0][0].numpy(), 'gray')
#         plt.axis('off')
#         plt.show()
#         plt.clf()
#         res=res[0][0].detach().numpy()
#         res=(res-res.min())/(res.max()-res.min())
#         res*=255.
#         res=res.astype(np.uint8)
#         cb_f,cr_f=other_channels_fusion(IR_ycrcb[:,:,2],VIS_ycrcb[:,:,2],IR_ycrcb[:,:,1],VIS_ycrcb[:,:,1])
#         result[:,:,0]=res
#         result[:,:,1]=cr_f
#         result[:,:,2]=cb_f
#         result=cv.cvtColor(result,cv.COLOR_YCrCb2BGR)
#         cv.imshow('this',result)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#         cv.imwrite('./results/focus.jpg',result)



