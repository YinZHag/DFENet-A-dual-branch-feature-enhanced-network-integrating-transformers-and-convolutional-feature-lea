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

IMAGE_SUFFIXES = ('.jpg', '.png', '.tif', '.svg')
FUSION_STRATEGIES = ('Addition', 'Average', 'Maximum', 'L1_Norm', 'LEM', 'LEGM')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def hyper_parametrs():
    parse = argparse.ArgumentParser(description='Created by liam Zhang')
    parse.add_argument('--model_path',type=str,default='',\
                       help='the pretrained model path')
    parse.add_argument('--mri_file',type=str,default=r'',\
                       help='the test mri dataset path')
    parse.add_argument('--spect_file',type=str,default=r'', \
                       help='the test spect dataset path')
    args = parse.parse_args()
    return args


def list_image_files(directory, description):
    if not os.path.isdir(directory):
        raise NotADirectoryError('The {} dataset directory does not exist: {!r}'.format(description, directory))
    names = [name for name in os.listdir(directory) if name.lower().endswith(IMAGE_SUFFIXES)]
    if not names:
        raise FileNotFoundError('No {} image found in {!r}, supported suffixes are {}'.format(
            description, directory, ', '.join(IMAGE_SUFFIXES)))
    try:
        names.sort(key=lambda x: int(x.split('.')[0]))
    except ValueError as error:
        raise ValueError('The {} images must be named with their index, e.g. 1.jpg, but got {}'.format(
            description, sorted(names))) from error
    return [os.path.join(directory, name) for name in names]


def read_image(path, flags):
    array = cv2.imread(path, flags)
    if array is None:
        raise OSError('The image could not be decoded: {}'.format(path))
    if array.shape[:2] != (256, 256):
        raise ValueError('The image {} must be 256x256 but its shape is {}'.format(path, array.shape))
    return array


if __name__ == '__main__':
    args = hyper_parametrs()
    if flag not in FUSION_STRATEGIES:
        raise ValueError('Unknown fusion strategy {!r}, expected one of {}'.format(flag, ', '.join(FUSION_STRATEGIES)))

    print('==>Loading the model')
    model = Net()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError('No pretrained model could be found at {!r}'.format(args.model_path))
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    print('==>Loading the test dataset')
    mri_paths = list_image_files(args.mri_file, 'mri')
    spect_paths = list_image_files(args.spect_file, 'spect')
    if len(mri_paths) != len(spect_paths):
        raise ValueError('The mri and spect datasets must be paired but hold {} and {} images'.format(
            len(mri_paths), len(spect_paths)))

    mri=[]
    spect=[]
    spect_tensor=[]
    for mri_path, spect_path in zip(mri_paths, spect_paths):
        mri_arr = read_image(mri_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        mri.append(torch.from_numpy(np.expand_dims(mri_arr,axis=0)).view(1,1,256,256))
        spect_arr = read_image(spect_path, cv2.IMREAD_COLOR)
        spect_ycrcb=cv2.cvtColor(spect_arr,cv2.COLOR_BGR2YCrCb)
        spect_ycrcb=spect_ycrcb.astype(np.float32)/255.
        spect_tensor.append(torch.from_numpy(np.expand_dims(spect_ycrcb[:,:,0],axis=0)).view(1,1,256,256))
        spect.append(spect_ycrcb)

    os.makedirs(RESULT_DIR, exist_ok=True)
    start=time.time()
    with torch.no_grad():
        model.eval()
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
            else:
                f = LEGM(L1,H1)
            res = model.Transform_Decoder(f)
            res = res[0][0].detach().numpy()
            res = (res - res.min()) / (res.max() - res.min())
            spect[i][:,:,0] = res
            spect[i] *= 255.
            spect[i] = spect[i].astype(np.uint8)
            result = cv2.cvtColor(spect[i],cv2.COLOR_YCrCb2BGR)
            result_path = os.path.join(RESULT_DIR, str(i+1)+'.jpg')
            if not cv2.imwrite(result_path, result):
                raise OSError('The fused image could not be written to {}'.format(result_path))
    end=time.time()
    print('the total time: %.4fs'%(end-start))
