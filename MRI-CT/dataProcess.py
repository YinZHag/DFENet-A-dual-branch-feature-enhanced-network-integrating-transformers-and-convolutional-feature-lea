import matplotlib.pyplot as plt
from PIL import Image
import  cv2
import numpy as np
import decompose
import pathlib

def detail_enhance(img: np.ndarray):
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError('detail_enhance requires a non-empty numpy image array')
    img = cv2.resize(img, (256, 256), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    imgY = img.copy()
    M, Ds = decompose.edge_preserving_decompose(imgY, 1)
    #TODO: enhanced factor
    enhanced_factor = 1.5
    detail_enhanced_img = decompose.clip_and_convert_to_uint8(M + enhanced_factor * np.sum(Ds, axis=0))
    return detail_enhanced_img


def load_train_img(ImgFilePath):
    """
    :param ImgFilePath: the input path of the training dataset
    """
    if not isinstance(ImgFilePath,(str,pathlib.Path)):
        raise TypeError('The ImgFilePath should be required for str or pathlib type, got {}'.format(type(ImgFilePath).__name__))
    img=Image.open(ImgFilePath).convert('L')
    index=np.random.randint(1,4,1)
    if index==1:
        img=img.rotate(90)
    elif index==2:
        img=img.rotate(180)
    elif index==3:
        img=img.rotate(270)
    else:
        pass
    enhanced=detail_enhance(np.array(img))
    enhanced=enhanced.astype(np.float32)/255.
    enhanced=np.reshape(enhanced,newshape=(256,256,1))
    return img,enhanced


def load_test_img(Mri_File_Path,Pet_File_Path):
    """
    :param Mri_file_path: the input file path of Magnetic Resonance Imageing
    :param Pet_file_path: the input file path of Positron Emission Tomography Imaging
    """
    if not isinstance(Mri_File_Path,(str,pathlib.Path)):
        raise TypeError('The Mri_File_Path should be required for str or pathlib type, got {}'.format(type(Mri_File_Path).__name__))
    if not isinstance(Pet_File_Path,(str,pathlib.Path)):
        raise TypeError('The Pet_File_Path should be required for str or pathlib type, got {}'.format(type(Pet_File_Path).__name__))
    mri=Image.open(Mri_File_Path).convert('L')
    pet=Image.open(Pet_File_Path).convert('L')
    index=np.random.randint(1,4,1)
    if index==1:
        pet=pet.rotate(90)
        mri=mri.rotate(90)
    elif index==2:
        mri=mri.rotate(180)
        pet=pet.rotate(180)
    elif index==3:
        mri=mri.rotate(270)
        pet=pet.rotate(270)
    else:
        pass
    if mri.size[0]!=256 or mri.size[1]!=256:
        mri=mri.resize((256,256),resample=Image.BICUBIC)
    if pet.size[0]!=256 or pet.size[1]!=256:
        pet = pet.resize((256, 256), resample=Image.BICUBIC)
    return mri,pet

