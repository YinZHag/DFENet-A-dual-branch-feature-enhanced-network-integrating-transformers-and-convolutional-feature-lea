from torch.utils.data import Dataset
from torchvision import transforms
import os
from dataProcess import load_train_img,load_test_img
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])
def LR_Transform_Train():
    return transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop((256,256))])
def LR_Transform_Eval():
    return transforms.Compose([transforms.ToTensor()])

class Train_dataSet(Dataset):
    def __init__(self,mscoco_path,transform=LR_Transform_Train(),transform1=LR_Transform_Eval()):
        super(Train_dataSet, self).__init__()
        self.train_path=[os.path.join(mscoco_path,i) for i in os.listdir(mscoco_path)if is_image_file(i)][:30000]#取30000张图片进行训练
        self.transform=transform
        self.transform1=transform1
    def __getitem__(self, item):
        img_path=self.train_path[item]
        img,enhanced=load_train_img(img_path)
        return self.transform(img),self.transform1(enhanced)
    def __len__(self):
        return len(self.train_path)

class Eval_dataSet(Dataset):
    def __init__(self,mri_path,pet_path,transform=LR_Transform_Eval()):
        super(Eval_dataSet, self).__init__()
        self.mri_path=[os.path.join(mri_path,i)for i in os.listdir(mri_path)if is_image_file(i)]
        self.pet_path=[os.path.join(pet_path,j)for j in os.listdir(pet_path)if is_image_file(j)]
        self.transform=transform
    def __getitem__(self, item):
        mri_path=self.mri_path[item]
        pet_path=self.pet_path[item]
        mri_img,pet_img=load_test_img(mri_path,pet_path)
        return self.transform(mri_img),self.transform(pet_img)
    def __len__(self):
        return len(self.mri_path)

