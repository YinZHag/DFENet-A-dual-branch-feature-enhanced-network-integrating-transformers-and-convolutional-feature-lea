import math

from scipy.io import loadmat
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch import  nn
from torch.nn.init import trunc_normal_
from torch.nn import functional as F



class Gradient_Net(nn.Module):
  def __init__(self,device):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)
    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x,stride=1,padding=1,groups=x.shape[1])
    grad_y = F.conv2d(x, self.weight_y,stride=1,padding=1,groups=x.shape[1])
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient
if __name__=='__main__':
    """
    This is a visualization of the .mat metrics file run according to the objective evaluation metrics 
    algorithm for images provided by link https://github.com/thfylsty/Objective-evaluation-for-image-fusion.
    """
    data  = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_Add_Revision.mat')
    data1 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_AVG_Revision.mat')
    data2 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_MAX_Revision.mat')
    data3 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_L1_Norm_Revision.mat')
    data4 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_LEM_Revision.mat')
    data5 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-SPECT\result\Noise_Data_LEGM_Revision.mat')
    data6 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-PET\result\les.mat')
    data7 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-PET\result\csr.mat')
    data8 = loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-PET\result\nsct.mat')
    data9= loadmat(r'F:\各种融合图像代码下载\test_fusion\MRI-PET\result\svt.mat')
    arr=['EI','CrossEntropy','SF','EN','Qabf','SCD','FMI_w','FMI_dct','SSIM','MS_SSIM','FMI_pixel','Nabf','MI','VIF','SD','EN','DF','QSF','QMI','QS','QY','QC','QNCIE','Q^{AB/F}','AG','MIabf','QG','CC','VIFF','QP','QW','QE','QCV','QCB']
    dict={}
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    dict5 = {}
    dict6 = {}
    dict7 = {}
    dict8 = {}
    dict9 = {}
    dict10 = {}
    for i in range(len(data['b'][0])):
        dict[arr[i]]=data['b'][0][i]
        dict1[arr[i]] = data1['b'][0][i]
        dict2[arr[i]] = data2['b'][0][i]
        dict3[arr[i]] = data3['b'][0][i]
        dict4[arr[i]] = data4['b'][0][i]
        dict5[arr[i]] = data5['b'][0][i]
    #     dict6[arr[i]] = data6['a'][10][i]
    #     dict7[arr[i]] = data7['a'][10][i]
    #     dict8[arr[i]] = data8['a'][10][i]
    #     dict9[arr[i]] = data9['a'][10][i]
    #     # dict10[arr[i]] = data10['a'][10][i]
    print('our:',dict,'\n','emfusion:',dict1,'\n','dsagan:',dict2,'\n','msenet:',dict3,'\n','ifcnn:',dict4,'\n','nsst_papcnn:',dict5,'\n','les:',dict6,'\n','csr:',dict7,'\n','nsct:',dict8,'\n','svt:',dict9,'\n','proposed:',dict10)

    '''
    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict={}
        for j in range(len(Data['a'][0])):
            dict[arr[j]]=Data['a'][i][j]
        dict1_.append(dict['SSIM'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['SSIM'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['SSIM'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['SSIM'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['SSIM'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['SSIM'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['SSIM'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['SSIM'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['SSIM'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['SSIM'])
    plt.plot(range(1,31),dict10_,linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='SVT')
    #settind the default label font-size
    plt.rcParams.update({'font.size': 10})
    plt.show()
    plt.clf()
    plt.plot(range(1,31),dict1_,linestyle=':',linewidth=2,color='b',marker='o',markersize=6,label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6, label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict4_,linestyle='-.', linewidth=2, color='c', marker='*', markersize=6, label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6, label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6, label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6, label='LES:{:.4f}'.format(data6['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict8_,linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6, label='CSR:{:.4f}'.format(data7['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict9_,linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6, label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('SSIM')]))
    plt.plot(range(1,31),dict10_,linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='SVT:{:.4f}'.format(data9['b'][0][arr.index('SSIM')]))
    plt.title('SSIM',fontstyle='italic',loc='center', fontsize=20,)
    plt.legend(bbox_to_anchor=(1,1), loc='upper left', borderaxespad=0.25,)
    # plt.legend(loc='upper left')
    plt.tick_params(axis='x',bottom=True,direction ='in',which='major')
    plt.tick_params(axis='y', left=True, direction='in',which='major')
    plt.tick_params(axis='both', top=True,right=True, direction='in',which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/SSIM.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['FMI_pixel'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['FMI_pixel'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['FMI_pixel'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['FMI_pixel'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['FMI_pixel'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['FMI_pixel'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['FMI_pixel'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['FMI_pixel'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['FMI_pixel'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['FMI_pixel'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('FMI_pixel')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('FMI_pixel')]))
    plt.title('FMI', fontstyle='italic',loc='center', fontsize=20, fontweight='light')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/FMI_pixel.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['Qabf'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['Qabf'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['Qabf'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['Qabf'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['Qabf'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['Qabf'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['Qabf'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['Qabf'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['Qabf'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['Qabf'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('Qabf')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('Qabf')]))
    plt.title('$Q^{AB/F}$', loc='center', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/Qabf.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['QMI'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['QMI'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['QMI'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['QMI'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['QMI'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['QMI'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['QMI'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['QMI'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['QMI'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['QMI'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('QMI')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('QMI')]))
    plt.title('$Q_{MI}$', loc='center', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/QMI.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['QNCIE'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['QNCIE'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['QNCIE'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['QNCIE'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['QNCIE'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['QNCIE'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['QNCIE'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['QNCIE'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['QNCIE'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['QNCIE'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('QNCIE')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('QNCIE')]))
    plt.title('$Q_{NCIE}$', loc='center', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/QNCIE.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    # dict1_ = []
    # dict2_ = []
    # dict3_ = []
    # dict4_ = []
    # dict5_ = []
    # dict6_ = []
    # dict7_ = []
    # dict8_ = []
    # dict9_ = []
    # dict10_ = []
    # for i in range(len(Data['a'])):
    #     dict = {}
    #     for j in range(len(Data['a'][0])):
    #         dict[arr[j]] = Data['a'][i][j]
    #     dict1_.append(dict['Nabf'])
    # print('\n')
    # print('\n')
    # for i in range(len(data1['a'])):
    #     dict = {}
    #     for j in range(len(data1['a'][0])):
    #         dict[arr[j]] = data1['a'][i][j]
    #     dict2_.append(dict['Nabf'])
    #
    # for i in range(len(data2['a'])):
    #     dict = {}
    #     for j in range(len(data2['a'][0])):
    #         dict[arr[j]] = data2['a'][i][j]
    #     dict3_.append(dict['Nabf'])
    #
    # for i in range(len(data3['a'])):
    #     dict = {}
    #     for j in range(len(data3['a'][0])):
    #         dict[arr[j]] = data3['a'][i][j]
    #     dict4_.append(dict['Nabf'])
    #
    # for i in range(len(data4['a'])):
    #     dict = {}
    #     for j in range(len(data4['a'][0])):
    #         dict[arr[j]] = data4['a'][i][j]
    #     dict5_.append(dict['Nabf'])
    #
    # for i in range(len(data5['a'])):
    #     dict = {}
    #     for j in range(len(data5['a'][0])):
    #         dict[arr[j]] = data5['a'][i][j]
    #     dict6_.append(dict['Nabf'])
    #
    # for i in range(len(data6['a'])):
    #     dict = {}
    #     for j in range(len(data6['a'][0])):
    #         dict[arr[j]] = data6['a'][i][j]
    #     dict7_.append(dict['Nabf'])
    #
    # for i in range(len(data7['a'])):
    #     dict = {}
    #     for j in range(len(data7['a'][0])):
    #         dict[arr[j]] = data7['a'][i][j]
    #     dict8_.append(dict['Nabf'])
    #
    # for i in range(len(data8['a'])):
    #     dict = {}
    #     for j in range(len(data8['a'][0])):
    #         dict[arr[j]] = data8['a'][i][j]
    #     dict9_.append(dict['Nabf'])
    #
    # for i in range(len(data9['a'])):
    #     dict = {}
    #     for j in range(len(data9['a'][0])):
    #         dict[arr[j]] = data9['a'][i][j]
    #     dict10_.append(dict['Nabf'])
    # # plt.figure(figsize=(13, 5))
    # plt.plot(dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6, label='Proposed')
    # plt.plot(dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6, label='Emfusion')
    # plt.plot(dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6, label='Dsagan')
    # plt.plot(dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6, label='Msdnet')
    # plt.plot(dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6, label='Ifcnn')
    # plt.plot(dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6, label='Nsst_papcnn')
    # plt.plot(dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6, label='Les')
    # plt.plot(dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6, label='Csr')
    # plt.plot(dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6, label='Nsct')
    # plt.plot(dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='Svt')
    # plt.title('Nabf', loc='center')
    # # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.2)
    # plt.legend(loc='upper left')
    # plt.savefig('E:/实验数据记录/对比指标图1/MRI-PET/Nabf.png')
    # plt.show()
    # plt.clf()

    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['QCV'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['QCV'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['QCV'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['QCV'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['QCV'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['QCV'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['QCV'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['QCV'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['QCV'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['QCV'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('QCV')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('QCV')]))
    plt.title('$Q_{CV}$', loc='center', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/QCV.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    dict1_ = []
    dict2_ = []
    dict3_ = []
    dict4_ = []
    dict5_ = []
    dict6_ = []
    dict7_ = []
    dict8_ = []
    dict9_ = []
    dict10_ = []
    for i in range(len(Data['a'])):
        dict = {}
        for j in range(len(Data['a'][0])):
            dict[arr[j]] = Data['a'][i][j]
        dict1_.append(dict['QCB'])
    print('\n')
    print('\n')
    for i in range(len(data1['a'])):
        dict = {}
        for j in range(len(data1['a'][0])):
            dict[arr[j]] = data1['a'][i][j]
        dict2_.append(dict['QCB'])

    for i in range(len(data2['a'])):
        dict = {}
        for j in range(len(data2['a'][0])):
            dict[arr[j]] = data2['a'][i][j]
        dict3_.append(dict['QCB'])

    for i in range(len(data3['a'])):
        dict = {}
        for j in range(len(data3['a'][0])):
            dict[arr[j]] = data3['a'][i][j]
        dict4_.append(dict['QCB'])

    for i in range(len(data4['a'])):
        dict = {}
        for j in range(len(data4['a'][0])):
            dict[arr[j]] = data4['a'][i][j]
        dict5_.append(dict['QCB'])

    for i in range(len(data5['a'])):
        dict = {}
        for j in range(len(data5['a'][0])):
            dict[arr[j]] = data5['a'][i][j]
        dict6_.append(dict['QCB'])

    for i in range(len(data6['a'])):
        dict = {}
        for j in range(len(data6['a'][0])):
            dict[arr[j]] = data6['a'][i][j]
        dict7_.append(dict['QCB'])

    for i in range(len(data7['a'])):
        dict = {}
        for j in range(len(data7['a'][0])):
            dict[arr[j]] = data7['a'][i][j]
        dict8_.append(dict['QCB'])

    for i in range(len(data8['a'])):
        dict = {}
        for j in range(len(data8['a'][0])):
            dict[arr[j]] = data8['a'][i][j]
        dict9_.append(dict['QCB'])

    for i in range(len(data9['a'])):
        dict = {}
        for j in range(len(data9['a'][0])):
            dict[arr[j]] = data9['a'][i][j]
        dict10_.append(dict['QCB'])
    plt.plot(range(1, 31), dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6,
             label='DFENet:{:.4f}'.format(Data['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6,
             label='EMFusion:{:.4f}'.format(data1['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6,
             label='DSAGAN:{:.4f}'.format(data2['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6,
             label='MSENet:{:.4f}'.format(data3['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6,
             label='IFCNN:{:.4f}'.format(data4['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6,
             label='NSST-PAPCNN:{:.4f}'.format(data5['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6,
             label='LES:{:.4f}'.format(data6['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6,
             label='CSR:{:.4f}'.format(data7['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6,
             label='NSCT:{:.4f}'.format(data8['b'][0][arr.index('QCB')]))
    plt.plot(range(1, 31), dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6,
             label='SVT:{:.4f}'.format(data9['b'][0][arr.index('QCB')]))
    plt.title('$Q_{CB}$', loc='center', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.25, )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(axis='x', bottom=True, direction='in', which='major')
    plt.tick_params(axis='y', left=True, direction='in', which='major')
    plt.tick_params(axis='both', top=True, right=True, direction='in', which='major')
    plt.savefig('F:/实验数据记录/对比指标图/MRI-CT/QCB.png', bbox_inches='tight')
    plt.show()
    plt.clf()


    # dict1_ = []
    # dict2_ = []
    # dict3_ = []
    # dict4_ = []
    # dict5_ = []
    # dict6_ = []
    # dict7_ = []
    # dict8_ = []
    # dict9_ = []
    # dict10_ = []
    # for i in range(len(Data['a'])):
    #     dict = {}
    #     for j in range(len(Data['a'][0])):
    #         dict[arr[j]] = Data['a'][i][j]
    #     dict1_.append(dict['QP'])
    # print('\n')
    # print('\n')
    # for i in range(len(data1['a'])):
    #     dict = {}
    #     for j in range(len(data1['a'][0])):
    #         dict[arr[j]] = data1['a'][i][j]
    #     dict2_.append(dict['QP'])
    # 
    # for i in range(len(data2['a'])):
    #     dict = {}
    #     for j in range(len(data2['a'][0])):
    #         dict[arr[j]] = data2['a'][i][j]
    #     dict3_.append(dict['QP'])
    # 
    # for i in range(len(data3['a'])):
    #     dict = {}
    #     for j in range(len(data3['a'][0])):
    #         dict[arr[j]] = data3['a'][i][j]
    #     dict4_.append(dict['QP'])
    # 
    # for i in range(len(data4['a'])):
    #     dict = {}
    #     for j in range(len(data4['a'][0])):
    #         dict[arr[j]] = data4['a'][i][j]
    #     dict5_.append(dict['QP'])
    # 
    # for i in range(len(data5['a'])):
    #     dict = {}
    #     for j in range(len(data5['a'][0])):
    #         dict[arr[j]] = data5['a'][i][j]
    #     dict6_.append(dict['QP'])
    # 
    # for i in range(len(data6['a'])):
    #     dict = {}
    #     for j in range(len(data6['a'][0])):
    #         dict[arr[j]] = data6['a'][i][j]
    #     dict7_.append(dict['QP'])
    # 
    # for i in range(len(data7['a'])):
    #     dict = {}
    #     for j in range(len(data7['a'][0])):
    #         dict[arr[j]] = data7['a'][i][j]
    #     dict8_.append(dict['QP'])
    # 
    # for i in range(len(data8['a'])):
    #     dict = {}
    #     for j in range(len(data8['a'][0])):
    #         dict[arr[j]] = data8['a'][i][j]
    #     dict9_.append(dict['QP'])
    # 
    # for i in range(len(data9['a'])):
    #     dict = {}
    #     for j in range(len(data9['a'][0])):
    #         dict[arr[j]] = data9['a'][i][j]
    #     dict10_.append(dict['QP'])
    # # plt.figure(figsize=(13, 5))
    # plt.plot(dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6, label='Proposed')
    # plt.plot(dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6, label='Emfusion')
    # plt.plot(dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6, label='Dsagan')
    # plt.plot(dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6, label='Msdnet')
    # plt.plot(dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6, label='Ifcnn')
    # plt.plot(dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6, label='Nsst_papcnn')
    # plt.plot(dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6, label='Les')
    # plt.plot(dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6, label='Csr')
    # plt.plot(dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6, label='Nsct')
    # plt.plot(dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='Svt')
    # plt.title('QP', loc='center')
    # # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.2)
    # plt.legend(loc='upper left')
    # plt.savefig('E:/实验数据记录/对比指标图1/MRI-CT/QP.png')
    # plt.show()
    # plt.clf()
    # 
    # dict1_ = []
    # dict2_ = []
    # dict3_ = []
    # dict4_ = []
    # dict5_ = []
    # dict6_ = []
    # dict7_ = []
    # dict8_ = []
    # dict9_ = []
    # dict10_ = []
    # for i in range(len(Data['a'])):
    #     dict = {}
    #     for j in range(len(Data['a'][0])):
    #         dict[arr[j]] = Data['a'][i][j]
    #     dict1_.append(dict['QW'])
    # print('\n')
    # print('\n')
    # for i in range(len(data1['a'])):
    #     dict = {}
    #     for j in range(len(data1['a'][0])):
    #         dict[arr[j]] = data1['a'][i][j]
    #     dict2_.append(dict['QW'])
    # 
    # for i in range(len(data2['a'])):
    #     dict = {}
    #     for j in range(len(data2['a'][0])):
    #         dict[arr[j]] = data2['a'][i][j]
    #     dict3_.append(dict['QW'])
    # 
    # for i in range(len(data3['a'])):
    #     dict = {}
    #     for j in range(len(data3['a'][0])):
    #         dict[arr[j]] = data3['a'][i][j]
    #     dict4_.append(dict['QW'])
    # 
    # for i in range(len(data4['a'])):
    #     dict = {}
    #     for j in range(len(data4['a'][0])):
    #         dict[arr[j]] = data4['a'][i][j]
    #     dict5_.append(dict['QW'])
    # 
    # for i in range(len(data5['a'])):
    #     dict = {}
    #     for j in range(len(data5['a'][0])):
    #         dict[arr[j]] = data5['a'][i][j]
    #     dict6_.append(dict['QW'])
    # 
    # for i in range(len(data6['a'])):
    #     dict = {}
    #     for j in range(len(data6['a'][0])):
    #         dict[arr[j]] = data6['a'][i][j]
    #     dict7_.append(dict['QW'])
    # 
    # for i in range(len(data7['a'])):
    #     dict = {}
    #     for j in range(len(data7['a'][0])):
    #         dict[arr[j]] = data7['a'][i][j]
    #     dict8_.append(dict['QW'])
    # 
    # for i in range(len(data8['a'])):
    #     dict = {}
    #     for j in range(len(data8['a'][0])):
    #         dict[arr[j]] = data8['a'][i][j]
    #     dict9_.append(dict['QW'])
    # 
    # for i in range(len(data9['a'])):
    #     dict = {}
    #     for j in range(len(data9['a'][0])):
    #         dict[arr[j]] = data9['a'][i][j]
    #     dict10_.append(dict['QW'])
    # # plt.figure(figsize=(13, 5))
    # plt.plot(dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6, label='Proposed')
    # plt.plot(dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6, label='Emfusion')
    # plt.plot(dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6, label='Dsagan')
    # plt.plot(dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6, label='Msdnet')
    # plt.plot(dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6, label='Ifcnn')
    # plt.plot(dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6, label='Nsst_papcnn')
    # plt.plot(dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6, label='Les')
    # plt.plot(dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6, label='Csr')
    # plt.plot(dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6, label='Nsct')
    # plt.plot(dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='Svt')
    # plt.title('QW', loc='center')
    # # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.2)
    # plt.legend(loc='upper left')
    # plt.savefig('E:/实验数据记录/对比指标图1/MRI-CT/QW.png')
    # plt.show()
    # plt.clf()
    # 
    # dict1_ = []
    # dict2_ = []
    # dict3_ = []
    # dict4_ = []
    # dict5_ = []
    # dict6_ = []
    # dict7_ = []
    # dict8_ = []
    # dict9_ = []
    # dict10_ = []
    # for i in range(len(Data['a'])):
    #     dict = {}
    #     for j in range(len(Data['a'][0])):
    #         dict[arr[j]] = Data['a'][i][j]
    #     dict1_.append(dict['QE'])
    # print('\n')
    # print('\n')
    # for i in range(len(data1['a'])):
    #     dict = {}
    #     for j in range(len(data1['a'][0])):
    #         dict[arr[j]] = data1['a'][i][j]
    #     dict2_.append(dict['QE'])
    # 
    # for i in range(len(data2['a'])):
    #     dict = {}
    #     for j in range(len(data2['a'][0])):
    #         dict[arr[j]] = data2['a'][i][j]
    #     dict3_.append(dict['QE'])
    # 
    # for i in range(len(data3['a'])):
    #     dict = {}
    #     for j in range(len(data3['a'][0])):
    #         dict[arr[j]] = data3['a'][i][j]
    #     dict4_.append(dict['QE'])
    # 
    # for i in range(len(data4['a'])):
    #     dict = {}
    #     for j in range(len(data4['a'][0])):
    #         dict[arr[j]] = data4['a'][i][j]
    #     dict5_.append(dict['QE'])
    # 
    # for i in range(len(data5['a'])):
    #     dict = {}
    #     for j in range(len(data5['a'][0])):
    #         dict[arr[j]] = data5['a'][i][j]
    #     dict6_.append(dict['QE'])
    # 
    # for i in range(len(data6['a'])):
    #     dict = {}
    #     for j in range(len(data6['a'][0])):
    #         dict[arr[j]] = data6['a'][i][j]
    #     dict7_.append(dict['QE'])
    # 
    # for i in range(len(data7['a'])):
    #     dict = {}
    #     for j in range(len(data7['a'][0])):
    #         dict[arr[j]] = data7['a'][i][j]
    #     dict8_.append(dict['QE'])
    # 
    # for i in range(len(data8['a'])):
    #     dict = {}
    #     for j in range(len(data8['a'][0])):
    #         dict[arr[j]] = data8['a'][i][j]
    #     dict9_.append(dict['QE'])
    # 
    # for i in range(len(data9['a'])):
    #     dict = {}
    #     for j in range(len(data9['a'][0])):
    #         dict[arr[j]] = data9['a'][i][j]
    #     dict10_.append(dict['QE'])
    # # plt.figure(figsize=(13, 5))
    # plt.plot(dict1_, linestyle=':', linewidth=2, color='b', marker='o', markersize=6, label='Proposed')
    # plt.plot(dict2_, linestyle='-.', linewidth=2, color='g', marker='v', markersize=6, label='Emfusion')
    # plt.plot(dict3_, linestyle='--', linewidth=2, color='r', marker='s', markersize=6, label='Dsagan')
    # plt.plot(dict4_, linestyle='-.', linewidth=2, color='c', marker='*', markersize=6, label='Msdnet')
    # plt.plot(dict5_, linestyle=':', linewidth=2, color='m', marker='H', markersize=6, label='Ifcnn')
    # plt.plot(dict6_, linestyle='-.', linewidth=2, color='k', marker='|', markersize=6, label='Nsst_papcnn')
    # plt.plot(dict7_, linestyle='-.', linewidth=2, color='y', marker='x', markersize=6, label='Les')
    # plt.plot(dict8_, linestyle='-.', linewidth=2, color='purple', marker='d', markersize=6, label='Csr')
    # plt.plot(dict9_, linestyle='-.', linewidth=2, color='#DA70D6', marker='+', markersize=6, label='Nsct')
    # plt.plot(dict10_, linestyle='-.', linewidth=2, color='brown', marker='_', markersize=6, label='Svt')
    # plt.title('QE', loc='center')
    # # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.2)
    # plt.legend(loc='upper left')
    # plt.savefig('E:/实验数据记录/对比指标图1/MRI-CT/QE.png')
    # plt.show()
    # plt.clf()
    '''


