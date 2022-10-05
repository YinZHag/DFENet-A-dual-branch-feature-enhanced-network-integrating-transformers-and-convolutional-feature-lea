import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import  argparse
from torch import  nn
from torch.autograd import Variable
import time
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.tensorboard import  SummaryWriter
from  net import Net
from dataSet import Train_dataSet,Eval_dataSet
from torch.optim import lr_scheduler
from pytorch_mssim import msssim
from torch.utils.data import DataLoader
from utils import LEGM
import math
import  copy
import numpy as np
parse=argparse.ArgumentParser(description='Liam zhang')
parse.add_argument('--train_path',type=str,default='D:\ZY\data\Ms_coco train2014\train2014',help='Train PATH')
# parse.add_argument('--train_path',type=str,default='D:/ZY/data/medical_images',help='Train PATH')
parse.add_argument('--eval_mri_path',type=str,default='D:/ZY/data/RFN_Nest_network-trainDatas/MR-PET/MRI',help='eval mri path')
parse.add_argument('--eval_pet_path',type=str,default='D:/ZY/data/RFN_Nest_network-trainDatas/MR-PET/FDG_PET',help='eval pet path')
parse.add_argument('--train_epoch',type=int,default=10,help='train epoch')
parse.add_argument('--train_batchsize',type=int,default=2,help='train batch_size')
parse.add_argument('--lr_rate',type=float,default=0.0005,help='lr_rate')
parse.add_argument('--eval_batchsize',type=int,default=4,help='eval batch_size')
parse.add_argument('--model_path',type=str,default=None,help='pretraining model')
# parse.add_argument('--fusionnet_path',type=str,default=None,help='fusionnet model')
# parse.add_argument('--lr_rate1',type=float,default=0.0005,help='fusionnet learning rate')
# parse.add_argument('--train_epoch1',type=int,default=500,help='fusionnet train epoch')
# parse.add_argument('--train_epoch2',type=int,default=1000,help='DataDistribution train epoch')
# parse.add_argument('--train_batchsize1',type=int,default=2,help='fusionnet train batchsize')

parse.add_argument('--param',type=float,help='the tradeoff between loss',default=0.5)
parse.add_argument('--param1', type=float, help='the tradeoff between loss', default=10)
parse.add_argument('--lrf',type=float,default=0.01)
arg=parse.parse_args()

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

def train_one_epoch(model,optimizer,dataloader,device,epoch,param=0.8):
    model.train()
    mse_loss=nn.L1Loss()
    loss_fea=torch.zeros(1).to(device)
    loss_gradient=torch.zeros(1).to(device)
    loss_all=torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader=tqdm(dataloader,file=sys.stdout)
    #TODO:Second-order Gradient operator
    gradient_operator=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float,device=device,requires_grad=False).view(1,1,3,3).contiguous()
    for step,Data in enumerate(data_loader):
        data,_ = Data
        data=data.to(device)
        data=Variable(data.clone(),requires_grad=False)
        f=model.Transform_Encoder(data)
        output=model.Transform_Decoder(f)
        output=(output-torch.min(output))/(torch.max(output)-torch.min(output))
        loss1=mse_loss(F.conv2d(output,gradient_operator,None,1,1,groups=output.shape[1]),F.conv2d(data,gradient_operator,None,1,1,groups=data.shape[1]))
        loss2=mse_loss(data,output)
        total_loss=param*loss1+loss2
        total_loss.backward()
        loss_fea+=loss2.detach()
        loss_gradient+=loss1.detach()
        loss_all+=total_loss.detach()
        data_loader.desc="[train epochs {}] gradient_loss{:.6f} fea_loss {:.6f}  total_loss {:.6f} ".format(epoch+1,loss_gradient.item()/(step+1),loss_fea.item()/(step+1)
                                                                                                        ,loss_all.item()/(step+1))
        if not torch.isfinite(total_loss):
            print('WARNING: non-finite loss, ending training ', total_loss.item())
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return loss_fea.item()/(step+1),loss_gradient.item()/(step+1),loss_all.item()/(step+1)

"""
def fine_tune(maml,dataloader,device,epoch):
    maml.train()
    data_loader=tqdm(dataloader,file=sys.stdout)
    Loss_all=0
    Mri_loss=0
    Pet_loss=0
    for step,(mri_spt,other_spt,mri_qry,other_qry) in enumerate(dataloader):
        mri_spt,other_spt,mri_qry,other_qry=mri_spt.squeeze(dim=0).to(device),other_spt.squeeze(dim=0).to(device),mri_qry.squeeze(dim=0).to(device),other_qry.squeeze(dim=0).to(device)
        loss_all,mri_loss,pet_loss=maml.finetunning(mri_spt,other_spt,mri_qry,other_qry)
        Loss_all+=loss_all
        Mri_loss+=mri_loss
        Pet_loss+=pet_loss
        print("[finetune epochs {}] total_loss {:.8f} mri_loss {:.8f} pet_loss {:.8f} ".format(
            epoch + 1, Loss_all/ (step + 1),Mri_loss/(step+1),Pet_loss/(step+1)))
        # data_loader.desc = "[finetune epochs {}] total_loss {:.8f} mri_loss {:.8f} pet_loss {:.8f} ".format(
        #     epoch + 1, Loss_all/ (step + 1),Mri_loss/(step+1),Pet_loss/(step+1))
        if np.isnan(loss_all) or np.isinf(loss_all):
            print('WARNING: non-finite loss, ending training ', loss_all)
            sys.exit(1)
    return Loss_all/(step+1),Mri_loss/(step+1),Pet_loss/(step+1)
"""

@torch.no_grad()
def eval_one_epoch(model,dataloader,device,epoch):
    model.eval()
    ssim_loss=msssim
    mse_loss=nn.MSELoss()
    dataloader=tqdm(dataloader,file=sys.stdout)
    loss_sim=torch.zeros(1).to(device)
    loss_psnr=torch.zeros(1).to(device)
    loss_all=torch.zeros(1).to(device)
    for step,data in enumerate(dataloader):
        mri,pet=data
        mri=mri.to(device)
        pet=pet.to(device)
        mri_1= model.Transform_Encoder(mri)
        pet_1= model.Transform_Encoder(pet)
        f=LEGM(mri_1,pet_1)
        output=model.Transform_Decoder(f)
        output=(output-torch.min(output))/(torch.max(output)-torch.min(output))
        loss1=(ssim_loss(mri,output,normalize=True)+ssim_loss(pet,output,normalize=True))/2.
        loss2=mse_loss(output,mri)+mse_loss(output,pet)
        psnr=10*torch.log10(1/loss2.mean())
        loss_sim+=loss1
        loss_psnr+=psnr
        loss_all+=loss1+psnr
        dataloader.desc="[epoch {}] ssim {:.6f} psnr {:.6f} all {:.6f}".format(epoch+1,loss_sim.item()/(step+1),loss_psnr.item()/(step+1),loss_all.item()/(step+1))
    return loss_sim.item()/(step+1),loss_psnr.item()/(step+1),loss_all.item()/(step+1)

if __name__=='__main__':
    print('==>Loading the training and testing datasets')
    train_dataset = Train_dataSet(arg.train_path)
    val_dataset = Eval_dataSet(arg.eval_mri_path, arg.eval_pet_path)
    train_dataloader = DataLoader(train_dataset, batch_size=arg.train_batchsize, shuffle=True, pin_memory=True,
                                  num_workers=4)
    eval_dataloader = DataLoader(val_dataset, batch_size=arg.eval_batchsize, shuffle=False, pin_memory=True,
                                 num_workers=1)
    torch.backends.cudnn.benchmark = True

    print('==>Loading the model')
    model=Net()
    model.apply(_init_vit_weights)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
        torch.cuda.manual_seed(1314)
    else:
        device = torch.device('cpu')
        model.to(device)
    if arg.model_path is not None:
        model.load_state_dict(torch.load(arg.model_path))
        model.eval()
    pr = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(pr, lr=arg.lr_rate, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / arg.train_epoch)) / 2) * (1 - arg.lrf) + arg.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tb_writer = SummaryWriter(log_dir='../log_runs')
    eval_stand = 0
    best_ws = copy.deepcopy(model.state_dict())
    start = time.time()
    for epoch in range(arg.train_epoch):
        train_fea,train_gradient,train_loss_all = train_one_epoch(model, optimizer, train_dataloader,
                                                                                device, epoch)
        scheduler.step()
        eval_ssim, eval_psnr, eval_all = eval_one_epoch(model, eval_dataloader, device, epoch)

        if eval_all > eval_stand:
            best_ws = copy.deepcopy(model.state_dict())
            eval_stand = eval_all

        tags = ['train_fea_loss', 'train_gradient_loss','train_ssim' ,'train_loss_all', 'test_ssim', 'test_psnr',
                'test_all']
        tb_writer.add_scalar(tags[0], train_fea, epoch+1)
        tb_writer.add_scalar(tags[1], train_gradient, epoch+1)
        # tb_writer.add_scalar(tags[2], train_ssim, epoch+1)
        tb_writer.add_scalar(tags[3], train_loss_all, epoch+1)
        tb_writer.add_scalar(tags[5], eval_psnr, epoch+1)
        tb_writer.add_scalar(tags[6], eval_all, epoch+1)
        torch.save(model.state_dict(), '../Checkpoints/' + 'ckpt_' + str(epoch +1) + '_0.5.pth')
    model.load_state_dict(best_ws)
    model.eval()
    torch.save(model.state_dict(), '../Checkpoints/best_model_ws.pth')
    print('training fininshed!!!')
    end = time.time()
    print('total consuming times: {:.4f}s'.format(end-start))









