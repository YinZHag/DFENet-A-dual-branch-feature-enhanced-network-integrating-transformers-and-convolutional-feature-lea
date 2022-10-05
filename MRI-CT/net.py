import torch
from torch import nn
from functools import partial
from collections import OrderedDict
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256,patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size,patch_size), stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        # [batch_size, num_patches , total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches , 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches , 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches , embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches , embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches ]
        # @: multiply -> [batch_size, num_heads, num_patches , num_patches ]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches , embed_dim_per_head]
        # transpose: -> [batch_size, num_patches , num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches , total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    #x的shape为(batch_size ,num_patches,total_embed_dim->chaneels*patch_size*patch_size)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Multiscale_VisionTransformer(nn.Module):
    def __init__(self, img_size=256,in_c=1,embed_dim=32, depth=[3,4,6,3], num_heads=[1,2,4,8],mlp_ratios=[4,4,4,4], embeding_dims=[64,128,256,512],qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Multiscale_VisionTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.depths=depth
        #patch_embedding
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_c,
                                             embed_dim=embeding_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embeding_dims[0],
                                              embed_dim=embeding_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embeding_dims[1],
                                              embed_dim=embeding_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embeding_dims[2],
                                              embed_dim=embeding_dims[3])
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio,sum(depth))]  # stochastic depth decay rule
        cur=0
        self.block1 = nn.ModuleList([Block(
            dim=embeding_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[cur + i], norm_layer=norm_layer,act_layer=act_layer)
            for i in range(self.depths[0])])
        self.norm1 = norm_layer(embeding_dims[0])

        cur += self.depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embeding_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[cur + i], norm_layer=norm_layer,act_layer=act_layer)
            for i in range(self.depths[1])])
        self.norm2 = norm_layer(embeding_dims[1])

        cur += self.depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embeding_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[cur + i], norm_layer=norm_layer,act_layer=act_layer)
            for i in range(self.depths[2])])
        self.norm3 = norm_layer(embeding_dims[2])

        cur += self.depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embeding_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[cur + i], norm_layer=norm_layer,act_layer=act_layer)
            for i in range(self.depths[3])])
        self.norm4 = norm_layer(embeding_dims[3])

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
    def forward_features(self, x):
        assert x.size()[1]==1 and x.size()[2]==256 and x.size()[3]==256,"image x.size must be 256*256 and 1 channel"
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        outputs=[]
        outputs.append(x)
        B=x.shape[0]
        x,h1,w1= self.patch_embed1(x)  # [B,h1*w1,64]
        for i, blk in enumerate(self.block1):
            x = blk(x)
        x = self.norm1(x)
        x = x.reshape(B, h1, w1, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, h2, w2 = self.patch_embed2(x)  # [B,h2*w2,128]
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        x = x.reshape(B, h2, w2, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, h3, w3 = self.patch_embed3(x)  # [B,h2*w2,128]
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        x = x.reshape(B, h3, w3, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, h4, w4 = self.patch_embed4(x)  # [B,h2*w2,128]
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        x = x.reshape(B, h4, w4, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        if self.dist_token is None:
            outputs[1],outputs[2],outputs[3],outputs[4]=self.pre_logits(outputs[1][:,:,:,:]),self.pre_logits(outputs[2][:,:,:,:]),self.pre_logits(outputs[3][:,:,:,:]),self.pre_logits(outputs[4][:,:,:,:])
            return outputs
        else:
            return None
    def forward(self, x):
        #x的shape为: 【【batch_size，c,h,w】,【batch_size，c1,h1,w1】，【batch_size，c2,h2,w2】,【batch_size，c3,h3,w3】,【batch_size，c4,h4,w4】】
        x = self.forward_features(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride,is_last=False):
        super(ConvLayer, self).__init__()
        padding_size=int(kernel_size//2)
        self.padding=nn.ReflectionPad2d(padding_size)
        self.conv=nn.Conv2d(in_c,out_c,kernel_size,stride)
        self.prelu=nn.PReLU()
        self.is_last=is_last
    def forward(self,x):
        x=self.padding(x)
        x=self.conv(x)
        if self.is_last==False:
            x=self.prelu(x)
        return x

#TODO: dilated Conv
class ConvLayer1(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride,dilation,is_last=False):
        super(ConvLayer1, self).__init__()
        padding_size=int((kernel_size+(dilation-1)*(kernel_size-1))//2)
        self.padding=nn.ReflectionPad2d(padding_size)
        self.conv=nn.Conv2d(in_c,out_c,kernel_size,stride,dilation=dilation)
        self.prelu=nn.PReLU()
        self.is_last=is_last
    def forward(self,x):
        x=self.padding(x)
        x=self.conv(x)
        if self.is_last==False:
            x=self.prelu(x)
        return x

#CBAM(Convlution block attention module)
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
        #sigmoid函数会自动保留sigmoid前的tensor的shape
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

#Res2Net's bottle neck layers seem like the residual building layer
class Bottle2neck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        #输入的通道数必须大于等于3，否则width==0
        width = int(np.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class res2net(nn.Module):
    def __init__(self,bottle2neck,inplanes,planes,layers,baseWidth=26,scale=4):
        super(res2net, self).__init__()
        self.baseWidth=baseWidth
        self.inplanes=inplanes
        self.planes=planes
        self.scale=scale
        self.layer1=self._make_layer(bottle2neck,self.planes,layers)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.PReLU(),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.layer1(x)
        return x
#CNN Encoder
class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.conv=nn.Conv2d(1,32,3,1,1)
        self.prelu=nn.PReLU()
        self.dilate_conv1=ConvLayer1(32,64,1,1,2)
        self.dilate_conv2=ConvLayer1(32,96,3,1,2)
        self.dilate_conv3=ConvLayer1(32,128,5,1,2)
        self.res2block1 = res2net(Bottle2neck,64,64,4)
        self.res2block2 = res2net(Bottle2neck,96,96,4)
        self.res2block3 = res2net(Bottle2neck,128,128,4)
    def forward(self,x):
        x_conv=self.conv(x)
        x_prelu=self.prelu(x_conv)
        x_1=self.dilate_conv1(x_prelu)
        x_2=self.dilate_conv2(x_prelu)
        x_3=self.dilate_conv3(x_prelu)
        x_1=self.res2block1(x_1)
        x_2=self.res2block2(x_2)
        x_3=self.res2block3(x_3)
        out=torch.cat((x_1,x_2,x_3),1)
        out=torch.mul(x,out)+out
        return out

#Transoformer Encoder
class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.vit=Multiscale_VisionTransformer()
        self.up1=nn.Upsample(scale_factor=2,mode='bicubic',align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)
        self.conv4_3=ConvLayer(512,256,3,1)
        self.conv4_2 = ConvLayer(512, 128, 3, 1)
        self.conv4_1 = ConvLayer(512, 64, 3, 1)
        self.conv3_2=ConvLayer(512,128,3,1)
        self.conv3_1 = ConvLayer(512,64, 3, 1)
        self.conv3=ConvLayer(512,512,1,1)
        self.conv2_1=ConvLayer(384,64,3,1)
        self.conv2=ConvLayer(384,384,1,1)
        self.conv1=ConvLayer(256,256,3,1)
        self.conv=ConvLayer(256,64,3,1)
        self.cbam=CBAMLayer(64)
    def forward(self,x):
        l1, l2, l3, l4, l5 = self.vit(x)
        print(l1.shape,l2.shape,l3.shape,l4.shape,l5.shape)
        db4_3=self.conv4_3(l5)
        db4_2 = self.conv4_2(l5)
        db4_1 = self.conv4_1(l5)
        db3=self.conv3(torch.cat((l4,self.up1(db4_3)),dim=1))
        db3_2=self.conv3_2(db3)
        db3_1=self.conv3_1(db3)
        db2=self.conv2(torch.cat((l3,self.up2(db4_2),self.up1(db3_2)),dim=1))
        db2_1=self.conv2_1(db2)
        db1=self.up2(self.conv1(torch.cat((l2,self.up3(db4_1),self.up2(db3_1),self.up1(db2_1)),dim=1)))
        db1=torch.mul(l1,db1)+db1
        output=self.conv(db1)
        output=self.cbam(output)
        return output

class transorm_encoder(nn.Module):
    def __init__(self):
        super(transorm_encoder, self).__init__()
        self.net2=net2()
        self.net3=net3()
        self.cbam=CBAMLayer(352)
    def forward(self,x):
        cnn_features=self.net2(x)
        transformer_features=self.net3(x)
        result=torch.cat([cnn_features,transformer_features],dim=1)
        result=self.cbam(result)
        return result

class transform_decoder(nn.Module):
    def __init__(self):
        super(transform_decoder, self).__init__()
        self.conv1=ConvLayer1(352,64,3,1,2)
        self.conv2=ConvLayer1(64,64,3,1,2)
        self.conv3=ConvLayer1(128,64,3,1,2)
        self.conv4=ConvLayer1(128,64,3,1,2)
        self.conv5=ConvLayer(64,1,3,1,True)
    def forward(self,x):
        db1=self.conv1(x)
        db2=self.conv2(db1)
        db3=self.conv3(torch.cat([db1,db2],dim=1))
        db4=self.conv4(torch.cat([db2,db3],dim=1))
        output=self.conv5(db4)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder=transorm_encoder()
        self.decoder=transform_decoder()
    def Transform_Encoder(self,x):
        return self.encoder(x)
    def Transform_Decoder(self,x):
        return self.decoder(x)


