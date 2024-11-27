import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import init as init

# from .UNet import *
# from .Resnet import *
# from .UNet import Unet
from .Resnet import ResNet

# modified from DSU.py https://github.com/lixiaotong97/DSU/blob/main/dsu.py


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x

        return x, w


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

    return f_cor, B


# Calcualate Cross Covarianc of two feature maps
# reference : https://github.com/shachoi/RobustNet
def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape

    B, C, H, W = f_map1.shape
    HW = H * W

    if eye is None:
        eye = torch.eye(C).cuda()

    # feature map shape : (B,C,H,W) -> (B,C,HW)
    f_map1 = f_map1.contiguous().view(B, C, -1)
    f_map2 = f_map2.contiguous().view(B, C, -1)

    # f_cor shape : (B, C, C)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW - 1) + (eps * eye)

    return f_cor, B


def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape

    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).cuda()

    # get diagonal values of covariance matrix
    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).cuda()
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    diag_loss = diag_loss / B

    return diag_loss


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Unet(nn.Module):

    def __init__(self, args, in_ch=2, out_ch=4):
        print('22222222', in_ch, out_ch)
        super(Unet, self).__init__()

        self.args = args
        n1 = 64
        filters = [64, 128, 256, 512, 1024]

        # self.Pad = nn.ConstantPad2d((92, 92, 92, 92), 0)
        self.Pad = nn.ConstantPad2d((94, 94, 94, 94), 0)  # xiugai 0505

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.IN = nn.InstanceNorm2d(1024)

        self.Up4 = up_conv(filters[4], 4)
        self.Up_conv4 = conv_block(516, filters[3])

        self.Up3 = up_conv(filters[3], 4)
        self.Up_conv3 = conv_block(260, filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        # x = tensor_list.tensors
        x = tensor_list

        x1 = self.Pad(x)
        e1 = self.Conv1(x1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        q_1 = self.IN(e5)

        d4 = self.Up4(e5)
        # e4_cropped = e4[:,:,4:38,4:38]#4
        e4_cropped = e4[:, :, 5:37, 5:37]  # xiugai 0505
        d4 = torch.cat((d4, e4_cropped), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # e3_cropped = e3[:,:,16:76,16:76]#16
        e3_cropped = e3[:, :, 18:74, 18:74]  # xiugai 0505
        d3 = torch.cat((d3, e3_cropped), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # e2_cropped = e2[:,:,40:152,40:152]#40
        e2_cropped = e2[:, :, 42:146, 42:146]  # xiugai 0505
        d2 = torch.cat((d2, e2_cropped), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        # e1_cropped = e1[:,:,88:304,88:304]#88
        e1_cropped = e1[:, :, 92:292, 92:292]  # xiugai 0505
        d1 = torch.cat((d1, e1_cropped), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        if self.args.model == 'Unet':
            out = self.active(norm_out)
        else:
            out = norm_out

        return out, q_1


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def DSU_make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
        layers.append(DistributionUncertainty())

    return nn.Sequential(*layers)


def default_conv(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, strides,
        padding=(kernel_size // 2), bias=bias)
class ResBlock(nn.Module):
    def __init__(
            self, conv=default_conv, n_feats=64, kernel_size=3,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class DSU_ResNet(nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=5,
                 bn=False):
        super(DSU_ResNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.pertubration_first = DistributionUncertainty()
        self.body = DSU_make_layer(ResBlock, num_block, n_feats=num_feat, bn=bn)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.pertubration_last = DistributionUncertainty()

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_last], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.pertubration_first(self.conv_first(x)))
        out = self.body(feat)
        out = self.pertubration_last(self.conv_last(self.lrelu(out)))
        out += x
        return out


class DSU_Unet(nn.Module):

    def __init__(self, args, in_ch=3, out_ch=6):
        super(DSU_Unet, self).__init__()

        self.args = args
        n1 = 64
        filters = [64, 128, 256, 512, 1024]

        # self.Pad = nn.ConstantPad2d((92, 92, 92, 92), 0)
        self.Pad = nn.ConstantPad2d((94, 94, 94, 94), 0)  # DSU0918

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.pertubration1 = DistributionUncertainty()
        self.pertubration2 = DistributionUncertainty()
        self.pertubration3 = DistributionUncertainty()
        self.pertubration4 = DistributionUncertainty()
        self.pertubration5 = DistributionUncertainty()

        self.up_pertubration1 = DistributionUncertainty()
        self.up_pertubration2 = DistributionUncertainty()
        self.up_pertubration3 = DistributionUncertainty()
        self.up_pertubration4 = DistributionUncertainty()


        self.IN= nn.InstanceNorm2d(1024)

        self.Up4 = up_conv(filters[4], 4)
        self.Up_conv4 = conv_block(516, filters[3])

        self.Up3 = up_conv(filters[3], 4)
        self.Up_conv3 = conv_block(260, filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        # x = tensor_list.tensors
        x = tensor_list

        x1 = self.Pad(x)
        e1 = self.pertubration1(self.Conv1(x1))

        e2 = self.Maxpool1(e1)
        e2 = self.pertubration2(self.Conv2(e2))

        e3 = self.Maxpool2(e2)
        e3 = self.pertubration3(self.Conv3(e3))

        e4 = self.Maxpool3(e3)
        e4 = self.pertubration4(self.Conv4(e4))

        e5 = self.Maxpool4(e4)
        e5 = self.pertubration5(self.Conv5(e5))

        q_1 = self.IN(e5)

        d4 = self.Up4(e5)
        # e4_cropped = e4[:, :, 4:38, 4:38]  # 4
        e4_cropped = e4[:, :, 5:37, 5:37]  # xiugai 0505
        d4 = torch.cat((d4, e4_cropped), dim=1)
        d4 = self.up_pertubration4(self.Up_conv4(d4))

        d3 = self.Up3(d4)
        # e3_cropped = e3[:, :, 16:76, 16:76]  # 16
        e3_cropped = e3[:, :, 18:74, 18:74]  # xiugai 0505
        d3 = torch.cat((d3, e3_cropped), dim=1)
        d3 = self.up_pertubration3(self.Up_conv3(d3))

        d2 = self.Up2(d3)
        # e2_cropped = e2[:, :, 40:152, 40:152]  # 40
        e2_cropped = e2[:, :, 42:146, 42:146]  # xiugai 0505
        d2 = torch.cat((d2, e2_cropped), dim=1)
        d2 = self.up_pertubration2(self.Up_conv2(d2))

        d1 = self.Up1(d2)
        # e1_cropped = e1[:, :, 88:304, 88:304]  # 88
        e1_cropped = e1[:, :, 92:292, 92:292]  # xiugai 0505
        d1 = torch.cat((d1, e1_cropped), dim=1)
        d1 = self.up_pertubration1(self.Up_conv1(d1))

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        if self.args.model == 'Unet':
            out = self.active(norm_out)
        else:
            out = norm_out

        return out, q_1


class SMCAnet(nn.Module):
    def __init__(self, args):
        super(SMCAnet, self).__init__()

        self.args = args
        # self.num_classes = args.num_classes
        self.num_classes = 6

        # reconstruct clean image x and infer noise
        # self.res_clean = DSU_ResNet(num_out_ch = 1)
        # self.res_noise = DSU_ResNet(num_out_ch = 1, num_block=6, bn=True)
        # pred mu and log var unit for seg_masks: B x K x W x H
        # self.unet = DSU_Unet(args, out_ch=1 * self.num_classes)

        # self.DSU_ResNet = DSU_ResNet(num_in_ch=3, num_out_ch=3)  #2222222
        self.unet = Unet(args, in_ch=3, out_ch=6)
        self.pertubration = DistributionUncertainty()

        # postprecess
        self.softmax = nn.Softmax(dim=1)

    def forward(self, samples: torch.Tensor, task):
        # x = self.res_clean(samples)
        B, C, H, W = samples.shape
        aug_samples = self.pertubration(samples)
        pred_q, q = self.unet(samples)
        pred_k, k = self.unet(aug_samples)
        out = {'pred_masks_q': self.softmax(pred_q), 'feat_q':q, 'pred_masks_k': self.softmax(pred_k), 'feat_k':k}
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for SMCAnet.
    """

    def __init__(self, losses, weight_dict, args):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.args = args

    def loss_AvgDice(self, outputs, targets):
        src_masks, src_masks_aug = outputs["pred_masks_q"],outputs["pred_masks_k"]
        src_masks = src_masks.argmax(1)
        src_masks_aug = src_masks_aug.argmax(1)
        targets_masks = targets.argmax(1)
        avg_dice = 0
        for i in range(1, 6, 1):
            dice1 = (2 * torch.sum((src_masks == i) * (targets_masks == i), (1, 2)).float()) / (
                        torch.sum(src_masks == i, (1, 2)).float() + torch.sum(targets_masks == i,
                                                                              (1, 2)).float() + 1e-10)
            dice2 = (2 * torch.sum((src_masks_aug == i) * (targets_masks == i), (1, 2)).float()) / (
                    torch.sum(src_masks_aug == i, (1, 2)).float() + torch.sum(targets_masks == i,
                                                                          (1, 2)).float() + 1e-10)
            avg_dice += (dice1.mean() + dice2.mean())/2
        return {"loss_AvgDice": avg_dice / 5}

    def loss_CrossEntropy(self, outputs, targets, eps=1e-12):
        src_masks, src_masks_aug = outputs["pred_masks_q"],outputs["pred_masks_k"]
        y_labeled = targets[:, 0:6, :, :]
        cross_entropy = -torch.sum(y_labeled * torch.log(src_masks + eps), dim=1)
        cross_entropy_aug = -torch.sum(y_labeled * torch.log(src_masks_aug + eps), dim=1)
        # criterion = nn.BCEWithLogitsLoss(reduction='none')
        # raw_loss = criterion(src_masks, y_labeled)
        losses = {
            "loss_CrossEntropy": cross_entropy.mean() + cross_entropy_aug.mean(),
        }
        return losses


    def loss_Inclusive(self, outputs, targets):
        scar_edema = outputs["pred_masks_q"]
        scar = scar_edema[:, 4:5, ...]
        edema = scar_edema[:, 4:5, ...] + scar_edema[:, 5:6, ...]

        gd_scar = targets[:, 4:5, ...]
        gd_edema = targets[:, 4:5, ...] + targets[:, 5:6, ...]

        gd_myo = targets[:, 2:3, ...]
        constraint_loss_myo_scar = -1 * (1 - gd_myo) * torch.log((1 - scar) + 1e-10)
        constraint_loss_myo_scar = constraint_loss_myo_scar.sum() / ((1 - gd_myo).sum() + 1e-10)
        constraint_loss_myo_edema = -1 * (1 - gd_myo) * torch.log((1 - edema) + 1e-10)
        constraint_loss_myo_edema = constraint_loss_myo_edema.sum() / ((1 - gd_myo).sum() + 1e-10)
        # (self, scar, gd_edema)
        inclusive_loss_scar = -1 * (1 - gd_edema) * torch.log((1 - scar) + 1e-10)
        inclusive_loss_scar = inclusive_loss_scar.sum() / ((1 - gd_edema).sum() + 1e-10)
        # (self, edema, gd_scar)
        inclusive_loss_edema = -1 * gd_scar * torch.log(edema + 1e-10)
        inclusive_loss_edema = inclusive_loss_edema.sum() / (gd_scar.sum() + 1e-10)

        # return {"loss_Inclusive": 0.1 * inclusive_loss_scar + 0.1 * inclusive_loss_edema + 0.8 * constraint_loss_myo_edema + 0.1 * constraint_loss_myo_scar}
        return {"loss_Inclusive": 0.2 * inclusive_loss_scar + 0.2 * inclusive_loss_edema + 1.0 * constraint_loss_myo_edema + 0.1 * constraint_loss_myo_scar}
        # return {"loss_Inclusive": 1.0 * constraint_loss_myo_edema + 0.5 * constraint_loss_myo_scar}

    def loss_Inclusive_aug(self, outputs, targets):
        scar_edema = outputs["pred_masks_k"]
        scar = scar_edema[:, 4:5, ...]
        edema = scar_edema[:, 4:5, ...] + scar_edema[:, 5:6, ...]

        gd_scar = targets[:, 4:5, ...]
        gd_edema = targets[:, 4:5, ...] + targets[:, 5:6, ...]

        gd_myo = targets[:, 2:3, ...]
        constraint_loss_myo_scar = -1 * (1 - gd_myo) * torch.log((1 - scar) + 1e-10)
        constraint_loss_myo_scar = constraint_loss_myo_scar.sum() / ((1 - gd_myo).sum() + 1e-10)
        constraint_loss_myo_edema = -1 * (1 - gd_myo) * torch.log((1 - edema) + 1e-10)
        constraint_loss_myo_edema = constraint_loss_myo_edema.sum() / ((1 - gd_myo).sum() + 1e-10)
        # (self, scar, gd_edema)
        inclusive_loss_scar = -1 * (1 - gd_edema) * torch.log((1 - scar) + 1e-10)
        inclusive_loss_scar = inclusive_loss_scar.sum() / ((1 - gd_edema).sum() + 1e-10)
        # (self, edema, gd_scar)
        inclusive_loss_edema = -1 * gd_scar * torch.log(edema + 1e-10)
        inclusive_loss_edema = inclusive_loss_edema.sum() / (gd_scar.sum() + 1e-10)

        # return {"loss_Inclusive": 0.1 * inclusive_loss_scar + 0.1 * inclusive_loss_edema + 0.8 * constraint_loss_myo_edema + 0.1 * constraint_loss_myo_scar}
        return {"loss_Inclusive": 0.2 * inclusive_loss_scar + 0.2 * inclusive_loss_edema + 1.0 * constraint_loss_myo_edema + 0.1 * constraint_loss_myo_scar}

    def loss_CA(self, outputs, targets, eps=1e-12):
        k_maps, q_maps = outputs["feat_k"],outputs["feat_q"]
        # detach original images
        # k_maps = k_maps.detach()d

        k_cor, _ = get_covariance_matrix(k_maps)
        q_cor, _ = get_covariance_matrix(q_maps)
        cov_loss = F.mse_loss(k_cor, q_cor)  # LCM
        crosscov_loss = cross_whitening_loss(k_maps, q_maps)  # LCC
        CML_CCL = cov_loss + crosscov_loss
        # CML_CCL = cov_loss
        # CML_CCL = crosscov_loss

        # if k_maps.shape[0] == q_maps.shape[0]:
        #     # k_cor, _ = get_covariance_matrix(k_maps)
        #     # q_cor, _ = get_covariance_matrix(q_maps)
        #     # cov_loss = F.mse_loss(k_cor, q_cor)
        #     crosscov_loss = cross_whitening_loss(k_maps, q_maps)
        #     # CML_CCL = cov_loss + crosscov_loss
        #     CML_CCL =  crosscov_loss
        #     # print('aaaaaaaaaa')
        #
        # else:
        #     # k_cor, _ = get_covariance_matrix(k_maps[:q_maps.shape[0],...])
        #     # q_cor, _ = get_covariance_matrix(q_maps)
        #     # cov_loss = F.mse_loss(k_cor, q_cor)
        #     crosscov_loss = cross_whitening_loss(k_maps[:q_maps.shape[0],...], q_maps)
        #     # CML_CCL = cov_loss + crosscov_loss
        #     CML_CCL =  crosscov_loss
        #     # print('bbbbbbbbbbbbbbbbb')


        losses = {
            "loss_CA": CML_CCL.mean(),
        }

        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {'CrossEntropy': self.loss_CrossEntropy,
                    'AvgDice': self.loss_AvgDice,
                    # 'Inclusive': self.loss_Inclusive,
                    'CA': self.loss_CA,
                    }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
        return results


class Visualization(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs, 'inputs', epoch, writer)
        self.save_image(outputs.float(), 'outputs', epoch, writer)
        self.save_image(labels.float(), 'labels', epoch, writer)
        self.save_image(others['recon'].float(), 'recon', epoch, writer)
        self.save_image(others['noise'].float(), 'noise', epoch, writer)
        self.save_image(others['logit'].float(), 'logit', epoch, writer)
        self.save_image(others['lines'].float(), 'lines', epoch, writer)
        self.save_image(others['contour'].float(), 'contour', epoch, writer)


def build(args):
    device = torch.device(args.device)
    model = SMCAnet(args)
    weight_dict = {
        'loss_CrossEntropy': args.CrossEntropy_loss_coef,
        'loss_Inclusive': args.Inclusive_loss_coef,
        # 'loss_AvgDice': args.AvgDice_loss_coef,
        # 'loss_Bayes':args.Bayes_loss_coef,
    }
    # losses = ['CrossEntropy', 'AvgDice', 'Inclusive', 'CA']
    losses = ['CrossEntropy', 'AvgDice', 'CA']  # ab
    # losses = ['CrossEntropy', 'AvgDice', 'Inclusive']  # ab
    criterion = SetCriterion(losses=losses, weight_dict=weight_dict, args=args)
    criterion.to(device)
    visualizer = Visualization()

    return model, criterion, visualizer


