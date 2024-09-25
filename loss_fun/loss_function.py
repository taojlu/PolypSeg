import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import long
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


# https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
class DiceLoss(nn.Module):
    def __init__(self, smooth=0.001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.act = nn.Sigmoid()

    def forward(self, predict_flat, mask_flat):
        predict_flat = self.act(predict_flat)

        predict_flat = predict_flat.type(torch.FloatTensor)
        mask_flat = mask_flat.type(torch.FloatTensor)
        predict_flat = torch.flatten(predict_flat)
        mask_flat = torch.flatten(mask_flat)

        total = (predict_flat + mask_flat).sum()
        intersection = (predict_flat * mask_flat).sum()
        union = total - intersection

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return torch.clamp((1 - dice).mean(), 0, 1)
        # return 1. - dice


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, in_put, target):
        inputs = torch.flatten(in_put)
        targets = torch.flatten(target)

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        # return F.binary_cross_entropy_with_logits(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return bce_loss


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=0.001):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.act = nn.Sigmoid()

    def forward(self, in_put, target):
        in_put = self.act(in_put)

        inputs = torch.flatten(in_put)
        targets = torch.flatten(target)

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        total = (inputs + targets).sum()
        intersection = (inputs * targets).sum()
        union = total - intersection
        dice_loss_value = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        bce_loss_value = self.bce(inputs, targets)
        # print("bce_loss_value: ", bce_loss_value)

        dice_bce_loss = dice_loss_value + bce_loss_value

        return dice_bce_loss


class CustomizedDiceLoss(nn.Module):
    def __init__(self):
        super(CustomizedDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, in_put, target):
        loss_value = self.dice_loss(in_put, target)
        return torch.log(torch.cosh(loss_value))


class FocalLoss(nn.Module):
    def __init__(self, gamma=10):
        super().__init__()
        self.gamma = gamma

    def forward(self, in_put, target):
        if not (target.size() == in_put.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), in_put.size()))
        max_val = (-in_put).clamp(min=0)
        loss = in_put - in_put * target + max_val + \
               ((-max_val).exp() + (-in_put - max_val).exp()).log()
        inv_probs = F.logsigmoid(-in_put * (target * 2.0 - 1.0))
        loss = (inv_probs * self.gamma).exp() * loss
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.focal = FocalLoss(gamma)

    def forward(self, in_put, target):
        loss = self.alpha * self.focal(in_put, target) + torch.log(self.dice_loss(in_put, target))
        return loss.mean()


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.cls_weights = None

    def forward(self, in_put, target):
        target = target.float()
        loss = F.cross_entropy(in_put, target)
        return loss


class CeDiceLoss(nn.Module):
    def __init__(self):
        super(CeDiceLoss, self).__init__()
        self.ce_loss = CELoss()
        self.dice_loss = DiceLoss()

    def forward(self, in_put, target):
        ce_dice_loss = 0.5 * self.ce_loss(in_put, target) + 0.5 * self.dice_loss(in_put, target)
        return ce_dice_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.8, beta=0.2, smooth=0.001):
        super(TverskyLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)

        return 1 - tversky


class JacardDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=0.001):
        super(JacardDiceLoss, self).__init__()
        self.smooth = smooth
        self.act = nn.Sigmoid()

    def forward(self, inputs, targets, ):
        inputs = self.act(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        Jarcad = (intersection + self.smooth) / (union + self.smooth)
        Jarcad_loss = 1 - Jarcad

        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        jacard_dice = Jarcad_loss * 0.4 + dice_loss * 0.6

        return jacard_dice


# https://github.com/uci-cbcl/RP-Net/blob/4d7b880c758634512c78daaa461799fb96753a25/net/unet.py
class GHMDice(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
            self,
            bins=4,
            momentum=2,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMDice, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label

        # pred = F.sigmoid(pred)
        # print(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        label_weight = label_weight.view(-1)

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        I = (pred * target).sum()
        S = pred.sum() + target.sum()
        g = torch.abs(2 * I / S * pred.detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = 1 - (2 * pred * target * weights).sum() / S
        return loss * self.loss_weight


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, predict, target):
        p0 = torch.flatten(predict)
        g0 = torch.flatten(target)
        # foreground
        num = torch.sum(p0 * g0)
        den = torch.sum(p0) + torch.sum(g0) + 1e-5

        loss_fore = 1 - num / (den + 1e-5)

        # background
        loss_back = - torch.sum((1 - p0) * (1 - g0)) / (torch.sum(1 - p0) + torch.sum(1 - g0) + 1e-5)

        loss = loss_fore + loss_back

        if g0.sum() == 0:
            loss = loss * 0

        return loss


class BinaryDiceLoss2(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss2, self).__init__()

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # print("pred: ", pred.shape)
        # print("target: ", target.shape)
        N, C = pred.shape
        pred = pred.sigmoid()
        losses = []
        alpha = 0.5
        beta = 0.5
        for i in range(C):
            p0 = (pred[:, i]).float()
            g0 = target[:, i]

            # foreground
            num = torch.sum(p0 * g0)
            den = torch.sum(p0) + torch.sum(g0) + 1e-5

            loss_fore = 1 - num / (den + 1e-5)

            # background
            loss_back = - torch.sum((1 - p0) * (1 - g0)) / (torch.sum(1 - p0) + torch.sum(1 - g0) + 1e-5)

            loss = loss_fore + loss_back

            if g0.sum() == 0:
                loss = loss * 0
            # else:
            #     loss = loss / weight[i]

            losses.append(loss)
            # print("losses: ", losses)
        return losses[0]


class GHMDiceLoss(nn.Module):
    def __init__(self, gamma=None):
        super(GHMDiceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predict, target):
        if predict.dim() > 2:
            predict = predict.view(predict.size(0), predict.size(1), -1)  # N,C,H,W => N,C,H*W
            predict = predict.transpose(1, 2)  # N,C,H*W => N,H*W,C
            predict = predict.contiguous().view(-1, predict.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        N, C = predict.shape
        predict = predict.sigmoid()
        losses = []
        for i in range(C):
            p0 = (predict[:, i]).float()
            g0 = target[:, i]

            loss = GHMDice()(p0, g0, torch.ones_like(p0))
            losses.append(loss)
        return losses[0]


class CustomizedGHMDiceLoss(nn.Module):
    def __init__(self):
        super(CustomizedGHMDiceLoss, self).__init__()
        self.ghm_dice_loss = GHMDiceLoss()

    def forward(self, in_put, target):
        loss_value = self.ghm_dice_loss(in_put, target)
        return torch.log(torch.cosh(loss_value))


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, inputs, targets, smooth=0.001, alpha=2., beta=5., gamma=0.75):
        # flatten label and prediction tensors
        inputs = self.act(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


# biao zhun de tu xiang fen ge focal loss
# https://zhuanlan.zhihu.com/p/103426335
class FocalLoss3(nn.Module):
    def __init__(self, gamma=2, alpha=0.9, size_average=True):
        super(FocalLoss3, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, in_put, target):
        if in_put.dim() > 2:
            in_put = in_put.view(in_put.size(0), in_put.size(1), -1)  # N,C,H,W => N,C,H*W
            in_put = in_put.transpose(1, 2)  # N,C,H*W => N,H*W,C
            in_put = in_put.contiguous().view(-1, in_put.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        log_pt = F.log_softmax(in_put)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != in_put.data.type():
                self.alpha = self.alpha.type_as(in_put.data)
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CrossEntropyLossRCF(nn.Module):
    def __init__(self):
        super(CrossEntropyLossRCF, self).__init__()
        self.alpha = 0.5
        self.act = nn.Sigmoid()

    def forward(self, prediction, label):
        prediction = self.act(prediction)

        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduce=False)
        # print("cost: ", cost.shape)
        return torch.sum(cost)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, in_put, target):
        in_put = self.act(in_put)
        # in_put[in_put >= 0.5] = 1
        # in_put[in_put < 0.5] = 0
        label = target.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        print("mask :", mask.shape)
        cost = torch.nn.functional.binary_cross_entropy(
            in_put.float(), label.float(), weight=mask, reduce=False)
        print("cost: ", cost.shape)
        cost_sum = torch.sum(cost)
        print("cost_sum: ", cost_sum)
        print("========")
        loss = torch.log(torch.cosh(cost_sum))
        print("loss", loss)
        return loss


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DiceCrossEntropyLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, in_put, target):
        dice_loss = self.dice_loss(in_put, target)

        in_put[in_put >= 0.5] = 1
        in_put[in_put < 0.5] = 0
        label = target.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy(
            in_put.float(), label.float(), weight=mask, reduce=False)
        cost_sum = torch.sum(cost)

        loss = dice_loss + cost_sum
        return torch.log(torch.cosh(loss))


class FocalLoss4(nn.Module):
    def __init__(self):
        super(FocalLoss4, self).__init__()
        self.alpha = 0.25
        self.gamma = 2.
        self.mask = None

    def forward(self, in_put, targets):
        p = torch.sigmoid(in_put)
        ce_loss = F.binary_cross_entropy_with_logits(
            in_put, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.mask is not None:
            loss = torch.einsum("bfn,bf->bfn", loss, self.mask)

        return loss


class StructureLoss(nn.Module):
    """
    https://github.com/AngeLouCN/CaraNet/blob/main/Train.py
    """

    def __init__(self):
        super(StructureLoss, self).__init__()
        self.act = nn.Sigmoid()
        # self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        # print("mask: ", mask.dtype)
        mask = mask.type(torch.cuda.FloatTensor)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # dice_loss = self.dice_loss(pred, mask)

        pred = self.act(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()
        # return (wbce + wiou + dice_loss).mean()


def som(loss, ratio):
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)

    top_loss, _ = loss.reshape(-1).topk(num_hns, largest=True)
    loss_mask = (top_loss != 0)
    loss = torch.sum(top_loss[loss_mask]) / (torch.sum(loss_mask) + 1e-6)
    return loss


class StructureLossSom(nn.Module):
    """
    https://github.com/AngeLouCN/CaraNet/blob/main/Train.py
    """

    def __init__(self):
        super(StructureLossSom, self).__init__()
        self.act = nn.Sigmoid()
        # self.dice_loss = DiceLoss()
        self.ratio = 0.8

    def forward(self, pred, mask):
        # print("mask: ", mask.dtype)
        mask = mask.type(torch.cuda.FloatTensor)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # dice_loss = self.dice_loss(pred, mask)

        pred = self.act(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        losses = wbce + wiou
        loss = som(losses, self.ratio)
        return loss


class OhemCELoss(nn.Module):
    """
    https://blog.csdn.net/chen1234520nnn/article/details/122812038
    """

    def __init__(self, thresh=0.003, ignore_lb=255, ohem_ratio=0.7, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

        self.ohem_ratio = ohem_ratio
        self.ignore_lb = ignore_lb
        self.criteria = DiceLoss()
        self.act = nn.Sigmoid()

    def forward(self, pred, mask):
        h, w = pred.shape[2], pred.shape[3]
        n_min = int(h * w * self.ohem_ratio)

        mask = mask.type(torch.cuda.FloatTensor)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

        wbce = (weit * wbce) / weit
        # print("wbce: ", wbce)

        loss = wbce.view(-1)
        loss, _ = torch.sort(loss, descending=True)  # 排序
        # print("loss: ", loss[0: 5])
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:n_min]

        loss_1 = torch.mean(loss)

        pred = self.act(pred)
        dice_loss = self.criteria(pred, mask)

        loss = loss_1 + dice_loss

        return loss


class OhemFocalLoss(nn.Module):
    """
    https://blog.csdn.net/CaiDaoqing/article/details/90457197
    """

    def __init__(self, alpha=2, gamma=4, OHEM_percent=0.5):
        super(OhemFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.OHEM_percent = OHEM_percent

    def forward(self, output, target):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = self.alpha * (invprobs * self.gamma).exp() * loss

        OHEM, _ = focal_loss.topk(k=int(self.OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()


class BoundaryBCELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.act = nn.Sigmoid()
        self.ignore_index = ignore_index

    def forward(self, edge, target, boundary):
        # print("target: ", target.shape)
        # print("boundary: ", boundary.shape)

        # edge = edge.squeeze(dim=1)
        # mask = target != self.ignore_index
        mask = target
        # print("mask: ", mask)
        pos_mask = (boundary == 1.0) & mask

        neg_mask = (boundary == 0.0) & mask
        num = torch.clamp(mask.sum(), min=1)
        pos_weight = neg_mask.sum() / num
        # print("pos_weight: ", type(pos_weight))
        neg_weight = pos_mask.sum() / num

        weight = torch.zeros_like(boundary)
        weight = weight.type(torch.FloatTensor)
        weight[pos_mask] = pos_weight
        weight[neg_mask] = neg_weight

        edge = edge.type(torch.FloatTensor)
        edge = self.act(edge)
        # print(edge)
        boundary = boundary.type(torch.FloatTensor)
        # print(boundary)

        loss = F.binary_cross_entropy(edge, boundary, weight, reduction='sum') / num
        # print("loss: ", loss)
        return loss


class ActiveContourLoss(nn.Module):
    def __init__(self):
        super(ActiveContourLoss, self).__init__()
        self.weight = 10

    def forward(self, y_pred, y_true):
        '''
        y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        # region term
        C_in = torch.ones_like(y_pred)
        C_out = torch.zeros_like(y_pred)

        region_in = torch.mean(y_pred * (y_true - C_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - y_pred) * (y_true - C_out) ** 2)
        region = region_in + region_out

        loss = self.weight * lenth + region

        return loss


class CombinedFocal_Dice_Loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, beta=1.5):
        """
        Combined loss function to reduce false positives.

        Parameters:
        - gamma: focusing parameter for Focal Loss.
        - alpha: weight for positive class in Focal Loss.
        - beta: weight for false positives.
        """
        super(CombinedFocal_Dice_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        # Focal Loss
        prob = torch.sigmoid(output)
        focal_weight = torch.where(target == 1, self.alpha * (1 - prob), (1 - self.alpha) * prob)
        focal_loss = -focal_weight * torch.pow((1 - prob), self.gamma) * target * torch.log(prob) - \
                     focal_weight * torch.pow(prob, self.gamma) * (1 - target) * torch.log(1 - prob)

        # Dice Loss
        intersection = torch.sum(prob * target)
        dice_loss = 1 - (2 * intersection + 1) / (torch.sum(prob) + torch.sum(target) + 1)

        # False Positive Penalty
        fp_penalty = self.beta * torch.sum((1 - target) * prob)

        # Combine losses
        combined_loss = focal_loss.mean() + dice_loss + fp_penalty

        return combined_loss


class StructureFalsePositiveLoss(nn.Module):
    """
    https://github.com/AngeLouCN/CaraNet/blob/main/Train.py
    """

    def __init__(self):
        super(StructureFalsePositiveLoss, self).__init__()
        self.act = nn.Sigmoid()
        # self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        # print("mask: ", mask.dtype)
        mask = mask.type(torch.cuda.FloatTensor)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # dice_loss = self.dice_loss(pred, mask)

        y_pred = self.act(pred)
        inter = ((y_pred * mask) * weit).sum(dim=(2, 3))
        union = ((y_pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        # 计算假阳性像素的损失
        fp_loss = torch.mean((y_pred - mask) ** 2 * (1 - mask))

        return (wbce + wiou).mean() + fp_loss


if __name__ == "__main__":
    # Example usage:
    model = CombinedFocal_Dice_Loss()
    output = torch.randn(1, 1, 256, 256)  # Random logits
    target = torch.randint(0, 2, (1, 1, 256, 256))  # Random binary target
    loss = model(output, target)
    pdb.set_trace()
    print(loss)
