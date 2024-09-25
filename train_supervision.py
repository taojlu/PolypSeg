import pdb

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
# from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.load_model import loading_mode
import torch.nn.functional as F
import numpy as np
import logging
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from utils.save_logs import SaveModelParam, SaveTrainingMetrics

from loss_fun.loss import structure_loss
import loss_fun.loss as loss_arch


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        if opt.deep_supervision:
            res = res[0]+res[1]+res[2]+res[-1]
        else:
            res = res

        # metrics Dice
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1


def train(train_loader, model, criterion_loss, optimizer, epoch, test_path):
    scaler = GradScaler()
    model.train()
    global best
    # size_rates = [0.75, 1, 1.25]
    size_rates = [1]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            pdb.set_trace()
            # pdb.set_trace()
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            with autocast():
                # ---- forward ----
                P2 = model(images)
                # P1, P2 = model(images)
                # ---- loss function ----
                # loss_P1 = structure_loss(P1, gts)
                # pdb.set_trace()
                if opt.deep_supervision:
                    loss_list = []
                    for i in range(len(P2)):
                        # print("深监督")?\
                        loss_i = criterion_loss(P2[i], gts)
                        loss_list.append(loss_i)
                    loss_P2 = sum(loss_list) / len(loss_list)
                else:
                    loss_P2 = structure_loss(P2, gts)
                # loss = loss_P2
            # ---- backward ----
            # loss.backward()
            # clip_gradient(optimizer, opt.clip)
            # optimizer.step()
            scaler.scale(loss_P2).backward()
            clip_gradient(optimizer, opt.clip)
            scaler.step(optimizer)
            scaler.update()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batch_size)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model
    # pdb.set_trace()
    save_path = opt.train_save
    train_loss = '{:0.2f}'.format(loss_P2_record.show())
    torch.save(model.state_dict(), save_path + str(epoch) + '_train_loss_{}.pth'.format(train_loss))
    # choose the best model

    global dict_plot
    dice_dict = {}
    test1path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset'
    if (epoch + 1) % 1 == 0:
        for dataset in ['Kvasir', 'ETIS-LaribPolypDB', 'CVC-ColonDB', 'CVC-ClinicDB', 'CVC-300']:
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dice_dict.update({'{}'.format(dataset): dataset_dice})
            dict_plot[dataset].append(dataset_dice)
            meandice = test(model, test_path, dataset)

            dict_plot['test'].append(meandice)
            if meandice > best:
                best = meandice
                # torch.save(model.state_dict(), save_path + 'net_{}.pth'.format(meandice))
                torch.save(model.state_dict(),
                           save_path + str(epoch) + '_net_dice_{}-best.pth'.format(round(meandice, 2)))
                print('##############################################################################best', best)
                logging.info(
                    '##############################################################################best:{}'.format(
                        best))
        # pdb.set_trace()
        save_training_metrics = SaveTrainingMetrics(model_save_path=opt.train_save,
                                                    timestamp=timestamp,
                                                    epoch=epoch,
                                                    train_loss=train_loss,
                                                    CVC_300_dice=dice_dict["CVC-300"],
                                                    CVC_ClinicDB_dice=dice_dict["CVC-ClinicDB"],
                                                    Kvasir_dice=dice_dict["Kvasir"],
                                                    CVC_ColonDB_dice=dice_dict["CVC-ColonDB"],
                                                    ETIS_LaribPolypDB_dice=dice_dict["ETIS-LaribPolypDB"],
                                                    )
        save_training_metrics.write_metrics()


def plot_train(dict_plot=None, name=None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,
                     'ETIS-LaribPolypDB': 0.733, 'test': 0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('metrics.png')
    # plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    # #################model_name#############################
    model_name = 'MaCbamSuperbisionFormer_1'
    ###############################################

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default="False", help='choose to do random flip rotation')

    parser.add_argument('--batch_size', type=int,
                        default=4, help='training batch size')

    parser.add_argument('--deep_supervision', action='store_true',
                        default=True, help='use deep supervision mechanism')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/root/nfs/gaobin/wt/Checkpoints/Polyp_Project_2/' + model_name + '/' + '{}'.format(
                            timestamp) + '/')
    # parser.add_argument('--train_save', type=str,
    #                     default=None,
    #                     help='path to save training checkpoints')

    opt = parser.parse_args()
    if opt.train_save:
        if not os.path.exists(opt.train_save):
            os.makedirs(opt.train_save)
        print("Train save path:", opt.train_save)
    else:
        print("Warning: 'train_save' path not provided.")

    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = PolypPVT().cuda()
    model = loading_mode(model_name)
    print("model name: ", model_name)

    best = 0

    params = model.parameters()
    model_total_para = sum(param.numel() for param in model.parameters())
    model_name = model.__class__.__name__

    criterion_loss = loss_arch.StructureLoss()
    loss_name = criterion_loss.__class__.__name__

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batch_size, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    save_model_param = SaveModelParam(model_path=opt.train_save,
                                      timestamp=timestamp,
                                      model_name=model_name,
                                      model_params=model_total_para,
                                      loss_name=loss_name,
                                      batch_size=opt.batch_size,
                                      learning_rate=opt.lr,
                                      optimizer=optimizer,
                                      augmentation='False')
    save_model_param.write_model()

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, criterion_loss, optimizer, epoch, opt.test_path)

    # plot the metrics.png in the training stage
    # plot_train(dict_plot, name)
