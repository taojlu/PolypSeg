import pdb

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.load_model import loading_mode
from utils.dataloader import test_dataset
import cv2

if __name__ == '__main__':

    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    check_path = '/root/nfs/gaobin/wt/Checkpoints/Polyp_Project_2'
    # check_path = '/sda1/wangtao/Checkpoints/Polyp_Project/Polyp_Data '

    # 定义模型名称到文件夹、子文件夹和文件名的映射
    model_info = {
        'U_Net': ('U_Net', '2024-09-09_21-37-13', '148_net_dice_0.87-best.pth'),
        'NestedUNet': ('NestedUNet', '2024-09-09_21-40-24', '74_net_dice_0.88-best.pth'),
        'M2SNet': ('M2SNet', '2024-09-09_21-19-25', '80_net_dice_0.84-best.pth'),
        'MSNet': ('MSNet', '2024-09-10_23-15-27', '186_net_dice_0.85-best.pth'),
        'AttU_Net': ('AttU_Net', '2024-04-03_21-57-42', '99_train_loss_0.05.pth'),
        'SwimGroupFormer': ('SwimGroupFormer', '2024-04-03_22-06-23', '99_train_loss_0.10.pth'),
        'SwimGroupMixFormer': ('SwimGroupMixFormer', '2024-04-04_16-31-10', '99_train_loss_0.27.pth'),
        'GroupMixFormerSupervise': ('GroupMixFormerSupervise', '2024-04-03_21-55-44', '99_train_loss_0.13.pth'),
        'GroupMixFormer': ('GroupMixFormer', '2024-04-13_17-01-23', '133_net_dice_0.9-best.pth'),
        'PolypPVT': ('PolypPVT', '2024-07-29_09-21-37', 'PolypPVT.pth'),
        # 'M2SNet': ('M2SNet', '2024-04-03_22-02-31', '99_train_loss_0.05.pth'),
        'HighResolutionNetOCR': ('HighResolutionNetOCR', '2024-04-04_17-14-51', '98_net_dice_0.92-best.pth'),
        'PVT_UNet': ('PVT_UNet', '2024-08-01_06-16-59', '118_net_dice_0.93-best.pth'),
        'PVT_UNet_NewAtt': ('PVT_UNet_NewAtt', '2024-08-06_02-48-17', '153_net_0.7-best.pth'), # 需要更换的
        'MaCbamSuperbisionFormer': ('MaCbamSuperbisionFormer', '2024-08-12_14-31-41', '32_net_0.93-best.pth'),
        'MaCbamFormer': ('MaCbamFormer', '2024-08-31_11-56-32', '24_net_0.9-best.pth'),
        'VMUNet': ('VMUNet', '2024-09-09_23-00-05', '34_net_dice_0.83-best.pth'),
        'PvtFormer': ('PvtFormer', '2024-08-09_22-44-22', '21_net_dice_0.91-best.pth'),
        'PvtMaFormer': ('PvtMaFormer', '2024-08-10_01-37-06', '26_net_dice_0.92-best.pth'),
        'MaCbamFormer': ('MaCbamFormer', '2024-08-10_03-14-52', '22_net_dice_0.93-best.pth'),
    }

    model_name = 'MaCbamFormer'
    model = loading_mode(model_name)
    # 使用get方法获取模型信息，如果模型名不存在于字典中，则返回None
    model_data = model_info.get(model_name)
    if model_data:
        model_folder, model_subfolder, model_file = model_data
        print(f"Model folder: {model_folder}")
        print(f"Model subfolder: {model_subfolder}")
        print(f"Model file: {model_file}")
    else:
        print(f"Model name '{model_name}' not found.")

    model_path = os.path.join(check_path, model_folder, model_subfolder, model_file)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.cuda()
    model.eval()
    # pdb.set_trace()
    print('model name: ', model_name)
    pth_name = model_file.split('.')[0]
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        ##### put data_path here #####
        data_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/{}/{}/{}/'.format(model_name,
                                                                                         model_subfolder + 'and' + pth_name,
                                                                                         _data_name)
        # pdb.set_trace()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P2 = model(image)
            # P2 = P2[0] + P2[1] + P2[2] + P2[-1]
            # pdb.set_trace()
            res = F.upsample(P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)
        print(_data_name, 'Finish!')
