import pdb

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.load_model import loading_mode
from utils.edge_dataloader import edge_test_dataset
import cv2

if __name__ == '__main__':

    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    check_path = '/sda1/wangtao/Checkpoints/Polyp_Project_2'
    model_name = 'SwimGroupMixFormer'
    model = loading_mode(model_name)
    if model_name == 'UNet':
        model_folder = model_name
        model_subfolder = '2024-03-17_20-27-56'
        model_file = '87_net_0.9-best.pth'
    elif model_name == 'SwimGroupFormer':
        model_folder = model_name
        model_subfolder = '2024-03-18_12-35-32'
        model_file = '60_net_0.91-best.pth'
    elif model_name == 'SwimGroupMixFormer':
        model_folder = model_name
        model_subfolder = '2024-03-18_19-09-17'
        model_file = '99_train.pth'
    elif model_name == 'GroupMixFormerSupervise':
        model_folder = model_name
        model_subfolder = '2024-03-18_17-59-58'
        model_file = '15_net_0.9-best.pth'
    elif model_name == 'GroupMixFormer':
        model_folder = model_name
        model_subfolder = '2024-03-18_17-52-54'
        model_file = '81_net_0.89-best.pth'

    model_path = os.path.join(check_path, model_folder, model_subfolder, model_file)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    # pdb.set_trace()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        ##### put data_path here #####
        data_path = '/home/wangtao/DataSet/Polyp/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = '/sda1/wangtao/DataSets/Polyp_Project_2/RESULT_MAP/{}/{}/{}/'.format(model_name,model_subfolder, _data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = edge_test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P2 = model(image)
            res = F.upsample(P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
