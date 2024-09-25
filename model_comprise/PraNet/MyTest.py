import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/root/nfs/gaobin/wt/Checkpoints/Polyp_Project_2/PraNet_Res2Net/PraNet-199.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}/'.format(_data_name)
    save_path = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/PraNet_Res2Net/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # misc.imsave(save_path+name, res)
        # imageio.imwrite(save_path + name, res)
        imageio.imwrite(save_path + name, ((res > .5) * 255).astype(np.uint8))