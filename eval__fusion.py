import os
import argparse
import pdb
import pandas as pd
import tqdm
from tqdm import tqdm

from PIL import Image


# filepath = os.path.split(os.path.abspath(__file__))[0]
# repopath = os.path.split(filepath)[0]
# sys.path.append(repopath)

# from metrics.eval_functions import *
from utils.utils import *
from utils.eval_functions import *


def evaluate(args):
    Thresholds = np.linspace(1, 0, 256)
    headers = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen',
               'meanSpe', 'maxSpe']

    results = []

    # if args.verbose is True:
    #     print('#' * 20, 'Start Evaluation', '#' * 20)
    #     datasets = tqdm.tqdm(args.datasets, desc='Expr - ' + method, total=len(
    #         args.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    # else:
    #     datasets = args.datasets
    metrics = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen',
               'meanSpe', 'maxSpe']
    # 初始化一个空的 metrics_data 字典
    metrics_data = {metric: {} for metric in metrics}

    datasets = os.listdir(args.fusion_root)
    datasets.sort(reverse=True)
    # pdb.set_trace()

    for dataset in tqdm(datasets):
        fusion_root = os.path.join(args.fusion_root, dataset)
        gt_root = os.path.join(args.gt_root, dataset, 'masks')

        preds = os.listdir(fusion_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        # threshold_Precision = np.zeros((len(preds), len(Thresholds)))
        # threshold_Recall = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        if args.verbose is True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            pred_mask = np.array(Image.open(os.path.join(fusion_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))
            gt_mask = gt_mask.astype(np.uint8) * 255
            # pdb.set_trace()
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]

            assert pred_mask.shape == gt_mask.shape
            # pdb.set_trace()
            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255
            pred_mask = (pred_mask > 0.5).astype(np.float64)
            # pdb.set_trace()

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[
                    j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        # 更新 metrics_data 字典中的值
        metrics_data['meanDic'][dataset] = meanDic
        metrics_data['meanIoU'][dataset] = meanIoU
        metrics_data['wFm'][dataset] = wFm
        metrics_data['Sm'][dataset] = Sm
        metrics_data['meanEm'][dataset] = meanEm
        metrics_data['mae'][dataset] = mae
        metrics_data['maxEm'][dataset] = maxEm
        metrics_data['maxDic'][dataset] = maxDic
        metrics_data['maxIoU'][dataset] = maxIoU
        metrics_data['meanSen'][dataset] = meanSen
        metrics_data['maxSen'][dataset] = maxSen
        metrics_data['meanSpe'][dataset] = meanSpe
        metrics_data['maxSpe'][dataset] = maxSpe

    df = pd.DataFrame(metrics_data)
    model_name, model_subfolder = args.fusion_root.split('/')[-2:]
    file_name = model_name + '-' + model_subfolder
    # pdb.set_trace()
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    df.to_excel('{}/{}.xlsx'.format(args.result_path, file_name))


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='FUSION')
        parser.add_argument('--fusion_root', type=str, default='/root/nfs/gaobin/wt/Datasets/Polyp/Results/FUSION')
        parser.add_argument('--gt_root', type=str, default='/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset')
        parser.add_argument('--result_path', type=str, default='/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_EXCELS')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--verbose', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=False)
        args = parser.parse_args()

        return args


    args = parse_args()

    model_name = 'FUSION'

    # 定义模型名称到文件夹和子文件夹的映射
    model_mappings = {
        'FUSION': 'fusion_6'

    }

    # 通过模型名称获取文件夹和子文件夹
    model_folder = model_name
    model_subfolder = model_mappings.get(model_name, 'default_subfolder')
    # pdb.set_trace()
    # Update args information
    args_dict = vars(args)  # Convert args namespace to dictionary
    args_dict['model_name'] = model_name
    # ==================== 单独模型预测评价 ===================
    # args_dict['fusion_root'] = os.path.join(args.fusion_root, model_folder, model_subfolder)
    # ==================== 多任务预测评价 ===================
    args_dict['fusion_root'] = os.path.join(args.fusion_root, model_subfolder)
    # ==================== 计算结果保存 ===================
    args_dict['result_path'] = os.path.join(args.result_path, model_name, model_subfolder)

    # Recreate args object with updated information
    args = argparse.Namespace(**args_dict)
    print('model name: ', model_name)
    evaluate(args)
