import sys
import concurrent.futures

import numpy as np
import os
import metrics as M
from PIL import Image

class test_dataloader:
    def __init__(self, salmap_root, gt_root):
        salmap_files = {f for f in os.listdir(salmap_root) if f.endswith(('.jpg', '.png'))}
        gt_files = {f for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png'))}
        common_files = salmap_files.intersection(gt_files)
        # print(len(common_files))
        self.salmap = [os.path.join(salmap_root, file) for file in common_files]
        self.gts = [os.path.join(gt_root, file) for file in common_files]
        self.salmap = sorted(self.salmap)
        self.gts = sorted(self.gts)
        self.size = len(self.salmap)

        self.index = 0

    def load_data(self):
        salmap = self.binary_loader(self.salmap[self.index])
        salmap = np.asarray(salmap, np.float32)
        gt = self.binary_loader(self.gts[self.index])
        gt = np.asarray(gt, np.float32)
        name_sal = self.salmap[self.index].split('/')[-1]
        name_gt = self.gts[self.index].split('/')[-1]
        assert name_gt == name_sal, print(self.salmap[self.index])
        assert gt.shape == salmap.shape, print(self.salmap[self.index])

        self.index += 1
        return salmap, gt, name_sal

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


dataset_path = '/root/autodl-tmp/TSOD_results/'
test_datasets = {
        # 'DUTS-Random2500': '/root/autodl-tmp/SOD_datasets/DUTS/Test_Random2500/mask',
        # 'TEOS10K-Random2500': '/root/autodl-tmp/TSOD_dataset/Test_Random2500/mask',
        # 'DUT-OMRON': '/root/autodl-tmp/SOD_datasets/DUT-OMRON/Test/mask',
        # 'ECSSD': '/root/autodl-tmp/SOD_datasets/ECSSD/Test/mask',
        # 'HKU-IS': '/root/autodl-tmp/SOD_datasets/HKU-IS/Test/mask',
        # 'PASCAL-S': '/root/autodl-tmp/SOD_datasets/PASCAL-S/Test/mask',
        # 'SOD': '/root/autodl-tmp/SOD_datasets/SOD/Test/mask',
        'DUTS-TE': '/root/autodl-tmp/SOD_dataset/DUTS/Test/mask'
        }
model_list = [
    'Tramba-S-SOD',
]

def evaluate_model(model):
    results_list = []
    for dataset, gt_root in test_datasets.items():
        # salmap_root = os.path.join(dataset_path, model, dataset + '_' + model + '/')
        salmap_root = os.path.join(dataset_path, model, 'SOD/')
        print(salmap_root)
        test_loader = test_dataloader(salmap_root, gt_root)

        FM = M.Fmeasure_and_FNR()
        WFM = M.WeightedFmeasure()
        SM = M.Smeasure()
        EM = M.Emeasure()
        MAE = M.MAE()
        for i in range(test_loader.size):
            pred, gt, name = test_loader.load_data()
            gt /= (gt.max() + 1e-8)
            pred = pred / 255

            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            MAE.step(pred=pred, gt=gt)

        fm = FM.get_results()[0]['fm']
        precision = FM.get_results()[0]['pr']['p']
        precision = np.array(precision, dtype=np.float32)
        recall = FM.get_results()[0]['pr']['r']
        recall = np.array(recall, dtype=np.float32)
        precision_save = os.path.join(dataset_path, model, dataset + '_precision.npy')
        recall_save = os.path.join(dataset_path, model, dataset + '_recall.npy')
        np.save(precision_save, precision)
        np.save(recall_save, recall)

        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']
        fnr = FM.get_results()[1]

        results = {
            'model': model,
            'dataset': dataset,
            'Smeasure_r': sm.round(4),
            'Wmeasure_r': wfm.round(4),
            'MAE_r': mae.round(4),
            'adpEm_r': em['adp'].round(4),
            'meanEm_r': em['curve'].mean().round(4),
            'maxEm_r': em['curve'].max().round(4),
            'adpFm_r': fm['adp'].round(4),
            'meanFm_r': fm['curve'].mean().round(4),
            'maxFm_r': fm['curve'].max().round(4),
            'fnr_r': fnr.round(4)
        }
        results_list.append(results)
        print('model: {} | dataset: {} || & {} & {} & {} & {}'.format(results['model'],
                                                                      results['dataset'],
                                                                      str(results['Wmeasure_r']),
                                                                      str(results['maxEm_r']),
                                                                      str(results['Smeasure_r']),
                                                                      str(results['MAE_r'])))
    return results_list


with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
    futures = [executor.submit(evaluate_model, model) for model in model_list]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]