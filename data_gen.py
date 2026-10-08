from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import math
import os

def load_data(data_sign, data_path_prefix):
    if data_sign == "IP":
        data = sio.loadmat('%s/IP/Indian_pines.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/IP/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "UP":
        data = sio.loadmat('%s/UP/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/UP/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt']
    elif data_sign=='HongHu':
        data = sio.loadmat('%s/HongHu/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/HongHu/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    elif data_sign=='Houston':
        data = sio.loadmat('%s/Houston/Houston.mat' % data_path_prefix)['Houstondata']
        labels = sio.loadmat('%s/Houston/Houston_GT.mat' % data_path_prefix)['Houstonlabel']
    elif data_sign=='SA':
        data = sio.loadmat('%s/SA/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/SA/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
    elif data_sign=='HanChuan':
        data = sio.loadmat('%s/HanChuan/WHU_Hi_HanChuan.mat' % data_path_prefix)['WHU_Hi_HanChuan']
        labels = sio.loadmat('%s/HanChuan/WHU_Hi_HanChuan_gt.mat' % data_path_prefix)['WHU_Hi_HanChuan_gt']
    else:
        raise ValueError(f"Unknown dataset: {data_sign}")
    return data, labels


# 按照数量固定划分训练集和测试集
def gen(data_sign, train_num_per_class, val_num_per_class, data_path_prefix, max_percent=0.5):
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}

    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                class2data.setdefault(labels[i, j], []).append([i, j])

    TR = np.zeros_like(labels)
    VAL = np.zeros_like(labels)
    TE = np.zeros_like(labels)

    split_info=[]

    for cl in range(1, class_num+1):
        ll = class2data[cl]
        total = len(ll)
        # 如果样本数量不足，按比例划分
        real_train_num = train_num_per_class
        real_val_num = val_num_per_class
        if total <= train_num_per_class + val_num_per_class:
            real_train_num = max(1, int(total * max_percent))
            real_val_num = max(1, int(total * max_percent / 2))
        if real_train_num + real_val_num >= total:
            real_val_num = max(1, total - real_train_num - 1)

        all_indices = list(range(total))
        random.shuffle(all_indices)
        train_indices = all_indices[:real_train_num]
        val_indices = all_indices[real_train_num:real_train_num + real_val_num]
        test_indices = all_indices[real_train_num + real_val_num:]

        for idx in train_indices:
            i, j = ll[idx]
            TR[i, j] = cl
        for idx in val_indices:
            i, j = ll[idx]
            VAL[i, j] = cl
        for idx in test_indices:
            i, j = ll[idx]
            TE[i, j] = cl

        split_info.append({
            "class": cl,
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "total": total
        })

    target = {'TR': TR, 'VAL': VAL, 'TE': TE, 'input': data}

    ntr = TR[TR > 0].shape[0]
    nval = VAL[VAL > 0].shape[0]
    nte = TE[TE > 0].shape[0]
    print(f'train={ntr}, val={nval}, test={nte}, total={labels[labels > 0].shape[0]}')

    print("\n===== Split Details =====")
    print("Class\tTrain\tVal\tTest\tTotal")
    for item in split_info:
        print(f"{item['class']}\t{item['train']}\t{item['val']}\t{item['test']}\t{item['total']}")
    print("=========================\n")

    return target


#执行函数
def run():
    signs = ['Houston']    # ['IP', 'UP', 'HongHu', 'Houston', 'SA']
    data_path_prefix = './data'
    train_num_per_class = 10  # 1, 3，5, 10
    val_num_per_class = 4    # 1, 1, 2, 4
    for data_sign in signs:
        save_dir = f'./data/{data_sign}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/{data_sign}_t{train_num_per_class}_v{val_num_per_class}_split.mat'
        target = gen(data_sign, train_num_per_class, val_num_per_class, data_path_prefix)
        sio.savemat(save_path, target)
        print(f'saved {save_path}')


if __name__ == "__main__":
    print(os.getcwd())
    run()