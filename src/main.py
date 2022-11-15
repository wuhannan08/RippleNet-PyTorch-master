import argparse
import csv

import numpy as np
from data_loader import load_data
from train import train
import pandas as pd

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=5, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')

# default settings for Book-Crossing
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''

parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
parser.add_argument('--cuda', type=int, default=2, help='which gpu to use')
args = parser.parse_args()


train_auc, train_acc, train_precision, train_recall, train_F1 = [], [], [], [], []
eval_auc, eval_acc, eval_precision, eval_recall, eval_F1 = [], [], [], [], []
test_auc, test_acc, test_precision, test_recall, test_F1 = [], [], [], [], []
for i in range(1):
    show_loss = False
    # 获取 train_data, eval_data, test_data, n_entity, n_relation, ripple_set
    data_info = load_data(args)
    last_epoch_result, each_epoch_result = train(args, data_info, show_loss)

    # 自己的代码
    train_auc.append(last_epoch_result.get("train_auc"))
    train_acc.append(last_epoch_result.get("train_acc"))
    train_precision.append(last_epoch_result.get("train_precision"))
    train_recall.append(last_epoch_result.get("train_recall"))
    train_F1.append(last_epoch_result.get("train_F1"))
    eval_auc.append(last_epoch_result.get("eval_auc"))
    eval_acc.append(last_epoch_result.get("eval_acc"))
    eval_precision.append(last_epoch_result.get("eval_precision"))
    eval_recall.append(last_epoch_result.get("eval_recall"))
    eval_F1.append(last_epoch_result.get("eval_F1"))
    test_auc.append(last_epoch_result.get("test_auc"))
    test_acc.append(last_epoch_result.get("test_acc"))
    test_precision.append(last_epoch_result.get("test_precision"))
    test_recall.append(last_epoch_result.get("test_recall"))
    test_F1.append(last_epoch_result.get("test_F1"))

# 平均结果计算
auc_result = [np.array(train_auc).mean(), np.array(eval_auc).mean(), np.array(test_auc).mean()]
acc_result = [np.array(train_acc).mean(), np.array(eval_acc).mean(), np.array(test_acc).mean()]
precision_result = [np.array(train_precision).mean(), np.array(eval_precision).mean(), np.array(test_precision).mean()]
recall_result = [np.array(train_recall).mean(), np.array(eval_recall).mean(), np.array(test_recall).mean()]
F1_result = [np.array(train_F1).mean(), np.array(eval_F1).mean(), np.array(test_F1).mean()]
print("average train auc: " + str(auc_result[0]) + "\taverage eval auc: " + str(auc_result[1]) + "\taverage test auc: " + str(auc_result[2]))
print("average train acc: " + str(acc_result[0]) + "\taverage eval acc: " + str(acc_result[1]) + "\taverage test acc: " + str(acc_result[2]))
print("average train precision: " + str(precision_result[0]) + "\taverage eval precision: " + str(precision_result[1]) + "\taverage test precision: " + str(precision_result[2]))
print("average train recall: " + str(recall_result[0]) + "\taverage eval recall: " + str(recall_result[1]) + "\taverage test recall: " + str(recall_result[2]))
print("average train F1: " + str(F1_result[0]) + "\taverage eval F1: " + str(F1_result[1]) + "\taverage test F1: " + str(F1_result[2]))

# 保存每一个epoch的数据
pd.DataFrame(np.array(each_epoch_result).T).to_csv("../data/result_without_topic.csv", index=False,
                                                   header=["train_auc", "train_acc", "train_precision", "train_recall", "train_F1",
                                                           "eval_auc", "eval_acc", "eval_precision", "eval_recall", "eval_F1",
                                                           "test_auc", "test_acc", "test_precision", "test_recall", "test_F1"])
