import numpy as np
import torch

from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    n_user = data_info[6]   # myCode

    model = RippleNet(args, n_entity, n_relation, n_user)
    if args.use_cuda:
        model.to(torch.device("cuda:1"))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )
    # 我的代码
    last_epoch_result = dict()      # 保存最后一个epoch计算得出的结果
    each_epoch_result = [[], [], [], [], [],
                         [], [], [], [], [],
                         [], [], [], [], []]  # 保存每一个epoch的结果，每一个list是一种结果，如auc所有epoch的结果
    for step in range(args.n_epoch):
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
        # while start < 0:
            # get_feed_dict()获取一个batch的数据：items, labels, memories_h, memories_r, memories_t
            # 其中items、labels分别为item和对应的评分，形状是[batch_size, 1]
            # memories分别是每一个user每一阶的头实体、尾实体、关系集合，[torch.LongTensor([1-hop ripple_set]), torch.LongTensor([2-hop ripple_set])...]
            # memories的形状是[batch_size, h-hop]
            # 将get_feed_dict()返回的一个batch的数据送入模型
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('已处理%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))


        # evaluation
        if args.dataset == "github":
            train_auc, train_acc, train_precision, train_recall, train_F1 = my_evaluation(args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc, eval_precision, eval_recall, eval_F1 = my_evaluation(args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_acc, test_precision, test_recall, test_F1 = my_evaluation(args, model, test_data, ripple_set, args.batch_size)

            # top-k预测
            test_len = test_data.shape[0]
            test_indices = np.random.choice(test_len, size=int(test_len * 0.01), replace=False)
            top_k(args, model, test_data[test_indices], ripple_set, args.batch_size, 10)


            print('epoch %d    train auc: %.4f  acc: %.4f  precision: %.4f  recall: %.4f  F1: %.4f      \n'
                  'eval auc: %.4f  acc: %.4f  precision: %.4f  recall: %.4f  F1: %.4f      \n'
                  'test auc: %.4f  acc: %.4f  precision: %.4f  recall: %.4f  F1: %.4f'
                  % (step, train_auc, train_acc, train_precision, train_recall, train_F1,
                     eval_auc, eval_acc, eval_precision, eval_recall, eval_F1,
                     test_auc, test_acc, test_precision, test_recall, test_F1))

            last_epoch_result = {"train_auc": train_auc, "train_acc": train_acc, "train_precision": train_precision,
                      "train_recall": train_recall, "train_F1": train_F1,
                      "eval_auc": eval_auc, "eval_acc": eval_acc, "eval_precision": eval_precision,
                      "eval_recall": eval_recall, "eval_F1": eval_F1,
                      "test_auc": test_auc, "test_acc": test_acc, "test_precision": test_precision,
                      "test_recall": test_recall, "test_F1": test_F1}

            r = [train_auc, train_acc, train_precision, train_recall, train_F1,
                 eval_auc, eval_acc, eval_precision, eval_recall, eval_F1,
                 test_auc, test_acc, test_precision, test_recall, test_F1]
            for i in range(15):
                each_epoch_result[i].append(r[i])

        else:
            train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
            last_epoch_result = {"train_auc": train_auc, "train_acc": train_acc,
                      "eval_auc": eval_auc, "eval_acc": eval_acc,
                      "test_auc": test_auc, "test_acc": test_acc}
    # 我的代码
    return last_epoch_result, each_epoch_result

def get_feed_dict(args, model, data, ripple_set, start, end):
    # 获取一个batch的item和ratings，shape=[batch_size, 1]
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])

    memories_h, memories_r, memories_t = [], [], []
    users = torch.LongTensor(data[start:end, 0])  # 存放user  mycode
    # 取出这个batch中user对应的h阶ripple_set放入三个list中，memories_h[i]就是第i-hop的头实体集合
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.to(torch.device("cuda:1"))
        labels = labels.to(torch.device("cuda:1"))
        memories_h = list(map(lambda x: x.to(torch.device("cuda:1")), memories_h))
        memories_r = list(map(lambda x: x.to(torch.device("cuda:1")), memories_r))
        memories_t = list(map(lambda x: x.to(torch.device("cuda:1")), memories_t))
        users = users.to(torch.device("cuda:1"))

    return items, labels, memories_h, memories_r, memories_t, users


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    while start < data.shape[0]:
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))

def my_evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    F1_list = []

    model.eval()
    while start < data.shape[0]:
        auc, acc, precision, recall, F1 = model.my_evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(precision_list)), float(np.mean(recall_list)), float(np.mean(F1_list))

# data是：用户 项目 rating
def construct_rating_matrix(data):
    user, item = set(data.T[0]), set(data.T[1])
    user_dict = {u: i for i, u in enumerate(user)}
    item_dict = {item: i for i, item in enumerate(item)}
    user_item_matrix = np.zeros(shape=(len(user), len(item)), dtype=np.float32)

    # 初始化 用户-项目 交互矩阵
    for line in data:
        user_item_matrix[user_dict.get(line[0])][item_dict.get(line[1])] = line[2]

    user_item_predict = np.zeros_like(user_item_matrix)
    return user_dict, item_dict, user_item_matrix, user_item_predict  # 最后一个返回值用于存储预测值


# top-k 推荐
# 1.构造测试集的  用户-item 交互矩阵  和   用户-item 预测矩阵
# 2.利用测试集构造预测评分数据集   (用户  item) 以供后续计算用户与每个item的相似度得分
# 3.使用训练好的模型预测  (用户 item score)，使用score更新 用户-item 预测矩阵
# 4.根据 用户-item 预测矩阵 计算用户的top-k项目，然后与 用户-item 交互矩阵 对比，得到Top-k推荐的准确率
def top_k(args, model, data, ripple_set, batch_size, k):
    # (1)
    user_item_matrix_info = construct_rating_matrix(data)
    user_dict, item_dict = user_item_matrix_info[0], user_item_matrix_info[1]
    user_item_matrix = user_item_matrix_info[2]
    # (2)
    # test_data = np.zeros(shape=(user_item_matrix.shape[0] * user_item_matrix.shape[1], 2), dtype=int)
    test_data = []
    for user in user_dict.keys():
        for item in item_dict.keys():
            test_data.append([user, item, -1])   # 每一条数据后加 -1 是为了复用get_feed_dict方法
    test_data = np.array(test_data)

    # (3)
    start = 0
    model.eval()
    while start < test_data.shape[0]:
        # 拿到一个batch的数据
        model.top_k_evaluate(*get_feed_dict(args, model, test_data, ripple_set, start, start + batch_size), user_item_matrix_info)
        start += batch_size
    model.train()

    # 将 用户-item-predict 矩阵中每行的top-k元素替换为1，其余替换为0
    user_item_predict = user_item_matrix_info[3]    # 获取预测打分矩阵
    # 取出top-k分数的索引
    _, indices = torch.topk(input=torch.tensor(user_item_predict), k=k, dim=1, largest=True)
    predict_result = torch.zeros_like(torch.tensor(user_item_predict), dtype=torch.int32)
    predict_result = torch.scatter(input=predict_result, dim=1, index=indices, value=1)  # 把top-k的位置替换为1

    result = predict_result & torch.tensor(user_item_matrix, dtype=torch.int32)    # 两个矩阵都为1的地方为1，否则为0

    precision_k = format((torch.sum(result) / (k * len(user_dict))).item(), ".4f")
    recall_k = format((torch.sum(result) / np.sum(user_item_matrix)).item(), ".4f")
    print("precision_" + str(k) + "：" + str(precision_k))
    print("recall_" + str(k) + "：" + str(recall_k))
    return precision_k, recall_k