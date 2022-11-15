import collections
import os
import numpy as np


def load_data(args):
    # 获取训练集、验证集、测试集、用户历史交互字典{user_index:[item_list]}
    train_data, eval_data, test_data, user_history_dict, n_user = load_rating(args)

    # 获取实体数、关系数和知识图谱
    n_entity, n_relation, kg = load_kg(args)

    # 获取每一个user在图谱中对应的h阶ripple_set
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set, n_user # Mycode

# 通过调用dataset_split()函数返回：训练集、验证集、测试集、用户历史交互字典{user_index:[item_list]}
def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))

    return dataset_split(rating_np)     # 划分数据集

# 划分训练集：验证集：测试集 = 6：2：2；并在验证集和测试集中筛选掉训练集中没有见过的user
def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]  # 有多少交互条数据

    # 从数据集中随机选取20%验证集、20%测试集、剩下的是训练集
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    # 将训练数据正样本集存入字典user_history_dict，键是user索引，值是正样本item_list
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    # 排除在训练集中没有见过的user相关的数据
    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # 获取对应的数据条目
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    '''
            My Code
    '''
    n_user = len(set(rating_np[:, 0]))

    return train_data, eval_data, test_data, user_history_dict, n_user


# 返回实体数、关系数、和构造的kg
def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    if args.dataset == "github":
        n_entity = 72616    # 直接给指定，有一部分实体可能不存在关系，所以没有进入kg_final.txt文件中，会导致下标越界
    # 求出实体数和关系数 | 是求set集合的并集
    else:
        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    # kg_np 即读取的kg_final.txt中的数据
    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


# 构造kg；键是head，值是元素为tuple的list：[(tail1, relation1),(tail2, relation2)]
def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    # 默认字典，当访问不存在的key时，会返回默认值，默认值由传入的类型或工厂函数决定，这里默认值是空list[]
    kg = collections.defaultdict(list)

    # 处理kg_final.txt中的数据为字典：键是头实体，值是尾实体和关系组成的元组list
    # defaultdict(<class 'list'>, {'head': [('tail1', 'relation1'), ('tail2', 'relation2')]})
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


# ***返回每一个user对应的ripple_set
# 从user_history_dict获取每一个user的历史交互集 item_list
# 对于历史交互集中的每一个item，去寻找以它为head的三元组，从这些三元组中随机选取固定个数的三元组，并放入ripple_set中
# 一直重复到指定的hop数；

# ripple_set是一个dict字典，键是user，值是user在图谱中对应的ripple_set
# user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
# 其中每个元组中的hop_0_heads, hop_0_relations, hop_0_tails都是一个list，代表每一个hop的所有头尾实体/关系
def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    # 其中每个元组中的hop_0_heads, hop_0_relations, hop_0_tails都是一个list，代表每一个hop的所有头尾实体/关系
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            # 获取上一hop的尾实体，tails_of_last_hop是一个尾实体list
            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2] # 获取上一跳的tail

            # 对于尾实体中的每一个实体都去KG中寻找以它为head的三元组，并分别放入memories集合中
            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            # 如果当前hop的RippleSet为空，则使用上一hop的
            if len(memories_h) == 0:
                # 如果历史交互集都没有出度，则就使用 【user->历史交互集】 作为rippleSet
                if len(ripple_set[user]) == 0:
                    memories_h = [user for i in range(len(tails_of_last_hop))]
                    memories_r = [t[1] for t in kg[user]]
                    memories_t = tails_of_last_hop
                    replace = len(memories_h) < args.n_memory
                    indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))
                else:
                    ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                # 从RippleSet的头、尾实体和关系中随机选择 n_memory个放入最终的RippleSet
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
