import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})

## 将 item索引 和 实体id 读取到字典中
def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]    # 获取 item 索引
        satori_id = line.strip().split('\t')[1]     # 获取 实体id
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1

'''
     使用 用户-item 交互数据构造正样本集和负样本集
     正样本集：用户评分大于等于阈值的item；
     负样本集：用户没有评分的item中随机选取与正样本相同数量的item；
     
     ***会生成一个ratings_final.txt文件，里面是正负样本集合
'''
def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values()) # 值为：0-item的数量
    user_pos_ratings = dict()   # 正样本：用户评分过的item
    user_neg_ratings = dict()   # 负样本：用户没有评分过的item，或评分低于阈值的item

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])    # 取出交互数据的每一行，以分隔符切成数组

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array)) # 去掉array中每个元素的第一个和最后一个字符，即去掉数字前后的“”

        # 如果当前评分的图书没有在子图谱中（即item_index2entity_id_rehashed.txt文件中），则跳过；
        # 如果在子图谱中，则获取对应的实体id、用户id和评分rating
        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        # 构造用户评分正样本集合 dict(user_index_old:{item1,item2...})，键是用户id，值是用户评分大于阈值的item_index
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        # 构造负样本集 dict(user_index_old:{item1,item2...})，键是用户id，值是用户评分小于阈值的item_index
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    # 读取正样本集，将user_id作为键，user_cnt作为值存入user_index_old2new中，user_cnt即当前是第几个正样本
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        # 将当前用户的正样本集写入文件，每一行是：user_index    item_index  1
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        # 未观测集unwatched_set=所有item中用户没有评分的item集合
        # 从未观测集中随机选取和正样本集个数相同的item构造负样本
        # 负样本每一行是 user_index item_index 0
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


'''
    将kg_rehashed.txt中的内容转换为 head_id relation_id tail_id 的格式写入 kg_final.txt 文件中
'''
def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)   # 实体数量
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'movie':
        files.append(open('../data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]    # 头实体id

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]    # 尾实体id

            # relation_id2index 就是kg_rehashed.txt中的每条关系，
            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]  # 关系id

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
