import collections
import os
import numpy as np
import pandas as pd
import csv


entity_dict = dict()
relation_dict = {"Follow": 0, "star": 1, "create": 2, "fork": 3, "pr": 4,
                 "forked_from": 5, "containLib": 6, "containPlat": 7, "containStan": 8,
                 "useLanguage": 9, "mention": 10, "containTopic": 11}
# 实体文件list，文件顺序一样，生成的dict就一样
file_list = ["../../data/github/developers.csv", "../../data/github/projects.csv",
             "../../data/github/Platform.csv", "../../data/github/Standard.csv",
             "../../data/github/Library.csv", "../../data/github/Languages.csv",
             "../../data/github/topics.csv"]

# 测试项目id 与开发者id是否有重复的
def test_repeat():
    developers = pd.read_csv("../../data/github/developer.csv")
    # 将id这一列提取为numpy数组
    developers = developers["id"].to_numpy()
    dev = set(developers)
    print("开发者数：" + str(len(dev)))
    projects = pd.read_csv("../../data/github/project.csv")
    pro = set(projects["id"].to_numpy())
    print("项目数：" + str(len(pro)))

    result = dev | pro
    print("项目+开发者数：" + str(len(result)))

    flag = True
    if len(result) == len(dev) + len(pro):
        flag = False
    print("developer id 和 project id 是否有重叠：" + str(flag))

# 为没有写关系的csv新增关系列
def insert_relation(file_path, relation_type):
    Relation = pd.read_csv(file_path)
    Relation.insert(loc=2, column="Type", value=relation_type)
    print(Relation)

    Relation.to_csv(file_path[: -4] + "1.csv", sep=",", index=False)
    print("写入csv成功")

# 对实体文件去重并构造每个文件对应的字典
# 对于开发者和项目，键就是id，对于其他实体键是name
def get_each_entity_dict(file_path, index):
    entities = pd.read_csv(file_path).to_numpy().reshape(-1)
    if file_path.__contains__("topic"):
        entities = set(e for e in entities)
    else:
        entities = set(e.lower() if isinstance(e, str) else e for e in entities)
    print(file_path.split("/")[-1][:-4] + "去重后实体数：" + str(len(entities)))

    # 不同实体类型的索引是连续的
    for e in entities:
        if e not in entity_dict:
            entity_dict[e] = index
            index += 1
    # print(entity)
    return index

# 获取所有类型实体组成的字典，返回{实体名:id}
def construct_all_entity_dict(file_list):
    index = 0
    for file in file_list:
        index = get_each_entity_dict(file, index)
    print("总实体数：" + str(len(entity_dict)))
    # print(entity_dict)

# 处理mention关系的方法
def process_mention():
    project = pd.read_csv("../../data/github/project.csv")
    mentionRelation = pd.read_csv("../../data/github/mentionRelation.csv")
    print("提及关系有：" + str(mentionRelation.shape[0]))

    print("重复项目名数：" + str(project.shape[0] - len(set(project["name"].to_numpy().reshape(-1)))))

    result = mentionRelation.merge(project, on="name", how="inner")
    print("结果数：" + str(result.shape[0]))
    result[["NodeA", "id", "Type"]].to_csv("../../data/github/relation/mentionRelation1.csv", sep=",", index=False)
    print("处理完成")

# 将关系文件转换为： 实体索引    关系索引     实体索引
def construct_kg():
    file_path = "../../data/github/relation/"
    file_dir = os.path.abspath(file_path)   # 根据相对路径返回绝对路径

    print("construct kg......")

    nodeA, nodeB, relation = [], [], []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            print(file + "  constructing...")
            filename = file_path + file
            df = pd.read_csv(filename)
            for id in df.iloc[:, 0]:
                nodeA.append(entity_dict[id])
            for n in df.iloc[:, 1]:
                if file.__contains__("topic"):
                    nodeB.append(entity_dict[n])
                else:
                    # 如果第二列是数字则直接去查对应的实体索引；如果是字符串，则先转换成小写再去查索引
                    nodeB.append(entity_dict[n.lower() if isinstance(n, str) else n])
            for r in df.iloc[:, 2]:
                relation.append(relation_dict[r])

        # 将处理后的实体关系写回文件， 实体索引   关系索引    实体索引
        data = np.asarray([nodeA, relation, nodeB]).T
        writer = open("../../data/github/kg_final.txt", 'w', encoding="utf-8")
        for item in data:
            writer.write('%d\t%d\t%d\n' % (item[0], item[1], item[2]))
        writer.close()
        # np.savetxt("../../data/github/kg_final.txt", data, fmt="%d")
        print("构造kg_final.txt成功")
        print("kg_final.txt中的关系数为：" + str(data.shape[0]))


# 构造正负样本集
def construct_data():
    file = "../../data/github/relation/developer2project.csv"
    projects_file = "../../data/github/projects.csv"


    print('reading rating file ...')
    item_set = set()
    for item in pd.read_csv(projects_file).to_numpy().reshape(-1).tolist():
        item_set.add(entity_dict[item])   # 项目集合，每一个元素是项目id
    user_pos_ratings = dict()  # 正样本：开发者操作过的项目
    user_neg_ratings = dict()  # 负样本：用户没有操作过的项目

    # 读完文件正样本集就构造好了
    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(",")  # 取出交互数据的每一行，以分隔符切成数组

        user_index = entity_dict[int(array[0])]
        item_index = entity_dict[int(array[1])]
        if user_index not in user_pos_ratings:
            user_pos_ratings[user_index] = set()
        user_pos_ratings[user_index].add(item_index)    # 将项目放入用户正样本集

    writer = open("../../data/github/ratings_final.txt", 'w', encoding="utf-8")
    for user_index, pos_item_set in user_pos_ratings.items():
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))

        unwatched_item_set = item_set - pos_item_set
        # 从未观测集中随机选取负样本，如果正样本集大于未观测集，则可以选重复的数据即replace=True
        replace = False
        if len(pos_item_set) > len(unwatched_item_set):
            replace = True
        for item in np.random.choice(list(unwatched_item_set), size=len(pos_item_set), replace=replace):
            writer.write('%d\t%d\t0\n' % (user_index, item))

    writer.close()
    print("ratings_final.txt done...")

if __name__ == "__main__":

    construct_all_entity_dict(file_list)    # 为entity_dict赋值
    construct_kg()  # 构造知识图谱，执行该方法后会在github目录下生成kg_final.txt文件
    construct_data()    # 构造正负样本集，执行该方法后会在GitHub目录下生成ratings_final.txt文件