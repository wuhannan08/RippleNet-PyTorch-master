import collections
import torch
import pandas
import numpy as np
import sklearn


if __name__ == "__main__":
    # BX_Book_Ratings = pandas.read_csv(filepath_or_buffer="../data/book/BX-Book-Ratings.csv", names= ["user_id", "ISBN", "ratings"], sep=";", encoding='latin-1',low_memory=False)
    # print(BX_Book_Ratings.shape)
    # BX_Book_Ratings.drop(BX_Book_Ratings.index[0])
    # print("用户数：" + str(BX_Book_Ratings["user_id"].nunique()))
    # print("书籍数" + str(BX_Book_Ratings["ISBN"].nunique()))


    # # 默认字典测试
    # d = collections.defaultdict(list)
    # d["head"].append(("tail1", "relation1"))
    # d["head"].append(("tail2", "relation2"))
    # print(d)
    # print("=================")
    # d["ripple"].append((["h1", "h2", "h3"], ["r1", "r2", "r3"], ["t1", "t2", "t3"]))
    # print(d["ripple"])
    #
    # print(np.__version__)
    # print(sklearn.__version__)

    # 读取打分文件，是一个二维ndarray
    ratings = np.load("../data/movie/ratings_final.npy")
    users = set(ratings[:, 0])
    n_user = len(users)
    print(users)
    print("==============")
    print(n_user)