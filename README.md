# RippleNet

### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
- `src/`: implementations of RippleNet.



### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```

### 文件介绍
#### 1、preprocess.py文件
- （1）预处理数据，对BX-Book-Ratings.csv用户评分文件中的数据进行处理，从中挑出评分大于阈值的条目作为用户的**正样本集**，
然后从用户没有评分的图书或评分小于阈值的图书中随机选取相同数量的条目作为**负样本集**，最终将正样本集和负样本集放入文件**ratings_final.txt**中；
- （2）对kg_rehashed.txt文件中的数据进行处理，将kg_rehashed.txt中的内容转换为 head_id relation_id tail_id 的格式写入 kg_final.txt 文件中；
并统计了实体数和关系数。

#### 2、data_loader.py文件
- 将ratings_final.txt中的数据集进行划分，训练集：验证集：测试集=6:2：2
- 根据kg_final.txt文件中的数据构造知识图谱，并统计实体数、关系数
- 获取每个user在知识图谱中的h阶ripple_set