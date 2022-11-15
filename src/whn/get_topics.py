import pandas as pd
import numpy as np
from github import Github
import pandas as pd
import csv
import _thread
import time
import os
from gharchive import GHArchive


# 拆分project.csv为几个文件
def split_projects():
    projects = pd.read_csv("../../data/github/project.csv")["name"].tolist()

    i = 0
    limit = 6000
    f = open("../../data/github/temp/pro" + str(int(i / limit)) + ".csv", 'w', newline="")
    writer = csv.writer(f)
    for pro in projects:
        if (i + 1) % limit == 0:
            f.close()
            f = open("../../data/github/temp/pro" + str(int(i / limit)) + ".csv", "w", newline="")
            writer = csv.writer(f)
            writer.writerow(["name"])
        writer.writerow([pro])
        i += 1
    f.close()


# 获取项目对应的主题
def get_topic(access_token, count):
    g = Github(access_token)
    # print(g.get_repo("hjw541988478/RefreshLoadMoreRecyclerView").get_topics())

    f = open("../../data/github/temp/topic" + str(count) + ".csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["project", "topic"])

    projects = pd.read_csv("../../data/github/temp/pro"+ str(count) + ".csv")["name"].tolist()
    for pro in projects:
        try:
            topics = g.get_repo(pro).get_topics()
        except:
            print("没有这个project：" + pro)
            continue
        else:
            for topic in topics:
                writer.writerow([pro, topic])
    f.close()

# 合并topic关系文件，并统计topic实体数写入topics.csv
def merge_file():
    file_path = "../../data/github/temp/"
    f = open("../../data/github/topic_relation.csv", "a", newline="")
    writer = csv.writer(f)
    writer.writerow(["project", "topic"])

    topics = set()
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.__contains__("topic"):
                filename = file_path + file
                print("开始读取" + filename)
                with open(filename, "r") as f1:
                    f1.readline()
                    lines = f1.readlines()
                    for i in range(len(lines)):
                        if i != 0:  # 跳过header
                            line = lines[i].strip().split(",")
                            writer.writerow(line)
                            topics.add(line[1])
                print("读取完毕..........")
    f.close()

    # 写入topic实体文件
    with open("../../data/github/topics.csv", "w", newline="") as f:
        for topic in topics:
            f.write(topic)
            f.write("\n")



if __name__ == "__main__":

    # try:
    #     _thread.start_new_thread(get_topic, ("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", 7, ))
    #     # _thread.start_new_thread(get_topic, ("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", 6, ))
    #     # _thread.start_new_thread(get_topic, ("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", 7, ))
    # except:
    #     print("线程无法启动")
    # time.sleep(120 * 60)

    # merge_file()

    g = Github("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")
    events = g.get_user("wuhannan08").get_events()
    for event in events:
        print(event.type + "：" + event.repo.name + ", create_at：" + str(event.created_at))
        # break
    # stargazers = g.get_repo("XavierWww/Chinese-Medical-Entity-Recognition").get_stargazers_with_dates()
    # for people in stargazers:
    #     print(people.user.name)
    #     print(people.starred_at)

    # topic = g.search_topics("ios-device")
    # for i in range(topic.totalCount):
    #     print(topic.__getitem__(i))
