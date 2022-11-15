import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def set_axis(x, y):
    plt.plot(x, y, lw=16)
    # 使用gca获取当前图中的轴，gca：“get current axis”
    ax = plt.gca()
    ax.spines["top"].set_color("none")    # spines 就是获取那四根边框，通过left、right、top、bottom指定
    ax.spines["right"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")   # 设置bottom为x轴
    ax.yaxis.set_ticks_position("left")   # 设置left为y轴
    ax.spines["bottom"].set_position(("data", 0))   # 挪动x轴的位置到y轴的0点
    ax.spines["left"].set_position(("data", 1)) # 挪动y轴到x轴的1点

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(18)   # 设置刻度的字体
        # 设置刻度的框框， facecolor表示刻度框的底色，edgecolor表示边框颜色，alpha是透明度
        tick.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.show()

def set_legend(data: list):
    l1, = plt.plot(data[0], data[1], label="up")
    l2, = plt.plot(data[2], data[3], label="down", color="red", linewidth=2, linestyle="--")

    # handles属性指图例要handle的线，labels属性指定线对应的描述，loc属性指定图例的位置
    plt.legend(handles=[l1, l2], labels=["aa", "bb"], loc="best")
    plt.show()

def set_annotation(data):
    plt.plot(data[0], data[1])

    # 给出要标注的点
    x0 = 1
    y0 = x0 * 2 + 1
    plt.scatter(x0, y0) # 以点的方式绘图
    plt.plot([x0, x0], [y0, -6], ls="--", lw=2, c="black")
    # 设置标注
    plt.annotate(text="$2x+1=3$", xy=(x0, y0), xycoords="data",
                 textcoords="offset points", xytext=(+30, -30), # 注释text显示的位置：相对于points（+30， -30）
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))   # 注释和data的连接方式
    # 前两个数是标注开始显示的位置坐标
    plt.text(-3, 4, s=r"This is My Figure", fontdict=dict(size=16, color="r"))

    plt.show()

def scatter():
    x = np.random.normal(0, 1, 624)
    y = np.random.normal(0, 1, 624)
    T = np.arctan2(y, x)    # 设置散点的颜色
    # s=size, c=color，根据T的值不同设置不同的颜色
    plt.scatter(x, y, s=75, c=T, alpha=0.5)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.show()

def bar(x, height):
    plt.bar(x, height, width=1)    # 设置柱的宽度
    plt.ylim((0.5, 1))
    plt.ylabel("auc value")
    plt.show()

# 画实验结果图
def show_figure(x, y_list, x_label, y_label):
    plt.figure()
    plt.plot(x, y_list[0], label="train", color="red", lw="2")
    plt.plot(x, y_list[1], label="eval", color="green", lw="2")
    plt.plot(x, y_list[2], label="test", color="blue", lw="2")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()    # 显示图例

    plt.show()


def show_result():
    result = pd.read_csv("../../data/result_without_topic.csv")

    train_auc = result["train_auc"].tolist()
    train_acc = result["train_acc"].tolist()
    train_precision = result["train_precision"].tolist()
    train_recall = result["train_recall"].tolist()
    train_F1 = result["train_F1"].tolist()

    eval_auc = result["eval_auc"].tolist()
    eval_acc = result["eval_acc"].tolist()
    eval_precision = result["eval_precision"].tolist()
    eval_recall = result["eval_recall"].tolist()
    eval_F1 = result["eval_F1"].tolist()

    test_auc = result["test_auc"].tolist()
    test_acc = result["test_acc"].tolist()
    test_precision = result["test_precision"].tolist()
    test_recall = result["test_recall"].tolist()
    test_F1 = result["test_F1"].tolist()

    plt.figure()
    plt.plot(np.arange(6), train_auc, label="train_auc", color="red")
    plt.plot(np.arange(6), test_auc, label="test_auc", color="blue")
    plt.plot(np.arange(6), train_acc, label="train_acc", color="yellow")
    plt.plot(np.arange(6), test_acc, label="test_acc", color="green")
    plt.legend()
    plt.xlabel("num of epoch")
    plt.ylabel("value")
    plt.show()

    plt.figure()
    plt.plot(np.arange(6), train_precision, label="train_precision", color="orange")
    plt.plot(np.arange(6), test_precision, label="test_precision", color="purple")

    plt.plot(np.arange(6), train_recall, label="train_recall", color="brown")
    plt.plot(np.arange(6), test_recall, label="test_recall", color="olive")

    plt.plot(np.arange(6), train_F1, label="train_F1", color="pink")
    plt.plot(np.arange(6), test_F1, label="test_F1", color="cyan")
    plt.legend()
    plt.xlabel("num of epoch")
    plt.ylabel("value")
    plt.show()

if __name__ == "__main__":
    x = np.arange(start=0, stop=14, step=2) # 横坐标的范围
    x_label = ["none", "Topic", "Language", "Platform", "Library", "Standard", "combine"]   # 横坐标的含义
    x_1 = x   # 标注值的横坐标
    auc = [0.7873, 0.7982, 0.8010, 0.7938, 0.7831, 0.8125, 0.8007]

    plt.bar(x, auc, width=1)  # 设置柱的宽度
    plt.xticks(x, x_label, rotation=30)
    plt.ylim((0.5, 1))
    plt.ylabel("recall value")

    for a, b in zip(x_1, auc):
        plt.text(a, b + 0.01, "%.4f" % b, ha="center", va="bottom", fontsize=7)

    plt.show()