# 多线程测试

import _thread
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def test_thread(Thread_name):
    for i in range(5):
        print(Thread_name + "====================  " + str(i))
        time.sleep(2)

def update_array(arr, k):
    for j in range(len(arr[0][0])):
        arr[0][k][j] = j


if __name__ == "__main__":

    # try:
    #     _thread.start_new_thread(test_thread, ("thread-01", ))
    #     _thread.start_new_thread(test_thread, ("thread-02", ))
    # except:
    #     print("线程无法启动")
    # time.sleep(15)

    arr = np.array([[1, 2, 3, 2],
                    [4, 2, 6, 1],
                    [3, 5, 1, 2]
                    ])
    arr = torch.tensor(arr)
    values, indices = arr.topk(k=2, dim=1, largest=True)
    print(indices)
    print(values)
    arr = torch.zeros_like(arr)
    out = torch.scatter(arr, 1, indices, 1) # 按indices中的下标替换arr中的值
    print(out)

    arr1 = torch.tensor([[1, 0, 0],
                         [1, 0, 1]])
    arr2 = torch.tensor([[1, 0, 1],
                         [0, 0, 1]])
    result = arr1 & arr2
    print(result)