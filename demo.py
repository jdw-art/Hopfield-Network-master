# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import network

np.random.seed(1)
# Utils
# 为图像生成噪声等级为corruption_level的二值噪声图片
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

# 将输入的一维数据重新整形为二维矩阵
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

# 可视化训练数据、加入噪声后的数据（测试数据）、预测数据
def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

# 对图像进行预处理：调整尺寸、二值化、形状转换
# 可以看作是脉冲生成的过程，都是把实数值转换成二值
def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w, h), mode='reflect')

    # Thresholding 对图像进行二值化处理
    # 计算图像的均值阈值
    thresh = threshold_mean(img)
    # 转换成二值图像，大于阈值为True小于阈值为False
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape 将二维图像转换为一维数组
    flatten = np.reshape(shift, (w*h))
    # flatten = np.reshape(img, (w*h))
    return flatten

def main():
    # Load data
    camera = skimage.data.camera()
    astronaut = rgb2gray(skimage.data.astronaut())
    horse = skimage.data.horse()
    coffee = rgb2gray(skimage.data.coffee())

    # Marge data
    data = [camera, astronaut, horse, coffee]
    data1 = [camera, astronaut, horse, coffee]
    data2 = [coffee]

    # Preprocessing
    print("Start to data preprocessing...")
    data1 = [preprocessing(d) for d in data1]
    data2 = [preprocessing(d) for d in data2]

    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data1)

    # Generate testset
    test = [get_corrupted_input(d, 0.3) for d in data]
    # test = []
    # for i in range(0, len(data)):
    #     test.append(get_corrupted_input(data[len(data) - i - 1], 0.1))

    # 对引入噪声的图片进行预测
    predicted = model.predict(data2, threshold=0, asyn=False)
    print("Show prediction results...")
    # plot(data1, data2, predicted)
    # print("Show network weights matrix...")
    # model.plot_weights()

    data1 = [reshape(d) for d in data1]
    data2 = [reshape(d) for d in data2]
    predicted = [reshape(d) for d in predicted]

    # 绘制图像
    plt.subplot(1, 3, 1)
    plt.imshow(data1[0], cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(data2[0], cmap='gray')
    plt.title('Output Spike')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted[0], cmap='gray')
    plt.title('Output Feature')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
