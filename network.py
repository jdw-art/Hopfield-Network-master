# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class HopfieldNetwork(object):
    # 使用Hebb学习规则调整连接权重
    def train_weights(self, train_data):
        print("Start to train weights...")
        # train_data的形状是其中包含四个长度为16384数组的列表，因此，num_data=4
        num_data =  len(train_data)
        # 神经元总数为16384
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights 初始化权重矩阵，网络神经元个数和输入向量长度相同
        W = np.zeros((self.num_neuron, self.num_neuron))
        # 计算数据的稀疏性rho，求每个样本的总和，然后求所有样本的和，最后除以网络中所有神经元的数量
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule Hebb学习规则，每个神经元根据自身的输入和输出信息调整权重
        for i in tqdm(range(num_data)):
            # 计算训练样本和稀疏度之间的差值
            t = train_data[i] - rho
            # 计算自身的外积，以更新网络权重
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0 使权重的对角线元素都为0，也就是说神经元自身不存在连接
        # 提取主对角线上的权重
        diagW = np.diag(np.diag(W))
        # 用权重矩阵减去主对角线上的权重使得对角线上的权重为0
        W = W - diagW
        # 归一化权重
        W /= num_data
        
        self.W = W 

    # 对给定输入数据进行预测，通过调用_run函数对输入样本进行预测返回每个样本中每个神经元的状态列表
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted

    # 对吸引子网络中的状态进行更新和迭代，init_s表示出事的神经元状态向量
    def _run(self, init_s):
        # 同步更新神经元节点
        # 初始化神经元状态 s 为 init_s。
        # 计算初始状态的能量 e，并进行迭代更新。
        # 在每次迭代中，通过计算连接权重矩阵 self.W 与神经元状态 s 的乘积，然后与阈值向量 self.threshold 进行比较，得到新的神经元状态 s。
        # 计算新的状态能量 e_new，如果当前状态的能量与新的状态能量相等（即状态收敛到吸引子），则返回最终状态 s，否则继续进行下一次迭代，直到达到最大迭代次数
        if self.asyn==False:
            """
            Synchronous update
            """
            # Compute initial state energy 计算初神经元状态
            s = init_s

            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)

                # 当前状态已经收敛到吸引子
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        # 异步更新神经元节点
        else:
            """
            Asynchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron) 
                    # Update s
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
    

    # 吸引子网络的能量函数
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()