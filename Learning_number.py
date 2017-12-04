##MNISTからデータをランダムで選び出力するだけのスクリプト

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

mnist = datasets.fetch_mldata('MNIST original', data_home = '.')

n= len(mnist.data)
N = 20
indice = np.random.permutation(range(n))[:N]
X = mnist.data[indice]
y = mnist.target[indice]

for i in range(4):
    for j in range(4):
        number =i +4*j+1
        plt.subplot(4, 4, number)
        plt.imshow(np.reshape(X[number],[28,28]),cmap = 'gray')
        plt.title(int(y[number]))
        
        