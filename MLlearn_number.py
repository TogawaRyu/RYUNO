import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def ReLU(X):#ReLU関数の実装
    Y = np.zeros(len(X))
    for i in range (len(X)):
        Y[i] = max(0,X[i])     
    return Y

def diff_ReLU(X):#ReLU関数の微分を実装
    Y  = np.zeros(len(X))
    for i in range((len(X))):
        if X[i] > 0:
            Y[i] = 1
    return Y
    
def Softmax(X):#Softmax関数の実装
    Z = np.exp(X)
    SZ = np.sum(Z)
    Y = Z/SZ
    return Y


def CrossEntropy(X,t):#交差エントロピーの実装
    E = 0
    for n in range(len(t)):
        h1 = ReLU(np.dot(W1,X[n]) + b1)
        h2 = ReLU(np.dot(W2,h1) + b2)
        Y = Softmax(np.dot(W3,h2) + b3)
        for k in range(len(t[0])): 
            E = t[n,k]*np.log(Y[k]) + E
    return E*-1


def Percent(X,Y):#正答率を計算
    C = 0
    for i in range(len(Y)):
        h1 = ReLU(np.dot(W1,X[i]) + b1)
        h2 = ReLU(np.dot(W2,h1) + b2)
        h3 = Softmax(np.dot(W3,h2) + b3)
        if np.argmax(h3) == np.argmax(Y[i]):
            C = C + 1
    return C/len(Y)
    
    

if __name__ == '__main__':
    
    mnist = datasets.fetch_mldata('MNIST original', data_home = '.')

    n= len(mnist.data)
    N = 10000#使用するデータの数
    N_train = 9000#学習に使用するデータの数
    N_validation = 1000#学習中の評価に使用するデータの数
    indice = np.random.permutation(range(n))[:N]
    X = mnist.data[indice]
    X = X / 225.0#Xの値を最大値が1になるように正規化
    y = mnist.target[indice]
    Y = np.eye(10)[y.astype(int)]#1ofKに変換
    
    X_train, X_test, Y_train, Y_test =train_test_split(X,Y,train_size = N_train)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train,test_size =N_validation)
    
    neuron = 1000
    eta  = 0.005
    epoch = 100
    
    
    W1 = np.random.randn(neuron,len(X[0]))*0.01
    b1 = np.zeros(neuron)
    W2 = np.random.randn(neuron,neuron)*0.01
    b2 = np.zeros(neuron)
    W3 = np.random.randn(len(Y[0]),neuron)*0.01
    b3 = np.zeros(len(Y[0]))
    
    Entro = []
    Per = []
    batch_size = 200
    n_batch = (N_train-N_validation)//batch_size
    flag = False
    for counter in range(epoch):
        X_, Y_ = shuffle(X_train,Y_train)
        for i in range(n_batch):
            start = i*batch_size
            end = start + batch_size
            x = X_[start:end]
            t = Y_[start:end]
            dEdW3 = np.zeros([len(W3),len(W3[0])])
            dEdW2 = np.zeros([len(W2),len(W2[0])])
            dEdW1 = np.zeros([len(W1),len(W1[0])])
            dEdb3 = np.zeros(len(b3))
            dEdb2 = np.zeros(len(b2))
            dEdb1 = np.zeros(len(b1))
            for j in range(len(x)):
                h1 = ReLU(np.dot(W1,x[j]) + b1)
                h2 = ReLU(np.dot(W2,h1) + b2)
                h3 = Softmax(np.dot(W3,h2) + b3) #まれにこの値が下で飽和する
                if( h3[0] != h3[0] ):#ｈ３の値がnanだとエラーになるからnanになったらループをとめる
                    print('miss!')
                    flag = True
                    break
                
                #    print(h3[0])
        
                e0 = -(t[j] - h3)
                e1 = diff_ReLU(np.dot(W2,h1) + b2)*np.dot(W3.T,e0)
                e2 = diff_ReLU(np.dot(W1,x[j]) + b1)*np.dot(W2.T,e1)
                
                dEdW3 = dEdW3 + np.outer(e0,h2)
                dEdb3 = dEdb3 + e0
                dEdW2 = dEdW2 + np.outer(e1,h1)
                dEdb2 = dEdb2 + e1
                dEdW1 = dEdW1 + np.outer(e2,x[j])
                dEdb1 = dEdb1 + e2
                
                
            
            if flag ==True:
                break
            
            W3 = W3 - eta*dEdW3
            b3 = b3 - eta*dEdb3
            W2 = W2 - eta*dEdW2
            b2 = b2 - eta*dEdb2
            W1 = W1 - eta*dEdW1
            b1 = b1 - eta*dEdb1
#            print(i)
            
        if flag ==True:
            print('miss!')
            break    
        E = CrossEntropy(X_validation,Y_validation)
        Entro.append(E)
        P = Percent(X_validation,Y_validation)
        Per.append(P)
        print('epoch=',counter,'Entropy=',E,'Percent=',P)
    
    fig, ax1 = plt.subplots()
    ax1.plot(Entro)
    ax2 = ax1.twinx()  # 2つのプロットを関連付ける
    ax2.plot(Per)
    plt.show()
    print(CrossEntropy(X_test,Y_test),Percent(X_test,Y_test))
    