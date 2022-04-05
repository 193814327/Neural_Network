import numpy as np
from keras.datasets import mnist
from Neural_Network import Neural_Network
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
import pickle
#导入数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28*28)).astype('float')
test_images=test_images.reshape((10000,28*28)).astype('float')
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network = Neural_Network(input_size=784, hidden_size=50, output_size=10)  #输入层为784，隐藏层节点数50，输出层为10
train_loss_list,train_acc_list,test_loss_list,test_acc_list=network.train(train_images,train_labels,test_images,test_labels)

#可视化W1，可视化成50个28×28的图片
W1=network.params['W1'].T
for i in range(1,51):
    x=W1[i-1].reshape(28, 28)
    plt.subplot(5,10,i)
    plt.imshow(x)
plt.show()

#保存模型
pickle.dump(network.params, open('model.pkl', 'wb'))

# 画损失函数的变化
x1 = np.arange(len(train_loss_list))
ax1=plt.subplot(221)
plt.plot(x1, train_loss_list, label='train loss')
plt.xlabel("iteration")
plt.ylabel("train loss")

x2 = np.arange(len(test_loss_list))
ax2=plt.subplot(222)
plt.plot(x2, test_loss_list, label='test loss')
plt.xlabel("iteration")
plt.ylabel("test loss")

# 画训练精度，测试精度随着epoch的变化
x3 = np.arange(len(train_acc_list))
ax3=plt.subplot(223)
plt.plot(x3, train_acc_list, label='train acc')
plt.xlabel("epochs")
plt.ylabel("train accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

x4 = np.arange(len(test_acc_list))
ax4=plt.subplot(224)
plt.plot(x4, test_acc_list, label='test acc')
plt.xlabel("epochs")
plt.ylabel("test accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
