# 19307110233
Neural_Network
#一.构建两层神经网络分类器

##1.初始化网络参数
```
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
```
##2.激活函数，前向传播
激活函数选择sigmoid函数，输出层选择用softmax进行输出，然后进行前向传播
```commandline
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))
```
```
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```
```commandline
    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
```


##3.loss计算，L2正则化
选择交叉熵损失函数，并进行L2正则化，正则化强度选择1e-4大小
```commandline
    def loss(self,x, t):
        y = self.forward(x)
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        data_loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        W1, W2 = self.params['W1'], self.params['W2']
        reg = 1e-4      #正则化强度
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)  # L2正则化
        return data_loss + reg_loss
```
##4.反向传播计算梯度
```commandline
    def backward(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        # 反向传播
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
```

##5.计算准确率
```commandline
    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy
```

#二.训练并测试该网络
##1.导入数据集
从keras.datasets库中导入mnist数据集
```commandline
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28*28)).astype('float')
test_images=test_images.reshape((10000,28*28)).astype('float')
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
```
##2.训练与测试预准备
首先设定好必要的超参数，更新次数iters_num =6000,学习率learning_rate = 0.01
学习率衰减参数learning_rate_decay = 0.90
```
    def train(self,train_images,train_labels,test_images,test_labels,iters_num =6000,
              learning_rate = 0.01,learning_rate_decay = 0.90, verbose=False):
```
保存下来训练集，测试集的loss和accura
```commandline
        train_loss_list = []
        train_acc_list = []
        test_loss_list=[]
        test_acc_list=[]
```
选择epoch大小
```commandline
        train_size = train_images.shape[0]
        test_size=test_images.shape[0]
        batch_size = 100
        epoch = max(train_size / batch_size, 1)
```
输入层为784，隐藏层节点数50，输出层为10
```commandline
        network = Neural_Network(input_size=784, hidden_size=50, output_size=10)  # 输入层为784，隐藏层节点数50，输出层为10
```
##3.正式开始训练
下面正式开始训练，更新参数次数6000次，每次随机抽取100个训练图片来更新参数,并记录下每次的train_loss值
```commandline
        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = train_images[batch_mask]
            t_batch = train_labels[batch_mask]


            grad = network.backward(x_batch, t_batch)
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
```
##4.每隔一个epoch进行测试
最后每隔一个epoch输出train_acc，test_loss，test_acc，并使学习率下降
```commandline
            if i % epoch == 0:
                test_loss=network.loss(test_images,test_labels)
                test_loss_list.append(test_loss)
                print(test_loss)
                train_acc = network.accuracy(train_images, train_labels)
                train_acc_list.append(train_acc)
                test_acc=network.accuracy(test_images,test_labels)
                test_acc_list.append(test_acc)
                learning_rate *= learning_rate_decay
                print("train acc : %.7s, test acc : %.7s,test loss : %.7s"%(train_acc,test_acc,test_loss))
```
##5.画曲线及保存模型，可视化网络参数
最后将train_loss,test_loss,train_acc,test_acc曲线画出，并将W1可视化，保存模型
```commandline
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
```
