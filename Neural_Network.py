import numpy as np
#激活函数
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 计算sigmoid层的反向传播导数
def sigmoid_grad(x):
        return (1.0 - sigmoid(x)) * sigmoid(x)



class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #前向传播
    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    #反向传播
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

    #loss计算
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

    #计算准确率
    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy

    #训练
    def train(self,train_images,train_labels,test_images,test_labels,iters_num =6000,
              learning_rate = 0.01,learning_rate_decay = 0.90, verbose=False):
        train_loss_list = []
        train_acc_list = []
        test_loss_list=[]
        test_acc_list=[]

        train_size = train_images.shape[0]
        test_size=test_images.shape[0]
        batch_size = 100
        epoch = max(train_size / batch_size, 1)

        network = Neural_Network(input_size=784, hidden_size=50, output_size=10)  # 输入层为784，隐藏层节点数50，输出层为10

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = train_images[batch_mask]
            t_batch = train_labels[batch_mask]


            grad = network.backward(x_batch, t_batch)
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)


            if i % epoch == 0:
                test_loss=network.loss(test_images,test_labels)
                test_loss_list.append(test_loss)
                train_acc = network.accuracy(train_images, train_labels)
                train_acc_list.append(train_acc)
                test_acc=network.accuracy(test_images,test_labels)
                test_acc_list.append(test_acc)
                learning_rate *= learning_rate_decay
                print("train acc : %.7s, test acc : %.7s,test loss : %.7s"%(train_acc,test_acc,test_loss))

        return train_loss_list,train_acc_list,test_loss_list,test_acc_list

