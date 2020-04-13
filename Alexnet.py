import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

# 定义网络的超参数
learning_rate = 0.001 # 学习效率
training_iters = 50000 # 迭代次数
batch_size = 128
display_step = 5


# 定义网络的参数
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10 # 标记的维度 (0-9 digits，MNIST 类别，一共10类)
dropout = 0.75 # Dropout的概率，输出的可能性，训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。

# 输入数据
# 导入 MINST 数据集，tensorflow.examples.tutorials 现在已经被高版本tensorflow弃用了，这里使用tensorflow.keras.datasets
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# mnist数据集的 onehot 和 shuffle 功能
def onehot(y,start,end,categories='auto'):
    ohot = OneHotEncoder()
    a = np.linspace(start,end-1,end-start)
    b = np.reshape(a,[-1,1]).astype(np.int32)
    ohot.fit(b)
    c = ohot.transform(y).toarray()
    return c
def MNISTLable_TO_ONEHOT(X_Train,Y_Train,X_Test,Y_Test,shuff=True):
    Y_Train = np.reshape(Y_Train,[-1,1])
    Y_Test = np.reshape(Y_Test,[-1,1])
    Y_Train = onehot(Y_Train.astype(np.int32),0,n_classes)
    Y_Test = onehot(Y_Test.astype(np.int32),0,n_classes)
    if shuff ==True:
        X_Train,Y_Train = shuffle(X_Train,Y_Train)
        X_Test,Y_Test = shuffle(X_Test,Y_Test)
        return X_Train,Y_Train,X_Test,Y_Test

X_train,y_train,X_test,y_test = MNISTLable_TO_ONEHOT(X_train,y_train,X_test,y_test)



# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


#构建网络模型


# 定义卷积操作
def conv2d(name,x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x,name=name)  # 使用relu激活函数

# 定义池化层操作，使用重叠的最大池化。此前CNN中普遍使用平均池化，AlexNet全部使用最大池化，避免平均池化的模糊化效果。
def maxpool2d(name,x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name=name)     #最大值池化

# 归一化操作，LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,
                     beta=0.75, name=name)

# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


#定义 Alexnet 网络模型


# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # 向量转为矩阵 Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1', conv1, k=2)
    # 归一化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 最大池化（向下采样）
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 归一化
    norm3 = norm('norm3', conv3, lsize=4)

    # 第四层卷积
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 最大池化（向下采样）
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 归一化
    norm5 = norm('norm5', pool5, lsize=4)

    # 在AlexNet中主要是最后几个全连接层使用了Dropout。
    # 全连接层1，，先把特征图转为向量
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 =tf.add(tf.matmul(fc1, weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout，训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。
    fc1=tf.nn.dropout(fc1,dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 =tf.add(tf.matmul(fc2, weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout，训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。
    fc2=tf.nn.dropout(fc2,dropout)

    # 网络输出层
    out = tf.add(tf.matmul(fc2, weights['out']) ,biases['out'])
    return out

#构建模型，定义损失函数和优化器，并构建评估函数

# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
# 这里定义损失函数时调用tf.nn.softmax_cross_entropy_with_logits() 函数必须使用参数命名的方式来调用 (logits=pred, labels=y)不然会报错。
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#训练模型和评估模型


# 初始化变量
init = tf.global_variables_initializer()


# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    acc = 0
    # 开始训练，直到达到training_iters，即50000
    while step * batch_size < training_iters:
        #获取批量数据
        batch_x = X_train[step*batch_size:(step+1)*batch_size,:]
        batch_x = np.reshape(batch_x,[-1,28*28]) # batch_x的shape为[batch_size,28,28], 这不符合我们定义的输入占位符，所以需要reshape batch_x的shape
        batch_y = y_train[step*batch_size:(step+1)*batch_size,:]
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # 计算损失值和准确度，输出
            loss,acc = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算测试集的准确率
    X_test = np.reshape(X_test,[-1,28*28])
    print ("Testing Accuracy:",
           sess.run(accuracy, feed_dict={x: X_test,
                                         y: y_test,
                                         keep_prob: 1.}))
    # 展示其中30个实际值与预测值，进行对比
    print(sess.run(tf.argmax(y_test[:30],1)),"Real Number")
    print(sess.run(tf.argmax(pred[:30],1),feed_dict={x:X_test,y:y_test,keep_prob: 1.}),"Prediction Number")
