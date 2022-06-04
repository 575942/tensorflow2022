from tensorflow import keras
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np

#创建数据集s
fashion_mnist=keras.datasets.fashion_mnist
(train_imges,train_labels),(test_imges,test_labels)=fashion_mnist.load_data()  #划分数据集

#构建神经元模型
model=tk.Sequential([
    ##输入层
    tk.layers.Flatten(input_shape=(28,28)),  #全连接层
    #隐藏层
    tk.layers.Dense(units=128,                 #构建128个神经元
                    activation=tf.nn.relu  #激活函数
                    ),
    #输出层
    tk.layers.Dense(units=10,        
                    activation=tf.nn.softmax  #(0~1)之间
                    )
    ])
#数据归一化
train_imges=train_imges/255

#model.summary()  #查看模型
#定义优化器和损失函数
model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"]
                )
model.fit(x=train_imges,y=train_labels,epochs=10)
#数据归一化
train_imges=train_imges/255

#model.summary()  #查看模型
#定义优化器和损失函数
model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"]
                )
model.fit(x=train_imges,y=train_labels,epochs=10)

#模型测试
test_imges_sd=test_imges/255  #数据归一化
prediction=model.predict(test_imges_sd)[0]
print(np.argmax(prediction))  #取最大值
print(test_labels[0])
