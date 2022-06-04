from email.policy import default
from msilib.schema import Directory
from sqlalchemy import values
from tensorflow import keras
import tensorflow as tf
import numpy as np
#参数调优
from kerastuner.tuners import Hyperband
from  kerastuner.engine.hyperparameters import HyperParameters

#创建数据集s
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()  #划分数据集
#数据归一化
train_images=train_images/255
test_images=test_images/255
hp=HyperParameters()
#构建神经元模型
model=tf.keras.Sequential([
    #卷积层
    tf.keras.layers.Conv2D(hp.Choice("layer0",values=[32,64],defaul=16),   #卷积核
                        kernel_size=3,  #卷积核纬度
                        padding="same",  #padding
                        activation="relu", #激活函数
                        input_shape=(28,28,1)  #输入得纬度
                        ),
    #maxpooling
    tf.keras.layers.MaxPool2D(pool_size=2,  #纬度除2
                            strides=2),  #步长
    #卷积层
    tf.keras.layers.Conv2D(filters=64,   #卷积核
                        kernel_size=3,  #卷积核纬度
                        padding="same",  #padding
                        activation="relu" #激活函数
                        ),
    #maxpooling
    tf.keras.layers.MaxPool2D(pool_size=2,  #纬度除2
                            strides=2),  #步长
    #全连接层
    tf.keras.layers.Flatten(),  #数据扁平化
    tf.keras.layers.Dense(units=128,  #神经元个数
                        activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,  #神经元个数，因为有10个类别
                        activation=tf.nn.softmax)
])

#数据归一化
train_imges=train_images/255

#model.summary()  #查看模型
#定义优化器和损失函数
model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"]
                )
model.fit(x=train_images,y=train_labels,epochs=10)

#模型测试
test_imges_sd=test_images/255  #数据归一化
#模型评估
print("模型评估")
model_test=model.evaluate(test_images,test_labels)
print(model_test)

#查看分类是否正确
prediction=model.predict(test_imges_sd)[0]
print(np.argmax(prediction))  #取最大值
print(test_labels[0])
