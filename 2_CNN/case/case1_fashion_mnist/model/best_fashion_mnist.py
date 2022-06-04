from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
from tensorflow import keras

#创建数据集
fashion_mnist=keras.datasets.fashion_mnist
(train_imges,train_labels),(test_imges,test_labels)=fashion_mnist.load_data()  #划分数据集
#数据归一化
train_imges_sd=train_imges/255
test_imges_sd=test_imges/255
hp=HyperParameters()

#构建神经元模型
def cnn(hp):
    model=tf.keras.Sequential([
        #卷积层
        tf.keras.layers.Conv2D(filters=hp.Int("filter1",min_value=32,max_value=64,step=16),   #卷积核
                            kernel_size=3,  #卷积核纬度
                            padding="same",  #padding
                            activation="relu", #激活函数
                            input_shape=(28,28,1)  #输入得纬度
                            ),
        #maxpooling
        tf.keras.layers.MaxPool2D(pool_size=2,  #纬度除2
                                strides=2),  #步长
        #卷积层
        tf.keras.layers.Conv2D(filters=hp.Int("filter2",min_value=32,max_value=64,step=16),   #卷积核
                            kernel_size=3,  #卷积核纬度
                            padding="same",  #padding
                            activation="relu" #激活函数
                            ),
        #maxpooling
        tf.keras.layers.MaxPool2D(pool_size=2,  #纬度除2
                                strides=2),  #步长

        #全连接层
        tf.keras.layers.Flatten(),  #数据扁平化
        tf.keras.layers.Dense(units=hp.Int("hidden_units1",min_value=32,max_value=128,step=16),  #神经元个数
                            activation=tf.nn.relu),
        tf.keras.layers.Dense(units=hp.Int("output_units",min_value=32,max_value=128,step=16),  #神经元个数，因为有10个类别
                            activation=tf.nn.softmax)
    ])
    #定义优化器和损失函数
    model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"]
                )
    return model

tuner=Hyperband(
    hypermodel=cnn,  #优化对象
    objective="val_accuracy",  #用测试集得精确度作为参数选择的标准
    max_epochs=5,  #最大训练次数
    directory="fashion_parameters",  #保存参数的文件夹
    project_name='fashion',  #
    hyperparameters=hp  #使用的变量
)
tuner.search(x=train_imges,y=train_labels,epochs=3,validation_data=(test_imges_sd,test_labels)) #搜索参数

best_hp=tuner.get_best_hyperparameters(1)[0]  #获取最佳参数
print(best_hp.values)

#用这些参数构建最佳模型
best_model=tuner.hypermodel.build(best_hp)  #最佳模型
best_model.summary()

#模型训练
best_model.fit(x=train_imges_sd,y=train_labels,epochs=10,validation_data=(test_imges_sd,test_labels))
print(best_model.evaluate(test_imges,test_labels))

