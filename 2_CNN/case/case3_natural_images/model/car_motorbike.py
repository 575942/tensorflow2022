import tensorflow as tf
from keras.preprocessing.image import  ImageDataGenerator,load_img,img_to_array  #数据增强
import matplotlib.pyplot as plt 

inpt_shape=250
batch_size=100


#定义数据增增强器
train_datagen=ImageDataGenerator(
                            rescale=1/255,    #数据归一化
                            )   
path=r"tensroflow2022/2_CNN/case/case3_natural_images/data/train"
train_data=train_datagen.flow_from_directory(directory=path,
                                            batch_size=batch_size,      #批次
                                            target_size=(inpt_shape,inpt_shape),  #图片的纬度
                                            class_mode="sparse"  #二分类
                                            )
train_images, _ = next(train_data)  #转化后的数据

#测试数据
path1=r"tensroflow2022/2_CNN/case/case3_natural_images/data/test"
test_datagen=ImageDataGenerator(rescale=1/255)  
test_data=test_datagen.flow_from_directory(directory=path1,
                                            batch_size=batch_size,      #批次
                                            target_size=(inpt_shape,inpt_shape),  #图片的纬度
                                            class_mode="sparse"  #二分类
                                            )   

model=tf.keras.Sequential([
    #卷积层s
    tf.keras.layers.Conv2D(filters=64,
                            kernel_size=3,
                            activation="relu",
                            padding="same",
                            input_shape=(inpt_shape,inpt_shape,3) #3代表的是RGB图像
                            ),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),

    #全连接层
    tf.keras.layers.Flatten(),  #数据扁平化
    #hidden layer
    tf.keras.layers.Dense(units=100,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=200,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    #output layer
    tf.keras.layers.Dense(units=5,  
                        activation=tf.nn.softmax)  #二分类问题
])

model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',  #多
                metrics=["accuracy"]
                )
#model.summary()  #查看模型参数
 #模型训练

hist=model.fit_generator(train_data,epochs=10,validation_data=test_data)
#模型验证
print(model.evaluate_generator(test_data))
