import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 加载数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 数据预处理
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# 创建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # 打印模型摘要
# model.summary()

# 训练模型
# model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

# # 评估模型
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_acc}")

# # 保存模型
# model.save('model/cifar10_cnn.h5')

# 加载模型 可以在另一文件中使用
model = tf.keras.models.load_model('model/cifar10_cnn.h5')

# 打印模型摘要
model.summary()

# # 使用模型对单张图片进行预测
img = x_test[0]
img = img.reshape(1, 32, 32, 3)
pred = model.predict(img)
print(f"Predicted class: {pred.argmax()}")
# 对比真实标签
print(f"True class: {y_test[0].argmax()}")