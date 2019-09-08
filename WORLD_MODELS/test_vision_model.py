from vision_model import VisionModel
import tensorflow as tf
from tensorflow import keras
import cv2

batch_size = 64
height = 28
width = 28

fashion_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
shape = train_images.shape + (1,)
train_images = train_images.reshape(shape) / 255

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = VisionModel(sess)
model.build_model(height,width)

sess.run(tf.global_variables_initializer())

for i in range(50) :
    print('iteration : {}'.format(i))
    model.train(train_images)

for i in range(100) :
    test_shape = (1,) + train_images[i].shape
    test_image = train_images[i].reshape(test_shape)
    res = model.predict_decoder(test_image)[0] * 255
    cv2.imwrite('images/output_file' + str(i) + '.png',res)