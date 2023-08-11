import tensorflow as tf

from tensorflow import keras 
model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
#model = tf.keras.applications.VGG19(include_top=True, weights='imagenet', input_tensor-=None, input_shape=None,pooling=None, classes=1000,classifier_activation='softmax')
#model = tf.keras.applications.MobileNet(input_shape=None,alpha=1.0,depth_multiplier=1, dropout=0.001,include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000,classifier_activation='softmax')
#model = tf.keras.applications.ResNet152(include_top=True, weights='imagenet' , input_tensor=None, input_shape=None, pooling=None, classes=1000)

model.summary()

img_path = 'a.jpg'

from tensorflow.keras.preprocessing import image

img = image.load_img(img_path, target_size= (224,224))

x = image.img_to_array(img)
x.shape

import numpy as np

x = np.expand_dims(x, axis = 0)
x.shape

from keras.applications.vgg19 import preprocess_input
#from keras.applications.mobilenet import preprocess_input
#from keras.applications.resnet import preprocess_input
#from keras.applications.densenet import preprocess_input

x = preprocess_input(x)

from keras.applications.vgg19 import decode_predictions
#from keras.applications.mobilenet import decode_predictions
#from keras.applications.resnet import decode_predictions
#from keras.applications.densenet import decode_predictions

#preds = model.predict(x)
features = model.predict(x)

features.shape

#decode_predictions(preds, top =5)[0]
