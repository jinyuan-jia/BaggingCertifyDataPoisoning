from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np 
import os
import dataaug 


import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--start',default='0')
parser.add_argument('--end',default='1')
parser.add_argument('--k',default='30')
parser.add_argument('--gpu',default='0')
parser.add_argument('--gpum',default='0.1')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = float(args.gpum)
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


batch_size = 16
num_classes = 10
epochs = 200


# input image dimensions
img_rows, img_cols = 28, 28


(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape=x_train.shape[1:]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## here we use the same initialization for all models, and you can also use different initialization for different models
weights_initialize = model.get_weights()

### the parameter k is the number of training examples sampled from the training dataset to train each base model
k_value = int(args.k)

''' 
track the label frequency for each testing input, and the last dimension is used to save the true label, 
which is further used to compute the certified radius
'''
aggregate_result=np.zeros([x_test.shape[0],num_classes+1],dtype=np.int)


## data augmentation function
datagen = dataaug.DataGeneratorFunMNIST()

for repeat_time in range(int(args.start),int(args.end)):
    # sampling with replacement.
    sample_index=np.random.choice(x_train.shape[0],k_value,replace=True)
    
    x_train_sample=x_train[sample_index,:,:,:]
    y_train_sample=y_train[sample_index,:]
      
    # train the model
    model.fit_generator(datagen.flow(x_train_sample, y_train_sample, batch_size=batch_size),
                        epochs=epochs, verbose=0, workers=4)

    # evaluate the base model and you can also comment it without influencing the results.
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    prediction_label = np.argmax(model.predict(x_test),axis=1)
    aggregate_result[np.arange(0,x_test.shape[0]),prediction_label] += 1
    # reinitialize the model, note that you can also use different parameters to initialize the model
    model.set_weights(weights_initialize)
aggregate_result[np.arange(0,x_test.shape[0]),-1]=np.argmax(y_test,axis=1)

### save the results

tmp_folder = "./aggregate_result"
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)
tmp_folder +="/mnist"
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)
aggregate_folder = "./aggregate_result/mnist/k_"+args.k
if not os.path.exists(aggregate_folder):
    os.makedirs(aggregate_folder)
np.savez(aggregate_folder+"/aggregate_batch_k_"+args.k+"_start_"+args.start+"_end_"+args.end+".npz",x=aggregate_result)
