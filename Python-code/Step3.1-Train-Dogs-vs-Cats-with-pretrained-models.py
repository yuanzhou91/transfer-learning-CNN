#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('wget --no-check-certificate   https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip   -O cats_and_dogs_filtered.zip')


# In[13]:


get_ipython().system('unzip -o -qq cats_and_dogs_filtered.zip')


# In[1]:


from tensorflow import keras
import os, sys
from matplotlib import pyplot

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np


# In[2]:


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# In[3]:


# run the test harness for evaluating a model
def run_test_harness(model):
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('cats_and_dogs_filtered/train/',
        class_mode='binary', batch_size=64, target_size=(64, 64))
    test_it = test_datagen.flow_from_directory('cats_and_dogs_filtered/validation/',
        class_mode='binary', batch_size=64, target_size=(64, 64))
    # Create EarlyStopping callback which stopps after 5 epochs of non-increasing accuracy
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=100, callbacks=[early])
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it))
    print('Test Accuracy > %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


# In[4]:


os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[15]:



# define cnn model
def load_model_1_block_cnn():
    # load model
    model = keras.models.load_model('baseline.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[17]:


model_1_cnn = load_model_1_block_cnn()
model_1_cnn.summary()


# In[18]:


run_test_harness(model_1_cnn)


# In[5]:


def load_model_2_blocks_cnn():
    # load model
    model = keras.models.load_model('2blocks-cnn.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[6]:


model_2_cnn = load_model_2_blocks_cnn()
model_2_cnn.summary()


# In[7]:


run_test_harness(model_2_cnn)


# In[19]:


def load_model_3_blocks_cnn():
    # load model
    model = keras.models.load_model('3blocks-cnn.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[20]:


model_3_cnn = load_model_3_blocks_cnn()
model_3_cnn.summary()


# In[21]:


run_test_harness(model_3_cnn)


# In[22]:


def load_model_4_blocks_cnn():
    # load model
    model = keras.models.load_model('4blocks-cnn.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[23]:


model_4_cnn = load_model_4_blocks_cnn()
model_4_cnn.summary()


# In[43]:


run_test_harness(model_4_cnn)


# In[25]:


def load_model_4_blocks_cnn_50_epochs():
    # load model
    model = keras.models.load_model('4blocks-cnn-50-epochs.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[26]:


model_4_cnn_50_epochs = load_model_4_blocks_cnn_50_epochs()
model_4_cnn_50_epochs.summary()


# In[27]:


run_test_harness(model_4_cnn_50_epochs)


# In[7]:


# define cnn model
def load_model_5_blocks_cnn():
    # load model
    model = keras.models.load_model('5blocks-cnn.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[8]:


model_5_cnn = load_model_5_blocks_cnn()
model_5_cnn.summary()


# In[46]:


run_test_harness(model_5_cnn)


# In[9]:


# define cnn model
def load_model_5_blocks_cnn_dropout():
    # load model
    model = keras.models.load_model('5blocks-cnn-50epochs-dropout.h5')    
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-4].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[10]:


model_5_cnn_dropout = load_model_5_blocks_cnn()
model_5_cnn_dropout.summary()


# In[14]:


run_test_harness(model_5_cnn_dropout)


# In[ ]:




