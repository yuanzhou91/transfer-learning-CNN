#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget --no-check-certificate   https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip   -O cats_and_dogs_filtered.zip')


# In[ ]:


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
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')
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


# In[5]:


# define cnn model
def load_model_4_blocks_cnn_dropout():
    # load model
    model = keras.models.load_model('4blocks-cnn-double-convnet-dropout.h5')    
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


model_4_cnn_dropout = load_model_4_blocks_cnn_dropout()
model_4_cnn_dropout.summary()


# In[7]:


run_test_harness(model_4_cnn_dropout)


# ## Visualize model features map and saliency map

# In[8]:


from tensorflow.keras.preprocessing import image

img = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2468.jpg', target_size=(64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
pyplot.imshow(img, cmap="binary")


# In[9]:


layer_outputs = [layer.output for layer in model_4_cnn_dropout.layers]

activation_model = Model(inputs=model_4_cnn_dropout.input, outputs=layer_outputs)


# In[10]:


activations = activation_model.predict(img_tensor)
len(activations)


# In[11]:


# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model_4_cnn_dropout.layers:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    
    if layer_name == 'flatten': 
        break
    if 'conv' not in layer_name:
        continue
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    pyplot.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    pyplot.title(layer_name)
    pyplot.grid(False)
    pyplot.imshow(display_grid, aspect='auto', cmap='binary')
    
pyplot.show()


# In[12]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tf_keras_vis.saliency import Saliency
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize


# In[13]:


from tensorflow.keras.applications.vgg16 import preprocess_input


# Image titles
image_titles = ['Dog', 'Dog', 'Dog', 'Dog', 'Cat', 'Cat', 'Cat', 'Cat']

# Load images
img1 = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2470.jpg', target_size=(64, 64))
img2 = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2471.jpg', target_size=(64, 64))
img3 = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2472.jpg', target_size=(64, 64))
img4 = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2473.jpg', target_size=(64, 64))
img5 = image.load_img('cats_and_dogs_filtered/validation/cats/cat.2473.jpg', target_size=(64, 64))
img6 = image.load_img('cats_and_dogs_filtered/validation/cats/cat.2474.jpg', target_size=(64, 64))
img7 = image.load_img('cats_and_dogs_filtered/validation/cats/cat.2475.jpg', target_size=(64, 64))
img8 = image.load_img('cats_and_dogs_filtered/validation/cats/cat.2476.jpg', target_size=(64, 64))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3), 
                     np.array(img4), np.array(img5), np.array(img6), np.array(img7), np.array(img8)])

# Preparing input data
X = preprocess_input(images)

# Rendering
subplot_args = { 'nrows': 1, 'ncols': 8, 'figsize': (28, 4),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.show()


# In[14]:


# The `output` variable refer to the output of the model,
# so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
def loss(output):
    return (output[0][0], output[1][0], output[2][0], output[3][0], output[4][0], output[5][0], output[6][0], output[7][0])
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


# In[15]:


# Create Saliency object.
saliency = Saliency(model_4_cnn_dropout,
                    model_modifier=model_modifier,
                    clone=False)

# Generate saliency map with smoothing that reduce noise by adding noise
saliency_map = saliency(loss,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.
saliency_map = normalize(saliency_map)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(saliency_map[i], cmap='jet')
plt.tight_layout()
plt.savefig('smoothgrad_pretrained.png')
plt.show()


# In[16]:


from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Create Gradcam object
gradcam = Gradcam(model_4_cnn_dropout,
                  model_modifier=model_modifier,
                  clone=False)

# Generate heatmap with GradCAM
cam = gradcam(loss,
              X,
              penultimate_layer=-1, # model.layers number
             )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
plt.tight_layout()
plt.show()


# In[ ]:




