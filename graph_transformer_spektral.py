#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime
import numpy as np
import os
import pandas as pd
import spektral
import tensorflow as tf
import tensorflow_addons as tfa

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.input_layer import InputLayer
from keras.models import load_model
from keras.layers import Activation
from keras import backend as K
from scipy.sparse import csr_array, csr_matrix, load_npz

tf.keras.backend.set_floatx('float32')
print("GPUs: {}".format(len(tf.config.list_physical_devices("GPU"))))


# In[2]:


class GraphDataset(spektral.data.Dataset):
    def __init__(self, path, **kwargs):
        self.data_path = path
        super().__init__(**kwargs)

    def read(self):
        output = []
        for i in range(int(len(os.listdir(self.data_path))/2)):
            graph = np.load(self.data_path + "graph_{}.npz".format(i))
            adjacency = load_npz(self.data_path + "adjacency_{}.npz".format(i))
            output.append(spektral.data.graph.Graph(x=graph['x'],
                                                    a=adjacency,
                                                    y=graph['y']))
        return output


# In[3]:


train_ds = GraphDataset("dataset/train/")
test_ds = GraphDataset("dataset/test/")
val_ds = GraphDataset("dataset/validation/")


# In[14]:


loader_train = spektral.data.BatchLoader(train_ds, batch_size=1, shuffle=False)
loader_test = spektral.data.BatchLoader(test_ds, batch_size=1, shuffle=False)
loader_val = spektral.data.BatchLoader(val_ds, batch_size=1, shuffle=False)


# In[15]:


class GraphAttentionNetwork(tf.keras.models.Model):
    def __init__(self, nlayers=1, dim_features=10, dim_global_features=6, dropout=0.5):
        super().__init__()
        self.nlayers = nlayers
        self.dim_features = dim_features
        self.dim_global_features = dim_global_features
        self.dropout = dropout

        self.attention = []
        self.skip = []
        for i in range(self.nlayers):
            self.attention.append(spektral.layers.GATConv(channels=self.dim_features, attn_heads=1, dropout_rate=self.dropout))
            self.skip.append(tf.keras.layers.Dense(self.dim_features, activation="relu", use_bias=True))

        self.avgpool = spektral.layers.GlobalAvgPool()
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.regression = [tf.keras.layers.Dense(128, activation="relu", use_bias=True)]
        self.regression.append(tf.keras.layers.Dense(128, activation="relu", use_bias=True))
        self.regression.append(tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True)) 

        self.global_features = [tf.keras.layers.Dense(12, activation="relu", use_bias=True)]
        self.global_features.append(tf.keras.layers.Dense(12, activation="relu", use_bias=True))
        self.global_features.append(tf.keras.layers.Dense(6, activation="relu", use_bias=True))

    def call(self, inputs):
        x = inputs[0][:,1:,:]
        a = inputs[1][:,:-1,:-1]
        gf = inputs[0][:,0,:6]

        for (attention_layer, skip_layer) in zip(self.attention, self.skip):
            x_attention = attention_layer([x,a])
            x_skip = skip_layer(x)
            x = x_skip + x_attention
            #x = self.layernorm(x)

        x = self.avgpool(x)

        for layer in self.global_features:
            gf = layer(gf)
        
        x = self.concat([x, gf])
        
        for layer in self.regression:
            x = layer(x)
        return x


# In[16]:


layers = 1
model = GraphAttentionNetwork(nlayers=layers)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])


# In[17]:


log_dir = "logs/test/"
log_dir_train = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir_train, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss',
                               restore_best_weights=True, patience=50,
                               verbose=0, mode='min')


# In[18]:


model.fit(loader_train.load(),
          steps_per_epoch=loader_train.steps_per_epoch,
          epochs=50,
          validation_data=loader_val.load(),
          validation_steps=loader_val.steps_per_epoch,
          callbacks=[tensorboard, early_stopping],
          verbose=1)


# In[ ]:


model.summary()


# In[ ]:


model.save_weights(log_dir + "model_weights")

