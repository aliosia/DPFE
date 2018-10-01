
# coding: utf-8

# In[1]:

import caffe
import numpy as np


# In[2]:

model_tiny_pca = '../PCA/cut-after-pca/whole/train_val_inf_pca_visualization-alexnet.prototxt'
weight_tiny_pca  = '../../../models/visualization_models/Saved/_iter_210000.caffemodel'


# In[3]:

tiny_pca_Net    = caffe.Net( model_tiny_pca, weight_tiny_pca, caffe.TRAIN)


# In[4]:

print("\nblobs {}\nparams {}".format(tiny_pca_Net.blobs.keys(), tiny_pca_Net.params.keys()))


# In[6]:

tiny_pca_Net.params['flat_1_shifted'][1].data


# In[13]:

tiny_pca.params['flat_1_shifted'][1].data


# In[7]:

model_pca = '../../Yeh/PCA/cut-after-pca/train_val_inf_pca_visualization-alexnet.prototxt'
weight_pca  = '../../../models/visualization_models_before_pca/_iter_80000.caffemodel'


# In[8]:

tiny_pca    = caffe.Net( model_pca, weight_pca, caffe.TEST)


# In[9]:

print("\nblobs {}\nparams {}".format(tiny_pca_Net.blobs.keys(), tiny_pca_Net.params.keys()))


# In[10]:

tiny_pca_Net.params['flat_1_shifted'][0].data[...] = tiny_pca.params['flat_1_shifted'][0].data[...]
tiny_pca_Net.params['flat_1_shifted'][1].data[...] = tiny_pca.params['flat_1_shifted'][1].data[...]
tiny_pca_Net.params['pca_1'][0].data[...] = tiny_pca.params['pca_1'][0].data[...]
tiny_pca_Net.params['pca_1'][1].data[...] = tiny_pca.params['pca_1'][1].data[...]
tiny_pca_Net.params['flat_1_shifted_reconst'][0].data[...] = tiny_pca.params['flat_1_shifted_reconst'][0].data[...]
tiny_pca_Net.params['flat_1_shifted_reconst'][1].data[...] = tiny_pca.params['flat_1_shifted_reconst'][1].data[...]
tiny_pca_Net.params['flat_1_reconst'][0].data[...] = tiny_pca.params['flat_1_reconst'][0].data[...]
tiny_pca_Net.params['flat_1_reconst'][1].data[...] = tiny_pca.params['flat_1_reconst'][1].data[...]


# In[ ]:

tiny_pca_Net.save('tuned-pca.caffemodel')


# In[ ]:



