
# coding: utf-8

# In[3]:

import caffe
import numpy as np
from sklearn.externals import joblib


# In[8]:

modelAttr = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv4-2/Siamese/train_val_siamese_conv4-2.prototxt'
weightAttr  = '/media/dml/1TbyeAvailable/caffe-moon/result/conv4-2/model/[20]/siamese_[20]_iter_20000.caffemodel'


# In[9]:

AttrNet    = caffe.Net( modelAttr, weightAttr, caffe.TRAIN)


# In[10]:

print("\nblobs {}\nparams {}".format(AttrNet.blobs.keys(), AttrNet.params.keys()))


# In[11]:

modelAlex = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv4-2/Simple/train_val_simple_conv4-2.prototxt'
weightAlex  = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv4-2/Simple/trained-model/New_iter_170000.caffemodel'


# In[12]:

AlexNet    = caffe.Net( modelAlex, weightAlex, caffe.TEST)


# In[13]:

print("\nblobs {}\nparams {}".format(AlexNet.blobs.keys(), AlexNet.params.keys()))


# In[14]:

AttrNet.params['conv5_11'][0].data[...] = AlexNet.params['conv5_11'][0].data[...]  #copy filter across 2 nets
AttrNet.params['conv5_11'][1].data[...] = AlexNet.params['conv5_11'][1].data[...]  #copy bias
AttrNet.params['conv4_11'][0].data[...] = AlexNet.params['conv4_11'][0].data[...]  #copy filter across 2 nets
AttrNet.params['conv4_11'][1].data[...] = AlexNet.params['conv4_11'][1].data[...]  #copy bias
AttrNet.params['conv3_11'][0].data[...] = AlexNet.params['conv3_11'][0].data[...]  #copy filter across 2 nets
AttrNet.params['conv3_11'][1].data[...] = AlexNet.params['conv3_11'][1].data[...]  #copy bias
AttrNet.params['deconv5'][0].data[...] = AlexNet.params['deconv5'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv5'][1].data[...] = AlexNet.params['deconv5'][1].data[...]  #copy bias
AttrNet.params['deconv4'][0].data[...] = AlexNet.params['deconv4'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv4'][1].data[...] = AlexNet.params['deconv4'][1].data[...]  #copy bias
AttrNet.params['deconv3'][0].data[...] = AlexNet.params['deconv3'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv3'][1].data[...] = AlexNet.params['deconv3'][1].data[...]  #copy bias
AttrNet.params['deconv2'][0].data[...] = AlexNet.params['deconv2'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv2'][1].data[...] = AlexNet.params['deconv2'][1].data[...]  #copy bias
AttrNet.params['deconv1'][0].data[...] = AlexNet.params['deconv1'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv1'][1].data[...] = AlexNet.params['deconv1'][1].data[...]  #copy bias
AttrNet.params['deconv0'][0].data[...] = AlexNet.params['deconv0'][0].data[...]  #copy filter across 2 nets
AttrNet.params['deconv0'][1].data[...] = AlexNet.params['deconv0'][1].data[...]  #copy bias
AttrNet.params['defc5'][0].data[...] = AlexNet.params['defc5'][0].data[...]  #copy filter across 2 nets
AttrNet.params['defc5'][1].data[...] = AlexNet.params['defc5'][1].data[...]  #copy bias
AttrNet.params['defc6'][0].data[...] = AlexNet.params['defc6'][0].data[...]  #copy filter across 2 nets
AttrNet.params['defc6'][1].data[...] = AlexNet.params['defc6'][1].data[...]  #copy bias
AttrNet.params['defc7'][0].data[...] = AlexNet.params['defc7'][0].data[...]  #copy filter across 2 nets
AttrNet.params['defc7'][1].data[...] = AlexNet.params['defc7'][1].data[...]  #copy bias
AttrNet.params['fc5'][0].data[...]  = AlexNet.params['fc5'][0].data[...]
AttrNet.params['fc5'][1].data[...]  = AlexNet.params['fc5'][1].data[...]
#AttrNet.params['bn_conv5_1'][0].data[...] = AlexNet.params['bn_conv5_1'][0].data[...]
#AttrNet.params['bn_conv5_1'][1].data[...] = AlexNet.params['bn_conv5_1'][1].data[...]


# In[15]:

AttrNet.save('enc-siamese-conv4-2_dec-alexnet.caffemodel')


# In[11]:

AlexNet.params['fc5'][0].data[...]


# In[12]:

AttrNet.params['fc5'][0].data[...]


# In[ ]:



