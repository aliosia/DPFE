
# coding: utf-8

# In[1]:

import caffe
import numpy as np
from sklearn.externals import joblib


# In[2]:

modelAttr = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv6-1/Siamese/train_val_siamese_conv6-1.prototxt'
weightAttr  = '/media/dml/1TbyeAvailable/caffe-moon/result/conv6-1/model/[20]/general_[20]_iter_20000.caffemodel'


# In[3]:

AttrNet    = caffe.Net( modelAttr, weightAttr, caffe.TRAIN)


# In[4]:

print("\nblobs {}\nparams {}".format(AttrNet.blobs.keys(), AttrNet.params.keys()))


# In[5]:

modelAlex = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv6-1/Simple/train_val_simple_conv6-1.prototxt'
weightAlex  = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv6-1/Simple/trained-model/New_iter_90000.caffemodel'


# In[6]:

AlexNet    = caffe.Net( modelAlex, weightAlex, caffe.TEST)


# In[7]:

print("\nblobs {}\nparams {}".format(AlexNet.blobs.keys(), AlexNet.params.keys()))


# In[8]:

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
AttrNet.params['fc6_1'][0].data[...]  = AlexNet.params['fc6_1'][0].data[...]
AttrNet.params['fc6_1'][1].data[...]  = AlexNet.params['fc6_1'][1].data[...]
#AttrNet.params['bn_conv5_1'][0].data[...] = AlexNet.params['bn_conv5_1'][0].data[...]
#AttrNet.params['bn_conv5_1'][1].data[...] = AlexNet.params['bn_conv5_1'][1].data[...]


# In[9]:

AttrNet.save('enc-siamese-conv6-1_dec-alexnet.caffemodel')


# In[11]:

AlexNet.params['fc5'][0].data[...]


# In[12]:

AttrNet.params['fc5'][0].data[...]


# In[ ]:



