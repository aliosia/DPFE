
# coding: utf-8

# In[1]:

import numpy as np
import caffe
import Image
import math
import cv2
import scipy.misc
import scipy.io
caffe.set_mode_gpu()


# In[2]:

def combine_images(generated_images):
    num = generated_images.shape[0]
    print num
    #width = int(math.sqrt(num))
    width = 10
    #height = int(math.ceil(float(num)/width))
    height = 5
    shape = generated_images.shape[1:]
    print(shape)
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img
    image = 255 * image
    return image


# In[3]:

model = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv4-2/PCA/train_val_pca_conv4-2.prototxt'
weights = '/media/dml/1TbyeAvailable/caffe-moon/Visualization/Yeh/conv4-2/PCA/trained-model/New_iter_230000.caffemodel'


# In[4]:


net = caffe.Net(model, weights, caffe.TEST)


# In[11]:

out = net.forward(blobs=['data','deconv0_crop'])


# In[12]:

res = out['deconv0_crop']
orig = out['data']
res2 = np.transpose(res , (0,2,3,1))
orig2 = np.transpose(orig, (0,2,3,1))
result = np.concatenate([orig2,res2],axis=1)


# In[13]:

f = combine_images(result)
cv2.imwrite('train_conv4-2_pca3_New.jpg', f)


# In[ ]:




# In[ ]:



