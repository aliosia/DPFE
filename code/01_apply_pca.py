
# 1- Loading the original caffemodel (from https://github.com/camel007/caffe-moon)
# 2- Choosing the intermediate layer (here conv6-1)
# 3- Loading the output of intermediate layer
# 4- Apply PCA with d=10 on the output 
# 5- Create a new model with an embedded linear autoencoder initialized with PCA weights
# 6- Save this model
# 7- (not in this file, do it yourself) Finetune (train again) this model 
# 8- (not in this file, do it yourself) Save it in ../model/pca/pca_conv6-1.caffemodel

# The result is a model with the same efficiency of the main model but with a low dimensional intermediate layer 

import numpy as np
import copy
import caffe
from progressbar import ProgressBar
from sklearn import decomposition
caffe.set_mode_gpu()
caffe.set_device(0)


class CAFFENET:
    
    def __init__(self, name, definition, weight):
        self.net = caffe.Net(model, weights, caffe.TRAIN)
        self.definition = definition
        self.batch_size = 50
        self.name = name
        self.filt = np.ones(40).astype(bool)

    def load_middle_features(self, num_batch, layer_name):
        features = []
        attr_labels = []
        id_labels = []
        pbar = ProgressBar()
        for i in pbar(range(0, num_batch)):
            output = self.net.forward(blobs=[layer_name,'attribute_labels','id_labels'])
            features.append(copy.copy(output[layer_name]))
            attr_labels.append(copy.copy(output['attribute_labels']))
            id_labels.append(copy.copy(output['id_labels']))
        features = np.concatenate(features, axis=0)
        attr_labels = np.concatenate(attr_labels, axis=0)
        id_labels = np.concatenate(id_labels, axis=0)
        print('Features are loaded and saved')
        return features, attr_labels, id_labels
    
 
# Loading the main model
   
model = 'train_val_pca_conv7.prototxt'
weights = 'moon_tiny_iter_1000000.caffemodel' # this is ok
name = 'pca_conv6-1'

mynet = CAFFENET(name, model, weights)

print("\nblobs {}\nparams {}".format(mynet.net.blobs.keys(), mynet.net.params.keys()))


# Loading the output of intermediate layer

features, attr_labels, id_labels  = mynet.load_middle_features(1000, 'flat_1')



def my_pca(features, labels, num_pca):
    pca = decomposition.IncrementalPCA(batch_size=50)
    pca.n_components = num_pca
    pca.fit(features)
    X_reduced = pca.transform(features)
    print(len(np.unique(labels)))
    print('PCA is fitted')
    return pca, X_reduced


# Applying PCA
pca , temp = my_pca(features, id_labels[:,0,0,0], 10)


# Creating new caffemodel

net_params = {pr: (mynet.net.params[pr][0].data, mynet.net.params[pr][1].data) for pr in mynet.net.params.keys()[0:]}

# initialize weights of flat_shifted, pca layer
net_params['flat_1_shifted'][1][...] = - pca.mean_
net_params['pca_1'][0].flat = pca.components_
net_params['flat_1_shifted_reconst'][0][...] = pca.components_.T
net_params['flat_1_reconst'][1][...] = pca.mean_


pca_net = caffe.Net(model, caffe.TRAIN)


# put all the initial values of parameters of new net in a directory
pca_net_params = {pr: (pca_net.params[pr][0].data, pca_net.params[pr][1].data) for pr in pca_net.params.keys()[0:]}


for net_pr, pca_net_pr in zip(net_params, pca_net_params):
    pca_net_params[pca_net_pr][0].flat = net_params[net_pr][0].flat
    pca_net_params[pca_net_pr][1][...] = net_params[net_pr][1]

pca_net.save('moon_tiny_iter_1000000_pca_conv7.caffemodel')


# You should finetune this model later with caffe train

