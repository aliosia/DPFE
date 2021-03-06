# use it in Jupyter



# coding: utf-8

# In[1]:

import numpy as np
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
import copy
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from __future__ import division
from scipy.stats.mstats import rankdata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from google.protobuf import text_format
import caffe.draw
from caffe.proto import caffe_pb2
get_ipython().magic('matplotlib inline')
import time
import lmdb
from sklearn.cross_validation import train_test_split
from progressbar import ProgressBar
from scipy.stats import kde
import os
import ast


# In[2]:

class CAFFENET:
    
    def __init__(self, name, prototxt_path, weight_path, output_name = 'prob', mse = False):
        self.net = caffe.Net(prototxt_path, weight_path, caffe.TEST)
        self.definition = prototxt_path
        self.batch_size = 50
        self.name = name
        self.output_name = output_name
        self.filt = np.ones(40).astype(bool)
        self.mse = mse
    
    
    def print_structure(self):
        for item in self.net.blobs:
            print item, self.net.blobs[item].data.shape
    
    
    def draw_caffe_net(self, output_path):
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(self.definition).read(), net)
        caffe.draw.draw_net_to_file(net, output_path)

    
    def set_filter(self, new_filter):
        self.filt = new_filter.astype(bool)
        #print 'Filter set to: ', self.filt.astype(int)

        
    def get_accuracy(self, num_batch, separated=False):
        if not separated:
            total_error = 0
        else:
            total_error = np.zeros(40)
        for i in range(num_batch):
            output = self.net.forward(blobs=['attribute_labels',self.output_name])
            if not separated:
                total_error += self.batch_error(output['attribute_labels'], output[self.output_name])
            else:
                total_error += self.batch_error_separated(output['attribute_labels'], output[self.output_name]) 
        error = 100.0 * total_error/(num_batch * self.batch_size)
        self.accuracy = 100-error
        #print ('Accuracy: {}%'.format(self.accuracy))
        # if separated=False return one number and if separated=True return 40 dimensional separated accs.
        return self.accuracy
    
    def batch_error_separated(self, ground_blob, pred_blob): # separated errors on all attributes
        batch_size = ground_blob.shape[0]
        if self.mse:
            out = np.sign(pred_blob)
        else:
            out = np.sign(pred_blob - .5)
        attr = ground_blob[:,:,0,0]
        res = 0.5 * np.abs(attr - out)
        err = np.sum(res,axis=0)
        return err  
    
    def batch_error(self, ground_blob, pred_blob): # average error on filtered attributes
        batch_size = ground_blob.shape[0]
        if self.mse:
            out = np.sign(pred_blob)
        else:
            out = np.sign(pred_blob - .5)
        attr = ground_blob[:,:,0,0]
        out = out[:,self.filt] 
        attr = attr[:,self.filt]
        res = 0.5 * np.abs(attr - out)
        err = int(np.sum(res))/self.filt.sum()
        return err    
 

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
        save_name = '../result/features/' + self.name + '_' + str(num_batch)
        np.savez(save_name, features=features, attributes=attr_labels, ids = id_labels)
        print('Features are loaded and saved')
        return features, attr_labels, id_labels
    
    
    #attention: it does not works if there is two top layer for middle_layer so use the right prototxt (test_siamese_pca)
    def get_acc_from_middle(self, middle_layer_name, next_layer_name, features, labels, num_batch, separated=False):
        if not separated:
            total_error = 0
        else:
            total_error = np.zeros(40)
        for i in range(0, num_batch):
            self.net.blobs[middle_layer_name].data[...] = features[i*self.batch_size:(i+1)*self.batch_size,:] 
            self.net.blobs['attribute_labels'].data[...] = labels[i*self.batch_size:(i+1)*self.batch_size,:,:,:]
            output = self.net.forward(start=next_layer_name, end=self.output_name) 
            if not separated:
                total_error += self.batch_error(labels[i*self.batch_size:(i+1)*self.batch_size,:,:,:], output[self.output_name])
            else:
                total_error += self.batch_error_separated(labels[i*self.batch_size:(i+1)*self.batch_size,:,:,:], output[self.output_name])
        error = 100.0 * total_error/(num_batch * self.batch_size)
        self.middle_accuracy = 100-error
        #print ('Accuracy from middle layer: {}%'.format(self.middle_accuracy))
        # if separated=False return one number and if separated=True return 40 dimensional separated accs.
        return self.middle_accuracy
            
    def check_layers(self):
        output = self.net.forward(blobs=['attribute_labels', 'filtered_labels', self.output_name, 
                            'filtered_output', 'id_labels', 'attr_loss'])
        print output['attribute_labels'][15,20,0,0]
        print output['filtered_labels'][15,20,0,0]
        #print output['filtered_labels'][:,20,0,0] == output['filtered_labels_b'][:,20,0,0]
        print 'here'
        print output['id_labels'][:,0,0,0]
        #print output['id_labels_b'][:,0,0,0]
        #print output['id_not_same'][:,0,0,0]
        #print 'here'
        #print output['attr_same'][:,0,0,0]
        #print 'here'
        return output
   

    def check_weights(self):
        source_params = {}
        for pr in self.net.params.keys():
            source_params[pr] = []
            for i in range(len(self.net.params[pr])):
                source_params[pr].append(self.net.params[pr][i].data)
        print source_params.keys()
        a = ['conv1', 'bn_conv1', 'conv2', 'bn_conv2', 'conv3', 'bn_conv3', 'conv4_1', 'bn_conv4_1',
            'conv4_2', 'flat_1_shifted', 'pca_1', 'bn_pca_1']
        b = ['conv1_b', 'bn_conv1_b', 'conv2_b', 'bn_conv2_b', 'conv3_b', 'bn_conv3_b', 'conv4_1_b', 'bn_conv4_1_b',
            'conv4_2_b', 'flat_2_shifted', 'pca_2', 'bn_pca_2']
        for pa, pb in zip(a,b):
            print pa, '-', pb
            for i in range(len(source_params[pa])):
                print (source_params[pa][i]==source_params[pb][i]).all()
                if pa=='bn_pca_1':
                    print source_params[pa][i]
                    print source_params[pb][i]
    


# In[3]:

class PrivacyAnalyzer:
    
    # features, id_labels and attr_labels shoud be caffe blob data with 4 dimension. 
    # accuracy_method is a method passed to this class 
    # which can take 2 caffe blobs (f,z), estimate p(z|f), and compute accuracy. 
    def initialize(self, features, id_labels, attr_labels, accuracy_method):
        self.features = features
        self.cov = np.cov(features, rowvar=False)
        self.id_labels = id_labels
        self.attr_labels = attr_labels
        self.dim = features.shape[1]
        self.num = features.shape[0]
        self.accuracy_method = accuracy_method
        self.scales = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2.5, 4, 6, 8])
        self.noise_acc = {}
        self.noise_priv = {}
        self.eps = .8   

    
    
    def get_noisy_features(self, scale):
        if scale==0:
            return self.features + 0
        #noise = np.random.multivariate_normal(np.zeros([self.dim,]), noise_var*np.eye(self.dim), self.num)
        noise = np.random.multivariate_normal(np.zeros([self.dim,]), scale * self.cov, self.num)
        return self.features + noise
    
    
    def train_test(self, features, labels):
        x = np.array(features)
        y = np.array(labels)
        y = y[:,0,0,0]
        # removing ids with less than 10 images
        u, counts = np.unique(y, return_counts=True)
        mai = u[counts>18]
        a = np.isin(y,mai)
        x = x[a,:]
        y = y[a]
        # separating train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=14)
        # removing ids existed in test but not in train
        a = np.isin(y_test, y_train)
        y_test = y_test[a]
        x_test = x_test[a,:]
        return x_train, x_test, y_train, y_test 
    
    def privacy(self, features, labels):
        x_train, x_test, y_train, y_test = self.train_test(features, labels)
        knn_priv = self.knn_privacy(x_train, x_test, y_train, y_test)
        kde_priv = self.kde_privacy(x_train, x_test, y_train, y_test)
        priv = [knn_priv] + kde_priv
        return np.asarray(priv)
    
    # output 1nn error as the privacy (more 1nn error, more opt bayes error, more privacy and less accuracy)
    def knn_privacy(self, x_train, x_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train, y_train)
        predictions = knn.predict(x_test)
        return 1 - accuracy_score(y_test,predictions)
  
    
    def create_kdes(self, features, labels):
        kdes = {}
        for label in np.unique(labels):
            kdes[label] = kde.gaussian_kde(features[labels==label,:].T)
        return kdes
     
    
    def kde_privacy(self, x_train, x_test, y_train, y_test):
        num_test = y_test.shape[0]
        kdes = self.create_kdes(x_train, y_train)
        p_vals = []
        maps = {}
        cnt = 0
        for (l,k) in kdes.iteritems():
            p_vals.append(k.evaluate(x_test.T))
            maps[l] = cnt
            cnt += 1
        p_vals = np.asarray(p_vals) # num_label(num_kde) * num_data p(i,j)=p(data j comes from kde i)
        num_labels = p_vals.shape[0]
        probs = p_vals/np.max(p_vals,0)
        #tops = np.sum(probs>self.eps,0) 
        ranks = 1 + num_labels - rankdata(probs,0) #+1 is ok?
        trueranks = np.zeros(num_test)
        for i in range(num_test):
            trueranks[i] = ranks[maps[y_test[i]],i] 
        #ek_priv = np.maximum(tops/num_labels,trueranks/num_labels)
        #e_priv = tops/num_labels
        rank_priv = trueranks #/num_labels
        log_rank_priv = np.log(rank_priv)
        opt_bayes = 1 - sum(trueranks==1)/num_test
        priv = [num_labels, opt_bayes, np.mean(rank_priv), np.std(rank_priv), np.mean(log_rank_priv), np.std(log_rank_priv)]

        return priv
        
    
    def acc_priv(self, model_type):
        #middle_name = 'pca_1'
        #if model_type == 'dpfe':
        middle_name = 'bn_pca_1'
        for scale in self.scales:
            print scale
            current_features = self.get_noisy_features(scale)
            self.noise_acc[scale] = self.accuracy_method(middle_name, 'flat_1_shifted_reconst', 
                                                         current_features, self.attr_labels, 
                                                         num_batch=100, separated=True)
            #if scale==0: 
            #    print(self.noise_acc[scale])
            self.noise_priv[scale] = self.privacy(current_features, self.id_labels)
            
        #for scale in self.scales:
        #    print "%.2f, %.2f, %.4f" % (scale, self.noise_acc[scale], self.noise_priv[scale])
        
        self.noise_accpriv = {}
        for k in self.noise_priv.iterkeys():
            self.noise_accpriv[k] = np.concatenate((self.noise_acc[k],self.noise_priv[k]))
        return np.asarray([self.noise_accpriv[k] for k in sorted(self.noise_accpriv)])
            

    def plot_acc_priv(self):
        plt.plot([self.noise_priv[k] for k in sorted(self.noise_priv)], 
                 [self.noise_acc[k] for k in sorted(self.noise_acc)], 
                'g', label='acc_priv')
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('privacy')
        plt.title('accuracy vs. privacy')
        plt.show()


# In[5]:

layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
#layer_names = ['conv7']
num_batch = 411
for layer_folder_name in layer_names:
    print layer_folder_name
    attr_names = ['[20]', '[20-39]', '[20-39-31]', '[20-39-31-6]', '[20-39-31-6-7]']
    #attr_names = ['all']
    privacy_save_path = os.path.join('../result', layer_folder_name , 'privacy')


    for attr_name in attr_names:
    
        iter_name = '15000'
        dpfe_prototxt_path = os.path.join('../result', layer_folder_name , 'prototxt', 
                                          'test_pca_bn_ce_' + layer_folder_name + '.prototxt')
        dpfe_model_path = os.path.join('../result', layer_folder_name , 'model', attr_name , 
                                       'dpfe_finetune_m10_' + attr_name + '_iter_' + iter_name + '.caffemodel')
        dpfe_net = CAFFENET(layer_folder_name + attr_name + '_dpfe', dpfe_prototxt_path, dpfe_model_path, output_name='prob', mse=False)
        mypriv = PrivacyAnalyzer()
        features, attr_labels, id_labels  = dpfe_net.load_middle_features(num_batch, 'bn_pca_1')
        mypriv.initialize(features, id_labels, attr_labels, dpfe_net.get_acc_from_middle)
        dpfe_acc_priv = mypriv.acc_priv(model_type = 'dpfe')   
        np.save(os.path.join(privacy_save_path,  attr_name + '_dpfe_finetune_m10_accpriv.npy') , dpfe_acc_priv)
        


# test pca models version 0
layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
for layer_folder_name in layer_names:
    prototxt_path = os.path.join('../result', layer_folder_name , 'prototxt', 
                                     'test_pca_' + layer_folder_name + '.prototxt')
    attr_model_path = os.path.join('../result', layer_folder_name , 'model', 'pca' , 
                                           'pca_' + layer_folder_name + '.caffemodel')
    attr_net = CAFFENET('pca_v0', prototxt_path, attr_model_path, output_name='moon-fc', mse=True)
    print('average accuracy of the version 0 pca model on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))
    filt = np.zeros(40)
    filt[20] = 1
    attr_net.set_filter(filt)
    print('average accuracy of the version 0 pca model on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))



# test pca models with bn and ce
layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
iter_name = '3000'
for layer_folder_name in layer_names:
    prototxt_path = os.path.join('../result', layer_folder_name , 'prototxt', 
                                 'test_pca_bn_ce_' + layer_folder_name + '.prototxt')
    attr_model_path = os.path.join('../result', layer_folder_name , 'model', 'pca' , 
                                   'pca_'  + 'iter_' + iter_name + '.caffemodel')
    attr_net = CAFFENET('pca_bn_ce', prototxt_path, attr_model_path, output_name='prob', mse=False)
    print('average accuracy of pca model with bn and ce on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))
    filt = np.zeros(40)
    filt[20] = 1
    attr_net.set_filter(filt)
    print('average accuracy of pca model with bn and ce on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))


# In[4]:

# test simple models
layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
iter_name = '3000'
attr_name = '[20-39-31]'
for layer_folder_name in layer_names:
    prototxt_path = os.path.join('../result', layer_folder_name , 'prototxt', 
                                 'test_pca_bn_ce_' + layer_folder_name + '.prototxt')
    attr_model_path = os.path.join('../result', layer_folder_name , 'model', attr_name , 
                                       'simple_finetune_' + attr_name + '_iter_' + iter_name + '.caffemodel')
    attr_net = CAFFENET('simple', prototxt_path, attr_model_path, output_name='prob', mse=False)
    print('average accuracy of pca model with bn and ce on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))
    filt = np.zeros(40)
    filt[20] = 1
    attr_net.set_filter(filt)
    print('average accuracy of pca model with bn and ce on %s is %.2f'
          %(layer_folder_name,attr_net.get_accuracy(20, separated=False)))





