
# coding: utf-8

# In[1]:

import caffe 

def cut_prototxt(layer_name, source_prototxt_path, mobile_prototxt_path):
    print "name: "+"\""+layer_name+"\""
    source_file = open(source_prototxt_path, "r")
    mobile_file = open(mobile_prototxt_path, "w+")
    explored = 0
    with source_file as f:
        for line in f:
            if explored and "layer" in line:
                break
            if "name: "+"\""+layer_name+"\"" in line:
                explored = 1
            mobile_file.writelines(line)
    mobile_file.close()
    source_file.close()

#test:
#layer_name = 'conv5_1'
#source_prototxt_path = 'deploy.prototxt'
#mobile_prototxt_path = 'mobile_'+layer_name+'.prototxt'
#cut_prototxt(layer_name, source_prototxt_path, mobile_prototxt_path)

def copy_weights(source_prototxt_path, source_caffemodel_path, mobile_prototxt_path):
    source_net = caffe.Net(source_prototxt_path, source_caffemodel_path, caffe.TEST)
    mobile_net = caffe.Net(mobile_prototxt_path, caffe.TEST)
    mobile_params = {pr: (mobile_net.params[pr][0].data, mobile_net.params[pr][1].data) for pr in mobile_net.params.keys()}
    source_params = {pr: (source_net.params[pr][0].data, source_net.params[pr][1].data) for pr in source_net.params.keys()}
    for pr in mobile_params.keys():
        mobile_params[pr][0].flat = source_params[pr][0].flat 
        mobile_params[pr][1][...] = source_params[pr][1]
    mobile_net.save(mobile_prototxt_path.split('.')[0]+'.caffemodel')
    print 'caffemodel created'
    
#test:   
#copy_weights('deploy.prototxt', 'models/moon_tiny_iter_1000000.caffemodel', 'mobile_conv5_1.prototxt')


# In[ ]:



