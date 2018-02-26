
# Assuming for each intermediate layer e.g. conv4-2 we have a model which has a low dimensional private-feature layer with batch normalization and cross entropy loss (saved in conv4-2/model/pca/pca_iter_3000.caffemodel), now we are ready to create Simple and DPFE models. In order to do this we should finetune the original model with the specified attribute set (e.g. {gender, age}) and with or without contrastive loss (for simple and dpfe). These models needs their own prototxt file and here we create all the required prototxt file for simple and dpfe models.

# See the factory folder.

layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
attr_names = ['[20]', '[20-39]', '[20-39-31]', '[20-39-31-6]', '[20-39-31-6-7]']
factory_addr = '../result/factory/'
kinds = ['simple', 'dpfe']


for layer_name in layer_names:
    for attr_name in attr_names:
        for kind in kinds:
            f = open('../result/%s/prototxt/train_%s_%s.prototxt'%(layer_name, kind, attr_name), "w")
            with open(factory_addr+'%s_%s.init'%(kind, layer_name)) as infile:
                f.write(infile.read())
            with open(factory_addr+'%s.slice'%(attr_name)) as infile:
                f.write(infile.read())

