# use it in Jupyter

# coding: utf-8

# In[2]:

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().magic('matplotlib inline')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
init_notebook_mode(connected=True)
import ast


# In[3]:

def well_shaped(priv_acc):
    temp = [priv_acc[:,0]]
    p = priv_acc[0,0]
    a = priv_acc[1,0]
    for i in range(priv_acc.shape[1]-1):
        if priv_acc[0,i+1] > p and priv_acc[1,i+1] < a:
            p = priv_acc[0,i+1]
            a = priv_acc[1,i+1]
            temp.append(priv_acc[:,i+1])
    return np.asarray(temp).T


# In[6]:

# DPFE vs Simple

#accpriv vector
#[accuracies(0:39), 1nn-priv(40), num_class(41), kde-bayes(42), rank(43), rank-std(44), log-rank(45), log-rank-std(46)]  
layer_names = ['conv4-2', 'conv7']
#layer_names = ['conv4-2']
attr_names_list = ['[20,39]', '[20,39,31,6,7]']
attr_names = ['[20-39]', '[20-39-31-6-7]']


xtitles = {'[20]':'G', '[20-39]':'GA', '[20-39-31]':'GAS', '[20-39-31-6]':'GASL', '[20-39-31-6-7]':'GASLN' }

#attr_names = ['[20]', '[20,39]', '[20, 39, 31]', '[20, 39, 31, 6]', '[20, 39, 31, 6, 7]']
#real_names = ['Gender', 'Gender & Age', 'Gender & Age & Smiling', 'Gender & Age & Smiling & Big Lips', 'Gender & Age & Smiling & Big Lips & Big Nose']
methods = {}
methods['1NN'] = 40
methods['LRP'] = 45

for l,layer_folder_name in enumerate(layer_names):
    cnt = 0
    for i,attr_name in enumerate(attr_names):
        for method_name,priv_col in methods.iteritems():
            cnt += 1
            privacy_save_path = os.path.join('../result', layer_folder_name , 'privacy')
            pca_name = attr_name + '_simple_finetune_accpriv.npy'
            fine_name = attr_name + '_dpfe_finetune_m10_accpriv.npy'
            accpriv_pca = np.load(os.path.join(privacy_save_path, pca_name))
            accpriv_fine = np.load(os.path.join(privacy_save_path, fine_name))

            pca_tmp = np.zeros((accpriv_pca.shape[0],2))
            pca_tmp[:,0] = accpriv_pca[:,priv_col]
            if method_name == 'LRP':
                pca_tmp[:,0] = pca_tmp[:,0] / np.log(accpriv_pca[0,41])
            pca_tmp[:,1] = accpriv_pca[:,ast.literal_eval(attr_names_list[i])].mean(axis=1)
            pca_priv_acc = well_shaped(pca_tmp.T)
            
            fine_tmp = np.zeros((accpriv_fine.shape[0],2))
            fine_tmp[:,0] = accpriv_fine[:,priv_col]
            if method_name == 'LRP':
                print(accpriv_fine[0,41])
                fine_tmp[:,0] = fine_tmp[:,0] / np.log(accpriv_fine[0,41])
            fine_tmp[:,1] = accpriv_fine[:,ast.literal_eval(attr_names_list[i])].mean(axis=1)
            fine_priv_acc = well_shaped(fine_tmp.T)
            #pca_priv_acc = pca_tmp.T
            #fine_priv_acc = fine_tmp.T
            #m = np.max([np.max(pca_priv_acc[1,:]), np.max(fine_priv_acc[1,:])])
            #pca_priv_acc[1,:] = pca_priv_acc[1,:] * (m / np.max(pca_priv_acc[1,:]))
            #fine_priv_acc[1,:] = fine_priv_acc[1,:] * (m / np.max(fine_priv_acc[1,:]))
            
            trace1 = go.Scatter(x=pca_priv_acc[1,:], y=pca_priv_acc[0,:], 
                                     name='simple', mode = 'lines',
                                     line=dict(shape='spline', dash='dash', color='1F77B4'))
            trace2 = go.Scatter(x=fine_priv_acc[1,:], y=fine_priv_acc[0,:], 
                                     name='dpfe', mode = 'lines',
                                     line=dict(shape='spline', color='FF7F0E'))
            m = np.asarray([np.max(pca_priv_acc[1,:]),np.max(fine_priv_acc[1,:])])
            mymax = np.min(m)
            layout = dict(title = layer_folder_name, 
                          yaxis = dict(title = method_name, showline=True, mirror=True), 
                          xaxis = dict(title = xtitles[attr_name] + ' Acc (%)', showline=True, mirror=True, range = [mymax-10,mymax]),
                          legend = dict(x=.01, y=.01), width=350, height=300, font=dict(family='Palatino', size=12),
                          margin = go.Margin(l = 50, r = 40, b = 50, t = 40, pad = 5))
            fig = dict(data=[trace1, trace2], layout=layout)
            iplot(fig, filename=layer_folder_name + '_' + attr_name + '_' + method_name, 
                   image_height=300, image_width=350)
            
            


# In[26]:

# Li > L(i-1)

#accpriv vector
#[accuracies(0:39), 1nn-priv(40), num_class(41), kde-bayes(42), rank(43), rank-std(44), log-rank(45), log-rank-std(46)]  
layer_names = ['conv4-2', 'conv5-1', 'conv6-1' ,'conv7']
#layer_names = ['conv4-2']
attr_names = ['[20, 39, 31, 6, 7]', '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]']
attr_names = ['[20]', '[20-39-31-6]' ]
attr_names_list = ['[20]', '[20,39,31,6]']

xtitles = {'[20]':'G', '[20-39]':'GA', '[20-39-31]':'GAS', '[20-39-31-6]':'GASL', '[20-39-31-6-7]':'GASLN' }

#attr_names = ['[20]', '[20,39]', '[20, 39, 31]', '[20, 39, 31, 6]', '[20, 39, 31, 6, 7]']
#real_names = ['Gender', 'Gender & Age', 'Gender & Age & Smiling', 'Gender & Age & Smiling & Big Lips', 'Gender & Age & Smiling & Big Lips & Big Nose']
methods = {}
methods['1NN'] = 40
methods['LRP'] = 45


mydashes = ["dot", "dash", "longdash", "solid"]
#mycolors = ['purple', 'green', 'blue', 'red']
mycolors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9575D2']



for i,attr_name in enumerate(attr_names):
    cnt = 0
    for method_name,priv_col in methods.iteritems():
        cnt += 1
        traces = []
        m = []
        for l,layer_folder_name in enumerate(layer_names):

            privacy_save_path = os.path.join('../result', layer_folder_name , 'privacy')
            #pca_name = attr_name + '_accpriv_pca.npy'
            fine_name = attr_name + '_dpfe_accpriv.npy'
            #accpriv_pca = np.load(os.path.join(privacy_save_path, pca_name))
            accpriv_fine = np.load(os.path.join(privacy_save_path, fine_name))
            fine_tmp = np.zeros((accpriv_fine.shape[0],2))
            fine_tmp[:,0] = accpriv_fine[:,priv_col]
            if method_name == 'LRP':
                fine_tmp[:,0] = fine_tmp[:,0] / np.log(accpriv_fine[0,41])
            fine_tmp[:,1] = accpriv_fine[:,ast.literal_eval(attr_names_list[i])].mean(axis=1)
            fine_priv_acc = well_shaped(fine_tmp.T)
            
            trace = go.Scatter(x=fine_priv_acc[1,:], y=fine_priv_acc[0,:], 
                               name=layer_folder_name, mode = 'lines',
                               line=dict(shape='spline', dash=mydashes[l], color=mycolors[l]))
            traces.append(trace)
            m.append(np.max(fine_priv_acc[1,:]))
        mymax = np.min(np.asarray(m))
        layout = dict(title = '', 
                      yaxis = dict(title = method_name, showline=True, mirror=True), 
                      xaxis = dict(title = xtitles[attr_name] + ' Acc (%)', showline=True, mirror=True, range = [mymax-10,mymax]),
                      legend = dict(x=.01, y=.01), width=350, height=300, font=dict(family='Palatino', size=12),
                      margin = go.Margin(l = 50, r = 40, b = 50, t = 40, pad = 5))
        fig = dict(data=traces, layout=layout)
        iplot(fig, filename='layer_comparison_'+ attr_name + '_' + method_name, 
              image='svg', image_height=300, image_width=350)


# In[49]:

# [1,2]_1 > [1,2,3]_1 
layer_names = ['conv7']

#accpriv vector
#[accuracies(0:39), 1nn-priv(40), num_class(41), kde-bayes(42), rank(43), rank-std(44), log-rank(45), log-rank-std(46)]  
attr_names = ['[20]', '[20-39]', '[20-39-31]', '[20-39-31-6]', '[20-39-31-6-7]', 'all']
attr_names_list = ['[20]', '[20]', '[20]', '[20]', '[20]', '[20]']

xtitles = {'[20]':'G', '[20-39]':'GA', '[20-39-31]':'GAS', '[20-39-31-6]':'GASL', '[20-39-31-6-7]':'GASLN' , 'all':'ALL'}

#attr_names = ['[20]', '[20,39]', '[20, 39, 31]', '[20, 39, 31, 6]', '[20, 39, 31, 6, 7]']
#real_names = ['Gender', 'Gender & Age', 'Gender & Age & Smiling', 'Gender & Age & Smiling & Big Lips', 'Gender & Age & Smiling & Big Lips & Big Nose']
methods = {}
methods['1NN'] = 40
methods['LRP'] = 45


mydashes = ["solid", "dot", "dashdot", "dash", "longdash", "solid"]
#mycolors = ['purple', 'green', 'blue', 'red']
mycolors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9575D2', '#8c564b']



for l,layer_folder_name in enumerate(layer_names):
    cnt = 0
    for method_name,priv_col in methods.iteritems():
        cnt += 1
        traces = []
        for i,attr_name in enumerate(attr_names):
            cnt += 1
            privacy_save_path = os.path.join('../result', layer_folder_name , 'privacy')
            fine_name = attr_name + '_dpfe_finetune_m10_accpriv.npy'
            accpriv_fine = np.load(os.path.join(privacy_save_path, fine_name))
            
            fine_tmp = np.zeros((accpriv_fine.shape[0],2))
            fine_tmp[:,0] = accpriv_fine[:,priv_col]
            if method_name == 'LRP':
                fine_tmp[:,0] = fine_tmp[:,0] / np.log(accpriv_fine[0,41])
            fine_tmp[:,1] = accpriv_fine[:,ast.literal_eval(attr_names_list[i])].mean(axis=1)
            fine_priv_acc = well_shaped(fine_tmp.T)
            #pca_priv_acc = pca_tmp.T
            #fine_priv_acc = fine_tmp.T
            #m = np.max([np.max(pca_priv_acc[1,:]), np.max(fine_priv_acc[1,:])])
            #pca_priv_acc[1,:] = pca_priv_acc[1,:] * (m / np.max(pca_priv_acc[1,:]))
            #fine_priv_acc[1,:] = fine_priv_acc[1,:] * (m / np.max(fine_priv_acc[1,:]))
            
            trace = go.Scatter(x=fine_priv_acc[1,:], y=fine_priv_acc[0,:], 
                               name=xtitles[attr_name], mode = 'lines',
                               line=dict(shape='spline', dash=mydashes[i], color=mycolors[i]))
            
            traces.append(trace)
            #m = np.asarray([np.max(pca_priv_acc[1,:]),np.max(fine_priv_acc[1,:])])
            #mymax = np.min(m)
        if cnt==7:
            ya = dict(title = method_name, showline=True, mirror=True, range=[.15,.8])
        else:
            ya = dict(title = method_name, showline=True, mirror=True)
        layout = dict(title = '', 
                      yaxis = ya, 
                      xaxis = dict(title ='Gender Acc (%)', showline=True, mirror=True, range=[85,95]),
                      legend = dict(x=.01, y=.01), width=350, height=300, font=dict(family='Palatino', size=12),
                      margin = go.Margin(l = 50, r = 40, b = 50, t = 40, pad = 5))
        fig = dict(data=traces, layout=layout)
        iplot(fig, filename='restriction_comparison_[20]_' + method_name, 
              image='svg', image_height=300, image_width=350)

     


# In[75]:

#std
#[accuracies(0:39), 1nn-priv(40), num_class(41), kde-bayes(42), rank(43), rank-std(44), log-rank(45), log-rank-std(46)]  

layer_names = ['conv7']
attr_names = ['[20]']
cols = np.array([43,20,44])


mydashes = ["dot", "dash", "longdash", "solid"]
#mycolors = ['purple', 'orange', 'green', 'blue', 'red']
mycolors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9575D2']

for i,attr_name in enumerate(attr_names):
    traces = []
    for layer_folder_name in layer_names:
        privacy_save_path = os.path.join('../result', layer_folder_name , 'privacy')
        pca_name = attr_name + '_simple_finetune_accpriv.npy'
        fine_name = attr_name + '_dpfe_finetune_m10_accpriv.npy'
        accpriv_pca = np.load(os.path.join(privacy_save_path, pca_name))
        accpriv_fine = np.load(os.path.join(privacy_save_path, fine_name))
        pca_priv_acc = well_shaped(accpriv_pca[:,cols].T)
        fine_priv_acc = well_shaped(accpriv_fine[:,cols].T)
        pca_priv_acc[0,:] = pca_priv_acc[0,:] / accpriv_pca[0,41]
        pca_priv_acc[2,:] = pca_priv_acc[2,:] / accpriv_pca[0,41]
        fine_priv_acc[0,:] = fine_priv_acc[0,:] / accpriv_fine[0,41]
        fine_priv_acc[2,:] = fine_priv_acc[2,:] / accpriv_fine[0,41]

        
        y_l_pca = pca_priv_acc[0,:] - pca_priv_acc[2,:]
        y_u_pca = pca_priv_acc[0,:] + pca_priv_acc[2,:]
        
        traces.append(go.Scatter(x=pca_priv_acc[1,:], y=y_l_pca, 
                                 name=layer_folder_name+'_pca_lower', mode = 'lines',
                                 line=dict(width=0, shape='spline'), showlegend = False))
        
        traces.append(go.Scatter(x=pca_priv_acc[1,:], y=pca_priv_acc[0,:], 
                                 name='simple', mode = 'lines',
                                 line=dict(width=2, shape='spline', color='rgb(31, 119, 180)', dash='dot'),
                                 fillcolor='rgba(31,119,180,.15)',fill='tonexty'))
        
        traces.append(go.Scatter(x=pca_priv_acc[1,:], y=y_u_pca, 
                                 name=layer_folder_name+'_pca_upper', mode = 'lines',
                                 line=dict(width=0, shape='spline', color='rgb(31, 119, 180)', dash='dot'),
                                 fillcolor='rgba(31,119,180,.15)', fill='tonexty', showlegend = False))
        
        y_l_fine = fine_priv_acc[0,:] - fine_priv_acc[2,:]
        y_u_fine = fine_priv_acc[0,:] + fine_priv_acc[2,:]
        
        traces.append(go.Scatter(x=fine_priv_acc[1,:], y=y_l_fine, 
                                 name=layer_folder_name+'_fine_lower', mode = 'lines',
                                 line=dict(width=0, shape='spline'), showlegend = False))
        
        traces.append(go.Scatter(x=fine_priv_acc[1,:], y=fine_priv_acc[0,:], 
                                 name='dpfe', mode = 'lines',
                                 line=dict(width=2, shape='spline', color='rgb(255,127,14)'),
                                 fillcolor='rgba(255,127,14,.15)',fill='tonexty'))
        
        traces.append(go.Scatter(x=fine_priv_acc[1,:], y=y_u_fine, 
                                 name=layer_folder_name+'_fine_upper', mode = 'lines',
                                 line=dict(width=0, shape='spline', color='rgb(255,127,14)'),
                                 fillcolor='rgba(255,127,14,.15)', fill='tonexty', showlegend = False))
        
        
    traces.append(go.Scatter(x=[90], y=[0.3], mode='markers',name='', showlegend=False, 
                             marker=dict(size=12, color='rgb(255,127,14)')))
    
    
    layout = dict(title = 'Rank mean/std comparison', showlegend=True, width=560, height=480, 
                  margin = go.Margin(l = 50, r = 50, b = 50, t = 40, pad = 5),
                  font=dict(family='Palatino', size=12), legend = dict(x=.01, y=.01),
                  yaxis = dict(title = 'Rank', showline=True, zeroline=False, mirror=True, range=[0,0.7]), 
                  xaxis = dict(title = 'Gender Acc. (%)', showline=True, mirror=True, range = [80,96]),
                  shapes=[ {'type': 'line','x0': 90,'y0': 0,'x1': 90,'y1': .7,
                            'line': {'color': 'gray','width': .7}}] )
    
 
    fig = dict(data=traces, layout=layout)
    iplot(fig, filename='std', image='svg', image_height=480, image_width=560)


     

