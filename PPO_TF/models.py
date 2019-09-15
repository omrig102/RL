import tensorflow as tf
import numpy as np

'''
    input.shape = [None,input_size]
'''
def create_mlp_layers(input,hidden_layers,hidden_size,hidden_activation,output_size=None,output_activation=None,output_name=None) :
    x = input
    for i in range(hidden_layers) :
        x = tf.layers.dense(x,units=hidden_size[i],activation=hidden_activation,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))

    if(output_size is not None) :
        if(output_name is None) :
            x = tf.layers.dense(x,units=output_size,activation=output_activation,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        else :
            x = tf.layers.dense(x,units=output_size,activation=output_activation,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name=output_name)
    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_conv2d_layers(input,hidden_layers,hidden_size,kernel_size,strides,hidden_activation,flatten=True) :
    x = input

    for i in range(hidden_layers) :
      x = tf.layers.conv2d(x,hidden_size[i],kernel_size=kernel_size[i],strides=strides[i],activation=hidden_activation,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))  
    if(flatten) :
        x = tf.layers.flatten(x)

    return x

'''
    input.shape = [timestamps,None,input_size]
'''
def create_lstm_layers(input,hidden_layers,hidden_size,unit_type) :
    x = input
    if(unit_type == 'lstm') :
        rnn = tf.contrib.cudnn_rnn.CudnnLSTM(hidden_layers, hidden_size)
    elif(unit_type == 'gru') :
        rnn = tf.contrib.cudnn_rnn.CudnnGRU(hidden_layers, hidden_size)
    else :
        raise Exception('unit type not supported : {}'.format(unit_type))
    x,_ = rnn(x)   

    return x

'''
    input.shape = [None,input_size]
'''
def create_mlp_network(input,hidden_layers,hidden_size,hidden_activation,output_size=None,output_activation=None,output_name=None) :
    x = input
    x = create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name)
    return x

'''
    input_shape = [None,timestamps,input_size]
'''
def create_network_lstm(input,lstm_hidden_layers,lstm_hidden_size,unit_type
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size=None,output_activation=None,output_name=None) :
    input_shape = input.get_shape().as_list()
    x = tf.transpose(input,[1,0,2])
    x = create_lstm_layers(x,lstm_hidden_layers,lstm_hidden_size,unit_type)
    x = tf.transpose(x,[1,0,2])
    x = x[:,-1,:]
    x = tf.reshape(x,[-1,lstm_hidden_size])
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name)

    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_network_pixels_mlp(input,hidden_layers,hidden_size,hidden_activation,output_size=None,output_activation=None,output_name=None) :
    input_shape = input.get_shape().as_list()
    x = input
    #x = tf.reshape(input,[-1,input_shape[1] * input_shape[2] * input_shape[3]])
    return create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name)

'''
    input.shape = [None,height,width,channels]
'''      
def create_network_pixels_conv(input,conv_hidden_layers,conv_hidden_size,kernel_size,strides,conv_hidden_activation
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size=None,output_activation=None,output_name=None) :
    input_shape = input.get_shape().as_list()
    x = input
    x = create_conv2d_layers(x,conv_hidden_layers,conv_hidden_size,kernel_size,strides,conv_hidden_activation,flatten=True)
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name)

    return x