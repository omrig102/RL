import tensorflow as tf

'''
    input.shape = [None,input_size]
'''
def create_mlp_layers(input,hidden_layers,hidden_size,hidden_activation,output_size=None,output_activation=None,output_name=None,use_noise=False,l2=None) :
    x = input
    for i in range(hidden_layers) :
        x = tf.layers.dense(x,units=hidden_size[i],activation=hidden_activation)

    if(output_size is not None) :
        if(output_name is None) :
            x = tf.layers.dense(x,units=output_size,activation=output_activation)
        else :
            x = tf.layers.dense(x,units=output_size,activation=output_activation,name=output_name)
    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_conv2d_layers(input,hidden_layers,hidden_size,hidden_activation,flatten=True,l2=None) :
    x = input
    if(l2 is None):
      reg = None
    else :
      reg = tf.contrib.layers.l2_regularizer(l2)
    for i in range(hidden_layers) :
        x = tf.layers.conv2d(x,filters=hidden_size[i],kernel_size=3,strides=1,activation=hidden_activation,kernel_regularizer=reg)
        
    x = tf.layers.max_pooling2d(x,pool_size=[2,2,],strides=2)
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
def create_mlp_network(input,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name=None,use_noise=False,l2=None) :
    x = input
    x = create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name,use_noise,l2)
    return x

'''
    input_shape = [None,timestamps,input_size]
'''
def create_network_lstm(input,lstm_hidden_layers,lstm_hidden_size,unit_type
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name,use_noise=False,l2=None) :
    input_shape = input.get_shape().as_list()
    x = tf.transpose(input,[1,0,2])
    x = create_lstm_layers(x,lstm_hidden_layers,lstm_hidden_size,unit_type)
    x = tf.transpose(x,[1,0,2])
    x = x[:,-1,:]
    x = tf.reshape(x,[-1,lstm_hidden_size])
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name,use_noise,l2)

    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_network_pixels_mlp(input,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name,use_noise=False,l2=None) :
    input_shape = input.get_shape().as_list()
    x = input
    #x = tf.reshape(input,[-1,input_shape[1] * input_shape[2] * input_shape[3]])
    return create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation,output_name,use_noise,l2)

'''
    input.shape = [None,height,width,channels]
'''      
def create_network_pixels_conv(input,conv_hidden_layers,conv_hidden_size,conv_hidden_activation
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name,use_noise=False,l2=None) :
    input_shape = input.get_shape().as_list()
    x = input
    x = create_conv2d_layers(x,conv_hidden_layers,conv_hidden_size,conv_hidden_activation,flatten=True,l2=l2)
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation,output_name,use_noise,l2)

    return x
