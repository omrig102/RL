import tensorflow as tf

'''
    input.shape = [None,input_size]
'''
def create_mlp_layers(input,hidden_layers,hidden_size,hidden_activation,output_size=None,output_activation=None) :
    x = input
    for i in range(hidden_layers) :
        x = tf.layers.dense(x,units=hidden_size,activation=hidden_activation)

    if(output_size is not None) :
        x = tf.layers.dense(x,units=output_size,activation=output_activation)
    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_conv2d_layers(input,hidden_layers,hidden_size,hidden_activation,flatten=True) :
    x = input
    for i in range(hidden_layers) :
        x = tf.layers.conv2d(x,filters=hidden_size,kernel_size=3,strides=1,activation=hidden_activation)

    if(flatten) :
        x = tf.layers.flatten(x)

    return x

'''
    input.shape = [timestamps,None,input_size]
'''
def create_lstm_layers(input,hidden_layers,hidden_size) :
    x = input
    rnn = tf.contrib.cudnn_rnn.CudnnLSTM(hidden_layers, hidden_size)
    x,_ = rnn(x)   

    return x

'''
    input.shape = [None,input_size]
'''
def create_mlp_network(input,hidden_layers,hidden_size,hidden_activation,output_size,output_activation) :
    x = input
    x = create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation)
    return x

'''
    input_shape = [None,timestamps,input_size]
'''
def create_network_lstm(input,lstm_hidden_layers,lstm_hidden_size
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation) :
    input_shape = input.get_shape().as_list()
    x = tf.transpose(input,[1,0,2])
    x = create_lstm_layers(x,lstm_hidden_layers,lstm_hidden_size)
    x = tf.transpose(x,[1,0,2])
    x = x[:,-1,:]
    x = tf.reshape(x,[-1,lstm_hidden_size])
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation)

    return x

'''
    input.shape = [None,height,width,channels]
'''
def create_network_pixels_mlp(input,hidden_layers,hidden_size,hidden_activation,output_size,output_activation) :
    input_shape = input.get_shape().as_list()
    x = input
    #x = tf.reshape(input,[-1,input_shape[1] * input_shape[2] * input_shape[3]])
    return create_mlp_layers(x,hidden_layers,hidden_size,hidden_activation,output_size,output_activation)

'''
    input.shape = [None,height,width,channels]
'''      
def create_network_pixels_conv(input,conv_hidden_layers,conv_hidden_size,conv_hidden_activation
,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation) :
    input_shape = input.get_shape().as_list()
    x = input
    x = create_conv2d_layers(x,conv_hidden_layers,conv_hidden_size,conv_hidden_activation,flatten=True)
    x = create_mlp_layers(x,mlp_hidden_layers,mlp_hidden_size,mlp_hidden_activation,output_size,output_activation)

    return x

