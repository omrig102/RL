class Config() :

    episodes = 10000
    rollout_size = 64
    

    #vision
    vision_batch_size = 64
    vision_learning_rate = 0.0001
    vision_encoder_output_size = 5
    vision_decoder_input_size = 3
    vision_conv_layers = 3
    vision_conv_filters = 64 