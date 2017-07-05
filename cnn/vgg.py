#! python3

def build(in_placeholder):
    print(in_placeholder.get_shape())
    assert in_placeholder.get_shape() == (None, 32, 32, 3)
    
    conv1_1 = conv_layer(bgr, "conv1_1")
    conv1_2 = conv_layer(conv1_1, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1")
    conv2_2 = conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1")
    conv3_2 = conv_layer(conv3_1, "conv3_2")
    conv3_3 = conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1")
    conv4_2 = conv_layer(conv4_1, "conv4_2")
    conv4_3 = conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1")
    conv5_2 = conv_layer(conv5_1, "conv5_2")
    conv5_3 = conv_layer(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    fc6 = fc_layer(pool5, "fc6")
    assert fc6.get_shape().as_list()[1:] == [4096]
    relu6 = tf.nn.relu(fc6)

    fc7 = fc_layer(relu6, "fc7")
    relu7 = tf.nn.relu(fc7)

    fc8 = fc_layer(relu7, "fc8")

    prob = tf.nn.softmax(fc8, name="prob")
