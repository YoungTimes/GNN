# https://github.com/tkipf/keras-gcn
import tensorflow as tf

class GraphConvolutionLayer(tf.keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, input_dim, output_dim, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None):
        super(GraphConvolutionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation

    def build(self, nodes_shape):
        self.kernel = self.add_weight(shape = (self.input_dim, self.output_dim),
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer = self.bias_regularizer)
        else:
            self.bias = None
            
        self.built = True

    def call(self, nodes, edges):
        support = tf.matmul(nodes, self.kernel) 

        output = tf.matmul(edges, support)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)
            
        return output


class GraphConvolutionModel(tf.keras.Model):
    def __init__(self):
        super(GraphConvolutionModel, self).__init__()

        self.graph_conv_1 = GraphConvolutionLayer(1433, 16,
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.graph_conv_2 = GraphConvolutionLayer(16, 7)

    def call(self, x, training=False):

        nodes = x[0]
        edges = x[1]

        h = self.graph_conv_1(nodes, edges)
        logit = self.graph_conv_2(h, edges)

        return logit