import tensorflow as tf

# class EmbeddingLayer(tf.keras.layers.Layer):
#     def __init__(self, args):
#         super(EmbeddingLayer, self).__init__()
#         self.in_dim = 2
#         self.out_dim = args.embedding_size

#         self.weight = self.add_weight(shape=(self.in_dim, self.out_dim),
#                                      initializer='glorot_uniform',
#                                      name='embedding_weight')

#         self.bias = self.add_weight(shape=(self.out_dim,),
#                                      initializer='zeros',
#                                      name='embedding_weight')

#         self.activation = tf.keras.activations.relu

#     def call(self, x):
#         x = tf.matmul(x, self.weight)
#         x = tf.add(x, self.bias)

#         output = self.activation(x)

#         return output

# https://stackoverflow.com/questions/60624960/tf-keras-layers-rnn-vs-tf-keras-layers-stackedrnncells-tensorflow-2

class Model(tf.keras.Model):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # dim = tf.zeros([args.batch_size, args.rnn_size])
        # rnn_cells = [tf.keras.layers.LSTMCell(args.rnn_size) for _ in range(args.num_layers)]

        # for rnn_cell in rnn_cells:
        #     print("+++++++++++++{}".format(rnn_cell.state_size))

        # # stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        # # initial_state = stacked_lstm.get_initial_state(batch_size=args.batch_size, dtype=tf.float32)
        # self.lstm_layer = tf.keras.layers.RNN(rnn_cells, return_sequences=True, return_state=True)
        # Output size is the set of parameters (mu, sigma, corr)
        self.output_size = 5  # 2 mu, 2 sigma and 1 corr

        # self.input_layer = tf.keras.layers.Input(batch_shape=(batch_size, None, 2))

        self.embedding_layer = tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu) # EmbeddingLayer(args)

        self.lstm_layer = tf.keras.layers.LSTM(args.rnn_size, return_sequences=True,
                                            return_state = True,
                                            stateful=True)

        self.dense = tf.keras.layers.Dense(self.output_size)

        # self.initial_state = None

        # self.lstm_layer = tf.keras.Sequential([
        #     tf.keras.layers.Dense(args.embedding_size,
        #         activation = tf.keras.activations.relu),
        # ])

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu,
                batch_input_shape = [args.batch_size, None, 2]),
        # self.model.add(tf.keras.layers.LSTM(args.rnn_size, return_sequences=True,
        #                                     return_state = True,
        #                                     stateful=True))
            tf.keras.layers.GRU(args.rnn_size,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.output_size)
        ])

    def call(self, x):
        # print("=========================x shape:{}".format(x.shape))

        # inputs = tf.split(x, self.args.seq_length, 1)
        # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # x = self.input_layer(x)
        # IN:(50, 10, 2), OUT:(50, 10, 128)
        # x = self.embedding_layer(x)

        # # print("=========================x11 shape:{}".format(x.shape))
        # # # IN: (50, 10, 128), OUT:()
        # outputs, state_h, state_c = self.lstm_layer(x)

        # # self.initial_state = [state_h, state_c]

        # # print("===================outputs shape:{}".format(outputs.shape))

        # output = self.dense(outputs)

        output = self.model(x)

        # print("=====================output shape:{}".format(output.shape))

        return output
        # inputs = tf.split(x, self.args.seq_length, 1)
        # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # embedded_inputs = []
        # for input in inputs:
        #     # Each x is a 2D tensor of size numPoints x 2
        #     # Embedding layer
        #     embedded_x = self.embedding_layer(input)

        #     print("===================embedded shape:{}".format(embedded_x.shape))

        #     embedded_inputs.append(embedded_x)

        # whole_seq_output, final_memory_state, final_carry_state = self.lstm_layer(embedded_inputs)

        # outputs, last_state = self.lstm_layer(embedded_inputs)

        # output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        # output = self.dense(output)

