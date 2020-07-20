import tensorflow as tf

class SocialModel(tf.keras.Model):

    def __init__(self, args):
        super(SocialModel, self).__init__()
        self.args = args

        self.cell = tf.keras.layers.LSTMCell(args.rnn_size)

        self.spatial_embedding = tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu) 
        self.tensor_embedding = tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu)

        self.output_size = 5 
        self.output_layer = tf.keras.layers.Dense(self.output_size)

    def get_social_tensor(self, grid, hidden_states):
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor

    def call(self, frame_datas, ped_lists, grid_frame_datas, ped_indexs):
        num_peds = len(ped_indexs)
        outputs = tf.zeros(self.seq_length * num_peds, self.output_size)

        # [args.seq_length, args.max_num_peds, 3]
        for frame_num, frame in enumerate(frame_data):
            # grid_frame_data = grid_frame_datas[frame_num]

            node_ids = [int(node_id) for node_id in ped_lists[frame_num]]

            if len(node_ids) == 0:
                continue

            list_of_nodes = [ped_indexs[x] for x in node_ids]

            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            social_tensor = self.get_social_tensor(grid_current, hidden_states_current)

            # Embed inputs
            input_embedded = self.spatial_embedding(nodes_current)
            tensor_embedded = self.tensor_embedding(social_tensor)

            # Concat input
            concat_embedded = tf.concat([input_embedded, tensor_embedded], axis = 1)

            h_nodes = self.cell(concat_embedded, (hidden_states_current))

            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes

            cell_states[corr_index.data] = c_nodes

        for frame_num in range(self.seq_length):
            for node in range(num_nodes):
                outputs_return[frame_num, node, :] = outputs[frame_num * num_nodes + node, :]

