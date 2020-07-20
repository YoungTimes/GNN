import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from social_model import SocialModel
from social_dataset import SocialDataLoader
from grid import get_sequence_grid_mask

def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=5,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=2,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--max_num_peds', type=int, default=27,
                        help='Maximum Number of Pedestrians')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    args = parser.parse_args()
    train(args)

def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    '''
    Function that implements the PDF of a 2D normal distribution
    params:
    x : input x points
    y : input y points
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''

    # eq 3 in the paper
    # and eq 24 & 25 in Graves (2013)
    # Calculate (x - mux) and (y-muy)
    normx = tf.math.subtract(x, mux)
    normy = tf.math.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.math.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.math.square(tf.math.divide(normx, sx)) + tf.math.square(tf.math.divide(normy, sy)) - 2*tf.math.divide(tf.math.multiply(rho, tf.math.multiply(normx, normy)), sxsy)
    negRho = 1 - tf.math.square(rho)
    # Numerator
    result = tf.math.exp(tf.math.divide(-z, 2*negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.math.multiply(sxsy, tf.math.sqrt(negRho))
    # Final PDF calculation
    result = tf.math.divide(result, denom)

    return result

def get_coef(output):
    # eq 20 -> 22 of Graves (2013)
    # TODO : (resolve) Does Social LSTM paper do this as well?
    # the paper says otherwise but this is essential as we cannot
    # have negative standard deviation and correlation needs to be between
    # -1 and 1

    z = output
    # Split the output into 5 parts corresponding to means, std devs and corr
    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, -1)

    # The output must be exponentiated for the std devs
    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    # Tanh applied to keep it in the range [-1, 1]
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    '''
    Function to calculate given a 2D distribution over x and y, and target data
    of observed x and y points
    params:
    z_mux : mean of the distribution in x
    z_muy : mean of the distribution in y
    z_sx : std dev of the distribution in x
    z_sy : std dev of the distribution in y
    z_rho : Correlation factor of the distribution
    x_data : target x points
    y_data : target y points
    '''
    step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

    # Calculate the PDF of the data w.r.t to the distribution
    result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

    result0 = tf.math.divide(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
    result0 = tf.math.multiply(tf.math.multiply(result0, step), step)

    # For numerical stability purposes
    epsilon = 1e-20

    # TODO: (resolve) I don't think we need this as we don't have the inner
    # summation
    # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
    # Apply the log operation
    result1 = -tf.math.log(tf.math.maximum(result0, epsilon))  # Numerical stability

    # TODO: For now, implementing loss func over all time-steps
    # Sum up all log probabilities for each data point
    return tf.reduce_sum(result1)


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # Extract mean
    mean = [mux, muy]
    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def sample(self, sess, traj, num = 10):
    '''
    Given an initial trajectory (as a list of tuples of points), predict the future trajectory
    until a few timesteps
    Params:
    sess: Current session of Tensorflow
    traj: List of past trajectory points
    num: Number of time-steps into the future to be predicted
    '''
    # Initial state with zeros
    state = sess.run(self.cell.zero_state(1, tf.float32))

    # Iterate over all the positions seen in the trajectory
    for pos in traj[:-1]:
        # Create the input data tensor
        data = np.zeros((1, 1, 2), dtype=np.float32)
        data[0, 0, 0] = pos[0]  # x
        data[0, 0, 1] = pos[1]  # y

        # Create the feed dict
        feed = {self.input_data: data, self.initial_state: state}
        # Get the final state after processing the current position
        [state] = sess.run([self.final_state], feed)

    ret = traj

    # Last position in the observed trajectory
    last_pos = traj[-1]

    # Construct the input data tensor for the last point
    prev_data = np.zeros((1, 1, 2), dtype=np.float32)
    prev_data[0, 0, 0] = last_pos[0]  # x
    prev_data[0, 0, 1] = last_pos[1]  # y

    for t in range(num):
        # Create the feed dict
        feed = {self.input_data: prev_data, self.initial_state: state}

        # Get the final state and also the coef of the distribution of the next point
        [o_mux, o_muy, o_sx, o_sy, o_corr, state] = sess.run([self.mux, self.muy, self.sx, self.sy, self.corr, self.final_state], feed)

        # Sample the next point from the distribution
        next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
        # Append the new point to the trajectory
        ret = np.vstack((ret, [next_x, next_y]))

    # Set the current sampled position as the last observed position
    prev_data[0, 0, 0] = next_x
    prev_data[0, 0, 1] = next_y

    return ret

def train(args):
    datasets = list(range(2))

    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.max_num_peds, datasets, forcePreProcess = True)

    model = SocialModel(args)

    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate, decay = 5e-4)

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()

        for batch in range(data_loader.num_batches):
            start = time.time()

            x, y, d, num_peds, ped_ids = data_loader.next_batch()

            for batch in range(data_loader.batch_size):

                x_batch, y_batch, d_batch, num_ped_batch, ped_id_batch = x[batch], y[batch], d[batch], num_peds[batch], ped_ids[batch]

                if d_batch == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                print(ped_id_batch)

                print(num_ped_batch)

                grid_batch = get_sequence_grid_mask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                # print("grid batch size:{}".format(grid_batch.shape))
                # print(np.where(grid_batch > 0))

                # ped_ids_index = dict(zip(ped_id_batch, range(0, len(ped_id_batch))))
                x_batch, ped_ids_index = data_loader.convert_proper_array(x_batch, num_ped_batch, ped_id_batch)

                train_loss = 0.0
                with tf.GradientTape() as tape:
                    tensor_x = tf.convert_to_tensor(x_batch, dtype=tf.float32)

                    logits = model(tensor_x, ped_id_batch, ped_ids_index)

                    [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                    # reshape target data so that it aligns with predictions
                    tensor_y = tf.convert_to_tensor(y, dtype=tf.float32)

                    # flat_target_data = tf.reshape(tensor_y, [-1, 2])
                    # Extract the x-coordinates and y-coordinates from the target data
                    [x_data, y_data] = tf.split(tensor_y, 2, -1)

                    # Compute the loss function
                    loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                    # Compute the cost
                    train_loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                    grads = tape.gradient(train_loss, model.trainable_variables)

                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                end = time.time()
                # Print epoch, batch, loss and time taken
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(e * data_loader.num_batches + batch,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))


if __name__ == '__main__':
    main()