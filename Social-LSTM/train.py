import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime

from model import Model
from dataset import DataLoader

from visual import plot_trajectories

def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Number of layers parameter
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Type of recurrent unit parameter
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    # Gradient value at which it should be clipped
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    # Dropout not implemented.
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1,
                        help='Dataset to be tested on')

    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

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

    print("xxxxxxxxxxxxxxxxxxoutput:{}".format(output.shape))
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

def get_mean_error(pred_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''

    # print("===================pred_traj===============")
    # print(pred_traj)

    # print("===================true_traj================")
    # print(true_traj)

    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = pred_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

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

    # print("mean:{}".format(mean))

    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

# def sample_traj(muxs, muys, sxs, sys, rhos):

#     num_points = len(muxs)

#     for num in range(num_points):
#         x, y = sample_gaussian_2d(muxs[num], muys[num], sxs[num], sys[num], rhos[num])

def test(args, model):
    # Dataset to get data from
    dataset = [args.test_dataset]

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoader(1, args.pred_length + args.obs_length, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    # Maintain the total_error until now
    total_error = 0
    counter = 0
    model.reset_states()
    for b in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        x, y = data_loader.next_batch()

        # The observed part of the trajectory
        obs_observed_traj = x[0][:args.obs_length]
        obs_observed_traj = tf.expand_dims(obs_observed_traj, 0)

        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyy")
        # print(obs_observed_traj)

        complete_traj = x[0][:args.obs_length]

        for idx in range(args.pred_length):
            tensor_x = tf.convert_to_tensor(obs_observed_traj)

            logits = model(tensor_x)

            # print("logit:{}".format(logits))

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

            # print("o_mux:{}".format(o_mux))
            # print("o_muy:{}".format(o_muy))
            # print("o_sx:{}".format(o_sx))
            # print("o_sy:{}".format(o_sy))
            # print("o_corr:{}".format(o_corr))

            next_x, next_y = sample_gaussian_2d(o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0])

            obs_observed_traj = tf.expand_dims([[next_x, next_y]], 0)

            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx shape:{}".format(obs_observed_traj.shape))
            # print(obs_observed_traj)

            complete_traj = np.vstack((complete_traj, [next_x, next_y]))

        plot_trajectories(complete_traj, x[0])

        exit(0)

        total_error += get_mean_error(complete_traj, y[0], args.obs_length)

        print("Processed trajectory number: {} out of {} trajectories".format(b, data_loader.num_batches))

    # Print the mean error across all the batches
    print("Total mean error of the model is {}".format(total_error/data_loader.num_batches))


def train(args):
    datasets = list(range(2))
    # Remove the leaveDataset from datasets
    #datasets.remove(args.leaveDataset)

    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    # Save the arguments int the config file
    # with open(os.path.join('save', 'config.pkl'), 'wb') as f:
    #     pickle.dump(args, f)

    # Create a Vanilla LSTM model with the arguments
    model = Model(args)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states()

        for batch in range(data_loader.num_batches):
            start = time.time()
            # Get the source and target data of the current batch
            # x has the source data, y has the target data
            # x, y = data_loader.next_batch()

            x, y = data_loader.next_batch()

            with tf.GradientTape() as tape:
                tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)

                print("Input tensor x shape:{}".format(tensor_x.shape))

                logits = model(tensor_x)

                [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                # reshape target data so that it aligns with predictions
                tensor_y = tf.convert_to_tensor(y, dtype=tf.float32)

                    # flat_target_data = tf.reshape(tensor_y, [-1, 2])
                    # Extract the x-coordinates and y-coordinates from the target data
                [x_data, y_data] = tf.split(tensor_y, 2, -1)

                # Compute the loss function
                loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                # Compute the cost
                loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.lr.assign(args.learning_rate * (args.decay_rate ** e))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                tf.print("learning rate:{} {}/{}".format(optimizer.lr, batch, e))

                train_loss(loss)

            end = time.time()
            # Print epoch, batch, loss and time taken
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(e * data_loader.num_batches + batch,
                            args.num_epochs * data_loader.num_batches,
                            e, loss, end - start))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    test(args, model)


if __name__ == '__main__':
    main()