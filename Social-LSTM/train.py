import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime

from model import Model
from dataset import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    parser.add_argument('--test_dataset', type=int, default=4,
                        help='Dataset to be tested on')
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

    args = parser.parse_args()
    # test(args)
    train(args)


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
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
    z = output

    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, -1)

    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):

    result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

    epsilon = 1e-20

    result1 = -tf.math.log(tf.math.maximum(result0, epsilon))  # Numerical stability

    return tf.reduce_sum(result1)

def get_mean_error(pred_traj, true_traj, observed_length):
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = pred_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

def get_final_error(pred_traj, true_traj):

    error = np.linalg.norm(pred_traj[-1, :] - true_traj[-1, :])

    # Return the mean error
    return error


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    # Extract mean
    mean = [mux, muy]

    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def test(args):
    checkpoint_dir = './training_checkpoints'

    # Dataset to get data from
    dataset = [args.test_dataset]

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoader(1, args.pred_length + args.obs_length, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    tf.train.latest_checkpoint(checkpoint_dir)

    args.batch_size = 1

    test_model = build_model(args) # Model(args)

    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    test_model.build(tf.TensorShape([1, None,  2]))

    # Maintain the total_error until now
    total_error = 0
    counter = 0
    final_error = 0.0

    truth_trajs = []
    pred_trajs = []
    gauss_params = []

    for b in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        x, y = data_loader.next_batch()

        base_pos = np.array([[e_x[0] for _ in range(len(e_x))] for e_x in x])
        x = x - base_pos

        # The observed part of the trajectory
        obs_observed_traj = x[0][:args.obs_length]
        obs_observed_traj = tf.expand_dims(obs_observed_traj, 0)

        complete_traj = x[0][:args.obs_length]

        test_model.reset_states()

        # test_model.initial_state = None
        gauss_param = np.array([])

        for idx in range(args.pred_length):
            tensor_x = tf.convert_to_tensor(obs_observed_traj)

            logits = test_model(tensor_x)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

            next_x, next_y = sample_gaussian_2d(o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0])

            obs_observed_traj = tf.expand_dims([[next_x, next_y]], 0)

            if len(gauss_param) <=0:
                gauss_param = np.array([o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])
            else:
                gauss_param = np.vstack((gauss_param, [o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]]))


            complete_traj = np.vstack((complete_traj, [next_x, next_y]))

        total_error += get_mean_error(complete_traj + base_pos[0], x[0] + base_pos[0], args.obs_length)
        final_error += get_final_error(complete_traj + base_pos[0], x[0] + base_pos[0])

        pred_trajs.append(complete_traj)
        truth_trajs.append(x[0])
        gauss_params.append(gauss_param)

        print("Processed trajectory number: {} out of {} trajectories".format(b, data_loader.num_batches))

    # Print the mean error across all the batches
    print("Total mean error of the model is {}".format(total_error/data_loader.num_batches))
    print("Total final error of the model is {}".format(final_error/data_loader.num_batches))

    data_file = "./pred_results.pkl"
    f = open(data_file, "wb")
    pickle.dump([pred_trajs, truth_trajs, gauss_params], f)
    f.close()

def build_model(args):
    output_size = 5
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu,
            batch_input_shape = [args.batch_size, None, 2]),
        tf.keras.layers.GRU(args.rnn_size,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(output_size)
    ])

    return model

def calc_prediction_error(mux, muy, sx, sy, corr, offset_positions, args):

    traj_nums = mux.shape[0]

    pred_nums = mux.shape[1]

    mean_error = 0.0
    final_error = 0.0
    for index in range(traj_nums):
        pred_traj = np.zeros((pred_nums, 2))
        for pt_index in range(pred_nums):
            next_x, next_y = sample_gaussian_2d(mux[index][pt_index][0],
                            muy[index][pt_index][0], sx[index][pt_index][0],
                            sy[index][pt_index][0], corr[index][pt_index][0])

            pred_traj[pt_index][0] = next_x
            pred_traj[pt_index][1] = next_y

        mean_error += get_mean_error(pred_traj, offset_positions[index], args.obs_length)
        final_error += get_final_error(pred_traj, offset_positions[index])

    mean_error = mean_error / traj_nums
    final_error = final_error / traj_nums

    return mean_error, final_error


def train(args):
    datasets = list(range(4))

    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    # Create a Vanilla LSTM model with the arguments
    model = build_model(args)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # 检查点保存至的目录
    checkpoint_dir = './training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states()

        for batch in range(data_loader.num_batches):
            start = time.time()

            x, y = data_loader.next_batch()

            base_pos = np.array([[e_x[0] for _ in range(len(e_x))] for e_x in x])

            x_offset = x - base_pos
            y_offset = y - base_pos

            with tf.GradientTape() as tape:
                tensor_x = tf.convert_to_tensor(x_offset, dtype=tf.float32)

                logits = model(tensor_x)

                [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                tensor_y = tf.convert_to_tensor(y_offset, dtype=tf.float32)

                [x_data, y_data] = tf.split(tensor_y, 2, -1)

                # Compute the loss function
                loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                mean_error, final_error = calc_prediction_error(o_mux, o_muy, o_sx, o_sy, o_corr, tensor_y, args)

                loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.lr.assign(args.learning_rate * (args.decay_rate ** e))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss(loss)

            end = time.time()
            # Print epoch, batch, loss and time taken
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, mean error = {}, final_error = {}"
                    .format(e * data_loader.num_batches + batch,
                            args.num_epochs * data_loader.num_batches,
                            e, loss, end - start, mean_error, final_error))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        model.save_weights(checkpoint_prefix.format(epoch=e))


    test(args)


if __name__ == '__main__':
    main()