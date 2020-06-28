import os
import pickle
import numpy as np
import random

class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, max_num_peds=40, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''
        # List of data directories where raw data resides
        # self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
        #                  './data/ucy/zara/zara01', './data/ucy/zara/zara02',
        #                  './data/ucy/univ']
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.maxNumPeds = max_num_peds

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # pedId, x, y , in each frame is stored
        all_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:

            # Define path of the csv file of the current dataset
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            # Number of frames
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the numpy array for the current dataset
            all_frame_data.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            # index to maintain the current frame
            curr_frame = 0
            for frame in frameList:
                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()

                # Helper print statement to figure out the maximum number of peds in any frame in any dataset
                # if len(pedsList) > 1:
                # print len(pedsList)
                # DEBUG
                #    continue

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                # Add the details of all the peds in the current frame to all_frame_data
                all_frame_data[dataset_index][curr_frame, 0:len(pedsList), :] = np.array(pedsWithPos)
                # Increment the frame index
                curr_frame += 1
            # Increment the dataset index
            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length+2))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]

                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 3))

                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]

                        if pedID == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])
                            if sped.size != 0:
                                sourceData[seq, ped, :] = sped
                            if tped.size != 0:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)
                self.frame_pointer += self.seq_length
                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer()

        return x_batch, y_batch, d

    def tick_batch_pointer(self):
        '''
        Advance the dataset pointer
        '''
        # Go to the next dataset
        self.dataset_pointer += 1
        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.dataset_pointer >= len(self.data):
            self.dataset_pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset all pointers
        '''
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0
        self.frame_pointer = 0

    # def preprocess(self, data_dirs, data_file):
    #     '''
    #     The function that pre-processes the pixel_pos.csv files of each dataset
    #     into data that can be used
    #     params:
    #     data_dirs : List of directories where raw data resides
    #     data_file : The file into which all the pre-processed data needs to be stored
    #     '''
    #     # all_ped_data would be a dictionary with mapping from each ped to their
    #     # trajectories given by matrix 3 x numPoints with each column
    #     # in the order x, y, frameId
    #     # Pedestrians from all datasets are combined
    #     # Dataset pedestrian indices are stored in dataset_indices
    #     all_ped_data = {}
    #     dataset_indices = []
    #     current_ped = 0
    #     # For each dataset
    #     for directory in data_dirs:
    #         # Define the path to its respective csv file
    #         file_path = os.path.join(directory, 'pixel_pos.csv')

    #         print("processing data: {}".format(file_path))
    #         # Load data from the csv file
    #         # Data is a 4 x numTrajPoints matrix
    #         # where each column is a (frameId, pedId, y, x) vector
    #         data = np.genfromtxt(file_path, delimiter=',')

    #         # Get the number of pedestrians in the current dataset
    #         numPeds = np.size(np.unique(data[1, :]))

    #         # For each pedestrian in the dataset
    #         for ped in range(1, numPeds+1):
    #             # Extract trajectory of the current ped
    #             traj = data[:, data[1, :] == ped]
    #             # Format it as (x, y, frameId)
    #             traj = traj[[3, 2, 0], :]

    #             # Store this in the dictionary
    #             all_ped_data[current_ped + ped] = traj

    #         # Current dataset done
    #         dataset_indices.append(current_ped+numPeds)
    #         current_ped += numPeds

    #     # The complete data is a tuple of all pedestrian data, and dataset ped indices
    #     complete_data = (all_ped_data, dataset_indices)
    #     # Store the complete data into the pickle file
    #     f = open(data_file, "wb")
    #     pickle.dump(complete_data, f, protocol=2)
    #     f.close()

    # def load_preprocessed(self, data_file):
    #     '''
    #     Function to load the pre-processed data into the DataLoader object
    #     params:
    #     data_file : The path to the pickled data file
    #     '''

    #     # Load data from the pickled file
    #     f = open(data_file, "rb")
    #     self.raw_data = pickle.load(f)
    #     f.close()

    #     # Get the pedestrian data from the pickle file
    #     all_ped_data = self.raw_data[0]
    #     # Not using dataset_indices for now
    #     # dataset_indices = self.raw_data[1]

    #     # Construct the data with sequences(or trajectories) longer than seq_length
    #     self.data = []
    #     counter = 0

    #     # For each pedestrian in the data
    #     for ped in all_ped_data:
    #         # Extract his trajectory
    #         traj = all_ped_data[ped]
    #         # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
    #         if traj.shape[1] > (self.seq_length+2):
    #             # TODO: (Improve) Store only the (x,y) coordinates for now
    #             self.data.append(traj[[0, 1], :].T)
    #             # Number of batches this datapoint is worth
    #             counter += int(traj.shape[1] / ((self.seq_length+2)))

    #     # Calculate the number of batches (each of batch_size) in the data
    #     self.num_batches = int(counter / self.batch_size)

    # def next_batch(self):
    #     '''
    #     Function to get the next batch of points
    #     '''
    #     # List of source and target data for the current batch
    #     x_batch = []
    #     y_batch = []
    #     # For each sequence in the batch
    #     for i in range(self.batch_size):
    #         # Extract the trajectory of the pedestrian pointed out by self.pointer
    #         traj = self.data[self.pointer]
    #         # Number of sequences corresponding to his trajectory
    #         n_batch = int(traj.shape[0] / (self.seq_length+2))
    #         # Randomly sample a index from which his trajectory is to be considered
    #         idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
    #         # Append the trajectory from idx until seq_length into source and target data
    #         x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
    #         y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))

    #         if random.random() < (1.0/float(n_batch)):
    #             # Adjust sampling probability
    #             # if this is a long datapoint, sample this data more with
    #             # higher probability
    #             self.tick_batch_pointer()

    #     return x_batch, y_batch

    # def tick_batch_pointer(self):
    #     '''
    #     Advance the data pointer
    #     '''
    #     self.pointer += 1
    #     if (self.pointer >= len(self.data)):
    #         self.pointer = 0

    # def reset_batch_pointer(self):
    #     '''
    #     Reset the data pointer
    #     '''
    #     self.pointer = 0