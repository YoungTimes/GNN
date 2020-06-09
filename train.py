from graph import GraphConvolutionLayer, GraphConvolutionModel
from utils import *

import time

import tensorflow as tf

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 2
PATIENCE = 10  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]

    # X: (2708, 1433), A_:(2708, 2708)

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k

else:
    raise Exception('Invalid filter type.')

# X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
# H = Dropout(0.5)(X_in)
# H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
# H = Dropout(0.5)(H)
# Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = GraphConvolutionModel()
# model.compile(loss='categorical_crossentropy',
#         optimizer=tf.keras.optimizers.Adam(lr=0.01),
#         metrics=['accuracy'])
def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Helper variables for main training loop
wait = 0
best_val_loss = 99999

# Fit
# for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    # t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
model.fit(graph, y_train, sample_weight=train_mask,
            batch_size=X.shape[0], epochs=200, shuffle=False, verbose=2)

    # # Predict on full dataset
    # preds = model.predict(graph, batch_size=A.shape[0])

    # # Train / validation scores
    # train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
    #                                                [idx_train, idx_val])
    # print("Epoch: {:04d}".format(epoch),
    #       "train_loss= {:.4f}".format(train_val_loss[0]),
    #       "train_acc= {:.4f}".format(train_val_acc[0]),
    #       "val_loss= {:.4f}".format(train_val_loss[1]),
    #       "val_acc= {:.4f}".format(train_val_acc[1]),
    #       "time= {:.4f}".format(time.time() - t))

    # # Early stopping
    # if train_val_loss[1] < best_val_loss:
    #     best_val_loss = train_val_loss[1]
    #     wait = 0
    # else:
    #     if wait >= PATIENCE:
    #         print('Epoch {}: early stopping'.format(epoch))
    #         break
    #     wait += 1

# Testing
# test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
# print("Test set results:",
#       "loss= {:.4f}".format(test_loss[0]),
#       "accuracy= {:.4f}".format(test_acc[0]))