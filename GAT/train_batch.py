import tensorflow as tf

import numpy as np
from model import GraphAttentionModel
from dataset import CoraData
from collections import namedtuple

import matplotlib.pyplot as plt

# INPUT_DIM = 1433    # 输入维度
# # Note: 采样的邻居阶数需要与GCN的层数保持一致
# HIDDEN_DIM = [128, 7]   # 隐藏单元节点数
# NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
# assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16     # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # 学习率
NUM_HEADS = 8
INPUT_DIM = 1433
HIDDEN_DIM = 24
OUTPUT_DIM = 7

Data = namedtuple('Data', ['x', 'y', 'traj','train_mask', 'val_mask', 'test_mask'])

data = CoraData().data()

train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0]
val_index = np.where(data.val_mask)[0]

model = GraphAttentionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_HEADS)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-4)

# 记录过程值，以便最后可视化
train_loss_results = []
train_accuracy_results = []
train_val_results = []
train_test_results = []

def train():
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = train_label[batch_src_index].astype(float)

            batch_sampling_x = data.x[batch_src_index]
            batch_adj = data.adj[np.ix_(batch_src_index, batch_src_index)]

            loss = 0.0
            with tf.GradientTape() as tape:
                batch_train_logits = model([batch_sampling_x, batch_adj], training = True)
                loss = loss_object(batch_src_label, batch_train_logits)
                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss))
        
        train_accuracy = test(train_index)
        val_accuracy = test(val_index)
        test_accuracy = test(test_index)

        train_loss_results.append(loss)
        train_accuracy_results.append(train_accuracy)
        train_val_results.append(val_accuracy)
        train_test_results.append(test_accuracy)

        print("Epoch {:03d} train accuracy: {} val accuracy: {} test accuracy:{}".format(e, train_accuracy, val_accuracy, test_accuracy))
        
        # ISSUE: https://stackoverflow.com/questions/58947679/no-gradients-provided-for-any-variable-in-tensorflow2-0

    # 训练过程可视化
    # fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training Metrics')

    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(train_loss_results)

    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].plot(train_accuracy_results)

    # axes[2].set_ylabel("Val Acc", fontsize=14)
    # axes[2].plot(train_val_results)

    # axes[3].set_ylabel("Test Acc", fontsize=14)
    # axes[3].plot(train_test_results)

    # plt.show()

def test(index):
    test_x = data.x[index]

    test_adj = data.adj[np.ix_(index, index)]

    test_logits = model([test_x, test_adj], training = False)
    test_label = data.y[index]

    ll = tf.math.equal(tf.math.argmax(test_label, -1), tf.math.argmax(test_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

    return accuarcy


if __name__ == '__main__':
    train()