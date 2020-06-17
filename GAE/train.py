import tensorflow as tf

import numpy as np
from model import GraphSage
from dataset import CoraData
from sampling import multihop_sampling
from collections import namedtuple

INPUT_DIM = 1433    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 7]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16     # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # 学习率

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

data = CoraData().data()

train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0]
val_index = np.where(data.val_mask)[0]

model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-4)

def train():
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = train_label[batch_src_index].astype(float)

            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            batch_sampling_x = [data.x[np.array(idx.astype(np.int32))] for idx in batch_sampling_result]

            with tf.GradientTape() as tape:
                batch_train_logits = model(batch_sampling_x)
                loss = loss_object(batch_src_label, batch_train_logits)
                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss))
        
        train_accuracy = test(train_index)
        val_accuracy = test(val_index)
        test_accuracy = test(test_index)

        print("Epoch {:03d} train accuracy: {} val accuracy: {} test accuracy:{}".format(e, train_accuracy, val_accuracy, test_accuracy))
        
        # ISSUE: https://stackoverflow.com/questions/58947679/no-gradients-provided-for-any-variable-in-tensorflow2-0

def test(index):
    test_sampling_result = multihop_sampling(index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
    test_x = [data.x[idx.astype(np.int32)] for idx in test_sampling_result]
    test_logits = model(test_x)
    test_label = data.y[index]

    ll = tf.math.equal(tf.math.argmax(test_label, -1), tf.math.argmax(test_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

    return accuarcy


if __name__ == '__main__':
    train()