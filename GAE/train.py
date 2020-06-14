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
# x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0]

model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST)
print(model)
# criterion = nn.CrossEntropyLoss().to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-5)

def train():
    # model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            print("00000000000000000000000000000")
            print(batch_src_index)
            batch_src_label = train_label[batch_src_index]
            print(batch_src_label)

            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]

            batch_train_logits = model(batch_sampling_x)
            loss = loss_object(batch_train_logits, batch_src_label)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # optimizer.zero_grad()
            # loss.backward()  # 反向传播计算参数的梯度
            # optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()


def test():
    # model.eval()
    # with torch.no_grad():
    test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
    # test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
    test_x = [x[idx].float() for idx in test_sampling_result]
    test_logits = model(test_x)
    # test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
    test_label = data.y[test_index]
    predict_y = test_logits.max(1)[1]
    accuarcy = torch.eq(predict_y, test_label).float().mean().item()
    print("Test Accuracy: ", accuarcy)


if __name__ == '__main__':
    train()