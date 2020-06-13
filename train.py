from graph import GraphConvolutionLayer, GraphConvolutionModel
from dataset import CoraData

import time
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = CoraData()
features, labels, adj, train_mask, val_mask, test_mask = dataset.data()

graph = [features, adj]

model = GraphConvolutionModel()

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def loss(model, x, y, train_mask, training):

    y_ = model(x, training=training)

    test_mask_logits = tf.gather_nd(y_, tf.where(train_mask))
    masked_labels = tf.gather_nd(y, tf.where(train_mask))

    return loss_object(y_true=masked_labels, y_pred=test_mask_logits)


def grad(model, inputs, targets, train_mask):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, train_mask, training=True)
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def test(mask):
    logits = model(graph)

    test_mask_logits = tf.gather_nd(logits, tf.where(mask))
    masked_labels = tf.gather_nd(labels, tf.where(mask))

    ll = tf.math.equal(tf.math.argmax(masked_labels, -1), tf.math.argmax(test_mask_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

    return accuarcy

optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-5)

# 记录过程值，以便最后可视化
train_loss_results = []
train_accuracy_results = []
train_val_results = []
train_test_results = []

num_epochs = 200

for epoch in range(num_epochs):

    loss_value, grads = grad(model, graph, labels, train_mask)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuarcy = test(train_mask)
    val_acc = test(val_mask)
    test_acc = test(test_mask)

    train_loss_results.append(loss_value)
    train_accuracy_results.append(accuarcy)
    train_val_results.append(val_acc)
    train_test_results.append(test_acc)

    print("Epoch {} loss={} accuracy={} val_acc={} test_acc={}".format(epoch, loss_value, accuarcy, val_acc, test_acc))

# 训练过程可视化
fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].plot(train_accuracy_results)

axes[2].set_ylabel("Val Acc", fontsize=14)
axes[2].plot(train_val_results)

axes[3].set_ylabel("Test Acc", fontsize=14)
axes[3].plot(train_test_results)

plt.show()