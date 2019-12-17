import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets


def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
batchSize = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(batchSize)
train_db = train_db.map(preprocess)
train_db.repeat(20)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchSize).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)


def train():
    lr = 1e-2
    accs, losses = [], []

    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        if step % 80 == 0:
            print(step, 'loss', float(loss))
            losses.append(float(loss))

        if step % 80 == 0:
            total, total_correct = 0., 0

            for x, y in test_db:
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)

                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)

                out = h2 @ w3 + b3

                pred = tf.argmax(out, axis=1)
                y = tf.argmax(y, axis=1)

                correct = tf.equal(pred, y)

                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确性')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')


if __name__ == '__main__':
    train()
