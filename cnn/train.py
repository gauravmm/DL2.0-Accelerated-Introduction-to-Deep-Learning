#! python3

import logging

import tensorflow as tf
from data import cifar10

from . import vgg

logger = logging.getLogger(__name__)


NUM_BATCHES = 10000

# Set up training data:
images, labels = cifar10.get_train()

# Define the model:
n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")
n_label = tf.placeholder(tf.float32, shape=cifar10.get_shape_label(), name="label")

# Build the model
n_output = vgg.build(n_input)

# Define the loss function
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax")
accuracy, accuracy_update_op = tf.metrics.accuracy(labels=n_label, predictions=n_output, name="accuracy")

# Add summaries to track the state of training:
tf.summary.scalar('summary/loss', loss)
tf.summary.scalar('summary/accuracy', accuracy)
summaries = tf.summaries.merge_all()

# Define training operations:
global_step = tf.Variable(0, trainable=False, name='global_step')
inc_global_step = tf.assign(global_step, global_step+1)

train_op = tf.train.AdamOptimizer().minimize(loss)

logger.info("Loading training supervisor...")
sv = tf.train.Supervisor(logdir="train_log/", global_step=global_step, summary_op=None, save_model_secs=600)
logger.info("Done!")

with sv.managed_session() as sess:
    # Set up tensorboard logging:
    logwriter = tf.summary.FileWriter("train_logs/cifar-vgg/", sess.graph)
    logwriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=batch)

    batch = sess.run(global_step)
    logger.info("Starting training from batch {} to {}. Saving model every {}s.".format(batch, NUM_BATCHES, 600))

    while not sv.should_stop():
        if batch >= NUM_BATCHES:
            logger.info("Saving...")
            sv.saver.save(sess, "train_logs/cifar-vgg/model.ckpt", global_step=batch)
            sv.stop()
            break

        if batch > 0 and batch % 100 == 0:
            logger.debug('Step {} of {}.'.format(batch, NUM_BATCHES))

        summ, _ = sess.run((summary_gen, (train_op, inc_global_step, accuracy_update_op)), feed_dict={
            n_input: next(data.rand_vec),
            n_label: next(data.rand_label_vec)
        })
        batch += 1
        
        logwriter.add_summary(summ, global_step=batch)

logger.info("Halting.")