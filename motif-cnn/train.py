from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import random

from utils import *
from models import MotifCNN
from metrics import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'dblp-p', 'Dataset string.')
flags.DEFINE_integer('seed', 4, 'Random seed')
flags.DEFINE_integer('conv_layers', 3, 'Number of convolution-pooling layers.')
flags.DEFINE_string('hidden_sizes', '64,32,10', 'Hidden layer sizes')
flags.DEFINE_integer('attention_type', 0, 'Type of attention, 0 for dot product, 1 for single weight')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('dropout_fc', 0.5, 'Dropout rate for FC layer (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_bool('calc_motif', False, 'Calculate motif from scratch')
flags.DEFINE_string('motif_def', './motif_def_dblp_p.json', 'JSON file where motif definitions are given')
flags.DEFINE_string('motif', 'apv,pap,pp1,pp2', 'Motifs used for model')

# Set random seed
tf.reset_default_graph()
tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

motif_types = FLAGS.motif.split(',')
hidden_sizes = [int(x) for x in FLAGS.hidden_sizes.split(',')]


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2]


# Initialize session
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask =\
    load_data(FLAGS.dataset, motif_types, load_ind=True, calc_motif=FLAGS.calc_motif, motif_def=FLAGS.motif_def)

y_all = y_train + y_val + y_test

features = preprocess_features(features)

support = preprocess_adj(adj)
num_motifs = len(motif_types)
channels = features[2][1]

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_motifs)],
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'dropout_fc': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = MotifCNN(placeholders, input_dim=channels, hidden_sizes=hidden_sizes,
                 support=support, logging=True)

sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# Train model
cost_val = []
for epoch in range(FLAGS.epochs):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['dropout_fc']: FLAGS.dropout_fc})
    # Training step
    s = time.time()
    outs = sess.run([model.opt_op], feed_dict=feed_dict)
    e = time.time()
    # evaluate train iter
    outs = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
    macro_f1_t, micro_f1_t = compute_f1(outs[-1], y_train, train_mask)
    # Validation
    cost, acc_v, pred_val = evaluate(features, support, y_val, val_mask, placeholders)
    macro_f1_v, micro_f1_v = compute_f1(pred_val, y_val, val_mask)
    cost_val.append(cost)
    # Print results
    print('Epoch:', '%04d' % (epoch + 1), 'train_loss=', '{:.5f}'.format(outs[0]),
          'train_micro_f1=', '{:.5f}'.format(micro_f1_t),
          'train_macro_f1=', '{:.5f}'.format(macro_f1_t), 'time=', '{:.5f}'.format(e - s))
    print('val_loss=', '{:.5f}'.format(cost),
          'val_micro_f1=', '{:.5f}'.format(micro_f1_v),
          'val_macro_f1=', '{:.5f}'.format(macro_f1_v))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1): -1]):
        print('Early stopping...')
        break
print('Optimization Finished!')

cost, acc, pred_test = evaluate(features, support, y_test, test_mask, placeholders)
macro_f1, micro_f1 = compute_f1(pred_test, y_test, test_mask)

print('Test set results:', 'cost=', '{:.5f}'.format(cost),
      'micro_f1=', '{:.5f}'.format(micro_f1),
      'macro_f1=', '{:.5f}'.format(macro_f1))
