from __future__ import print_function
from layers import *
from metrics import *
from inits import glorot
import sys
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        ''' Wrapper for _build() '''
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, 'tmp/%s.ckpt' % self.name)
        print('Model saved in file: %s' % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        saver = tf.train.Saver(self.vars)
        save_path = 'tmp/%s.ckpt' % self.name
        saver.restore(sess, save_path)
        print('Model restored from file: %s' % save_path)


class MotifCNN(Model):
    def __init__(self, placeholders, input_dim, hidden_sizes, support, **kwargs):
        super(MotifCNN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes

        self.num_motifs = len(support)
        motif_positions = np.zeros(self.num_motifs)
        for i in range(self.num_motifs):
            motif_positions[i] = support[i][2][0]
        self.motif_positions = np.array(motif_positions).astype('int32')

        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay for fc layer
        for var in self.layers[-1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # softmax cross-entropy
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # For each motif convolution layer
        # Input: N * C tensor, sparse tensor for first layer
        # Output: M * N * H tensor
        n_layers = len(self.hidden_sizes)
        for i in range(n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_sizes[i - 1]
            sparse_inputs = (i == 0)
            self.layers.append(MotifConvolution(name='Conv_l' + str(i),
                                                input_dim=input_dim,
                                                output_dim=self.hidden_sizes[i],
                                                placeholders=self.placeholders,
                                                motif_positions=self.motif_positions,
                                                dropout=True,
                                                bias=True,
                                                sparse_inputs=sparse_inputs,
                                                logging=self.logging))
            if i < n_layers - 1:
                # For each motif attention layer
                # Input: M * N * H tensor
                # Output: N * H tensor
                self.layers.append(MotifAttention(name='Attn_l' + str(i),
                                                  hidden_size=self.hidden_sizes[i]))
                # self.layers.append(Concat(name='Concat_l' + str(i)))
            else:
                self.layers.append(Concat(name='Concat'))
        # For fully connected layer
        # Input: (M * N) * H tensor
        # Output: N * D tensor
        self.layers.append(Dense(name='FC',
                                 input_dim=self.hidden_sizes[-1] * self.num_motifs,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 bias=True,
                                 logging=self.logging))

    def predict(self):
        act = tf.nn.softmax(self.outputs)
        pred = tf.one_hot(tf.argmax(act, 1), self.output_dim, on_value=1, off_value=0)
        # pred = tf.nn.sigmoid(self.outputs)
        return pred
