import tensorflow as tf

from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    '''Helper function, assigns unique layer IDs.'''
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    '''Dropout for sparse tensors.'''
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    '''Wrapper for tf.matmul (sparse vs dense).'''
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    '''Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output) - main function used for the
            computation of the layer. This function is called by the
            global variables initializer function.
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    '''

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        # Set the name as Layer_x
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    '''Dense layer.'''

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout_fc']
        else:
            self.dropout = 0.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # print ('Shape of x ', x.get_shape(), 'input dim', self.input_dim,'output dim', self.output_dim)
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MotifConvolution(Layer):
    '''Motif convolution layer.'''

    def __init__(self, input_dim, output_dim, motif_positions, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.elu, bias=False, **kwargs):
        super(MotifConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']

        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.num_motifs = len(motif_positions)
        self.motif_positions = motif_positions  # number of positions for each motif

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for m in range(self.num_motifs):
                for k in range(0, self.motif_positions[m]):
                    self.vars['weights_' + str(m) + '_' + str(k)] =\
                        glorot([input_dim, output_dim],
                               name='weights_' + str(m) + '_' + str(k))
            if self.bias:
                for m in range(self.num_motifs):
                    self.vars['bias_' +
                              str(m)] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # motif conv
        new_activations = []
        # for each motif
        for m in range(self.num_motifs):
            x = inputs
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout,
                                   self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)
            adj_positions = tf.sparse_split(
                sp_input=self.support[m], num_split=self.motif_positions[m], axis=0)
            supports = list()
            # For each position
            for k in range(0, self.motif_positions[m]):
                XW = dot(x, self.vars['weights_' + str(m) + '_' + str(k)],
                         sparse=self.sparse_inputs)
                temp = tf.sparse_reduce_sum_sparse(adj_positions[k], axis=0)
                support = dot(temp, XW, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)
            if self.bias:
                output += self.vars['bias_' + str(m)]
            new_activations.append(self.act(output))
        return new_activations


class MotifAttention(Layer):
    '''Attention mechanism for multiple conv unit.'''

    def __init__(self, hidden_size, method='dot_product', num_motifs=1, **kwargs):
        super(MotifAttention, self).__init__(**kwargs)

        self.method = method

        if self.method not in ['dot_product', 'basic']:
            raise NotImplemented('Unsupported attention type')

        with tf.variable_scope(self.name + '_vars'):
            if self.method == 'dot_product':
                self.vars['attn_w'] = tf.Variable(tf.random_uniform((hidden_size,)),
                                                  name='attn_w')
            elif self.method == 'basic':
                self.vars['attn_w'] = tf.Variable(tf.ones((num_motifs,), name="attn_w"))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        if self.method == 'dot_product':
            attn_act = tf.tensordot(
                inputs, self.vars['attn_w'], axes=1)  # n_motif * n_nodes
            # attn_act = tf.multiply(attn_act, 1. / tf.norm(self.vars['attn_w']))
            attn_weights = tf.nn.softmax(
                tf.transpose(attn_act))    # n_nodes * n_motif
            # n_hidden * n_nodes * n_motif
            attended = tf.transpose(inputs, [2, 1, 0]) * attn_weights
            # n_motif * n_nodes * n_hidden
            attended = tf.transpose(attended, [2, 1, 0])
            attended = tf.reduce_sum(attended, axis=0)  # n_nodes * n_hidden
        elif self.method == 'basic':
            attended = tf.multiply(tf.transpose(inputs, [1, 2, 0]), tf.nn.softmax(self.vars['attn_w']))
            attended = tf.transpose(attended, [2, 0, 1])
            attended = tf.reduce_sum(attended, axis=0)
        return attended


class Concat(Layer):
    '''Concatenation layer for multiple motifs.'''

    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)

    def _call(self, inputs):
        # print('Concat inputs shape', inputs.get_shape())
        output = tf.concat(inputs, axis=1)
        return output
