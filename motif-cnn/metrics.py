import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics


def masked_softmax_cross_entropy(preds, labels, mask):
    '''Softmax cross-entropy loss with masking.'''
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_sigmoid_cross_entropy(preds, labels, mask):
    '''Sigmoid cross-entropy loss with masking. For multilabel classification'''
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    '''Accuracy with masking.
    Input: preds - output values from FC layer
           labels - one hot encoded class labels
           mask - 0-1 mask of indices
    Output: tf accuracy op'''
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_macro_f1(preds, labels, mask):
    '''Macro-F1 with masking.
    Input: preds - predictions given in 0-1 label indicator format
           labels - ground truth labels in 0-1 label indicator format
           mask - 0-1 mask of indices
    Output: macro f1 score'''
    inds = np.nonzero(mask)[0]
    preds_masked = preds[inds]
    labels_masked = labels[inds]
    return metrics.f1_score(labels_masked, preds_masked, average='macro')


def masked_micro_f1(preds, labels, mask):
    '''Micro-F1 with masking.'''
    inds = np.nonzero(mask)[0]
    preds_masked = preds[inds]
    labels_masked = labels[inds]
    return metrics.f1_score(labels_masked, preds_masked, average='micro')
