import tensorflow as tf

def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """
    d = tf.square(x-y)
    d = tf.sqrt(tf.reduce_sum(d, axis=1)+1e-8) # What about the axis ???
    return d

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels) 
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
