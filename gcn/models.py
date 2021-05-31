from gcn.layers import GraphConvolution, FullyConnected
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from copy import copy

class GCN_MLP(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = [tf.nn.relu for i in range(len(self.model_config['connection']))]
        self.act[-1] = lambda x: x

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim

        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf1.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
        )
        # self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=self.model_config['learning_rate'],
        # )
        # self.optimizer = tf.train.AdagradOptimizer(
        #     learning_rate=self.model_config['learning_rate'],
        # )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution, 'f': FullyConnected}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        sparse = isinstance(self.placeholders['features'], tf.SparseTensor)
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls, act) in enumerate(zip(layer_size[1:], layer_type, self.act)):
                # create Variables
                # self.activations[-1] = tf.layers.batch_normalization(self.activations[-1], momentum=0.9,
                #                                                      training=self.placeholders['training'])
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=act,
                                             dropout=True if i >= 1 else False,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             bias=True,
                                             conv_config={
                                                'conv'  : 'rnm',
                                                'k'     : 1,
                                             }))
                sparse = False
                # Build sequential layer model
                hidden = self.layers[-1]()  # build the graph, give layer inputs, return layer outpus
                self.activations.append(hidden)

            self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf1.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf1.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        update_ops = tf1.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.opt_op = tf.group([self.opt_op, update_ops])
        self.summary = tf1.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)

        self.cross_entropy_loss = masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        self.loss += self.cross_entropy_loss

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                  for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
