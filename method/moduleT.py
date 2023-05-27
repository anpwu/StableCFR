import tensorflow as tf
import numpy as np

class NetT(object):
    def __init__(self, n, x_dim, FLAGS):
        self.variables = {}
        self.wd_loss = 0
        self.n = n
        self.dims = [x_dim, FLAGS.dim_in, FLAGS.dim_out]

        ''' Initialize input placeholders '''
        self.tnet_x  = tf.placeholder("float", shape=[None, x_dim], name='tnet_x') # Features
        self.tnet_t  = tf.placeholder("float", shape=[None, 1], name='tnet_t')   # Treatent

        if FLAGS.activation.lower() == 'elu':
            self.activation = tf.nn.elu
        elif FLAGS.activation.lower() == 'tanh':
            self.activation = tf.nn.tanh
        else:
            self.activation = tf.nn.relu

        self._build_graph(FLAGS)

    def _add_variable(self, var, name):
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, FLAGS):

        x = self.tnet_x
        t = self.tnet_t

        dim_input = self.dims[0]
        dim_in = self.dims[1]

        weights_in = []; biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        ''' Construct input/representation layers '''
        with tf.name_scope("treatment"):
            h_in = [x]
            for i in range(0, FLAGS.n_in):
                if i==0:
                    ''' If using variable selection, first layer is just rescaling'''
                    if FLAGS.varsel:
                        weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                    else:
                        weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))

                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel and i==0:
                    biases_in.append([])
                    h_in.append(tf.mul(h_in[i],weights_in[i]))
                else:
                    biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                    z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]
                    h_in.append(self.activation(z))

            h_rep = h_in[len(h_in)-1]

            W = tf.Variable(tf.zeros([dim_in, 1]), name='W')
            b = tf.Variable(tf.zeros([1]), name='b')
            sigma = tf.nn.sigmoid(tf.matmul(h_rep, W) + b)
            pts = tf.multiply(t, sigma) + tf.multiply(1.0 - t, 1.0 - sigma)
            cost = -tf.reduce_mean(tf.multiply(t, tf.log(sigma + 1e-4)) + tf.multiply(1.0 - t, tf.log(
                1.0 - sigma + 1e-4))) + 1e-3 * tf.nn.l2_loss(W)

        self.pi_t1 = sigma
        self.cost_loss = cost
        self.pi_t0 = 1-sigma
        self.ipw = 1 / pts