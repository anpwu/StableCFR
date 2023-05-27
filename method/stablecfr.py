import os
import random
import numpy as np
import pandas as pd 
import scipy.stats as st
import tensorflow as tf
from .module import Net

def log(logfile, str_):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str_+'\n')
    print(str_)

class CausalDB(object):
    def __init__(self, data):
        self.x = data[:, :3]
        self.t = data[:, 3:4]
        self.mu0 = data[:, 4:5]
        self.mu1 = data[:, 5:6]
        self.yf = data[:, 6:7]


''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', 'StableCFR', """Model Name """)
tf.app.flags.DEFINE_integer('block_num', 10, """Batch size. """)
tf.app.flags.DEFINE_integer('top', 5, """Batch size. """)
tf.app.flags.DEFINE_float('param', 0.1, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('pi', 0.5, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_float('p_alpha', 0.0, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_integer('dim_in', 200, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_string('activation', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('detail', 1, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_boolean('saveModel', 0, """Whether to save model. """)
tf.app.flags.DEFINE_string('imb_fun', 'wass', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('experiments', 10, """Number of experiments. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('fairMode', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_string('outdir', 'results/example_ihdp', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', 'data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-100.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 200, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)

def pdist2(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)

def get_batch_idx(train_X, batch_size):
    x_1 = np.random.uniform(-2.0,2.0,size=(batch_size,1))
    x_2 = np.random.uniform(-0.1,2.0,size=(batch_size,1))
    x_3 = np.random.uniform(-2.0,0.1,size=(batch_size,1))
    x_c = np.concatenate([x_1,x_2,x_3], 1)
    D = pdist2(x_c, train_X)
    # print(D.min())
    return np.argmin(D,1)

def get_batch_idx_from_dist(train_X, batch_size, top=5, param=0.1, pi=0.5):
    x_1 = np.random.uniform(-2.0,2.0,size=(batch_size,1))
    x_2 = np.random.uniform(-0.1,2.0,size=(batch_size,1))
    x_3 = np.random.uniform(-2.0,0.1,size=(batch_size,1))
    x_c = np.concatenate([x_1,x_2,x_3], 1)
    D = pdist2(x_c, train_X)

    epsilon = np.random.rand()
    if epsilon > pi:
        return np.argmin(D,1)

    index_top = D.argsort()[:,:top]
    D.sort(-1)
    D_top = D[:,:top]
    D_prob = st.norm.pdf(D_top, loc=0, scale=param)
    D_prob = D_prob / np.sum(D_prob, axis=1, keepdims=True)

    idx_list = []
    for i in range(batch_size):
        idx_list.append(random.choices(
            population=index_top[i],
            weights=D_prob[i],
            k=1)[0])
    idx_list = np.array(idx_list)
    return idx_list

def get_batch_idx_from_normal(train_X, batch_size, top=5, param=0.1, pi=0.5):
    x_1 = np.random.uniform(-2.0,2.0,size=(batch_size,1))
    x_2 = np.random.uniform(-0.1,2.0,size=(batch_size,1))
    x_3 = np.random.uniform(-2.0,0.1,size=(batch_size,1))
    x_c = np.concatenate([x_1,x_2,x_3], 1)
    D = pdist2(x_c, train_X)

    if pi > 0.99:
        return np.argmin(D,1)

    index_top = D.argsort()[:,:top]
    D.sort(-1)
    D_top = D[:,:top]
    D_prob = st.norm.pdf(D_top, loc=0, scale=param)
    D_prob = D_prob / np.sum(D_prob, axis=1, keepdims=True)

    idx_list = []
    epsilon = np.random.rand(batch_size)
    for i in range(batch_size):
        if epsilon[i] > pi:
            idx_list.append(index_top[i][0])
        else:
            idx_list.append(random.choices(
                population=index_top[i],
                weights=D_prob[i],
                k=1)[0])
    idx_list = np.array(idx_list)

    return idx_list

class StableCFR(object):
    def __init__(self) -> None:
        self.config = {
                    'name': 'StableCFR',
                    'top': 10,
                    'param': 0.25, 
                    'pi': 0.6,
                    'p_alpha': 0.0,
                    'p_lambda': 1e-4, 
                    'n_in': 3,
                    'n_out': 3,
                    'dim_in': 128,
                    'dim_out': 128,
                    'dropout_in': 1.0,
                    'dropout_out': 1.0,
                    'iterations': 10000, 
                    'lrate': 2e-3, 
                    'activation': 'elu',
                    'batch_size': 100,
                    'batch_norm': 0,
                    'experiments': 10,
                    'output_delay': 100,
                    'detail': 1,
                    'imb_fun': 'wass',
                    'saveModel': False, 
                    'fairMode': True, 
                    'seed': 2022,
                    'block_num': 10,
                    }

    def set_Configuration(self, config):
        self.config = config

    def run(self, dataName, config=None):
        if config is None:
            config = self.config

        FLAGS.name = config['name']
        FLAGS.top = config['top']
        FLAGS.param = config['param']
        FLAGS.pi = config['pi']
        FLAGS.fairMode = config['fairMode']
        FLAGS.p_alpha= config['p_alpha']
        FLAGS.p_lambda = config['p_lambda']
        FLAGS.n_in = config['n_in']
        FLAGS.n_out = config['n_out']
        FLAGS.dim_in = config['dim_in']
        FLAGS.dim_out = config['dim_out']
        FLAGS.dropout_in = config['dropout_in']
        FLAGS.dropout_out = config['dropout_out']
        FLAGS.iterations = config['iterations']
        FLAGS.lrate = config['lrate']
        FLAGS.activation = config['activation']
        FLAGS.batch_size = config['batch_size']
        FLAGS.batch_norm = config['batch_norm']
        FLAGS.experiments = config['experiments']
        FLAGS.output_delay = config['output_delay']
        FLAGS.detail = config['detail']
        FLAGS.imb_fun = config['imb_fun']
        FLAGS.saveModel = config['saveModel']
        FLAGS.seed = config['seed']
        FLAGS.block_num = config['block_num']

        ''' Set random seeds '''
        random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        testData_ = np.load('./Data/{}/test/test.npy'.format(dataName))
        testData_ = testData_.reshape(-1, 7)
        testDB = CausalDB(testData_)

        self.rst_table = []
        for exp in range(FLAGS.experiments):
            self.train(exp, dataName, testDB, FLAGS)

        result_table = np.array(self.rst_table)
        print("test - mean:", result_table.mean(0)[-5:].round(4), "test - std:", result_table.std(0)[-5:].round(4))

        return result_table

    def train(self, exp, dataName, testDB, FLAGS):
        dataPath = './Data/{}/{}/'.format(dataName, exp)
        savePath = './Result/{}/'.format(dataName)

        os.makedirs(os.path.dirname(savePath+'/{}/checkpoints/{}/'.format(FLAGS.name, exp)), exist_ok=True)
        os.makedirs(os.path.dirname(savePath+'/{}/draw/'.format(FLAGS.name)), exist_ok=True)
        os.makedirs(os.path.dirname(savePath+'/{}/log/'.format(FLAGS.name)), exist_ok=True)
        os.makedirs(os.path.dirname(savePath+'/{}/result/'.format(FLAGS.name)), exist_ok=True)

        logfile  = savePath+'/{}/log/log_{}.txt'.format(FLAGS.name, exp)
        with open(logfile,'w') as f:
            f.write('')

        ''' Load dataset '''
        trainData_ = np.load(dataPath+'train_all.npy')
        validData_ = np.load(dataPath+'valid_all.npy')
        trainDB = CausalDB(trainData_)
        validDB = CausalDB(validData_)
        n, x_dim = trainDB.x.shape

        num4t1 = np.sum(trainDB.t)
        num4t0 = n - np.sum(trainDB.t)

        trainData_t0 = np.load(dataPath+'train_t0.npy')
        validData_t0 = np.load(dataPath+'valid_t0.npy')
        trainData_t1 = np.load(dataPath+'train_t1.npy')
        validData_t1 = np.load(dataPath+'valid_t1.npy')
        trainDBt0 = CausalDB(trainData_t0)
        validDBt0 = CausalDB(validData_t0)
        trainDBt1 = CausalDB(trainData_t1)
        validDBt1 = CausalDB(validData_t1)

        log(logfile, 'The data is from: {}. '.format(dataPath))
        log(logfile, 'The number of Control Group (training T=0): {:.0f}'.format(num4t0))
        log(logfile, 'The number of Treated Group (training T=1): {:.0f}'.format(num4t1))

        ''' Start Session '''
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

        ''' Define model graph '''
        REP = Net(n, x_dim, FLAGS)

        ''' Set up optimizer '''
        global_step = tf.Variable(0, trainable=False)
        NUM_ITERATIONS_PER_DECAY = 100
        lr = tf.train.exponential_decay(FLAGS.lrate, global_step, NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
        opt = tf.train.AdamOptimizer(lr)
        train_step = opt.minimize(REP.tot_loss,global_step=global_step)

        ''' Compute treatment probability'''
        p_treated = np.mean(trainDB.t)

        ''' Set up loss feed_dicts'''
        dict_train = {REP.x: trainDB.x, REP.t: trainDB.t, REP.y_: trainDB.yf, 
                        REP.do_in: 1.0, REP.do_out: 1.0, REP.r_alpha: FLAGS.p_alpha, REP.r_lambda: FLAGS.p_lambda, REP.p_t: p_treated}
        dict_valid   = {REP.x: validDB.x, REP.t: validDB.t, REP.y_: validDB.yf, 
                        REP.do_in: 1.0, REP.do_out: 1.0, REP.r_alpha: FLAGS.p_alpha, REP.r_lambda: FLAGS.p_lambda, REP.p_t: p_treated}

        # dict_test4T1 = {REP.x: testDB.x, REP.t: 1-testDB.t+testDB.t, REP.y_: testDB.mu1, REP.do_in: 1.0, REP.do_out: 1.0}
        # dict_test4T0 = {REP.x: testDB.x, REP.t: testDB.t-testDB.t,   REP.y_: testDB.mu0, REP.do_in: 1.0, REP.do_out: 1.0}
        # dict_train4T1 = {REP.x: trainDB.x, REP.t: 1-trainDB.t+trainDB.t, REP.y_: trainDB.mu1, REP.do_in: 1.0, REP.do_out: 1.0}
        # dict_train4T0 = {REP.x: trainDB.x, REP.t: trainDB.t-trainDB.t,   REP.y_: trainDB.mu0, REP.do_in: 1.0, REP.do_out: 1.0}


        ''' Initialize TensorFlow variables '''
        sess.run(tf.global_variables_initializer())

        ''' Train for multiple iterations '''
        saver = tf.train.Saver()
        objnan = False
        early_stop = 9999
        for i in range(FLAGS.iterations):

            ''' Fetch sample '''
            if FLAGS.fairMode:
                I = get_batch_idx_from_normal(trainDBt1.x, FLAGS.batch_size//2, FLAGS.top, FLAGS.param, FLAGS.pi)
                x_batcht1 = trainDBt1.x[I,:]
                t_batcht1 = trainDBt1.t[I]
                y_batcht1 = trainDBt1.yf[I]

                I = get_batch_idx_from_normal(trainDBt0.x, FLAGS.batch_size//2, FLAGS.top, FLAGS.param, FLAGS.pi)
                x_batcht0 = trainDBt0.x[I,:]
                t_batcht0 = trainDBt0.t[I]
                y_batcht0 = trainDBt0.yf[I]

                x_batch = np.concatenate([x_batcht0,x_batcht1], 0)
                t_batch = np.concatenate([t_batcht0,t_batcht1], 0)
                y_batch = np.concatenate([y_batcht0,y_batcht1], 0)
            else:
                I = random.sample(range(0, n), FLAGS.batch_size)
                x_batch = trainDB.x[I,:]
                t_batch = trainDB.t[I]
                y_batch = trainDB.yf[I]

            ''' Do one step of gradient descent '''
            if not objnan:
                sess.run(train_step, feed_dict={REP.x: x_batch, REP.t: t_batch, \
                    REP.y_: y_batch, REP.do_in: FLAGS.dropout_in, REP.do_out: FLAGS.dropout_out, \
                    REP.r_alpha: FLAGS.p_alpha, REP.r_lambda: FLAGS.p_lambda, REP.p_t: p_treated})

            ''' Compute loss every N iterations '''
            if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
                obj_loss,f_error,imb_err = sess.run([REP.tot_loss, REP.pred_loss, REP.imb_dist],feed_dict=dict_train)
                valid_obj, valid_f_error, valid_imb = sess.run([REP.tot_loss, REP.pred_loss, REP.imb_dist], feed_dict=dict_valid)


                if np.isnan(obj_loss):
                    print('Experiment {}: Objective is NaN. Skipping.'.format(exp))
                    objnan = True

                loss_str = str(i) + '\tTrtObj: %.3f,\tTrtY: %.3f,\tTrtImb: %.2g,\tValObj: %.3f,\tValY: %.3f,\tValImb: %.2g' \
                            % (obj_loss, f_error, imb_err, valid_obj, valid_f_error, valid_imb)
                log(logfile, loss_str)
                
                if (early_stop > valid_f_error and i >= FLAGS.iterations//2 and i >= FLAGS.iterations - 5000) or i == FLAGS.iterations//2:
                    if FLAGS.detail:
                        train_str, train_result, _, _, trainRE = self.test(logfile, i, sess, REP, trainDB.x, trainDB.t, trainDB.mu0, trainDB.mu1, 'train')
                        valid_str, valid_result, _, _, validRE = self.test(logfile, i, sess, REP, validDB.x, validDB.t, validDB.mu0, validDB.mu1, 'valid')
                        test_str,  test_result, _, _, testRE = self.test(logfile, i, sess, REP, testDB.x,  testDB.t,  testDB.mu0,  testDB.mu1,  'test ')

                    log(logfile, 'EarlyStop({}) from {} to {}.'.format(i, early_stop, valid_f_error))
                    early_stop = valid_f_error
                    results = []
                    results_Y0 = []
                    results_Y1 = []
                    for k in range(FLAGS.block_num ** 3):
                        _, result_list, test_hatY0_i, test_hatY1_i, _ = self.test(logfile, i, sess, REP, testDB.x[k*1000:(k+1)*1000], 
                        testDB.t[k*1000:(k+1)*1000], testDB.mu0[k*1000:(k+1)*1000], testDB.mu1[k*1000:(k+1)*1000], 'test', False)
                        results.append(result_list)
                        results_Y0.append(np.concatenate(test_hatY0_i, 1).T)
                        results_Y1.append(np.concatenate(test_hatY1_i, 1).T)
                    results = np.array(results)
                    results_Y0 = np.array(results_Y0)
                    results_Y1 = np.array(results_Y1)
                    if FLAGS.saveModel:
                        saver.save(sess, savePath+'/{}/checkpoints/{}/MyModel'.format(FLAGS.name, exp))         

        self.rst_table.append(train_result+valid_result+test_result)
        result_table = np.array(self.rst_table)

        np.save(savePath+'/{}/detail_rst.npy'.format(FLAGS.name), result_table)
        np.save(savePath+'/{}/result/results_{}.npy'.format(FLAGS.name, exp), results)
        np.save(savePath+'/{}/result/resultsY0_{}.npy'.format(FLAGS.name, exp), results_Y0)
        np.save(savePath+'/{}/result/resultsY1_{}.npy'.format(FLAGS.name, exp), results_Y1)

        np.save(savePath+'/{}/draw/'.format(FLAGS.name)+'train_{}.npy'.format(exp), np.concatenate(trainRE, 1))
        np.save(savePath+'/{}/draw/'.format(FLAGS.name)+'valid_{}.npy'.format(exp), np.concatenate(validRE, 1))
        np.save(savePath+'/{}/draw/'.format(FLAGS.name)+'test_{}.npy'.format(exp), np.concatenate(testRE, 1))

    def test(self, logfile, i, sess, REP, X, T, MU0, MU1, name='train', ifprint=True):
        dict4T1 = {REP.x: X, REP.t: 1-T+T, REP.y_: MU1, REP.do_in: 1.0, REP.do_out: 1.0}
        dict4T0 = {REP.x: X, REP.t: T-T,   REP.y_: MU0, REP.do_in: 1.0, REP.do_out: 1.0}

        hat_y0 = sess.run(REP.output, feed_dict=dict4T0)
        hat_y1 = sess.run(REP.output, feed_dict=dict4T1)
        ITE_from_mu = MU1 - MU0
        ITE_from_hat = hat_y1 - hat_y0
        ATE_from_mu = np.mean(ITE_from_mu)
        ATE_from_hat = np.mean(ITE_from_hat)
        ATE_bias = ATE_from_hat - ATE_from_mu
        PEHE = np.sqrt(np.mean(np.square(ITE_from_hat-ITE_from_mu)))
        MSE = np.mean( (ITE_from_mu - ITE_from_hat) **2 )
        T1_MSE = np.mean( (hat_y1 - MU1) **2 )
        T0_MSE = np.mean( (hat_y0 - MU0) **2 )

        detail_str = '{}({}) - T0_MSE: {:.4f}, T1_MSE: {:.4f}, MSE: {:.4f}, PEHE: {:.4f}, ATE_Bias: {:.4f}'.format(
               name, i, T0_MSE, T1_MSE, MSE, PEHE, ATE_bias)

        if ifprint:
            log(logfile, detail_str)

        return detail_str, [T0_MSE, T1_MSE, MSE, PEHE, ATE_bias], [hat_y0, MU0], [hat_y1, MU1], [X, MU0, hat_y0, MU1, hat_y1]

    