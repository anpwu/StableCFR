import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

local_rng = np.random.RandomState(0)

def potOutcome_t0(x1, x2, x3):
    poly_ = x1 * x2 * x3 + x1 ** 2 + (1-x2+x3)**2
    cos_  = np.cos(x1-x2+x3)
    abs_  = np.abs(x2**2 - x3**2)

    return - poly_ + 2*cos_ + abs_ 


def potOutcome_t1(x1, x2, x3):
    poly_ = x1 * x2 * x3 + x1 ** 2 + (1-x2+x3)**2
    sin_  = np.sin(x1-x2+x3)
    abs_  = np.abs(x2**2 - x3**2)

    return poly_ + 2*sin_ + abs_


def longTail(rng, num, bottom, top, scale=1.0):
    x = rng.normal(loc=0.0, scale=scale, size=50000)
    x = x[x<top]
    x = x[x>=bottom]
    x = x[:num].reshape(-1,1)
    return x


def longTailX(rng=local_rng,num=3000,scale=1.0,b1=-2,t1=2,b2=-0.1,t2=2,b3=-2,t3=0.1):
    x1 = longTail(rng, num, b1, t1, scale)
    x2 = longTail(rng, num, b2, t2, scale)
    x3 = longTail(rng, num, b3, t3, scale)

    return x1, x2, x3


def balancedX(rng=local_rng,num=3000,b1=-2,t1=2,b2=-0.1,t2=2,b3=-2,t3=0.1):
    x1 = rng.uniform(b1,t1, size=(num,1))
    x2 = rng.uniform(b2,t2, size=(num,1))
    x3 = rng.uniform(b3,t3, size=(num,1))

    return x1, x2, x3

def conf4pi(x1, x2, x3, confs):
    if confs == 0:
        pi = 1 / (1+np.exp(x1-x1))
    elif confs == 1:
        pi = 1 / (1+np.exp(x1))
    elif confs == 2:
        pi = 1 / (1+np.exp((x2+x3)/2))
    elif confs == 3:
        pi = 1 / (1+np.exp((x1+x2+x3)/3))

    return pi

def conf4t(rng, x1, x2, x3, confs, ifprint=False):
    pi = conf4pi(x1, x2, x3, confs)
    t  = rng.binomial(1, pi) 

    if ifprint:
        print('Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}. '.format(pi.min(),pi.max(),t.mean()))

    return t

class lowSyn(object):
    def __init__(self) -> None:
        self.config = {
                    'name': 'low',
                    'dist': 'long',   # 'long', 'bal'
                    'bound': [-2,2,-0.1,2,-2,0.1], 
                    'num': 3000,
                    'confs': 3,         # 0, 1, 2, 3
                    'reps': 10, 
                    'split': 0.8,
                    'block_num': 10,
                    'seedInit': 2022,
                    'scale': 1.0,
                    'seedMul': 666,
                    }

    def set_Configuration(self, config):
        self.config = config

    def run(self, config=None):
        if config is None:
            config = self.config

        self.name = config['name']
        self.dist = config['dist']
        self.bound = config['bound']
        self.num = config['num']
        self.confs = config['confs']
        self.reps = config['reps']
        self.seedInit = config['seedInit']
        self.seedMul = config['seedMul']
        self.split = config['split']
        self.block_num = config['block_num']
        self.scale = config['scale']

        print('Generate Datasets: {}_{}({})({})_{}. '.format(self.name, self.dist, self.confs, self.scale, self.num))
        for exp in range(self.reps):
            self.genTrain(exp)

        self.genTest()

    def show(self, show_path, x, name, exp):

        n,bins,patches=plt.hist(x, bins=40, density=0, color='steelblue', edgecolor="black", alpha=0.7)
        plt.xlabel('age')
        plt.ylabel("num")
        plt.savefig(show_path + f'{name}({exp}).png', dpi=400, bbox_inches = 'tight')
        plt.show()
        plt.close()

    def genTrain(self, exp=0):
        num = self.num
        scale = self.scale
        b1,t1,b2,t2,b3,t3 = self.bound

        # Init random seed.
        rng = np.random.RandomState(self.seedInit + exp * self.seedMul)

        # Generate X.
        if self.dist == 'long':
            x1, x2, x3 = longTailX(rng,num,scale,b1,t1,b2,t2,b3,t3)
        elif self.dist == 'bal':
            x1, x2, x3 = balancedX(rng,num,b1,t1,b2,t2,b3,t3)
        else:
            self.dist = 'bal'
            x1, x2, x3 = balancedX(rng,num,b1,t1,b2,t2,b3,t3)

            

        # Generate T.
        t = conf4t(rng, x1, x2, x3, self.confs, True)

        # Generate Y.
        y0 = potOutcome_t0(x1, x2, x3)
        y1 = potOutcome_t1(x1, x2, x3)
        y = t * y1 + (1-t) * y0 + rng.normal(loc=0.0, scale=0.1, size=(num,1))
        trainData = np.concatenate([x1,x2,x3,t,y0,y1,y],1)

        data_path = './Data/{}_{}({})({})_{}/{}/'.format(self.name, self.dist, self.confs, self.scale, self.num, exp)
        show_path = './Data/{}_{}({})({})_{}/draw/'.format(self.name, self.dist, self.confs, self.scale, self.num)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(os.path.dirname(show_path), exist_ok=True)

        self.show(show_path, x1, 'x1', exp)
        self.show(show_path, x2, 'x2', exp)
        self.show(show_path, x3, 'x3', exp)

        np.save(data_path+"train.npy", trainData)

        if self.split is None:
            return trainData
        else:
            trainData_t0, trainData_t1, trainData, validData_t0, validData_t1, validData = self.splitTrain(trainData, self.split)

            np.save(data_path+"train_t0.npy", trainData_t0)
            np.save(data_path+"train_t1.npy", trainData_t1)
            np.save(data_path+"train_all.npy", trainData)
            np.save(data_path+"valid_t0.npy", validData_t0)
            np.save(data_path+"valid_t1.npy", validData_t1)
            np.save(data_path+"valid_all.npy", validData)

            countData = self.counT(trainData)
            np.save(data_path+"countData.npy", countData)

            return trainData_t0, trainData_t1, trainData, validData_t0, validData_t1, validData

    def counT(self, data):
        b1,t1,b2,t2,b3,t3 = self.bound
        block_num = self.block_num
        
        list1 = np.linspace(b1,t1,num=block_num+1)
        list2 = np.linspace(b2,t2,num=block_num+1)
        list3 = np.linspace(b3,t3,num=block_num+1)

        prob1 = (norm.cdf(list1[1:]) - norm.cdf(list1[:-1])) / (norm.cdf(list1[-1]) - norm.cdf(list1[0]))
        prob2 = (norm.cdf(list2[1:]) - norm.cdf(list2[:-1])) / (norm.cdf(list2[-1]) - norm.cdf(list2[0]))
        prob3 = (norm.cdf(list3[1:]) - norm.cdf(list3[:-1])) / (norm.cdf(list3[-1]) - norm.cdf(list3[0]))

        diff1 = (list1[1:] - list1[:-1])/2
        diff2 = (list2[1:] - list2[:-1])/2
        diff3 = (list3[1:] - list3[:-1])/2

        xx1 = list1[:-1] + diff1
        xx2 = list2[:-1] + diff2
        xx3 = list3[:-1] + diff3

        data_list = []
        for i in range(block_num):
            j_list = []
            for j in range(block_num):
                k_list = []
                for k in range(block_num):
                    prob_block = prob1[i] * prob2[j] * prob3[k]
                    pi = conf4pi(xx1[i], xx2[j], xx3[k], self.confs)
                    k_list.append([prob_block, pi, prob_block*(1-pi), prob_block*pi])
                j_list.append(k_list)
            data_list.append(j_list)
        countData = np.array(data_list)
        
        return countData




    def genTest(self):
        b1,t1,b2,t2,b3,t3 = self.bound

        # Init random seed.
        rng = np.random.RandomState(self.seedInit)
        block_num = self.block_num

        list1 = np.linspace(b1,t1,num=block_num+1)
        list2 = np.linspace(b2,t2,num=block_num+1)
        list3 = np.linspace(b3,t3,num=block_num+1)

        data_list = []
        for i in range(block_num):
            j_list = []
            for j in range(block_num):
                k_list = []
                for k in range(block_num):
                    x1, x2, x3 = balancedX(rng, 1000, list1[i], list1[i+1], list2[j], list2[j+1], list3[k], list3[k+1])
                    t = conf4t(rng, x1, x2, x3, self.confs)
                    y0 = potOutcome_t0(x1, x2, x3)
                    y1 = potOutcome_t1(x1, x2, x3)
                    y = t * y1 + (1-t) * y0
                    testData = np.concatenate([x1,x2,x3,t,y0,y1,y],1)
                    k_list.append(testData)
                j_list.append(k_list)
            data_list.append(j_list)
        testData = np.array(data_list)
        print('testData: ', testData.shape)

        data_path = './Data/{}_{}({})({})_{}/test/'.format(self.name, self.dist, self.confs, self.scale, self.num)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        np.save(data_path+"test.npy", testData)

        return testData

    def splitTrain(self, trainData, split=None):
        if split is None:
            split = self.split

        trainData_t1 = trainData[trainData[:,3]>0.5]
        trainData_t0 = trainData[trainData[:,3]<=0.5]
        n1 = len(trainData_t1)
        n0 = len(trainData_t0)
        validData_t1 = trainData_t1[int(n1*split):]
        validData_t0 = trainData_t0[int(n0*split):]
        trainData_t1 = trainData_t1[:int(n1*split)]
        trainData_t0 = trainData_t0[:int(n0*split)]

        trainData = np.concatenate([trainData_t0, trainData_t1], 0)
        validData = np.concatenate([validData_t0, validData_t1], 0)

        return trainData_t0, trainData_t1, trainData, validData_t0, validData_t1, validData

        
        

if __name__=="__main__":
    Gen = lowSyn()
    Gen.run()
