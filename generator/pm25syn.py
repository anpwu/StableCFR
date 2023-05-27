import numpy as np
import os
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

local_rng = np.random.RandomState(0)

def potOutcome_t0(x):
    poly_1 = - np.sum(x[:,:2], 1, keepdims=True)
    poly_2 = np.sum(x[:,2:4]*x[:,4:6], 1, keepdims=True)/3
    poly_3 = np.sum(x[:,4:7]**2, 1, keepdims=True)/3
    cos_   = 2*np.cos(np.sum(x,1, keepdims=True))
    return poly_1+poly_2+poly_3 + cos_


def potOutcome_t1(x):
    poly_1 = np.sum(x[:,:2], 1, keepdims=True)
    poly_2 = np.sum(x[:,2:4]*x[:,5:7], 1, keepdims=True)/3
    poly_3 = np.sum(x[:,4:7]**2, 1, keepdims=True)/3
    sin_   = 2*np.cos(np.sum(x,1, keepdims=True))

    return poly_1+poly_2+poly_3+sin_


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

def readPM25(x_path, ty_path, year, scale=1):
    df_ty = pd.read_csv(ty_path, index_col=0)
    df_ty = df_ty[df_ty.Year.isin([year])]
    df_ty = df_ty.drop(['Year'], axis=1)
    df_ty.columns = ['FIPS','PM2.5_{}'.format(year),'CMR_{}'.format(year)]

    df_x = pd.read_csv(x_path, index_col=0)
    df_x = df_x.drop(['population_2000'], axis=1)
    df_x = df_x.drop(['healthfac_2005_1999'], axis=1)
    column_names = df_x.columns
    column_other = []
    column_1990 = []
    column_2000 = []
    column_2010 = []
    for item in column_names:
        if item[-4:] == '1990':
            column_1990.append(item)
        elif item[-4:] == '2000':
            column_2000.append(item)
        elif item[-4:] == '2010':
            column_2010.append(item)
        else:
            column_other.append(item)
    # column_names = column_other + column_2010 + column_2000 + column_1990
    column_names = column_other[0:1] + column_2010

    df_x = df_x[column_names]
    # df_x[column_1990] = df_x[column_2010].values-df_x[column_1990].values
    # df_x[column_2000] = df_x[column_2010].values-df_x[column_2000].values
    df_x_columns = column_names[1:]
    df_x[df_x_columns] = df_x[df_x_columns].apply(lambda x: (x - np.mean(x)) / (np.std(x)) * scale)
    df_x[df_x_columns] = df_x[df_x_columns].apply(lambda x: np.clip(x, -2, 2))
    
    df_tyx = pd.merge(df_ty,df_x,how='inner',on='FIPS')
    return df_tyx, column_names

class PM25(object):
    def __init__(self) -> None:
        self.config = {
                    'name': 'PM25SYN',
                    'num': 2132,
                    'reps': 10, 
                    'split': 0.8,
                    'block_num': 10,
                    'seedInit': 2022,
                    'scale': 1.0,
                    'seedMul': 666,
                    'ty_path': 'Data/Causal/PM25/County_annual_PM25_CMR.csv',
                    'x_path': 'Data/Causal/PM25/County_RAW_variables.csv', 
                    'year': 2010,
                    'drawif': False, 
                    }

    def set_Configuration(self, config):
        self.config = config

    def run(self, config=None):
        if config is None:
            config = self.config

        self.name = config['name']
        self.num = config['num']
        self.reps = config['reps']
        self.seedInit = config['seedInit']
        self.seedMul = config['seedMul']
        self.split = config['split']
        self.block_num = config['block_num']
        self.ty_path = config['ty_path']
        self.x_path  = config['x_path']
        self.year = config['year']
        self.scale = config['scale']
        self.drawif = config['drawif']
        
        df_tyx, column_names = readPM25(self.x_path, self.ty_path, self.year, self.scale)

        df_tyx = df_tyx.drop(['CMR_2010'], axis=1)
        pm25__ = df_tyx['PM2.5_2010'].values
        pi = (pm25__ - pm25__.min())/(pm25__.max() - pm25__.min()) * 0.6 + 0.2
        t  = local_rng.binomial(1, pi) 
        df_tyx['FIPS'] = pi
        df_tyx['PM2.5_2010'] = t
        df_tyx.rename(columns={'FIPS':'PI', 'PM2.5_2010':'T'},inplace=True) 

        print('Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}. '.format(pi.min(),pi.max(),t.mean()))

        if self.num < len(df_tyx):
            df_tyx = df_tyx[:self.num]
        else:
            self.num = len(df_tyx)
        print('Generate Datasets: {}({}_{}). '.format(self.name, self.scale, self.num))
        data_numpy = df_tyx.values

        for exp in range(self.reps):
            self.genTrain(exp, data_numpy)

        show_path = './Data/{}({}_{})/draw/'.format(self.name, self.scale, self.num)
        os.makedirs(os.path.dirname(show_path), exist_ok=True)
        bottom, top = self.show(show_path, df_tyx, self.drawif)
        self.genTest(bottom, top)
        self.genTest3D(bottom, top)

    def genTrain(self, exp, data_numpy):
        num = self.num
        data = copy.deepcopy(data_numpy)

        # Init random seed.
        rng = np.random.RandomState(self.seedInit + exp * self.seedMul)
        rng.shuffle(data)

        # Obtain data.
        pi = data[:, 0:1]
        t = data[:, 1:2]
        x = data[:, 2:]  

        # Generate Y.
        y0 = potOutcome_t0(x)
        y1 = potOutcome_t1(x)
        y = t * y1 + (1-t) * y0 + rng.normal(loc=0.0, scale=0.1, size=(num,1))
        trainData = np.concatenate([x,t,y0,y1,y,pi],1)

        data_path = './Data/{}({}_{})/{}/'.format(self.name, self.scale, self.num, exp)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

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

            return trainData_t0, trainData_t1, trainData, validData_t0, validData_t1, validData

    def show(self, show_path, data_df, drawif=True):
        bottom = data_df.min(0)[2:].values
        top    = data_df.max(0)[2:].values

        if drawif:
            for idx, item in enumerate(data_df.columns[2:]):
                x = data_df[item].values
                n,bins,patches=plt.hist(x, bins=40, density=0, color='steelblue', edgecolor="black", alpha=0.7)
                plt.xlabel(item+f'-{idx}')
                plt.ylabel("num")
                plt.savefig(show_path + f'{item}.png', dpi=400, bbox_inches = 'tight')
                plt.show()
                plt.close()

        np.save(show_path+"bottom.npy", bottom)
        np.save(show_path+"top.npy", top)
        
        return bottom, top

    def genTest3D(self, bottom, top, idx1=0, idx2=1, idx3=2):

        # Init random seed.
        rng = np.random.RandomState(self.seedInit)
        block_num = self.block_num
        baseNum = 1000

        test_lists = []
        for i in range(len(bottom)):
            test_lists.append(rng.uniform(bottom[i], top[i], size=(baseNum * (block_num ** 3),)))
        test_numpy = np.array(test_lists).T

        list1 = np.linspace(bottom[idx1],top[idx1],num=block_num+1)
        list2 = np.linspace(bottom[idx2],top[idx2],num=block_num+1)
        list3 = np.linspace(bottom[idx3],top[idx3],num=block_num+1)

        data_list = []
        for i in range(block_num):
            x1 = rng.uniform(list1[i], list1[i+1], size=(baseNum ,))
            for j in range(block_num):
                x2 = rng.uniform(list2[j], list2[j+1], size=(baseNum ,))
                for k in range(block_num):
                    x3 = rng.uniform(list3[k], list3[k+1], size=(baseNum ,))
                    index_now  =  i*baseNum*(block_num**2)  +  j*baseNum*block_num  +   k*baseNum
                    test_numpy[index_now:(index_now+baseNum), idx1] = x1
                    test_numpy[index_now:(index_now+baseNum), idx2] = x2
                    test_numpy[index_now:(index_now+baseNum), idx3] = x3

        x = test_numpy
        t = np.ones(shape=(baseNum*(block_num**3), 1))
        t[:int(baseNum*(block_num**3)//2),0] = 0
        pi = np.ones(shape=(baseNum*(block_num**3), 1)) * 0.5
        y0 = potOutcome_t0(x)
        y1 = potOutcome_t1(x)
        y = t * y1 + (1-t) * y0 + rng.normal(loc=0.0, scale=0.1, size=(baseNum*(block_num**3),1))

        testData = np.concatenate([x,t,y0,y1,y,pi],1)

        print('testData3D: ', testData.shape)

        data_path = './Data/{}({}_{})/test/'.format(self.name, self.scale, self.num)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        np.save(data_path+"test3D.npy", testData)

        return testData

    def genTest(self, bottom, top, idx1=0, idx2=1, idx3=2):

        # Init random seed.
        rng = np.random.RandomState(self.seedInit)
        block_num = self.block_num
        baseNum = 1000

        test_lists = []
        for i in range(len(bottom)):
            test_lists.append(rng.uniform(bottom[i], top[i], size=(baseNum * (block_num ** 2),)))
        test_numpy = np.array(test_lists).T

        list1 = np.linspace(bottom[idx1],top[idx1],num=block_num+1)
        list2 = np.linspace(bottom[idx2],top[idx2],num=block_num+1)

        data_list = []
        for i in range(block_num):
            x1 = rng.uniform(list1[i], list1[i+1], size=(baseNum ,))
            for j in range(block_num):
                x2 = rng.uniform(list2[j], list2[j+1], size=(baseNum ,))
                index_now  =  i*baseNum*block_num  +  j*baseNum
                test_numpy[index_now:(index_now+baseNum), idx1] = x1
                test_numpy[index_now:(index_now+baseNum), idx2] = x2

        x = test_numpy
        t = np.ones(shape=(len(x), 1))
        t[:int(len(x)//2),0] = 0
        pi = np.ones(shape=(len(x), 1)) * 0.5
        y0 = potOutcome_t0(x)
        y1 = potOutcome_t1(x)
        y = t * y1 + (1-t) * y0 + rng.normal(loc=0.0, scale=0.1, size=(len(x),1))

        testData = np.concatenate([x,t,y0,y1,y,pi],1)

        print('testData: ', testData.shape)

        data_path = './Data/{}({}_{})/test/'.format(self.name, self.scale, self.num)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        np.save(data_path+"test.npy", testData)

        return testData

    def splitTrain(self, trainData, split=None):
        if split is None:
            split = self.split
        trainData_t1 = trainData[trainData[:,7]>0.5]
        trainData_t0 = trainData[trainData[:,7]<=0.5]
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
    Gen = PM25()
    Gen.run()
