import os
import tensorflow as tf
from generator.low import lowSyn
from method.stablecfr import StableCFR

Gen = lowSyn()
Gen.config['block_num'] = 10
Gen.config['scale'] = 0.8
Gen.run()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
data_name = 'low_long(3)(0.8)_3000'
logfile = './Result/low_long(3)(0.8)_3000/log.txt'

################    Run StableCFR.   ##################
###################################################
model = StableCFR()
model.config['experiments'] = 10
model.config['top'] = 10
model.config['param'] = 0.25
model.config['pi'] = 0.6
model.config['fairMode'] = True
model.config['p_alpha'] = 0.0
model.config['name'] = 'StableCFR'
result_table = model.run(data_name)
print("#"*20)
print(model.config['name'])
print(result_table.mean(0)[-5:].round(4))
print(result_table.std(0)[-5:].round(4))
print("#"*20)



################   Run VANILLA.  ##################
###################################################
model = StableCFR()
model.config['experiments'] = 10
model.config['top'] = 10
model.config['param'] = 0.25
model.config['pi'] = 0.6
model.config['fairMode'] = False
model.config['p_alpha'] = 0.0
model.config['name'] = 'VANILLA'
result_table = model.run(data_name)
print("#"*20)
print(model.config['name'])
print(result_table.mean(0)[-5:].round(4))
print(result_table.std(0)[-5:].round(4))
print("#"*20)





################   Run CFRNet.   ##################
###################################################
model = StableCFR()
model.config['experiments'] = 10
model.config['top'] = 10
model.config['param'] = 0.25
model.config['pi'] = 0.6
model.config['fairMode'] = False
model.config['p_alpha'] = 1.0
model.config['name'] = 'CFRNet'
result_table = model.run(data_name)
print("#"*20)
print(model.config['name'])
print(result_table.mean(0)[-5:].round(4))
print(result_table.std(0)[-5:].round(4))
print("#"*20)