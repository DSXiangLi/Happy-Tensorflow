"""
@author xiang 

Cifar Trainer 
"""


from base.base_train import  BaseTrain

from tqdm import tqdm 
import numpy as np 
import tensorflow as tf 

class CifarTrain(BaseTrain): 
    def __init__(self, sess, config, data_lodaer, model, logger):
        """
        1. Initialize configuration, data loader, model and loggger
        """
        super(CifarTrain, self).__init__(sess, config, data_loader, model, logger)
        
        self.sess = sess
        
        self.config = config
        
        self.data_loader = data_loader
         
        self.model = model 
        
        self.summarize = logger 
        
        self.x, self.y, self.is_train = tf.get_collection('inputs')
        
        self.loss, self.accu, self.train_op = tf.get_collection('train') ## return list of collections in order 
    
        self.model.load(self.sess)
        
    def train(self):
        """
        Load model from last check point 
        Run multiple training epoch 
        """
        for i in tqdm(range(self.model.gloabl_step.eval(self.sess),self.config.n_epochs)):

            self.train_epochs(epoch = i )
            self.sess.run(self.model.global_step_increment)
            if i%10 ==0: 
                print('Current Epoch = {} - Total = {} '.format(self.global_step.eval(self.sess),
                                                   self.config.n_epoch))
    
    
    def train_epoch(self, epoch = None):
        """
        Train one epoch: iterate across all batches 
        Keep track of the Error Metric 
        save model 
        """
        
        loss = []
        accu = []
        
        for i in tqdm(self.data_loader.train_iteration):
            loss, accu = self.train_step()
            loss.append(loss)
            accu.appednd(accu)
            self.sess.run(self.model.epoch_step_increment) ## keep track of current epochs 
            
            if i%100 == 0:
                print('Current Batch = {} - Total = {}'.format(self.model.epoch_step_increment, 
                                                              self.data_loader.train_iteration))
                
        self.model.save()
        
            
    def train_step(self):
        """
        Run the TF session: pass in optimization Function.
        Initialize sess_run with is_triain and assign vaule to self.loss and self.accu 
        Data Iterator of each epoch is get inside of model 
        ? sess.run don't assign value to pass in 
        """
        loss, accu, _ = self.sess.run([self.loss, self.accu, self.train_op],
                      feed_dict = {'is_train': self.is_train})
        return loss,accu
    
            
