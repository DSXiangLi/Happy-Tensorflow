 # -*- coding: utf-8 -*-
"""
This file is used to store loader util function 
It will download data from URL, unzip the file and return [train, test]

@author: xiang
"""

import os 
import tarfile 
import pickle 

def url_loader():
    '''
    This function is used to download dataset 
    and unzip the file to specific dir 
    '''
    dataset = 'cifar-100-python.tar.gz' ## can specify in param
    dataurl = 'https://www.cs.toronto.edu/~kriz/' ## can specify in params 
    
    def check_data(dataset):
        try:
            # when run in the data folder 
            filepath = os.path.join(
                    os.path.split(os.path.abspath(__file__))[0],
                    dataset)
        except:
            # When run in the main parent directory 
            filepath = os.path.join(
                    os.getcwd(),
                    "data",
                    dataset)
            
        if( not os.path.isfile(filepath) ):
            from six.moves import urllib
            url = dataurl + dataset
            print('Downloading data from {} to {}'.format(url, filepath))
            urllib.request.urlretrieve(url)
        else:
            print("File exist at {}".format(filepath))
    
        return filepath
    
    filepath = check_data(dataset)
    f = tarfile.open(filepath, "r:gz")
    for member in f.getmembers():
        if 'train' in member.name:
            file = f.extractfile(member.name)
            dic_train = pickle.load(file, encoding = 'latin1')
        elif 'test' in member.name:
            file = f.extractfile(member.name)
            dic_test = pickle.load(file, encoding = 'latin1')
        else:
            pass 
    f.close()
    
    return ([dic_train, dic_test])



