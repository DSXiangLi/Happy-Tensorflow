"""
This file is for preparing cifar10 and extracting it from the binary files

# Please first download cifar100 dataset and extract it in data folder here!!
# Then run this script to prepare the data of cifar100

- Generates numpys
- Generates images
- Generates tfrecords
"""
import os
import numpy as np
import sys
import pickle

sys.path.insert(0,"./data_fetching/")
from Data_Download import url_loader
from Data_Save import save_imgs_to_disk, save_numpy_to_disk, save_img_to_tfrecord
project = 'cifar-100-python'

def main():
    dic_train, dic_test = url_loader()

    print(dic_train.keys())

    x_train = dic_train['data']
    x_test = dic_test['data']

    x_train_filenames = dic_train['filenames']
    x_test_filenames = dic_test['filenames']

    y_train = np.array(dic_train['fine_labels'], np.int32)
    y_test = np.array(dic_test['fine_labels'], np.int32)

    #x_train each row is ordered in n_c * n_b * n_h
    #return sample *n_b * n_h * n_c
    x_train = np.transpose(x_train.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
    x_test = np.transpose(x_test.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))


    x_train_filenames = ['imgs/' + name for name in x_train_filenames]
    x_test_filenames = ['imgs/' + name for name in x_test_filenames]

    ## sub folder for image. Can be different if input are in different format
    if not os.path.exists( os.path.join('data', project,'imgs') ):
        os.makedirs(os.path.join('data', project,'imgs'))

    print("Save Train/Test Filename as pkl..")
    with open(os.path.join('data',project,'x_train_filenames.pkl'), 'wb') as f:
        pickle.dump(x_train_filenames, f)
    with open(os.path.join( 'data',project,'x_test_filenames.pkl'), 'wb') as f:
        pickle.dump(x_test_filenames, f)

    print("Saving the imgs to the disk..")
    save_imgs_to_disk(os.path.join('data', project), x_train, x_train_filenames)
    save_imgs_to_disk(os.path.join('data', project), x_test, x_test_filenames)

    print("Saving the numpys to the disk..")
    save_numpy_to_disk(os.path.join('data', project, 'x_train.npy'), x_train)
    save_numpy_to_disk(os.path.join('data', project, 'y_train.npy'), y_train)
    save_numpy_to_disk(os.path.join('data', project, 'x_test.npy'), x_test)
    save_numpy_to_disk(os.path.join('data', project, 'y_test.npy'), y_test)

    print("Saving the data numpy pickle to the disk..")
    with open(os.path.join('data', project, 'data_numpy.pkl'), 'wb')as f:
        pickle.dump({'x_train': x_train,
                     'y_train': y_train,
                     'x_test': x_test,
                     'y_test': y_test,
                     }, f)

    print('saving tfrecord..')
    save_img_to_tfrecord(os.path.join('data',project,'train.tfrecord'), x_train, y_train)
    save_img_to_tfrecord(os.path.join('data',project,'test.tfrecord'), x_test, y_test)

if __name__ == '__main__':
    main()
