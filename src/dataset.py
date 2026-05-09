import gzip
import struct
import numpy as np
import urllib.request

def load_images(filepath):
    with gzip.open(filepath,'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII",f.read(16))
        images = np.frombuffer(f.read(),dtype=np.uint8)
        images = images.reshape(num,rows*cols)
    return images


def load_labels(filepath):
    with gzip.open(filepath,'rb') as f:
        struct.unpack(">II",f.read(8))
        labels = np.frombuffer(f.read(),dtype=np.uint8)
    return labels


def load_fashion_mnist(data_dir="data/fashion_mnist",val_ratio=0.1):
    x_train_all=load_images(r"data\fashion_mnist\train-images-idx3-ubyte.gz")
    y_train_all=load_labels(r"data\fashion_mnist\train-labels-idx1-ubyte.gz")
    x_test=load_images(r"data\fashion_mnist\t10k-images-idx3-ubyte.gz")
    y_test=load_labels(r"data\fashion_mnist\t10k-labels-idx1-ubyte.gz")
                       
    
    x_train_all= x_train_all.astype(np.float32)/255.0
    x_test= x_test.astype(np.float32)/255.0


    num_train=len(x_train_all)
    num_val=int(num_train*val_ratio)


    indices=np.random.permutation(num_train)
    val_idx=indices[:num_val]
    train_idx=indices[num_val:num_train]


    x_train=x_train_all[train_idx]
    x_val=x_train_all[val_idx]
    y_train=y_train_all[train_idx]
    y_val=y_train_all[val_idx]

    print(f"train:{x_train.shape},val:{x_val.shape},test:{x_test.shape}")
    return x_train,x_val,x_test,y_train,y_val,y_test


def get_batches(x,y,batch_size,shuffle=True):
    n=len(x)
    indices=np.random.permutation(n) if shuffle else np.arange(n)
    
    
    for start in range(0,n,batch_size):
        idx=indices[start:start+batch_size]
        yield x[idx],y[idx]









    