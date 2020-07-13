import gzip
import numpy as np

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def load_images(f):

    with gzip.GzipFile(fileobj=f) as bytestream: 
        # parse meta info
        magic = _read32(bytestream)
        number_of_samples = _read32(bytestream)
        col_dim = _read32(bytestream)
        row_dim = _read32(bytestream)

        # read in image data
        buf = bytestream.read(number_of_samples*col_dim*row_dim)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(number_of_samples, col_dim*row_dim)
                                           
    return data

def load_labels(f):

    with gzip.GzipFile(fileobj=f) as bytestream:

        # parse meta info
        magic = _read32(bytestream)
        number_of_samples = _read32(bytestream)


        # load in label data
        buf = bytestream.read(number_of_samples)
        data = np.frombuffer(buf, dtype=np.uint8)

    return data

def load_mnist_data(path_to_file):
    
    with open(path_to_file, "rb") as f:
        data = load_images(f) if "images" in path_to_file else load_labels(f)
    return data
