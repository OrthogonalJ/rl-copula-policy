import pickle

def dump_pickle(data, path):
    with open(path, 'wb') as fd:
        pickle.dump(data, fd)

def load_pickle(path):
    with open(path, 'rb') as fd:
        return pickle.load(fd)
