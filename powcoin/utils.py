import pickle

def serialize(coin):
    return pickle.dumps(coin)

def deserialize(serialized):
    return pickle.loads(serialized)
