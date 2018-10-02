import pickle

def serialize(coin):
    return pickle.dumps(coin)

def deserialize(serialized):
    return pickle.loads(serialized)

def to_disk(coin, filename):
    serialized = serialize(coin)
    with open(filename, "wb") as f:
        f.write(serialized)

def from_disk(filename):
    with open(filename, "rb") as f:
        serialized = f.read()
        return deserialize(serialized)
