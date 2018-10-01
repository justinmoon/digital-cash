import io
import pickle
from PIL import Image


###########
# Helpers #
###########

def handle_user_input(user_input):
    if user_input.lower() == "y":
        return True
    elif user_input.lower() == "n":
        return False
    else:
        user_input  = input('Please enter "y" or "n"')
        handle_user_input(user_input)

def request_and_handle_user_input():
    user_input  = input("Is this a valid minting signature? (y/n)")
    return handle_user_input(user_input)
    
def image_to_bytes(path, file_format):
    img = Image.open(path)
    with io.BytesIO() as output:
        img.save(output, file_format)
        contents = output.getvalue()
        return contents


############
# PNG Coin #
############

class PNGCoin:

    def __init__(self, transfers):
        self.transfers = transfers

    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, serialized):
        return pickle.loads(serialized)

    def to_disk(self, filename):
        serialized = self.serialize()
        with open(filename, "wb") as f:
            f.write(serialized)

    @classmethod
    def from_disk(cls, filename):
        with open(filename, "rb") as f:
            serialized = f.read()
            return cls.deserialize(serialized)

    def validate(self):
        for transfer in self.transfers:
            img = Image.open(io.BytesIO(transfer))
            img.show()
            if not request_and_handle_user_input():
                return False
        return True
