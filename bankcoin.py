import uuid
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize


def transfer_message(previous_signature, public_key):
    return serialize({
        "previous_signature": previous_signature,
        "next_owner_public_key": public_key,
    })


class Transfer:
    
    def __init__(self, signature, public_key):
        self.signature = signature
        self.public_key = public_key

    def __eq__(self, other):
        return self.signature == other.signature and \
               self.public_key.to_string() == \
               other.public_key.to_string()


class BankCoin:

    def __init__(self, transfers):
        self.id = uuid.uuid1()
        self.transfers = transfers

    def __eq__(self, other):
        return str(self.id) == str(other.id) and \
               self.transfers == other.transfers

    @property
    def last_transfer(self):
        return self.transfers[-1]

    def transfer(self, owner_private_key, recipient_public_key):
        message = transfer_message(self.last_transfer.signature, 
                                   recipient_public_key)
        transfer = Transfer(
            signature=owner_private_key.sign(message),
            public_key=recipient_public_key,
        )
        self.transfers.append(transfer)

    def validate(self):
        # Check the subsequent transfers
        previous_transfer = self.transfers[0]
        for transfer in self.transfers[1:]:
            # Check previous owner signed this transfer using their private key
            assert previous_transfer.public_key.verify(
                transfer.signature,
                transfer_message(previous_transfer.signature, transfer.public_key),
            )
            # Next loop we treat transfer as previous_transfer
            previous_transfer = transfer

class Bank:

    def __init__(self):
        # This is our single source of truth
        self.coins = {}

    def issue(self, public_key):
        # Create a message specifying who the coin is being issued to
        message = serialize(public_key)
        
        # Create the first transfer, signing with the banks private key
        transfer = Transfer(
            signature=None,
            public_key=public_key,
        )
        
        # Create and return the coin with just the issuing transfer
        coin = BankCoin(transfers=[transfer])

        # Put it in out database of coins
        self.coins[coin.id] = deepcopy(coin)
        return coin

    def observe_coin(self, coin):
        # Look up our copy of the coin
        last_observation = self.coins[coin.id]

        # New observation of this coin builds off what we know
        assert last_observation.transfers == \
               coin.transfers[:len(last_observation.transfers)]

        # Check that transfer history is good
        coin.validate()

        # Update bank database.
        self.coins[coin.id] = deepcopy(coin)

    def fetch_coins(self, public_key):
        coins = []
        for coin in self.coins.values():
            if coin.last_transfer.public_key.to_string() == public_key.to_string():
                coins.append(coin)
        return coins

