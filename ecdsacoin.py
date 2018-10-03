from ecdsa import SigningKey, SECP256k1
from utils import serialize


bank_private_key = SigningKey.generate(curve=SECP256k1)
bank_public_key = bank_private_key.get_verifying_key()


def transfer_message(previous_signature, public_key):
    return serialize({
        "previous_signature": previous_signature,
        "next_owner_public_key": public_key,
    })


class Transfer:
    
    def __init__(self, signature, public_key):
        self.signature = signature
        self.public_key = public_key


class ECDSACoin:
    
    def __init__(self, transfers):
        self.transfers = transfers

    def validate(self):
        # Check the first transfer
        transfer = self.transfers[0]
        message = serialize(transfer.public_key)
        assert bank_public_key.verify(transfer.signature, message)

        # Check the subsequent transfers
        previous_transfer = self.transfers[0]
        for transfer in self.transfers[1:]:
            # Check previous owner signed this transfer using their private key
            assert previous_transfer.public_key.verify(
                transfer.signature,
                transfer_message(previous_transfer.signature, transfer.public_key),

            )
            previous_transfer = transfer


def issue(public_key):
    # Create a message specifying who the coin is being issued to
    message = serialize(public_key)
    
    # Create the first transfer, signing with the banks private key
    signature = bank_private_key.sign(message)
    transfer = Transfer(
        signature=signature,
        public_key=public_key,
    )
    
    # Create and return the coin with just the issuing transfer
    coin = ECDSACoin(transfers=[transfer])
    return coin
