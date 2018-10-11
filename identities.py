from ecdsa import SigningKey, SECP256k1

alice_private_key = SigningKey.from_secret_exponent(1, curve=SECP256k1)
alice_public_key = alice_private_key.get_verifying_key()

bob_private_key = SigningKey.from_secret_exponent(2, curve=SECP256k1)
bob_public_key = bob_private_key.get_verifying_key()

def lookup_key(name):
    name_to_key = {
        "alice": alice_private_key,
        "bob": bob_private_key,
    }
    return name_to_key[name]

def bank_private_key(id):
    assert isinstance(id, int)
    assert id >= 0
    base = 1000
    return SigningKey.from_secret_exponent(base + id, curve=SECP256k1)

def bank_public_key(id):
    """Returns public key of a block producer given their ID"""
    private_key = bank_private_key(id)
    return private_key.get_verifying_key()
