from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from mybankcoin import Transfer, Bank
from utils import serialize

# The usual suspects
bob_private_key = SigningKey.generate(curve=SECP256k1)
alice_private_key = SigningKey.generate(curve=SECP256k1)
bob_public_key = bob_private_key.get_verifying_key()
alice_public_key = alice_private_key.get_verifying_key()

def test_valid_transfers():
    # Issue a coin and transfer it twice

    # Bank issues coins to Alice
    bank = Bank()
    coin = bank.issue(alice_public_key)
    initial_coin_copy = deepcopy(coin)

    assert bank.fetch_coins(alice_public_key) == [initial_coin_copy]
    assert bank.fetch_coins(bob_public_key) == []

    # Alice constructs transfer to Bob, but doesn't tell the bank
    coin.transfer(
        owner_private_key=alice_private_key,
        recipient_public_key=bob_public_key,
    )

    # Bank -- source of truth -- doesn't know about transfers until told
    assert bank.fetch_coins(alice_public_key) == [initial_coin_copy]
    assert bank.fetch_coins(bob_public_key) == []

    # Alice tells the bank, which updates it's balances
    bank.observe_coin(coin)
    assert bank.fetch_coins(alice_public_key) == []
    assert bank.fetch_coins(bob_public_key) == [coin]

    # Bob sends to Alice
    coin.transfer(
        owner_private_key=bob_private_key,
        recipient_public_key=alice_public_key,
    )
    bank.observe_coin(coin)
    assert bank.fetch_coins(alice_public_key) == [coin]
    assert bank.fetch_coins(bob_public_key) == []
