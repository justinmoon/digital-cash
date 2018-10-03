import pytest
from ecdsa import SigningKey, SECP256k1
import bankcoin
from bankcoin import Transfer, Bank, ECDSACoin, transfer_message
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
    bank.validate_transfers(coin)

    assert bank.fetch_coins(alice_public_key) == [coin]
    assert bank.fetch_coins(bob_public_key) == []

    # Alice sends to Bob
    coin.transfer(
        owner_private_key=alice_private_key,
        recipient_public_key=bob_public_key,
    )
    bank.validate_transfers(coin)

    assert len(bank.fetch_coins(alice_public_key)) == 1
    assert len(bank.fetch_coins(bob_public_key)) == 0

    # Update bank's observation and check balances
    bank.observe_coin(coin)
    assert bank.fetch_coins(alice_public_key) == []
    assert bank.fetch_coins(bob_public_key) == [coin]

    # Bob sends to Bank
    coin.transfer(
        owner_private_key=bob_private_key,
        recipient_public_key=bank.public_key,
    )
    bank.validate_transfers(coin)

    # Update bank's observation and check balances
    bank.observe_coin(coin)
    assert bank.fetch_coins(alice_public_key) == []
    assert bank.fetch_coins(bob_public_key) == []
