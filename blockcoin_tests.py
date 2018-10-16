import pytest, time, uuid, ecdsa
import identities
from blockcoin import *


def test_blocks():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))

    # Good block
    block = Block(txns=[])
    block.sign(bank.private_key)
    bank.handle_block(block)
    assert len(bank.blocks) == 1

    # Wrong bank signs
    block = Block(txns=[])
    wrong_private_key = identities.bank_private_key(1000) 
    block.sign(wrong_private_key)
    with pytest.raises(ecdsa.keys.BadSignatureError):
        bank.handle_block(block)

def test_bad_tx():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))
    tx = identities.airdrop_tx()
    bank.airdrop(tx)

    tx = prepare_simple_tx(
        utxos=bank.fetch_utxos(identities.alice_public_key),
        sender_private_key=identities.alice_private_key,
        recipient_public_key=identities.bob_public_key,
        amount=10,
    )
    # Put in a phony signature
    tx.tx_ins[0].signature = identities.alice_private_key.sign(b"bad")

    with pytest.raises(ecdsa.keys.BadSignatureError):
        bank.handle_tx(tx)

def test_airdrop():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))
    tx = identities.airdrop_tx()
    bank.airdrop(tx)

    assert 500_000 == bank.fetch_balance(identities.alice_public_key)
    assert 500_000 == bank.fetch_balance(identities.bob_public_key)


def test_utxo():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))
    tx = identities.airdrop_tx()
    bank.airdrop(tx)
    assert len(bank.blocks) == 1

    # Alice sends 10 to Bob
    tx = prepare_simple_tx(
        utxos=bank.fetch_utxos(identities.alice_public_key),
        sender_private_key=identities.alice_private_key,
        recipient_public_key=identities.bob_public_key,
        amount=10
    )
    block = Block(txns=[tx])
    block.sign(identities.bank_private_key(1))
    bank.handle_block(block)

    assert 500_000 - 10 == bank.fetch_balance(identities.alice_public_key)
    assert 500_000 + 10 == bank.fetch_balance(identities.bob_public_key)
