import pytest, time, uuid, ecdsa
import identities
from blockcoin import *


def airdrop_tx():
    id = str(uuid.uuid4()), 
    tx = Tx(
        id=id, 
        tx_ins=[], 
        tx_outs=[
            TxOut(tx_id=id, index=0, amount=500_000, public_key=identities.bob_public_key), 
            TxOut(tx_id=id, index=1, amount=500_000, public_key=identities.alice_public_key),
        ],
    )
    return tx


def test_blocks():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))

    # Good block
    block = Block(height=0, timestamp=time.time(), txns=[])
    block.sign(bank.private_key)
    bank.handle_block(block)
    assert len(bank.blocks) == 1

    # Block with bad height
    block = Block(height=0, timestamp=time.time(), txns=[])

    # FIXME: this should just be private_key_for_bank(bank)
    block.sign(bank.private_key)
    with pytest.raises(AssertionError):
        bank.handle_block(block)

    # Block with bad signature
    block = Block(height=1, timestamp=time.time(), txns=[])
    # Wrong block producer signs
    wrong_private_key = identities.bank_private_key(1000) 
    block.sign(wrong_private_key)
    with pytest.raises(ecdsa.keys.BadSignatureError):
        bank.handle_block(block)


def test_airdrop():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))
    tx = airdrop_tx()
    bank.airdrop(tx)

    assert 500_000 == bank.fetch_balance(identities.alice_public_key)
    assert 500_000 == bank.fetch_balance(identities.bob_public_key)


def test_utxo():
    bank = Bank(id=0, private_key=identities.bank_private_key(0))
    tx = airdrop_tx()
    bank.airdrop(tx)
    assert len(bank.blocks) == 1

    # Alice sends 10 to Bob
    tx = send_value(
        bank=bank,
        sender_private_key=identities.alice_private_key,
        recipient_public_key=identities.bob_public_key,
        amount=10
    )
    block = Block(height=1, timestamp=time.time(), txns=[tx])
    block.sign(bank.private_key)
    bank.handle_block(block)

    assert 500_000 - 10 == bank.fetch_balance(identities.alice_public_key)
    assert 500_000 + 10 == bank.fetch_balance(identities.bob_public_key)


def test_mempool():
    pass
