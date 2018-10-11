import pytest, time, uuid, ecdsa
import identities
from blockcoin import *

def make_tx(sender_private_key, recipient_public_key, amount):
    # FIXME UNFINISHED
    utxo = bank.fetch_utxo(sender_public_key)
    # FIXME: sort by ascending fee. Add transactions until amount is met ...
    utxo_sum = sum([u.amount for u in utxo])
    tx_ins = [
        TxIn(tx_id=coinbase.id, index=0, signature=None),
    ]
    tx_id = uuid.uuid4()
    tx_outs = [
        TxOut(tx_id=tx_id, index=0, amount=10, public_key=bob_public_key), 
        TxOut(tx_id=tx_id, index=1, amount=990, public_key=alice_public_key),
    ]
    tx = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
    tx.sign_input(0, sender_private_key)
    return tx


def test_blocks():
    bank = Bank(id=0)
    bank_private_key = identities.bank_private_key(bank.id)

    txns = []

    # Good block
    block = Block(height=0, timestamp=time.time(), txns=[])
    block.sign(bank_private_key)
    bank.handle_block(block)
    assert len(bank.blocks) == 1

    # Block with bad height
    block = Block(height=0, timestamp=time.time(), txns=[])

    # FIXME: this should just be private_key_for_bank(bank)
    block.sign(bank_private_key)
    with pytest.raises(AssertionError):
        bank.handle_block(block)

    # Block with bad signature
    block = Block(height=1, timestamp=time.time(), txns=[])
    # Wrong block producer signs
    wrong_private_key = identities.bank_private_key(1000) 
    block.sign(wrong_private_key)
    with pytest.raises(ecdsa.keys.BadSignatureError):
        bank.handle_block(block)


def test_utxo():
    pass


def test_block_production():
    pass
