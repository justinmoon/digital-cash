import pytest
from blockcoin import *


def test_block_heights():
    bank = Bank()

    # Good block
    block = Block(height=0)
    bank.handle_block(block)
    assert len(bank.blocks) == 1

    # Block with bad height
    block = Block(height=3)
    with pytest.raises(AssertionError):
        bank.handle_block(block)


def test_utxo():
    pass


def test_block_production():
    pass
