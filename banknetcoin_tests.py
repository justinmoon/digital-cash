import uuid, pytest
from ecdsa import SigningKey, VerifyingKey, SECP256k1
from ecdsa.keys import BadSignatureError
from bankutxocoin import TxIn, TxOut, Tx, Bank
# from mybanknetcoin import TxIn, TxOut, Tx, Bank

# The usual suspects
bob_private_key = SigningKey.generate(curve=SECP256k1)
alice_private_key = SigningKey.generate(curve=SECP256k1)
bob_public_key = bob_private_key.get_verifying_key()
alice_public_key = alice_private_key.get_verifying_key()


def test_bank_balances():
    # Create bank and issue Alice some coins
    bank = Bank()
    coinbase = bank.issue(1000, alice_public_key)

    # Alice sends 10 coins to Bob
    tx_ins = [
        TxIn(tx_id=coinbase.id, index=0, signature=None),
    ]
    tx_id = uuid.uuid4()
    tx_outs = [
        TxOut(tx_id=tx_id, index=0, amount=10, public_key=bob_public_key), 
        TxOut(tx_id=tx_id, index=1, amount=990, public_key=alice_public_key),
    ]
    alice_to_bob = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
    alice_to_bob.sign_input(0, alice_private_key)

    # After signing, Bob modifies transaction to get more $$$
    alice_to_bob.tx_outs[0].amount = 990
    alice_to_bob.tx_outs[1].amount = 10

    with pytest.raises(BadSignatureError):
        bank.handle_tx(alice_to_bob)


def test_public_key_comparisons():
    derived_bob_public_key = VerifyingKey.from_string(
        bob_public_key.to_string(),
        curve=SECP256k1,
    )

    assert bob_public_key.to_string() == derived_bob_public_key.to_string()

    assert bob_public_key == derived_bob_public_key
