from powcoin import *
import identities as ids

node = Node()

def send_tx(sender_private_key, recipient_public_key, amount):
    utxos = node.fetch_utxos(sender_private_key.get_verifying_key())
    return prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

# Alice mines the genesis block
genesis_coinbase = prepare_coinbase(public_key=ids.alice_public_key)
unmined_genesis_block = Block(txns=[genesis_coinbase], prev_id=None)
mined_genesis_block = mine_block(unmined_genesis_block)

# FIXME HACK
node.chains.append([mined_genesis_block])
node.active_chain_index = 0
node.update_utxo_set(genesis_coinbase)
print(mined_genesis_block)

# Bob mines next block
print(f"alice has {node.fetch_balance(ids.alice_public_key)}")
coinbase = prepare_coinbase(ids.bob_public_key)
alice_to_bob = send_tx(ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=mined_genesis_block.id)
first_mined_block = mine_block(unmined_block)
node.handle_block(first_mined_block)
print(first_mined_block)

### Bob and Alice both mine next block. Node discover's Bob's first.
# Bob's
coinbase = prepare_coinbase(ids.bob_public_key)
alice_to_bob = send_tx(ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=first_mined_block.id)
bob_fork_block = mine_block(unmined_block)
node.handle_block(bob_fork_block)
print(bob_fork_block)

# Alice's
coinbase = prepare_coinbase(ids.alice_public_key)
bob_to_alice = send_tx(ids.bob_private_key, ids.alice_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_alice], 
                      prev_id=first_mined_block.id)
alice_fork_block = mine_block(unmined_block)
node.handle_block(alice_fork_block)
print(alice_fork_block)

assert node.active_chain == [
    mined_genesis_block,
    first_mined_block,
    bob_fork_block,
]

### Again, they both mine next block. Node discover's Alice's first.
# Alice's
coinbase = prepare_coinbase(ids.alice_public_key)
bob_to_alice = send_tx(ids.bob_private_key, ids.alice_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_alice], 
                      prev_id=mined_block.id)
alice_second_fork_block = mine_block(unmined_block)
node.handle_block(alice_second_fork_block)
print(alice_second_fork_block)

# Bob's
coinbase = prepare_coinbase(ids.bob_public_key)
alice_to_bob = send_tx(ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=mined_block.id)
bob_second_fork_block = mine_block(unmined_block)
node.handle_block(bob_second_fork_block)
print(bob_second_fork_block)

assert node.active_chain == [
    mined_genesis_block,
    first_mined_block,
    alice_fork_block,
    alice_second_fork_block,
]

