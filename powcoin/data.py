from powcoin import *
from pprint import pprint
import identities as ids
from copy import deepcopy

node = Node("x")
alice_node = Node("x")
bob_node = Node("x")

def send_tx(n, sender_private_key, recipient_public_key, amount):
    utxos = n.fetch_utxos(sender_private_key.get_verifying_key())
    return prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

#################################
# Alice mines the genesis block #
#################################

mined_genesis_block = None

# FIXME HACK
print("Genesis mined")
for n in [node, alice_node, bob_node]:
    mined_genesis_block = mine_genesis_block(n, ids.alice_public_key)
print(mined_genesis_block)
print()

########################
# Bob mines next block #
########################

print("Bob mines first real block")
coinbase = prepare_coinbase(ids.bob_public_key, height=1)
alice_to_bob = send_tx(bob_node, ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=mined_genesis_block.id)
first_mined_block = mine_block(unmined_block)
node.handle_block(first_mined_block)
alice_node.handle_block(first_mined_block)
bob_node.handle_block(first_mined_block)
print(first_mined_block)
print()

###################################################################
# Bob and Alice both mine next block. Node discover's Bob's first #
###################################################################

# Bob's
print("Bob's first fork block:")
coinbase = prepare_coinbase(ids.bob_public_key, height=2)
alice_to_bob = send_tx(bob_node, ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=first_mined_block.id)
bob_fork_block = mine_block(unmined_block)
node.handle_block(bob_fork_block)
bob_node.handle_block(bob_fork_block)
print(bob_fork_block)
print()

# Alice's
print("Alice's first fork block:") 
coinbase = prepare_coinbase(ids.alice_public_key, height=2) 
bob_to_alice = send_tx(alice_node, ids.bob_private_key, ids.alice_public_key, 10) 
unmined_block = Block(txns=[coinbase, bob_to_alice], 
                      prev_id=first_mined_block.id)
alice_fork_block = mine_block(unmined_block)
node.handle_block(alice_fork_block)
alice_node.handle_block(alice_fork_block)
print(alice_fork_block)
print()

assert node.chain == [
    mined_genesis_block,
    first_mined_block,
    bob_fork_block,
]

#################################################
# Alice triggers reorg attempt with a bad block #
#################################################

old_chain = deepcopy(node.chain)
old_branches = deepcopy(node.branches)

print("Alice's bad second fork block:") 
coinbase = prepare_coinbase(ids.alice_public_key, height=3)
bob_to_alice = send_tx(alice_node, ids.bob_private_key, ids.alice_public_key, 10)
# FIXME: badsignatureerror is bad example b/c this can be checked w/o utxo db
bob_to_alice.tx_outs[0].amount = 100000
unmined_block = Block(txns=[coinbase, bob_to_alice], 
                      prev_id=alice_fork_block.id)
alice_second_fork_block = mine_block(unmined_block)
node.handle_block(alice_second_fork_block)
print(alice_second_fork_block)
print()


# Assert chain is unchanged
assert node.chain == [
    mined_genesis_block,
    first_mined_block,
    bob_fork_block,
]
assert node.chain == old_chain
assert node.branches == old_branches

###################################################################
# Again, they both mine next block. Node discover's Alice's first #
###################################################################

# Alice's
print("Alice's second fork block:")


coinbase = prepare_coinbase(ids.alice_public_key, height=3)
bob_to_alice = send_tx(alice_node, ids.bob_private_key, ids.alice_public_key, 10)
unmined_block = Block(txns=[coinbase, bob_to_alice], 
                      prev_id=alice_fork_block.id)
alice_second_fork_block = mine_block(unmined_block)
node.handle_block(alice_second_fork_block)
alice_node.handle_block(alice_second_fork_block)
print(alice_second_fork_block)
print()

expected = [
    mined_genesis_block,
    first_mined_block,
    alice_fork_block,
    alice_second_fork_block,
]
assert node.chain == expected


# Bob's
print("Bob's second fork block:")
coinbase = prepare_coinbase(ids.bob_public_key, height=3)
alice_to_bob = send_tx(bob_node, ids.alice_private_key, ids.bob_public_key, 10)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=bob_fork_block.id)
bob_second_fork_block = mine_block(unmined_block)
node.handle_block(bob_second_fork_block)
bob_node.handle_block(bob_second_fork_block)
print(bob_second_fork_block)
print()

expected = [
    mined_genesis_block,
    first_mined_block,
    alice_fork_block,
    alice_second_fork_block,
]
assert node.chain == expected

#################################
# Alice attempts a double-spend #
#################################

print("Alice's double-spend:")

# Collect initial data
alice_starting_balance = node.fetch_balance(ids.alice_public_key)
starting_chain_height = len(node.chain) - 1

# Attempt the double-spend
coinbase = prepare_coinbase(ids.alice_public_key, height=4)
# `alice_to_bob` has already been mined!
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=alice_second_fork_block.id)
alice_double_spend_block = mine_block(unmined_block)
try:
    node.handle_block(alice_double_spend_block)
except:
    print("error raised attempting double spend")

# Collect final data
alice_ending_balance = node.fetch_balance(ids.alice_public_key)
ending_chain_height = len(node.chain) - 1

# Assert that the block wasn't accepted, Alice's balance didn't change
assert alice_starting_balance == alice_ending_balance
assert starting_chain_height == ending_chain_height

####################
# Test the mempool #
####################

print()
print("Testing mempool")
print()

alice_to_bob = send_tx(node, ids.alice_private_key, ids.bob_public_key, 20)
node.handle_tx(alice_to_bob)
assert alice_to_bob in node.mempool
node.handle_tx(alice_to_bob)
assert alice_to_bob in node.mempool


coinbase = prepare_coinbase(ids.bob_public_key, height=4)
unmined_block = Block(txns=[coinbase, alice_to_bob], 
                      prev_id=alice_second_fork_block.id)
alice_third_block = mine_block(unmined_block)
node.handle_block(alice_third_block)

assert alice_to_bob not in node.mempool


