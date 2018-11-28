from copy import deepcopy
import pytest
import powcoin as p
import identities as ids

###########
# Helpers #
###########

# Set difficuly very low
p.POW_TARGET = 2** (256 - 2)

def send_tx(node, sender_private_key, recipient_public_key, amount):
    utxos = node.fetch_utxos(sender_private_key.get_verifying_key())
    return p.prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

def mine_block(node, miner_public_key, prev_block, mempool, height, nonce=0):
    coinbase = p.prepare_coinbase(miner_public_key, height)
    unmined_block = p.Block(
        txns=[coinbase] + deepcopy(mempool),
        prev_id=prev_block.id,
        nonce=nonce,
    )
    mined_block = p.mine_block(unmined_block)
    node.handle_block(mined_block)
    return mined_block

def new_node():
    node = p.Node(address="")

    # Bob mines height=0
    p.mine_genesis_block(node, ids.bob_public_key)

    # Alice mines height=1
    mine_block(node, ids.alice_public_key, node.chain[-1], [], 1)
    
    bob_balance = node.fetch_balance(ids.bob_public_key)
    alice_balance = node.fetch_balance(ids.alice_public_key)
    print(bob_balance, alice_balance)

    # Bob mines height=2,3,4 including one bob-to-alice txn
    mine_block(node, ids.bob_public_key, node.chain[-1], [], 2)
    bob_to_alice = send_tx(node, ids.bob_private_key, 
                           ids.alice_public_key, 10)
    mine_block(node, ids.bob_public_key, node.chain[-1], [bob_to_alice], 3)
    mine_block(node, ids.bob_public_key, node.chain[-1], [], 4)

    # Alice forks off 2 blocks at height=1 including one alice-to-bob txn
    mine_block(node, ids.alice_public_key, node.chain[1], [], 2)
    alice_to_bob = send_tx(node, ids.alice_private_key, 
                           ids.bob_public_key, 10)
    mine_block(node, ids.alice_public_key, node.branches[0][-1], 
               [alice_to_bob], 3)

    return node


def make_node():
    node = p.Node(address="")

    # Bob mines height=0
    p.mine_genesis_block(node, ids.bob_public_key)

    return node

#########
# Tests #
#########

def test_extend_chain():
    node = p.Node(address="")

    # Bob mines height=0,1
    p.mine_genesis_block(node, ids.bob_public_key)
    block = mine_block(node, ids.bob_public_key, node.chain[0], [], 1)
    
    # Alice's balance unchanged, Bob received block subsidy
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 2*p.BLOCK_SUBSIDY

    # Chain extended
    assert len(node.chain) == 2
    assert node.chain[-1] == block

    # Branches empty
    assert node.branches == []

def test_fork_chain():
    node = make_node()
    node = p.Node(address="")

    # Bob mines height=0,1
    p.mine_genesis_block(node, ids.bob_public_key)
    bob_block = mine_block(node, ids.bob_public_key, node.chain[0], [], 1)

    # Alice mines height=1 too
    alice_block = mine_block(node, ids.alice_public_key, node.chain[0], [], 1)

    # UTXO database unchanged
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 2*p.BLOCK_SUBSIDY

    # Chain unchanged
    assert len(node.chain) == 2
    assert alice_block not in node.chain

    # One more chain with one block on it
    assert len(node.branches) == 1
    assert node.branches[0] == [alice_block]

def test_block_extending_fork():
    node = p.Node(address="")

    # Bob mines height=0,1,2
    p.mine_genesis_block(node, ids.bob_public_key)
    mine_block(node, ids.bob_public_key, node.chain[0], [], 1)
    bob_block = mine_block(node, ids.bob_public_key, node.chain[1], [], 2)

    # Alice mines height=1
    alice_block = mine_block(node, ids.alice_public_key, node.chain[0], [], 1)
    # Alice mines block on top of her branch
    alice_block = mine_block(node, ids.alice_public_key, node.branches[0][0], [], 2)

    # UTXOs
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 3*p.BLOCK_SUBSIDY

    # Now new branches
    assert len(node.chain) == 3
    assert len(node.branches) == 1
    assert len(node.branches[0]) == 2

def test_block_forking_fork():
    node = p.Node(address="")

    # Bob mines height=0,1,2
    p.mine_genesis_block(node, ids.bob_public_key)
    mine_block(node, ids.bob_public_key, node.chain[0], [], 1)
    bob_block = mine_block(node, ids.bob_public_key, node.chain[1], [], 2)

    # Alice mines height=1
    first = mine_block(node, ids.alice_public_key, node.chain[0], [], 1)

    # Alice mines 2 separate blocks top of her branch, each at height 2
    second = mine_block(node, ids.alice_public_key, node.branches[0][0], [], 2)
    third = mine_block(node, ids.alice_public_key, node.branches[0][0], [], 
                       2, nonce=second.nonce+1)
     
    # UTXOs and chain unaffected
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 3*p.BLOCK_SUBSIDY

    # One more branch added, which contains alice's first block and this one
    assert len(node.chain) == 3
    assert len(node.branches) == 2
    assert node.branches[0] == [first, second]
    assert node.branches[1] == [first, third]

def test_successful_reorg():
    node = p.Node(address="")
    alice_node = p.Node(address="")

    # Bob mines height=0,1,2
    b0 = p.mine_genesis_block(node, ids.bob_public_key)
    b1 = mine_block(node, ids.bob_public_key, node.chain[0], [], 1)
    # height=2 contains a bob->alice txn
    bob_to_alice = send_tx(node, ids.bob_private_key, 
                           ids.alice_public_key, 10)
    b2 = mine_block(node, ids.bob_public_key, node.chain[1], 
                    [bob_to_alice], 2)

    # Alice accepts bob's first two blocks, but not the third
    p.mine_genesis_block(alice_node, ids.bob_public_key) # FIXME confusing
    alice_node.handle_block(b1)

    # FIXME just borrow everything up until this point from another test
    # Create and handle two blocks atop Alice's chain
    a2 = mine_block(alice_node, ids.alice_public_key, node.chain[1], [], 1)
    node.handle_block(a2)

    # Chains
    assert len(node.chain) == 3
    assert node.chain == [b0, b1, b2]
    assert len(node.branches) == 1
    assert node.branches[0] == [a2]

    # Balances
    assert (bob_to_alice.id, 0) in node.utxo_set
    assert (bob_to_alice.id, 1) in node.utxo_set
    assert node.fetch_balance(ids.alice_public_key) == 10
    assert node.fetch_balance(ids.bob_public_key) == 3*p.BLOCK_SUBSIDY - 10

    # Use alice's node to assemble this txn b/c she doesn't have any utxos in bob's view of world
    alice_to_bob = send_tx(alice_node, ids.alice_private_key, 
                           ids.bob_public_key, 20)
    a3 = mine_block(node, ids.alice_public_key, node.branches[0][0], 
                    [alice_to_bob], 2)


    # Chains
    assert len(node.chain) == 4
    assert node.chain == [b0, b1, a2, a3]
    assert len(node.branches) == 1
    assert node.branches[0] == [b2]

    # Balances
    assert (bob_to_alice.id, 0) not in node.utxo_set
    assert (bob_to_alice.id, 1) not in node.utxo_set
    assert (alice_to_bob.id, 0) in node.utxo_set
    assert (alice_to_bob.id, 1) in node.utxo_set
    assert node.fetch_balance(ids.alice_public_key) == 2*p.BLOCK_SUBSIDY - 20
    assert node.fetch_balance(ids.bob_public_key) == 2*p.BLOCK_SUBSIDY + 20

    # Mempool
    assert len(node.mempool) == 1
    assert bob_to_alice in node.mempool

def test_unsuccessful_reorg():
    node = p.Node(address="")
    alice_node = p.Node(address="")

    # Bob mines height=0,1,2
    b0 = p.mine_genesis_block(node, ids.bob_public_key)
    b1 = mine_block(node, ids.bob_public_key, node.chain[0], [], 1)
    b2 = mine_block(node, ids.bob_public_key, node.chain[1], [], 2)

    # Alice accepts bob's first two blocks, but not the third
    p.mine_genesis_block(alice_node, ids.bob_public_key) # FIXME confusing
    alice_node.handle_block(b1)

    # FIXME just borrow everything up until this point from another test
    # Create one valid block for Alice
    a2 = mine_block(alice_node, ids.alice_public_key, node.chain[1], [], 1)
    node.handle_block(a2)

    # Create one invalid block for Alice
    alice_to_bob = send_tx(alice_node, ids.alice_private_key, 
                           ids.bob_public_key, 20)
    # txn invalid b/c changing amount arbitrarily after signing ...
    alice_to_bob.tx_outs[0].amount = 100000

    initial_utxo_set = deepcopy(node.utxo_set)
    initial_chain = deepcopy(node.chain)
    initial_branches = deepcopy(node.branches)

    # This block shouldn't make it into branches or chain
    a3 = mine_block(node, ids.alice_public_key, node.branches[0][0], 
                    [alice_to_bob], 2)

    # UTXO, chain, branches unchanged
    assert str(node.utxo_set) == str(initial_utxo_set) # FIXME
    assert node.chain == initial_chain
    assert node.branches == initial_branches
