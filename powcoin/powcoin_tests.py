from copy import deepcopy
import pytest
import mypowcoin as p
import identities as ids

###########
# Helpers #
###########

# Set difficuly very low
p.POW_TARGET = 2 ** (256 - 2)

def send_tx(node, sender_private_key, recipient_public_key, amount):
    utxos = node.fetch_utxos(sender_private_key.get_verifying_key())
    return p.prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

def mine_block(node, miner_public_key, prev_block, mempool, nonce=0):
    coinbase = p.prepare_coinbase(miner_public_key)
    unmined_block = p.Block(
        txns=[coinbase] + deepcopy(mempool),
        prev_id=prev_block.id,
        nonce=nonce,
    )
    mined_block = p.mine_block(unmined_block)
    node.handle_block(mined_block)
    return mined_block

#########
# Tests #
#########

def test_duplicate():
    node = p.Node(address="")
    alice_node = p.Node(address="")

    # Bob mines height=0,1
    p.mine_genesis_block(node, ids.bob_public_key)
    p.mine_genesis_block(alice_node, ids.bob_public_key)

    block = mine_block(node, ids.bob_public_key, node.blocks[0], [])

    # Assert handling block already in blocks
    with pytest.raises(Exception):
        node.handle_block(block)

    assert alice_node.blocks[0] == node.blocks[0]
    block = mine_block(alice_node, ids.alice_public_key, node.blocks[0], [])
    node.handle_block(block)  # goes into branches
    assert len(node.branches) == 1

    # Assert handling block already in branches
    with pytest.raises(Exception):
        node.handle_block(block)

def test_extend_chain():
    node = p.Node(address="")

    # Bob mines height=0,1
    p.mine_genesis_block(node, ids.bob_public_key)
    block = mine_block(node, ids.bob_public_key, node.blocks[0], [])
    
    # Alice's balance unchanged, Bob received block subsidy
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 2*p.BLOCK_SUBSIDY

    # Chain extended
    assert len(node.blocks) == 2
    assert node.blocks[-1] == block

    # Branches empty
    assert node.branches == []

def test_fork_chain():
    node = p.Node(address="")

    # Bob mines height=0,1
    p.mine_genesis_block(node, ids.bob_public_key)
    bob_block = mine_block(node, ids.bob_public_key, node.blocks[0], [])

    # Alice mines height=1 too
    alice_block = mine_block(node, ids.alice_public_key, node.blocks[0], [])

    # UTXO database unchanged
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 2*p.BLOCK_SUBSIDY

    # Chain unchanged
    assert len(node.blocks) == 2
    assert alice_block not in node.blocks

    # One more chain with one block on it
    assert len(node.branches) == 1
    assert node.branches[0] == [alice_block]

def test_block_extending_fork():
    node = p.Node(address="")

    # Bob mines height=0,1,2
    p.mine_genesis_block(node, ids.bob_public_key)
    mine_block(node, ids.bob_public_key, node.blocks[0], [])
    bob_block = mine_block(node, ids.bob_public_key, node.blocks[1], [])

    # Alice mines height=1
    alice_block = mine_block(node, ids.alice_public_key, node.blocks[0], [])
    # Alice mines block on top of her branch
    alice_block = mine_block(node, ids.alice_public_key, node.branches[0][0], [])

    # UTXOs
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 3*p.BLOCK_SUBSIDY

    # Now new branches
    assert len(node.blocks) == 3
    assert len(node.branches) == 1
    assert len(node.branches[0]) == 2

def test_block_forking_fork():
    node = p.Node(address="")

    # Bob mines height=0,1,2
    p.mine_genesis_block(node, ids.bob_public_key)
    mine_block(node, ids.bob_public_key, node.blocks[0], [])
    bob_block = mine_block(node, ids.bob_public_key, node.blocks[1], [])

    # Alice mines height=1
    first = mine_block(node, ids.alice_public_key, node.blocks[0], [])

    # Alice mines 2 separate blocks top of her branch, each at height 2
    second = mine_block(node, ids.alice_public_key, node.branches[0][0], [])
    third = mine_block(node, ids.alice_public_key, node.branches[0][0], [], 
                       nonce=second.nonce+1)
     
    # UTXOs and chain unaffected
    assert node.fetch_balance(ids.alice_public_key) == 0 
    assert node.fetch_balance(ids.bob_public_key) == 3*p.BLOCK_SUBSIDY

    # One more branch added, which contains alice's first block and this one
    assert len(node.blocks) == 3
    assert len(node.branches) == 2
    assert node.branches[0] == [first, second]
    assert node.branches[1] == [first, third]

def test_successful_reorg():
    node = p.Node(address="")
    alice_node = p.Node(address="")

    # Bob mines height=0,1,2
    b0 = p.mine_genesis_block(node, ids.bob_public_key)
    b1 = mine_block(node, ids.bob_public_key, node.blocks[0], [])
    # height=2 contains a bob->alice txn
    bob_to_alice = send_tx(node, ids.bob_private_key, 
                           ids.alice_public_key, 10)
    b2 = mine_block(node, ids.bob_public_key, node.blocks[1], 
                    [bob_to_alice])

    # Alice accepts bob's first two blocks, but not the third
    p.mine_genesis_block(alice_node, ids.bob_public_key) # FIXME confusing
    alice_node.handle_block(b1)

    # FIXME just borrow everything up until this point from another test
    # Create and handle two blocks atop Alice's chain
    a2 = mine_block(alice_node, ids.alice_public_key, node.blocks[1], [])
    node.handle_block(a2)

    # Chains
    assert len(node.blocks) == 3
    print([b0, b1, b2])
    assert node.blocks == [b0, b1, b2]
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
                    [alice_to_bob])


    # Chains
    assert len(node.blocks) == 4
    assert node.blocks == [b0, b1, a2, a3]
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
    # FIXME: ideally this would assert that a reorg was attempted ...
    # passes even when reorgs were never tried ...
    node = p.Node(address="")
    alice_node = p.Node(address="")

    # Bob mines height=0,1,2
    b0 = p.mine_genesis_block(node, ids.bob_public_key)
    b1 = mine_block(node, ids.bob_public_key, node.blocks[0], [])
    b2 = mine_block(node, ids.bob_public_key, node.blocks[1], [])

    # Alice accepts bob's first two blocks, but not the third
    p.mine_genesis_block(alice_node, ids.bob_public_key) # FIXME confusing
    alice_node.handle_block(b1)

    # FIXME just borrow everything up until this point from another test
    # Create one valid block for Alice
    a2 = mine_block(alice_node, ids.alice_public_key, node.blocks[1], [])
    node.handle_block(a2)

    # Create one invalid block for Alice
    alice_to_bob = send_tx(alice_node, ids.alice_private_key, 
                           ids.bob_public_key, 20)
    # txn invalid b/c changing amount arbitrarily after signing ...
    alice_to_bob.tx_outs[0].amount = 100000

    initial_utxo_set = deepcopy(node.utxo_set)
    initial_chain = deepcopy(node.blocks)
    initial_branches = deepcopy(node.branches)

    # This block shouldn't make it into branches or chain
    # b/c it contains an invalid transaction that will only be discovered
    # during reorg
    a3 = mine_block(node, ids.alice_public_key, node.branches[0][0], 
                    [alice_to_bob])

    # UTXO, chain, branches unchanged
    assert str(node.utxo_set.keys()) == str(initial_utxo_set.keys()) # FIXME
    assert node.blocks == initial_chain
    assert node.branches == initial_branches
