from copy import deepcopy
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

def mine_block(node, miner_public_key, prev_block, mempool, height):
    coinbase = p.prepare_coinbase(miner_public_key, height)
    unmined_block = p.Block(
        txns=[coinbase] + deepcopy(mempool),
        prev_id=prev_block.id,
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
    node = make_node()

    # Bob mines height=1
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

    # Bob mines height=1
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
    # Mine one block atop Alice's last block

    # UTXOs and chain unaffected

    # Now new branches'

    # Alice's branch extended by one

    pass

def test_block_forking_fork():
    # Mine block atop Alice's first fork block
     
    # UTXOs and chain unaffected

    # One more branch added, which contains alice's first block and this one

    pass

def test_successful_reorg():
    # Create and handle two block atop Alice's chain

    # Check that utxo from bob's blocks 2,3,4 get removed

    # Check that utxo from alice's blocks 2,3,4,5 get added

    # Check that txs from blocks 2,3,4 get added back to mempool

    # Check that coinbases don't get added back to mempool

    # Check that bobs 2,3,4 are now preserved as a branch

    # Check that old branch was removed

    pass

def test_unsuccessful_reorg():
    # Create one valid block for Alice

    # Create one invalid block for Alice

    # Exception raised

    # UTXO, chain, branches unchanged

    pass
