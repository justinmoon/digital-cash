"""
BlockCoin

Usage:
  blockcoin.py serve
  blockcoin.py ping [--node <node>]
  blockcoin.py tx <from> <to> <amount> [--node <node>]
  blockcoin.py balance <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, hashlib

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import user_private_key, user_public_key, key_to_name

bits = 2
target = 1 << (256 - bits)


BLOCK_SUBSIDY = 50
NUM_BANKS = 3
BLOCK_TIME = 5   # in seconds
PORT = 10000
node = None

logging.basicConfig(
    level="INFO",
    format='%(asctime)-15s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


###########
# Classes #
###########

def spend_message(tx, index):
    outpoint = tx.tx_ins[index].outpoint
    return serialize(outpoint) + serialize(tx.tx_outs)

class Tx:

    def __init__(self, tx_ins, tx_outs):
        # FIXME: generate this by hashing the tx
        self.tx_ins = tx_ins
        self.tx_outs = tx_outs

    def sign_input(self, index, private_key):
        message = spend_message(self, index)
        signature = private_key.sign(message)
        self.tx_ins[index].signature = signature

    def verify_input(self, index, public_key):
        tx_in = self.tx_ins[index]
        message = spend_message(self, index)
        return public_key.verify(tx_in.signature, message)

    @property
    def is_coinbase(self):
        # return self.tx_ins[0].outpoint == (None, None)
        return isinstance(self.tx_ins[0].signature, int)

    @property
    def id(self):
        # FIXME repr
        return mining_hash(f"tx_ins={self.tx_ins}, tx_outs={self.tx_outs})")

    def __repr__(self):
        return f"Tx(id={self.id}, tx_ins={self.tx_ins}, tx_outs={self.tx_outs})"

class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def outpoint(self):
        return (self.tx_id, self.index)

    def __repr__(self):
        signature = self.signature if isinstance(self.signature, int) else "..."
        return f"TxIn(tx_id={self.tx_id}, index={self.index} {signature})"

class TxOut:

    def __init__(self, amount, public_key):
        self.amount = amount
        self.public_key = public_key

    def __repr__(self):
        return f"TxOut(amount={self.amount}, public_key={key_to_name(self.public_key)})"

class UnspentTxOut:

    def __init__(self, tx_id, index, amount, public_key):
        self.tx_id = tx_id
        self.index = index
        self.amount = amount
        self.public_key = public_key

    @property
    def outpoint(self):
        return (self.tx_id, self.index)

class Block:

    def __init__(self, txns, prev_id, nonce=0):
        self.txns = txns
        self.prev_id = prev_id
        self.nonce = nonce

    @property
    def id(self):
        return mining_hash(self.header(self.nonce))

    def header(self, nonce):
        return serialize([self.txns, nonce])

    def __repr__(self):
        return f"Block(prev_id={self.prev_id}, id={self.id} nonce={self.nonce})"

class Chain(list):

    # def __init__(self, blocks):
        # self.blocks = blocks

    @property
    def work(self):
        # FIXME
        return len(self)

    @property
    def tip(self):
        return self[-1]

    @property
    def height(self):
        return len(self) - 1

def txn_iterator(chain):
    return (
        (txn, block, height)
        for height, block in enumerate(chain) for txn in block.txns)

def get_last_shared_block(chain_one, chain_two):
    for height, (b1, b2) in enumerate(zip(chain_one, chain_two)):
        if b1.id != b2.id:
            return height - 1
    return min(len(chain_one), len(chain_two)) - 1

def total_work(chain):
    # FIXME
    return len(chain)

def tx_in_to_utxo(tx_in, chain):
    for tx, block, height in txn_iterator(chain):
        if tx.id == tx_in.tx_id:
            tx_out = tx.tx_outs[tx_in.index]
            return UnspentTxOut(tx_id=tx_in.tx_id, index=tx_in.index,
                   amount=tx_out.amount, public_key=tx_out.public_key)

class Node:

    def __init__(self):
        self.active_chain_index = 0
        self.chains = []
        self.utxo_set = {}
        self.mempool = []
        # TODO: just call this peers
        # TODO: add some way to handle "pending peers" who are handshaking
        self.peer_addresses = {(p, PORT) for p in os.environ.get('PEERS', '').split(',') if p}

    @property
    def active_chain(self):
        return self.chains[self.active_chain_index]

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx.tx_ins]

    @property
    def mempool_tx_ids(self):
        return [tx.id for tx in self.mempool]

    def update_utxo_set(self, tx):
        # Remove utxos that were just spent
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]
        # Save utxos which were just created
        for index, tx_out in enumerate(tx.tx_outs):
            utxo = UnspentTxOut(tx_id=tx.id, index=index, 
                amount=tx_out.amount, public_key=tx_out.public_key)
            self.utxo_set[utxo.outpoint] = utxo

    def rollback_utxo_set(self, tx):
        # tx.tx_ins put back in self.utxo_set
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                utxo = tx_in_to_utxo(tx_in, self.active_chain)
                self.utxo_set[utxo.outpoint] = utxo

        # tx.tx_outs removed from utxo_set
        for index in range(len(tx.tx_outs)):
            outpoint = (tx.id, index)
            del self.utxo_set[outpoint]

    def fetch_utxos(self, public_key):
        return [utxo for utxo in self.utxo_set.values() 
                if utxo.public_key == public_key]

    def fetch_balance(self, public_key):
        # Fetch utxos associated with this public key
        utxos = self.fetch_utxos(public_key)
        # Sum the amounts
        return sum([utxo.amount for utxo in utxos])

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0
        for index, tx_in in enumerate(tx.tx_ins):
            # TxIn spending an unspent output
            assert tx_in.outpoint in self.utxo_set, f"{tx_in.outpoint} not in {self.utxo_set}"

            # No pending transactions spending this same output
            assert tx_in.outpoint not in self.mempool_outpoints

            # Grab the tx_out
            tx_out = self.utxo_set[tx_in.outpoint]

            # Verify signature using public key of TxOut we're spending
            public_key = tx_out.public_key
            tx.verify_input(index, public_key)

            # Sum up the total inputs
            amount = tx_out.amount
            in_sum += amount

        for tx_out in tx.tx_outs:
            # Sum up the total outpouts
            out_sum += tx_out.amount

        # Check no value created or destroyed
        assert in_sum == out_sum

    def validate_coinbase(self, tx):
        assert len(tx.tx_ins) == 1
        assert len(tx.tx_outs) == 1
        assert tx.tx_outs[0].amount == BLOCK_SUBSIDY

    def handle_tx(self, tx, as_coinbase=False):
        if as_coinbase:
            self.validate_coinbase(tx)
        else:
            self.validate_tx(tx)
        self.mempool.append(tx)

    def find_block(self, block):
        # FIXME: HACK check the active_chain manually
        if self.active_chain[-1].id == block.prev_id:
            height = len(self.active_chain) - 1
            is_tip = height == len(self.active_chain) - 1
            return self.active_chain, self.active_chain_index, height, is_tip

        # longest chains first
        sorted_chains = sorted(self.chains, key=lambda c: len(c), reverse=True)
        for chain_index, chain in enumerate(sorted_chains):
            for height, _block in enumerate(chain):
                if _block.id == block.prev_id:
                    is_tip = height == len(chain) - 1
                    chain_index = self.chains.index(chain)
                    return chain, chain_index, height, is_tip

    def rollback_utxo(self, block):
        pass

    def sync_utxo(self, block):
        pass

    def validate_block(self, block):
        # Check POW

        # If we're extending the main chain, validate the transactions

        pass

    def chain_diffs(self, from_chain, to_chain):
        fork_height = get_last_shared_block(from_chain, to_chain)
        rollback_blocks = from_chain[fork_height+1:]
        sync_blocks = to_chain[fork_height+1:]
        return rollback_blocks, sync_blocks

    def create_branch(self, chain_index, height):
        # +1 b/c we want to include this block
        base_chain = self.chains[chain_index][:height+1]  
        self.chains.append(base_chain)
        new_chain_index = len(self.chains) - 1
        print(f"CREATED FORK (index={new_chain_index})")
        new_chain = self.chains[new_chain_index]
        return new_chain, new_chain_index

    def handle_block(self, block):

        # Validate the block

        # Create a new chain if needed

        # Save the block

        # Sync utxo -- condition here is if the hash of the tip changed?
        # * we need to get the diff between the old and new main chain

        # Set active_chain_index

        active_chain = deepcopy(self.active_chain)  # FIXME: inefficient

        # If this is a new fork, we need to create a new chain
        chain, chain_index, height, is_tip = self.find_block(block)
        if not is_tip:
            chain, chain_index = self.create_branch(chain_index, height)

        # Add to the chain
        chain.append(block)

        # Resync the UTXO database if the "work record" was broken
        if total_work(chain) > total_work(active_chain):
            print(f"ACTIVE BRANCH CHANGE: {self.active_chain_index} -> {chain_index}")

            rollback_blocks, sync_blocks = self.chain_diffs(active_chain, chain)

            print(f"Rolling back: {rollback_blocks}")
            print(f"Syncing: {sync_blocks}")
            add_to_mempool = []
            remove_from_mempool = []

            # Rollback every transaction in current active_chain but not in the new one
            # No exception handling here b/c failure would mean program is broken
            for block in rollback_blocks[::-1]:
                for tx in block.txns:
                    self.rollback_utxo_set(tx)
                    add_to_mempool.append(tx)
            
            
            # Attempt to update the UTXO set
            # Rollback if there are any problems
            updated_txns = []
            for block in sync_blocks:
                for index, tx in enumerate(block.txns):
                    print("updateing tx", tx)
                    try:
                        # Check the other transactions are valid
                        if index == 0:
                            self.validate_coinbase(tx)
                        else:
                            self.validate_tx(tx)
                        self.update_utxo_set(tx)
                        updated_txns.append(tx)
                        remove_from_mempool.append(tx)
                    except Exception as e:
                        print("PROBLEM ROLLING BACK")
                        import traceback
                        print(traceback.format_exc())
                        # Block is invalid
                        # Rollback all utxo changes
                        for tx in updated_txns:
                            self.rollback_utxo_set(tx)
                        
                        # re-add "rollback_blocks"
                        for block in rollback_blocks:
                            for tx in block.txns:
                                self.update_utxo_set(tx)
                        return

            # Add rolled-back transactions to the mempool
            for tx in add_to_mempool:
                if tx.id not in self.mempool_tx_ids:
                    self.mempool.append(tx)

            # Remove freshly synced transactions from mempool
            for tx in remove_from_mempool:
                if tx.id in self.mempool_tx_ids:
                    self.mempool.append(tx)
            
            # If everything worked update the "active chain"
            self.active_chain_index = chain_index

    def make_block(self):
        # Reset mempool
        txns = deepcopy(self.mempool)
        # FIXME: with powcoin lets allow handle_block manage mempool
        self.mempool = []
        block = Block(txns=txns)
        return block

    def submit_block(self):
        # Make the block
        block = self.make_block()

        # Save locally
        self.handle_block(block)

        # Tell peers
        for address in self.peer_addresses:
            send_message(address, "block", block)

###################
# Tx Construction #
###################

def prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount):
    sender_public_key = sender_private_key.get_verifying_key()

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxos:
        tx_ins.append(TxIn(tx_id=tx_out.tx_id, index=tx_out.index, signature=None))
        tx_in_sum += tx_out.amount
        if tx_in_sum > amount:
            break

    # Make sure sender can afford it
    assert tx_in_sum >= amount

    # Construct tx.tx_outs
    change = tx_in_sum - amount
    tx_outs = [
        TxOut(amount=amount, public_key=recipient_public_key), 
        TxOut(amount=change, public_key=sender_public_key),
    ]

    # Construct tx and sign inputs
    tx = Tx(tx_ins=tx_ins, tx_outs=tx_outs)
    for i in range(len(tx.tx_ins)):
        tx.sign_input(i, sender_private_key)

    return tx

def prepare_coinbase(public_key, height):
    return Tx(
        tx_ins=[
            # we'll use height as the signature so coinbase hashes
            # don't collide
            TxIn(None, None, height)
        ],
        tx_outs=[
            TxOut(amount=BLOCK_SUBSIDY, public_key=public_key), 
        ],
    )

##########
# Mining #
##########

def mining_hash(s):
    if not isinstance(s, bytes):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def mine_block(block):
    nonce = 0
    # FIXME: make this line more readable
    while int(mining_hash(block.header(nonce)), 16) >= target:
        nonce += 1
    block.nonce = nonce
    # print(f"Nonce found {block}")
    return block


def mine_forever():
    while True:
        unmined_block = Block(
            txns=[],
            prev_id=node.active_chain[-1].id,
        )
        mined_block = mine_block(unmined_block)
        
        # This is False if mining was interrupted
        # Perhaps an exception would be wiser ...
        if mined_block:
            node.active_chain.append(mined_block)


# def chain_is_valid():
    # current_block = chain[0]
    # for block in chain[1:]:
        # assert block.prev_id == current_block.id
        # assert int(block.id, 16) < target
        # current_block = block


##############
# Networking #
##############

def prepare_message(command, data):
    return {
        "command": command,
        "data": data,
    }

class TCPHandler(socketserver.BaseRequestHandler):

    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(serialize(response))

    def handle(self):
        message_bytes = self.request.recv(1024*4).strip()
        message = deserialize(message_bytes)
        command = message["command"]
        data = message["data"]

        logger.info(f"received {command}")

        if command == "ping":
            self.respond(command="pong", data="")

        if command == "block":
            node.handle_block(data)

        if command == "tx":
            node.handle_tx(data)

        if command == "balance":
            balance = node.fetch_balance(data)
            self.respond(command="balance-response", data=balance)

        if command == "utxos":
            utxos = node.fetch_utxos(data)
            self.respond(command="utxos-response", data=utxos)

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ('localhost', port)

def serve():
    server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def send_message(address, command, data, response=False):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(serialize(message))
        if response:
            return deserialize(s.recv(5000))

#######
# CLI #
#######

def main(args):
    if args["serve"]:
        global node
        node = Node()
        # FIXME: needs coinbase
        genesis_block = Block(
            txns=[],
            prev_id=None,
            nonce=0,
        )
        node.chains.append([genesis_block])
        node.active_chain_index = 0
        # serve()  # FIXME
        mine_forever()
    elif args["ping"]:
        address = address_from_host(args["--node"])
        send_message(address, "ping", "")
    elif args["balance"]:
        public_key = user_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "balance", public_key, response=True)
        print(response["data"])
    elif args["tx"]:
        # Grab parameters
        sender_private_key = user_private_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = user_private_key(args["<to>"])
        recipient_public_key = recipient_private_key.get_verifying_key()
        amount = int(args["<amount>"])
        address = external_address(args["--node"])

        # Fetch utxos available to spend
        response = send_message(address, "utxos", sender_public_key, response=True)
        utxos = response["data"]

        # Prepare transaction
        tx = prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

        # send to node
        send_message(address, "tx", tx)
    else:
        print("Invalid command")


if __name__ == '__main__':
    main(docopt(__doc__))
