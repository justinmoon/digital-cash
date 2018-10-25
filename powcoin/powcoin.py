"""
POWCoin

Usage:
  powcoin.py serve
  powcoin.py ping [--node <node>]
  powcoin.py tx <from> <to> <amount> [--node <node>]
  powcoin.py balance <name> [--node <node>]
  powcoin.py utxos <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, hashlib

import pprint

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import user_private_key, user_public_key, key_to_name, node_public_key


# blocks must supply a nonce such that integer interpretation
# of sha256 of serialization of the block is less than POW_TARGET:
# int(mining_hash(serialize(block)), 16) < POW_TARGET
# BITS = 2
BITS = 19
POW_TARGET = 1 << (256 - BITS)
BLOCK_SUBSIDY = 50
PORT = 10000
node = None

# logging.basicConfig(level="INFO", format="%(asctime)-15s %(levelname)s %(message)s")
# logging.basicConfig(level="INFO", format="%(message)s")
logging.basicConfig(level="DEBUG", format="%(message)s")
logger = logging.getLogger(__name__)


###########
# Classes #
###########

def spend_message(tx, index):
    outpoint = tx.tx_ins[index].outpoint
    return serialize(outpoint) + serialize(tx.tx_outs)

class Tx:

    def __init__(self, tx_ins, tx_outs):
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
        return isinstance(self.tx_ins[0].signature, int)

    @property
    def id(self):
        return mining_hash(f"Tx(tx_ins={self.tx_ins}, tx_outs={self.tx_outs})")

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

    def __repr__(self):
        return f"TxOut(tx_id={self.tx_id}, index={self.index} amount={self.amount}, public_key={key_to_name(self.public_key)})"
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

    def __init__(self, peers):
        self.active_chain_index = 0
        self.chains = []
        self.utxo_set = {}
        self.mempool = []
        self.peers = peers
        self.chain_lock = threading.Lock()

    @property
    def active_chain(self):
        return self.chains[self.active_chain_index]

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx.tx_ins]

    @property
    def mempool_tx_ids(self):
        return [tx.id for tx in self.mempool]

    def add_tx_to_utxo_set(self, tx):
        # Remove utxos that were just spent
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]
        # Save utxos which were just created
        for index, tx_out in enumerate(tx.tx_outs):
            utxo = UnspentTxOut(tx_id=tx.id, index=index, 
                amount=tx_out.amount, public_key=tx_out.public_key)
            self.utxo_set[utxo.outpoint] = utxo

    def remove_tx_from_utxo_set(self, tx):
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
            assert tx_in.outpoint in self.utxo_set, \
                   f"{tx_in} not in utxo_set"

            # # No pending transactions spending this same output
            # assert tx_in.outpoint not in self.mempool_outpoints

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

    def handle_tx(self, tx):
        # Validate it against our longest chain
        try:
            self.validate_tx(tx)
        except:
            logger.debug(f"rejecting invalid tx: {tx}")
            import traceback
            logger.info(traceback.format_exc())
            return

        # Add to our mempool if it passes validation and isn't already there
        print( tx.id not in self.mempool_tx_ids)
        if tx.id not in self.mempool_tx_ids:
            self.mempool.append(tx)
            logger.info("ADDED TX TO MEMPOOL")


            # Tell peers
            for peer in self.peers:
                send_message(peer, "tx", tx)

    def find_block(self, block):
        for chain_index, chain in enumerate(self.chains):
            for height, _block in enumerate(chain):
                if block.id == _block.id:
                    return chain_index, height
        return None, None

    def find_prev_block(self, block): # FIXME: HACK check the active_chain manually
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

    def sync_utxo_set(self, chain, active_chain):

        if self.active_chain_index != self.chains.index(chain):
            logger.info(f"ACTIVE BRANCH CHANGE: {self.active_chain_index} -> {self.chains.index(chain)}")

        rollback_blocks, sync_blocks = self.chain_diffs(active_chain, chain)

        # Rollback every transaction in current active_chain but not in the new one
        # No exception handling here b/c failure would mean program is broken
        # Iterate backwards because we're rolling BACK
        rollback_txns = []
        for block in rollback_blocks[::-1]:
            for tx in block.txns:
                self.remove_tx_from_utxo_set(tx)
                rollback_txns.append(tx)
        
        
        # Attempt to update the UTXO set
        sync_txns = []
        for block in sync_blocks:
            for index, tx in enumerate(block.txns):
                try:
                    if index == 0:
                        self.validate_coinbase(tx)
                    else:
                        self.validate_tx(tx)
                    self.add_tx_to_utxo_set(tx)
                    sync_txns.append(tx)
                except Exception as e:
                    import traceback
                    logger.info(traceback.format_exc())

                    # Block is invalid. Revert the entire operation.
                    logger.info("Invalid block")

                    # Reverse the rollbacks
                    for tx in rollback_txns:
                        self.add_tx_to_utxo_set(tx)

                    # Rollback the syncs
                    for tx in sync_txns:
                        self.remove_tx_from_utxo_set(tx)

                    # Remove this and future blocks from this chain
                    chain_index = self.chains.index(chain)
                    block_index = chain.index(block)
                    self.chains[chain_index] = self.chains[chain_index][:block_index]
                    return

        # Add rolled-back transactions to the mempool
        for tx in rollback_txns:
            if tx.id not in self.mempool_tx_ids:
                self.mempool.append(tx)

        # Remove freshly synced transactions from mempool
        for tx in sync_txns:
            if tx.id in self.mempool_tx_ids:
                # FIXME: implement Tx.__eq__ (just compare ids)
                index = self.mempool_tx_ids.index(tx.id)
                self.mempool.pop(index)
                logging.info("\n\nRemoved tx from mempool. Now contains {len(self.mempool)}\n\n")
        
        # If everything worked update the "active chain"
        self.active_chain_index = self.chains.index(chain)

        logging.info(f"Block accepted: index={self.active_chain_index} height={len(self.active_chain) - 1} txns={len(block.txns)}")

    def validate_block(self, block):
        # Check POW
        assert int(block.id, 16) < POW_TARGET, "Insufficient Proof-of-Work"

    def chain_diffs(self, from_chain, to_chain):
        """Calculate blocks unique to each chain"""
        fork_height = get_last_shared_block(from_chain, to_chain)
        rollback_blocks = from_chain[fork_height+1:]
        sync_blocks = to_chain[fork_height+1:]
        return rollback_blocks, sync_blocks

    def create_branch(self, chain_index, height):
        # +1 b/c we want to include this block
        base_chain = self.chains[chain_index][:height+1]  
        self.chains.append(base_chain)
        new_chain_index = len(self.chains) - 1
        logging.info(f"CREATED FORK (index={new_chain_index})")
        new_chain = self.chains[new_chain_index]
        return new_chain, new_chain_index

    def handle_block(self, block):
        # Claim the lock
        with self.chain_lock:

            # Validate the block
            self.validate_block(block)


            # YOLO
            active_chain = deepcopy(self.active_chain)  # FIXME

            # If this is a new fork, we need to create a new chain
            chain, chain_index, height, is_tip = self.find_prev_block(block)
            if not is_tip:
                chain, chain_index = self.create_branch(chain_index, height)

            # Add to the chain
            chain.append(block)

            # Resync the UTXO database if the "work record" was broken
            try:
                if total_work(chain) > total_work(active_chain):
                    self.sync_utxo_set(chain, active_chain)
            except:
                import traceback
                print(traceback.format_exc())

            # Tell peers
            for peer in self.peers:
                send_message(peer, "block", block)

        # FIXME
        logging.info(f"Active chain index is {self.active_chain_index}. Active chain height is {len(self.active_chain) - 1}")

    def submit_block(self):
        # Make the block
        block = self.make_block()

        # Save locally
        self.handle_block(block)


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

mining_interrupt = threading.Event()

def mining_hash(s):
    if not isinstance(s, bytes):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def mine_block(block):
    nonce = 0
    # FIXME: make this line more readable
    while int(mining_hash(block.header(nonce)), 16) >= POW_TARGET:
        nonce += 1
        if mining_interrupt.is_set():
            logger.info("Mining interrupted")
            mining_interrupt.clear()
            return
    block.nonce = nonce
    return block


def mine_forever(public_key):
    logging.info("Starting miner")
    while True:
        coinbase = prepare_coinbase(public_key, len(node.active_chain) - 1)
        logging.info(f"Top of mining loop. Mempool contains {len(node.mempool)} txns")
        # logging.info([coinbase] + deepcopy(node.mempool))
        unmined_block = Block(
            txns=[coinbase] + deepcopy(node.mempool),
            prev_id=node.active_chain[-1].id,
        )
        mined_block = mine_block(unmined_block)
        
        # This is False if mining was interrupted
        # Perhaps an exception would be wiser ...
        if mined_block:
            logging.info(f"Mined a block w/ txns")
            # logging.info(mined_block.txns)
            node.handle_block(mined_block)


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

        if command == "ping":
            self.respond(command="pong", data="")

        if command == "block":
            # If the block isn't new, ignore it
            chain_index, height = node.find_block(data)
            if chain_index == height == None:
                logging.info(f"Received block from peer")

                node.handle_block(data)
                # Tell the mining thread mine the new tip
                logger.info(f"in tcphandler. mempool has {len(node.mempool)}")
                mining_interrupt.set()

            # logging.info(f"Ignoring block: {data}")

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
        # FIXME: needs coinbase
        # genesis_block = Block(
            # txns=[],
            # prev_id=None,
            # nonce=0,
        # )
        # node.chains.append([genesis_block])
        # node.active_chain_index = 0

        # Set up the node (for convience, alice get coinbase coins)
        global node
        peers = {(p, PORT) for p in os.environ['PEERS'].split(',')}
        node = Node(peers)
        genesis_coinbase = prepare_coinbase(public_key=user_public_key("alice"), height=0)
        unmined_genesis_block = Block(txns=[genesis_coinbase], prev_id=None)
        mined_genesis_block = mine_block(unmined_genesis_block)
        node.chains.append([mined_genesis_block])
        node.active_chain_index = 0
        node.add_tx_to_utxo_set(genesis_coinbase)

        # Run the miner in a thread
        node_id = int(os.environ["ID"])
        mining_public_key = node_public_key(node_id)
        thread = threading.Thread(target=mine_forever, args=(mining_public_key,))
        thread.start()

        serve()
    elif args["ping"]:
        address = address_from_host(args["--node"])
        send_message(address, "ping", "")
    elif args["balance"]:
        public_key = user_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "balance", public_key, response=True)
        print(response["data"])

    elif args["utxos"]:
        public_key = user_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "utxos", public_key, response=True)
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
        print(tx)
    else:
        print("Invalid command")


if __name__ == '__main__':
    main(docopt(__doc__))
