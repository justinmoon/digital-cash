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

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, hashlib, random, re

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
BITS = 16
POW_TARGET = 1 << (256 - BITS)
BLOCK_SUBSIDY = 50
PORT = 10000
node = None
chain_lock = threading.Lock()

# logging.basicConfig(level="INFO", format="%(asctime)-15s %(levelname)s %(message)s")
# logging.basicConfig(level="INFO", format="%(message)s")
logging.basicConfig(level="DEBUG", format="%(threadName)-6s | %(message)s")
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
        # FIXME
        return mining_hash(f"Tx(tx_ins={self.tx_ins}, tx_outs={self.tx_outs})")

    def __repr__(self):
        return f"Tx(id={self.id}, tx_ins={self.tx_ins}, tx_outs={self.tx_outs})"

    def __eq__(self, other):
        return self.id == other.id

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
    # FIXME bad name
    # FIXME can we just use locate_block?
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

    def __init__(self, node_id=None):
        self.active_chain_index = 0
        self.chains = []
        self.utxo_set = {}
        self.mempool = []
        self.peers = set()
        self.address = (f"node{node_id}", PORT)
        self.syncing = False

    def join_network(self, peers):
        for peer in peers:
            response = send_message(peer, "join", self.address, response=True)

            # Let's say a "None" response turns us down
            if response is not None:
                self.peers.add(peer)

    @property
    def active_chain(self):
        return self.chains[self.active_chain_index]

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
                   f"{tx_in.outpoint} not in utxo_set"

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
            logger.info(f"rejecting invalid tx: {tx}")
            import traceback
            logger.info(traceback.format_exc())
            return

        # Add to our mempool if it passes validation and isn't already there
        if tx not in self.mempool:
            self.mempool.append(tx)
            logger.info("Added tx to mempool")

            # Tell peers
            for peer in self.peers:
                send_message(peer, "tx", tx)

    def locate_block(self, block_id):
        for chain_index, chain in enumerate(self.work_ordered_chains()):
            for height, block in enumerate(chain):
                if block.id == block_id:
                    is_tip = height == len(chain) - 1
                    return chain, chain_index, height, is_tip
        return None, None, None, None

    def work_ordered_chains(self):
        def key(chain):
             is_active = self.chains.index(chain) == self.active_chain_index
             return len(chain) + int(is_active)
        return sorted(self.chains, key=key, reverse=True)

    def sync_utxo_set(self, chain):

        if self.active_chain_index != self.chains.index(chain):
            logger.info(f"ACTIVE BRANCH CHANGE: {self.active_chain_index} -> {self.chains.index(chain)}")

        # FIXME have to treat active chain separately 
        # since it changes under our feet
        if self.chains.index(chain) == self.active_chain_index:
            sync_blocks = self.active_chain[-1:]
            rollback_blocks = []
        else:
            rollback_blocks, sync_blocks = self.chain_diffs(chain)

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
            if tx not in self.mempool and not tx.is_coinbase:
                try:
                    self.validate_tx(tx)
                    self.mempool.append(tx)
                except:
                    # Kinda hacky, but this will reject coinbase txns
                    logger.info("couldn't add back to mempool")
                    continue

        # Remove freshly synced transactions from mempool
        for tx in sync_txns:
            if tx in self.mempool:
                self.mempool.remove(tx)
                logging.info(f"Removed tx from mempool")

        # Sanity check
        prev_id = self.active_chain[0].id
        for block in self.active_chain[1:]:
            assert block.prev_id == prev_id
            prev_id = block.id

        # If everything worked update the "active chain"
        self.active_chain_index = self.chains.index(chain)

    def validate_block(self, block):
        # Check POW
        assert int(block.id, 16) < POW_TARGET, "Insufficient Proof-of-Work"

    def chain_diffs(self, to_chain):
        """Calculate blocks unique to each chain"""
        fork_height = get_last_shared_block(self.active_chain, to_chain)
        rollback_blocks = self.active_chain[fork_height+1:]
        sync_blocks = to_chain[fork_height+1:]
        return rollback_blocks, sync_blocks

    def create_branch(self, chain_index, height, base_chain):
        # +1 b/c we want to include this block
        base_chain = base_chain[:height+1]  
        assert height+1 == len(base_chain), f"{height+1} | {len(base_chain)}"
        self.chains.append(base_chain)
        new_chain_index = len(self.chains) - 1
        logging.info(f"CREATED FORK (index={new_chain_index})")
        new_chain = self.chains[new_chain_index]
        return new_chain, new_chain_index

    def handle_block(self, block):
        # Claim the lock
        with chain_lock:
            # see if it's new
            chain, _, _, _ = self.locate_block(block.id)
            if chain:
                # already know about it
                raise Exception("already seen it")

            # Validate the block
            self.validate_block(block)

            # If this is a new fork, we need to create a new chain
            chain, chain_index, height, is_tip = self.locate_block(block.prev_id)

            # FIXME: what to do if chain_index / height come back None???
            # (orphan blocks ...)
            # e.g. while doing ibd you get the tip of the real chain ...
            # this causes an exception right now ...
            if not is_tip:
                chain, chain_index = self.create_branch(chain_index, height, chain)

            # Add to the chain
            chain.append(block)

            # Resync the UTXO database if the "work record" was broken
            # Or if we're extending the active chain
            if total_work(chain) > total_work(self.active_chain) or \
                    self.chains.index(chain) == self.active_chain_index:
                try:
                    self.sync_utxo_set(chain)
                except:
                    import traceback
                    logger.info(traceback.format_exc())
                    return

            # Tell peers
            # time.sleep(random.random())
            for peer in self.peers:
                send_message(peer, "block", block)

            # FIXME
            logging.info(f"Block accepted: chain={self.active_chain_index} height={len(self.active_chain) - 1} txns={len(block.txns)}")

            # Sanity checks
            assert len(self.active_chain) == \
                   max([len(chain) for chain in self.chains])

            for chain in self.chains:
                assert len(set([block.id for block in chain])) == len(chain)

    def initial_block_download(self):
        # just talk to one peer for now
        # FIXME
        if len(self.peers):
            peer = next(iter(self.peers))
            send_message(peer, "get_blocks", self.active_chain[-1].id)


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


def mine_block(block, step=3):
    nonce = int(os.environ["ID"])
    # FIXME: make this line more readable
    while int(mining_hash(block.header(nonce)), 16) >= POW_TARGET:
        nonce += step  # Hack to make mining more competitive
        if mining_interrupt.is_set():
            logger.info("Mining interrupted")
            mining_interrupt.clear()
            return
    block.nonce = nonce
    return block


def mine_forever(public_key):
    logging.info("Starting miner")
    while True:
        with chain_lock:
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
            logger.info("")
            logger.info(f"Mined a block w/ txns")
            # FIXME
            try:
                node.handle_block(mined_block)
            except:
                import traceback
                logger.info(traceback.format_exc())


##############
# Networking #
##############

def prepare_message(command, data):
    return {
        "command": command,
        "data": data,
    }

class TCPHandler(socketserver.BaseRequestHandler):

    def peer(self):
        address = self.client_address[0]
        raw_host_info = socket.gethostbyaddr(address)
        hostname = re.search(r"_(.*?)_", raw_host_info[0]).group(1)
        return (hostname, PORT)

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
            if data == None:
                logger.info(f"Initial block download complete")
                node.syncing = False
                return

            logging.info(f"Received block from peer")

            try:
                node.handle_block(data)
                mining_interrupt.set()
            except:
                pass

            # If syncing, request next block
            if node.syncing:
                node.initial_block_download()

        if command == "tx":
            node.handle_tx(data)
    
        if command == "balance":
            balance = node.fetch_balance(data)
            self.respond(command="balance-response", data=balance)

        if command == "utxos":
            utxos = node.fetch_utxos(data)
            self.respond(command="utxos-response", data=utxos)

        if command == "join":
            node.peers.add(data)
            self.respond(command="peers", data=node.peers)
            logger.info("received join msg")

        if command == "peers":
            logger.info("received peer list")

        if command == "get_blocks":
            next_block = None
            # locate the block in the main chain
            # FIXME: this should call a general-purpose function
            for block in node.active_chain:
                if block.prev_id == data:
                    next_block = block
                    break

            # Says the IBD is done
            send_message(self.peer(), command="block", data=next_block)
            logger.info(f"sent 'block' message: {next_block}")


def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ('localhost', port)

def serve():
    server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def send_message(address, command, data, response=False, retries=3):
    if retries == 0:
        raise Exception("connection error")
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(address)
            s.sendall(serialize(message))
            if response:
                return deserialize(s.recv(5000))
        except:
            logger.info("retrying")
            time.sleep(.1)
            return send_message(address, command, data, response, 
                         retries=retries-1)
            

#######
# CLI #
#######

def main(args):
    if args["serve"]:
        threading.current_thread().name = "main"

        # FIXME: needs coinbase
        # genesis_block = Block(
            # txns=[],
            # prev_id=None,
            # nonce=0,
        # )
        # node.chains.append([genesis_block])
        # node.active_chain_index = 0

        node_id = int(os.environ["ID"])

        duration = 5 * node_id
        logger.info(f"sleeping {duration}")
        time.sleep(duration)
        logger.info("waking up")


        # Set up the node (for convience, alice get coinbase coins)
        global node
        node = Node(node_id=node_id)

        # Insert coinbase
        # FIXME: this is a mess
        genesis_coinbase = prepare_coinbase(public_key=user_public_key("alice"), height=0)
        unmined_genesis_block = Block(txns=[genesis_coinbase], prev_id=None)
        mined_genesis_block = mine_block(unmined_genesis_block, step=1)
        node.chains.append([mined_genesis_block])
        node.active_chain_index = 0
        node.add_tx_to_utxo_set(genesis_coinbase)

        # First thing, start server in another thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        # Join the network
        peers = {(p, PORT) for p in os.environ['PEERS'].split(',')}
        # first one will fail b/c no peers yet. 
        try:
            node.join_network(peers)
        except:
            pass

        # Do initial block download
        logger.info("starting ibd")
        node.syncing = True
        node.initial_block_download()

        # Run the miner in a thread
        logger.info("starting miner")
        node_id = int(os.environ["ID"])
        mining_public_key = node_public_key(node_id)
        miner_thread = threading.Thread(target=mine_forever, args=(mining_public_key,), name="miner")
        miner_thread.start()

    elif args["ping"]:
        address = address_from_host(args["--node"])
        send_message(address, "ping", "")
    elif args["balance"]:
        public_key = user_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "balance", public_key, response=True)
        logger.info(response["data"])


    
    elif args["utxos"]:
        public_key = user_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "utxos", public_key, response=True)
        logger.info(response["data"])

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
        logger.info(tx)
    else:
        logger.info("Invalid command")


if __name__ == '__main__':
    main(docopt(__doc__))
