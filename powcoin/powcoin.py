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
BITS = 15
POW_TARGET = 1 << (256 - BITS)
GET_BLOCKS_CHUNK = 10
BLOCK_SUBSIDY = 50
ACTIVE_CHAIN_INDEX = 0
PORT = 10000
node = None
chain_lock = threading.Lock()

logging.basicConfig(level="DEBUG", format="%(threadName)-12s | %(message)s")
logger = logging.getLogger(__name__)

def print_exc():
    import traceback
    logger.info(traceback.format_exc())


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

def total_work(chain):
    return len(chain)

def tx_in_to_utxo(tx_in, chain):
    for tx, block, height in txn_iterator(chain):
        if tx.id == tx_in.tx_id:
            tx_out = tx.tx_outs[tx_in.index]
            return UnspentTxOut(tx_id=tx_in.tx_id, index=tx_in.index,
                   amount=tx_out.amount, public_key=tx_out.public_key)


########
# Node #
########

class Node:

    def __init__(self, address):
        self.utxo_set = {}
        self.mempool = []
        self.peers = []
        self.address = address
        self.chain = []
        self.branches = []

    def connect(self, peer):
        if peer not in self.peers and peer != self.address:
            send_message(peer, "connect", None)

    def handle_peer(self, peer):
        if peer not in self.peers:
            node.peers.append(peer)

    def initial_block_download(self):
        for peer in self.peers:
            block_ids = [block.id for block in self.chain][-50:]  # HACK
            send_message(peer, "get_blocks", block_ids)

    def add_to_utxo_set(self, tx):
        # Remove utxos that were just spent
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]
        # Save utxos which were just created
        for index, tx_out in enumerate(tx.tx_outs):
            utxo = UnspentTxOut(tx_id=tx.id, index=index, 
                amount=tx_out.amount, public_key=tx_out.public_key)
            self.utxo_set[utxo.outpoint] = utxo

        # Remove from mempool
        if tx in self.mempool:
            self.mempool.remove(tx)
            logging.info(f"Removed tx from mempool")

    def remove_from_utxo_set(self, tx):
        # tx.tx_ins put back in self.utxo_set
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                utxo = tx_in_to_utxo(tx_in, self.chain)
                self.utxo_set[utxo.outpoint] = utxo

        # tx.tx_outs removed from utxo_set
        for index in range(len(tx.tx_outs)):
            outpoint = (tx.id, index)
            del self.utxo_set[outpoint]

        # Put it back in mempool
        if tx not in self.mempool and not tx.is_coinbase:
            self.mempool.append(tx)
            logging.info(f"Added tx to mempool")

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

        # Sum up the total outpouts
        for tx_out in tx.tx_outs:
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
            print_exc()
            return

        # Add to our mempool if it passes validation and isn't already there
        if tx not in self.mempool:
            self.mempool.append(tx)
            logger.info("Added tx to mempool")

            # Tell peers
            for peer in self.peers:
                send_message(peer, "tx", tx)

    def locate_block(self, block_id):
        # chain_index of 0: on self.chain. Greater means it's a branch
        chains = [self.chain] + self.branches
        for chain_index, chain in enumerate(chains):
            for height, block in enumerate(chain):
                if block.id == block_id:
                    return chain, chain_index, height
        return None, None, None

    def create_branch(self):
        self.branches.append([])
        chain = self.branches[-1]
        chain_index = len(self.branches)
        height = 0
        return chain, chain_index, height

    def validate_block(self, block):
        # Check POW
        assert int(block.id, 16) < POW_TARGET, "Insufficient Proof-of-Work"

        # Check txns if we're extending main chain
        if block.prev_id == self.chain[-1].id:
            self.validate_coinbase(block.txns[0])
            for tx in block.txns[1:]:
                self.validate_tx(tx)

    def handle_block(self, block):
        # Claim the lock
        with chain_lock:
            # Ignore if we've already seen it
            if self.locate_block(block.id)[0]:
                raise Exception("Duplicate block")

            # Add block to chain / branch. Updates utxos if chain.
            self.connect_block(block)

            # Attempt a reorg
            self.attempt_reorg()

            # Tell peers
            for peer in self.peers:
                disrupt(send_message, [peer, "blocks", [block]])

            # FIXME
            logging.info(f"Block accepted: height={len(self.chain) - 1} txns={len(block.txns)}")

    def attempt_reorg(self):
        for branch_index, branch in enumerate(self.branches):
            # Compare branch with self.chain since fork block
            _, _, fork_height = self.locate_block(branch[0].prev_id)
            chain_since_fork = self.chain[fork_height+1:]
            if total_work(branch) > total_work(chain_since_fork):
                logger.info(f'Attempting reorg')
                self.reorg(branch, branch_index, fork_height)

    def reorg(self, branch, branch_index, fork_index):
        fork_block = self.chain[fork_index]

        # Disconnect old blocks 
        disconnected_blocks = []
        while self.chain[-1].id != fork_block.id:
            block = self.disconnect_block()
            disconnected_blocks.insert(0, block)  # Prepend to preserve order

        # Replace branch with newly disconnected blocks
        self.branches[branch_index] = disconnected_blocks

        # Connect new blocks
        for block in branch:
            try:
                # This will now validate txns against utxo_set
                # Could fail now even if it previously passed limited validation
                self.connect_block(block)
            except:
                # If we can't connect block, undo all reorg changes
                self.reorg(disconnected_blocks, branch_index, fork_index)
                logger.info(f"Reorg failed")

    def connect_block(self, block):
        # Find the parent
        chain, chain_index, height = self.locate_block(block.prev_id)

        # If we don't know the parent, re-enter IBD to search for it
        if chain is None:
            logger.info("DOWNLOADING MISSING BLOCKS")
            self.initial_block_download()
            raise Exception("Can't connect block. Searching for parent.")

        # Validate the block
        self.validate_block(block)

        # If previous block isn't the end of it's chain, create a new one
        # (branches of branches not implemented)
        if height != len(chain) - 1:
            chain, chain_index, height = self.create_branch()
            logger.info(f"Creating branch #{len(self.branches)}")

        # Update utxo set if we're appending to main chain
        if block.prev_id == self.chain[-1].id:
            for tx in block.txns:
                self.add_to_utxo_set(tx)

        # Add block to chain
        chain.append(block)
        logger.info(f"Extending chain #{chain_index}")

    def disconnect_block(self):
        for tx in self.chain[-1].txns:
            self.remove_from_utxo_set(tx)
        return self.chain.pop()

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
            coinbase = prepare_coinbase(public_key, len(node.chain) - 1)
            logging.info(f"Top of mining loop. Mempool contains {len(node.mempool)} txns")
            # logging.info([coinbase] + deepcopy(node.mempool))
            unmined_block = Block(
                txns=[coinbase] + deepcopy(node.mempool),
                prev_id=node.chain[-1].id,
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

class TCPHandler(socketserver.BaseRequestHandler):

    def get_canonical_peer_address(self):
        ip = self.client_address[0]
        try:
            hostname = socket.gethostbyaddr(ip)
            hostname = re.search(r"_(.*?)_", hostname[0]).group(1)
        except:
            hostname = ip
        return (hostname, PORT)

    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(response)

    def handle(self):
        peer = self.get_canonical_peer_address()
        message = read_message(self.request)
        command = message["command"]
        data = message["data"]

        if command == "connect":
            node.handle_peer(peer)
            logger.info(f'Connected to {peer[0]}"')
            send_message(peer, "connect-response", None)

        if command == "connect-response":
            node.handle_peer(peer)
            logger.info(f'Connected to {peer[0]}"')

            # Request their peers
            send_message(peer, "peers", None)

        # assert peer in node.peers, \
                # "rejecting message from unconnected peer"

        if command == "peers":
            send_message(peer, "peers-response", node.peers)

        if command == "peers-response":
            for peer in data:
                try:
                    node.connect(peer)
                    logger.info(f'Connected to "{peer[0]}"')
                except:
                    logger.info(f'Node "{peer[0]}" offline')

        if command == "ping":
            self.respond(command="pong", data="")

        if command == "blocks":
            for block in data:
                try:
                    node.handle_block(block)
                    mining_interrupt.set()
                except:
                    # logger.info("Rejected block block")
                    pass

            # If syncing, request next block
            if len(data) == GET_BLOCKS_CHUNK:
                node.initial_block_download()

        if command == "tx":
            node.handle_tx(data)
    
        if command == "balance":
            balance = node.fetch_balance(data)
            logger.info("\n\nbalance req\n\n")
            self.respond(command="balance-response", data=balance)

        if command == "utxos":
            utxos = node.fetch_utxos(data)
            self.respond(command="utxos-response", data=utxos)

        if command == "get_blocks":
            # Find our most recent block they don't know about
            peer_block_ids = data
            for block in node.chain[::-1]:
                if block.id not in peer_block_ids \
                        and block.prev_id in peer_block_ids:
                    height = node.chain.index(block)
                    blocks = node.chain[height:height+GET_BLOCKS_CHUNK]
                    send_message(peer, command="blocks", data=blocks)
                    return

            logger.info("couldn't serve get_blocks request")
            send_message(peer, command="blocks", data=[])

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ('127.0.0.1', port)

def serve():
    # server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    server = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def read_message(s):
    data = b''
    # Our protocol is: first 4 bytes signify msg length.
    raw_msg_len = s.recv(4) or b"\x00"
    msg_len = int.from_bytes(raw_msg_len, 'big')

    while msg_len > 0:
        tdat = s.recv(1024)
        data += tdat
        msg_len -= len(tdat)

    return deserialize(data)

def prepare_message(command, data):
    d = {
        "command": command,
        "data": data,
    }
    ser = serialize(d)
    length = len(ser).to_bytes(4, "big")
    return length + ser


def disrupt(func, args):
    # Adds latency. Drops packets.
    if random.randint(1, 10) != 5:
        threading.Timer(random.random(), func, args).start()


def send_message(address, command, data, response=False, retries=3):
    if retries == 0:
        raise Exception("connection error")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(address)
            message = prepare_message(command, data)
            s.sendall(message)
            if response:
                return read_message(s)
        except:
            logger.info("retrying")
            # time.sleep(.1)
            return send_message(address, command, data, response, 
                         retries=retries-1)
            

#######
# CLI #
#######

def main(args):
    if args["serve"]:
        threading.current_thread().name = "main"

        node_id = int(os.environ["ID"])

        duration = 30 * node_id
        logger.info(f"sleeping {duration}")
        time.sleep(duration)
        logger.info("waking up")

        # Set up the node (for convience, alice get coinbase coins)
        global node
        address = (f"node{node_id}", PORT)
        node = Node(address=address)

        # Insert coinbase
        # FIXME: this is a mess
        genesis_coinbase = prepare_coinbase(public_key=user_public_key("alice"), height=0)
        unmined_genesis_block = Block(txns=[genesis_coinbase], prev_id=None)
        mined_genesis_block = mine_block(unmined_genesis_block, step=1)
        node.chain.append(mined_genesis_block)
        node.add_to_utxo_set(genesis_coinbase)

        # First thing, start server in another thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        # Join the network
        peers = [(p, PORT) for p in os.environ['PEERS'].split(',')]
        for peer in peers:
            try:
                node.connect(peer)
                logger.info(f'Connected to "{peer[0]}"')
            except:
                logger.info(f'Node "{peer[0]} offline"')

        # Do initial block download
        logger.info("starting ibd")
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
