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

import socketserver, socket, sys, argparse, time, os, logging, threading, hashlib, random, re, pickle

import pprint

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1



GET_BLOCKS_CHUNK = 50
BLOCK_SUBSIDY = 50
PORT = 10000
DIFFICULTY_BITS = 17
POW_TARGET = 2 ** (256 - DIFFICULTY_BITS)

node = None
mining_interrupt = threading.Event()
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
        return hashlib.sha256(serialize(self)).hexdigest()

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
    def header(self):
        return serialize([
            self.txns,
            self.prev_id,
            self.nonce,
        ])

    @property
    def id(self):
        return hashlib.sha256(self.header).hexdigest()

    @property
    def proof(self):
        return int(self.id, 16)

    def __repr__(self):
        return f"Block(prev_id={self.prev_id[:10] if self.prev_id else self.prev_id}... id={self.id[:10]}...)"

    def __eq__(self, other):
        return self.id == other.id


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
        if peer not in self.peers and peer != self.address:
            node.peers.append(peer)

    def initial_block_download(self):
        for peer in self.peers:
            block_ids = [block.id for block in self.chain][-50:]  # HACK
            send_message(peer, "sync", block_ids)

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

        # No value created or destroyed
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
            logger.info(f"Rejecting invalid tx: {tx}")
            return

        # Add to our mempool if it passes validation and isn't already there
        if tx not in self.mempool:
            self.mempool.append(tx)
            logger.info("Added tx to mempool")

            # Tell peers
            for peer in self.peers:
                send_message(peer, "tx", tx)

    def locate_in_branch(self, block_id):
        for chain_index, chain in enumerate(self.branches):
            for height, block in enumerate(chain):
                if block.id == block_id:
                    return chain, chain_index, height
        return None, None, None

    def validate_block(self, block, validate_txns=False):
        # Check POW
        assert block.proof < POW_TARGET, "Insufficient Proof-of-Work"

        # Check txns if we're extending main chain
        if validate_txns:
            self.validate_coinbase(block.txns[0])
            for tx in block.txns[1:]:
                self.validate_tx(tx)

    def handle_block(self, block):
        # Ignore if we've already seen it
        in_branch = self.locate_in_branch(block.id)[0]
        in_chain = block in self.chain
        if in_branch or in_chain:
            raise Exception("Duplicate block")

        # Conditions used in if-statements below
        branch, branch_index, height = self.locate_in_branch(block.prev_id)
        extends_chain = block.prev_id == self.chain[-1].id
        forks_chain = not branch and \
            block.prev_id in [block.id for block in self.chain[:-1]]
        extends_branch = branch and height == len(branch) -1
        forks_branch = branch and height != len(branch) -1

        # First of all, validate the block
        self.validate_block(block, validate_txns=extends_chain)

        if extends_chain:
            # Add it to the chain
            self.connect_block(block)
            logger.info(f"Extended chain to height {len(self.chain)}")
        elif forks_chain:
            # Create a new branch with just this block
            self.branches.append([block])
            logger.info(f"Created branch {len(self.branches)} to height {len(self.branches[-1])}")
        elif extends_branch:
            # Add it to the branch
            branch.append(block)
            logger.info(f"Extended branch {branch_index} to {len(branch)}")

            # Reorg if branch now has more work than main chain
            # FIXME
            chain_ids = [b.id for b in self.chain]

            chain_since_fork = self.chain[chain_ids.index(branch[0].prev_id)+1:]
            print(f"chain: {len(chain_since_fork)} branch: {len(branch)}")
            if total_work(branch) > total_work(chain_since_fork):
                logger.info(f"Reorging to branch {branch_index}")
                self.reorg(branch, branch_index, height)
        elif forks_branch:
            # Create a new branch with branch up to fork + this block
            self.branches.append(branch[:height+1] + [block])
            logger.info(f"Created (fork) branch {len(self.branches)} to height {len(self.branches[-1])}")
        else:
            # Re-enter IBD to find parent blocks
            logger.info("DOWNLOADING MISSING BLOCKS")
            self.initial_block_download()
            raise Exception("Can't connect block. Searching for parent.")

        # If there were no problems, tell peers
        for peer in self.peers:
            disrupt(send_message, [peer, "blocks", [block]])

    def reorg(self, branch, branch_index, fork_index):
        # Disconnect to fork block, preserving as a branch
        disconnected_blocks = []
        while self.chain[-1].id != branch[0].prev_id:
            block = self.chain.pop()
            for tx in block.txns:
                self.remove_from_utxo_set(tx)
            disconnected_blocks.insert(0, block)

        # Replace branch with newly disconnected blocks
        self.branches[branch_index] = disconnected_blocks

        # Connect new blocks, rollback if error encountered
        for block in branch:
            try:
                self.validate_block(block, validate_txns=True)
                self.connect_block(block)
            except:
                self.reorg(disconnected_blocks, branch_index, fork_index)
                logger.info(f"Reorg failed")
                return

    def connect_block(self, block):
        # Add to main chain
        self.chain.append(block)
        # Update utxo set
        for tx in block.txns:
            self.add_to_utxo_set(tx)

##################
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

def mine_block(block):
    while block.proof >= POW_TARGET:
        # TODO: accept interrupts here if tip changes
        if mining_interrupt.is_set():
            logger.info("Mining interrupted")
            mining_interrupt.clear()
            return
        block.nonce += 1
    return block


def mine_forever(public_key):
    logging.info("Starting miner")
    while True:
        with chain_lock:
            coinbase = prepare_coinbase(public_key, len(node.chain) -1)
            unmined_block = Block(
                txns=[coinbase] + deepcopy(node.mempool),
                prev_id=node.chain[-1].id,
                nonce=random.randint(0, 1000000000),
            )
        mined_block = mine_block(unmined_block)

        if mined_block:
            logger.info("")
            logger.info("Mined a block")
            with chain_lock:
                # TODO conceivably this could fail ...
                node.handle_block(mined_block)

def mine_genesis_block(node, public_key):
    coinbase = prepare_coinbase(public_key=public_key, height=0)
    unmined_block = Block(txns=[coinbase], prev_id=None, nonce=0)
    mined_block = mine_block(unmined_block)
    node.chain.append(mined_block)
    node.add_to_utxo_set(coinbase)
    return mined_block

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
            logger.info(f'Connected to {node.peers}')
            send_message(peer, "connect-response", None)

        if command == "connect-response":
            node.handle_peer(peer)
            logger.info(f'Connected to {peer[0]}"')

            # Request their peers
            send_message(peer, "peers", None)

        assert peer in node.peers, "Rejecting message from unconnected peer"

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
            with chain_lock:
                for block in data:
                    try:
                        node.handle_block(block)
                        mining_interrupt.set()
                    except Exception as e:
                        logger.info("Rejected block")

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

        if command == "sync":
            # Find our most recent block they don't know about
            peer_block_ids = data
            for block in node.chain[::-1]:
                if block.id not in peer_block_ids \
                        and block.prev_id in peer_block_ids:
                    height = node.chain.index(block)
                    blocks = node.chain[height:height+GET_BLOCKS_CHUNK]
                    send_message(peer, command="blocks", data=blocks)
                    return

            logger.info("couldn't serve sync request")
            send_message(peer, command="blocks", data=[])

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ('127.0.0.1', port)

def serve():
    server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    # server = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def serialize(obj):
    return pickle.dumps(obj)

def deserialize(obj):
    return pickle.loads(obj)

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
            logger.info("Retrying")
            return send_message(address, command, data, response, 
                         retries=retries-1)
            

#######
# CLI #
#######

def lookup_private_key(keyword):
    """
    Hacky way to predictibly lookup private keys for "node ids" 
    or "characters" ("bob" or "alice")
    """
    # Interpret integers as bank ID's
    if isinstance(keyword, int):
        # we add 100, so that first 100 integers can represent "characters"
        exponent = 100 + keyword

    # Otherwise, look up in this little registry of "characters"
    else:
        exponent = {"alice": 1, "bob": 2}[keyword]

    return SigningKey.from_secret_exponent(exponent, curve=SECP256k1)

def lookup_public_key(keyword):
    return lookup_private_key(keyword).get_verifying_key()

def main(args):
    if args["serve"]:
        threading.current_thread().name = "main"

        node_id = int(os.environ["ID"])

        duration = 3 * node_id
        logger.info(f"Sleeping {duration}")
        time.sleep(duration)
        logger.info("Waking up")

        # Set up the node
        global node
        address = (f"node{node_id}", PORT)
        node = Node(address=address)

        # Alice mines genesis block
        mine_genesis_block(node, lookup_public_key("alice"))

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
        mining_public_key = lookup_public_key(node_id)
        miner_thread = threading.Thread(target=mine_forever, args=(mining_public_key,), name="miner")
        miner_thread.start()

    elif args["ping"]:
        address = address_from_host(args["--node"])
        send_message(address, "ping", "")
    elif args["balance"]:
        public_key = lookup_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "balance", public_key, response=True)
        logger.info(response["data"])
    
    elif args["utxos"]:
        public_key = lookup_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "utxos", public_key, response=True)
        logger.info(response["data"])

    elif args["tx"]:
        # Grab parameters
        sender_private_key = lookup_public_key(args["<from>"])
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
