"""
POW P2P Coin

Usage:
  powp2pcoin.py.py serve
  powp2pcoin.py.py ping [--node <node>]
  powp2pcoin.py.py tx <from> <to> <amount> [--node <node>]
  powp2pcoin.py.py balance <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, hashlib, random, re

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import user_private_key, user_public_key


PORT = 10000
node = None

logging.basicConfig(level="INFO", format='%(threadName)-6s | %(message)s')
logger = logging.getLogger(__name__)


def spend_message(tx, index):
    outpoint = tx.tx_ins[index].outpoint
    return serialize(outpoint) + serialize(tx.tx_outs)

class Tx:

    def __init__(self, id, tx_ins, tx_outs):
        self.id = id
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

class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def outpoint(self):
        return (self.tx_id, self.index)

class TxOut:

    def __init__(self, tx_id, index, amount, public_key):
        self.tx_id = tx_id
        self.index = index
        self.amount = amount
        self.public_key = public_key

    @property
    def outpoint(self):
        return (self.tx_id, self.index)

class Block:

    def __init__(self, txns, prev_id, nonce):
        self.txns = txns
        self.prev_id = prev_id
        self.nonce = nonce

    @property
    def header(self):
        return serialize(self)

    @property
    def id(self):
        return hashlib.sha256(self.header).hexdigest()

    @property
    def proof(self):
        return int(self.id, 16)

    def __repr__(self):
        return f"Block(prev_id={self.prev_id[:10]}... id={self.id[:10]}...)"

class Node:

    def __init__(self, address):
        self.blocks = []
        self.utxo_set = {}
        self.mempool = []
        # FIXME ugly
        self.address = address
        self.peers = set()
        self.syncing = False

    def join_network(self, peers):
        for peer in peers:
            if peer not in self.peers and peer != self.address:
                try:
                    send_message(peer, "join", None)
                except:
                    logger.info(f'Peer "{peer} offline"')
                    continue

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx.tx_ins]

    def fetch_utxos(self, public_key):
        return [tx_out for tx_out in self.utxo_set.values() 
                if tx_out.public_key == public_key]

    def update_utxo_set(self, tx):
        # Remove utxos that were just spent
        for tx_in in tx.tx_ins:
            del self.utxo_set[tx_in.outpoint]
        # Save utxos which were just created
        for tx_out in tx.tx_outs:
            self.utxo_set[tx_out.outpoint] = tx_out

    def fetch_balance(self, public_key):
        # Fetch utxos associated with this public key
        utxos = self.fetch_utxos(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in utxos])

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0
        for index, tx_in in enumerate(tx.tx_ins):
            # TxIn spending an unspent output
            assert tx_in.outpoint in self.utxo_set

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

    def handle_tx(self, tx):
        self.validate_tx(tx)
        self.mempool.append(tx)

    def validate_block(self, block):
        assert block.proof < POW_TARGET, "Insufficient Proof-of-Work"
        assert block.prev_id == self.blocks[-1].id

    def handle_block(self, block):
        # Check work, chain ordering
        self.validate_block(block)

        # Check the transactions are valid
        for tx in block.txns:
            self.validate_tx(tx)

        # If they're all good, update self.blocks and self.utxo_set
        for tx in block.txns:
            self.update_utxo_set(tx)
        
        # Add the block to our chain
        self.blocks.append(block)

        logger.info(f"Block accepted: height={len(self.blocks) - 1}")

        # Block propogation
        for peer in self.peers:
            send_message(peer, "block", block)

    def initial_block_download(self):
        # just talk to one peer for now
        if self.peers:
            peer = next(iter(self.peers))
            send_message(peer, "get_blocks", self.blocks[-1].id)
        else:
            logger.info("(ibd) no peers")

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
    tx_id = uuid.uuid4()
    change = tx_in_sum - amount
    tx_outs = [
        TxOut(tx_id=tx_id, index=0, amount=amount, public_key=recipient_public_key), 
        TxOut(tx_id=tx_id, index=1, amount=change, public_key=sender_public_key),
    ]

    # Construct tx and sign inputs
    tx = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
    for i in range(len(tx.tx_ins)):
        tx.sign_input(i, sender_private_key)

##########
# Mining #
##########

DIFFICULTY_BITS = 20
POW_TARGET = 2 ** (256 - DIFFICULTY_BITS)
mining_interrupt = threading.Event()


def mine_block(block):
    while block.proof >= POW_TARGET:
        # TODO: accept interrupts here if tip changes
        if mining_interrupt.is_set():
            logger.info("Mining interrupted")
            mining_interrupt.clear()
            return
        block.nonce += 1
    return block


def mine_forever():
    logging.info("Starting miner")
    while True:
        unmined_block = Block(
            txns=node.mempool,
            prev_id=node.blocks[-1].id,
            nonce=random.randint(0, 1000000000),
        )
        mined_block = mine_block(unmined_block)

        if mined_block:
            logger.info("")
            logger.info("Mined a block")
            node.handle_block(mined_block)

def mine_genesis_block():
    global node
    unmined_block = Block(txns=[], prev_id=None, nonce=0)
    mined_block = mine_block(unmined_block)
    node.blocks.append(mined_block)
    # TODO: update utxo set, award coinbase, etc


##############
# Networking #
##############

def prepare_message(command, data):
    return {
        "command": command,
        "data": data,
    }

class TCPHandler(socketserver.BaseRequestHandler):

    def get_canonical_peer_address(self):
        address = self.client_address[0]
        try:
            raw_host_info = socket.gethostbyaddr(address)
            hostname = re.search(r"_(.*?)_", raw_host_info[0]).group(1)
        except:
            hostname = address
        return (hostname, PORT)


    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(serialize(response))

    def handle(self):
        peer = self.get_canonical_peer_address()
        message_bytes = self.request.recv(1024*4).strip()
        message = deserialize(message_bytes)
        command = message["command"]
        data = message["data"]

        # logger.info(f"received {command}")

        if command == "join":
            # Accept the connection. Add them to our list of peers
            node.peers.add(peer)
            logger.info(f'Now connected to {peer[0]}"')
            send_message(peer, "join-response", None)

        if command == "join-response":
            node.peers.add(peer)
            logger.info(f'Now connected to {peer[0]}"')
            # Request the peer's peers
            send_message(peer, "peers", None)

        assert peer in node.peers, \
                "rejecting message from unconnected peer"

        if command == "peers":
            send_message(peer, "peers-response", node.peers)

        if command == "peers-response":
            node.join_network(data)
            logger.info(f"received peers: {data}")

        # Non-handshake messages require authentication
        if command == "ping":
            self.respond(command="pong", data="")

        if command == "block":
            if data == None:
                logger.info(f"Initial block download complete")
                node.syncing = False
                return

            if data.prev_id == node.blocks[-1].id:
                node.handle_block(data)
                # Interrupt mining thread
                mining_interrupt.set()

            # If syncing, request next block
            if node.syncing:
                node.initial_block_download()

        if command == "get_blocks":
            next_block = None
            # locate the block in the main chain
            # FIXME: this should call a general-purpose function
            for block in node.blocks:
                if block.prev_id == data:
                    next_block = block
                    break

            send_message(peer, command="block", data=next_block)
            logger.info(f"continue ibd")

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
    logger.info("Starting server")
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
    logger.info(os.environ)
    if args["serve"]:
        threading.current_thread().name = "main"

        global node
        hostname = os.environ['ME']
        address = (hostname, PORT)
        node = Node(address)

        node_id = int(hostname[-1])
        duration = 10 * node_id
        time.sleep(duration)
        
        # TODO: mine genesis block
        mine_genesis_block()

        # Start server thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        # Join the network
        peers = {(p, PORT) for p in os.environ['PEERS'].split(',')}

        try:
            node.join_network(peers)
        except:
            # Peer not online yet
            pass

        # time for responses
        time.sleep(1)

        # Do initial block download
        logger.info("starting ibd")
        node.syncing = True
        node.initial_block_download()
        
        while node.syncing:
            time.sleep(1)

        # Start miner thread
        miner_thread = threading.Thread(target=mine_forever, name="miner")
        miner_thread.start()

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
