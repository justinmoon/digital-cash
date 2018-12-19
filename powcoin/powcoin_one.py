"""
POWCoin Part 1
* Add code to simulate network latency and packet loss

Usage:
  powcoin_one.py serve
  powcoin_one.py ping [--node <node>]
  powcoin_one.py tx <from> <to> <amount> [--node <node>]
  powcoin_one.py balance <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, hashlib, random, re, pickle

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1

PORT = 10000
GET_BLOCKS_CHUNK = 10
BLOCK_SUBSIDY = 50
node = None
lock = threading.Lock()

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

    @property
    def is_coinbase(self):
        return self.tx_ins[0].tx_id is None

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
        self.peers = []
        self.pending_peers = []
        self.address = address

    def connect(self, peer):
        if peer not in self.peers and peer != self.address:
            logger.info(f'(handshake) Sent "connect" to {peer[0]}')
            try:
                send_message(peer, "connect", None)
                self.pending_peers.append(peer)
            except:
                logger.info(f'(handshake) Node {peer[0]} offline')

    def sync(self):
        blocks = self.blocks[-GET_BLOCKS_CHUNK:]
        block_ids = [block.id for block in blocks]
        for peer in self.peers:
            send_message(peer, "sync", block_ids)

    def fetch_utxos(self, public_key):
        return [tx_out for tx_out in self.utxo_set.values() 
                if tx_out.public_key == public_key]

    def update_utxo_set(self, tx):
        # Remove utxos that were just spent
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]

        # Save utxos which were just created
        for tx_out in tx.tx_outs:
            self.utxo_set[tx_out.outpoint] = tx_out

        # Clean up mempool
        if tx in self.mempool:
            self.mempool.remove(tx)

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
        assert len(tx.tx_ins) == len(tx.tx_outs) == 1
        assert tx.tx_outs[0].amount == BLOCK_SUBSIDY

    def handle_tx(self, tx):
        if tx not in self.mempool:
            self.validate_tx(tx)
            self.mempool.append(tx)

            # Propogate transaction
            for peer in self.peers:
                send_message(peer, "tx", tx)

    def validate_block(self, block):
        assert block.proof < POW_TARGET, "Insufficient Proof-of-Work"
        assert block.prev_id == self.blocks[-1].id

    def handle_block(self, block):
        # Check work, chain ordering
        self.validate_block(block)

        # Validate coinbase separately
        self.validate_coinbase(block.txns[0])

        # Check the transactions are valid
        for tx in block.txns[1:]:
            self.validate_tx(tx)

        # If they're all good, update self.blocks and self.utxo_set
        for tx in block.txns:
            self.update_utxo_set(tx)
        
        # Add the block to our chain
        self.blocks.append(block)

        logger.info(f"Block accepted: height={len(self.blocks) - 1}")

        # Block propogation
        for peer in self.peers:
            disrupt(func=send_message, args=[peer, "blocks", [block]])

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

    return tx

def prepare_coinbase(public_key, tx_id=None):
    if tx_id is None:
        tx_id = uuid.uuid4()
    return Tx(
        id=tx_id,
        tx_ins=[
            TxIn(None, None, None),    
        ],
        tx_outs=[
            TxOut(tx_id=tx_id, index=0, amount=BLOCK_SUBSIDY,
                  public_key=public_key),
        ],
    )

##########
# Mining #
##########

DIFFICULTY_BITS = 15
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


def mine_forever(public_key):
    logging.info("Starting miner")
    while True:
        coinbase = prepare_coinbase(public_key)
        unmined_block = Block(
            txns=[coinbase] + node.mempool,
            prev_id=node.blocks[-1].id,
            nonce=random.randint(0, 1000000000),
        )
        mined_block = mine_block(unmined_block)

        if mined_block:
            logger.info("")
            logger.info("Mined a block")
            with lock:
                node.handle_block(mined_block)

def mine_genesis_block(public_key):
    global node
    coinbase = prepare_coinbase(public_key, tx_id="abc123")
    unmined_block = Block(txns=[coinbase], prev_id=None, nonce=0)
    mined_block = mine_block(unmined_block)
    node.blocks.append(mined_block)
    node.update_utxo_set(coinbase)

##############
# Networking #
##############

def serialize(coin):
    return pickle.dumps(coin)

def deserialize(serialized):
    return pickle.loads(serialized)

def read_message(s):
    message = b''
    # Our protocol is: first 4 bytes signify message length
    raw_message_length = s.recv(4) or b"\x00"
    message_length = int.from_bytes(raw_message_length, 'big')

    while message_length > 0:
        chunk = s.recv(1024)
        message += chunk
        message_length -= len(chunk)

    return deserialize(message)

def prepare_message(command, data):
    message = {
        "command": command,
        "data": data,
    }
    serialized_message = serialize(message)
    length = len(serialized_message).to_bytes(4, 'big')
    return length + serialized_message

def disrupt(func, args):
    if random.randint(0, 10) != 0:
        threading.Timer(random.random(), func, args).start()

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
        message = read_message(self.request)
        command = message["command"]
        data = message["data"]

        peer = self.get_canonical_peer_address()

        # Handshake / Authentication
        if command == "connect":
            if peer not in node.pending_peers and peer not in node.peers:
                node.pending_peers.append(peer)
                logger.info(f'(handshake) Accepted "connect" request from "{peer[0]}"')
                send_message(peer, "connect-response", None)
        elif command == "connect-response":
            if peer in node.pending_peers and peer not in node.peers:
                node.pending_peers.remove(peer)
                node.peers.append(peer)
                logger.info(f'(handshake) Connected to "{peer[0]}"')
                send_message(peer, "connect-response", None)

                # Request their peers
                send_message(peer, "peers", None)
            
        # else:
            # assert peer in node.peers, \
                # f"Rejecting {command} from unconnected {peer[0]}"

        # Business Logic
        if command == "peers":
            send_message(peer, "peers-response", node.peers)

        if command == "peers-response":
            for peer in data:
                node.connect(peer)

        if command == "ping":
            self.respond(command="pong", data="")

        if command == "sync":
            # Find our most recent block peer doesn't know about,
            # But which build off a block they do know about.
            peer_block_ids = data
            for block in node.blocks[::-1]:
                if block.id not in peer_block_ids \
                        and block.prev_id in peer_block_ids:
                    height = node.blocks.index(block)
                    blocks = node.blocks[height:height+GET_BLOCKS_CHUNK]
                    send_message(peer, "blocks", blocks)
                    logger.info('Served "sync" request')
                    return

            logger.info('Could not serve "sync" request')

        if command == "blocks":

            for block in data:
                try:
                    with lock:
                        node.handle_block(block)
                    mining_interrupt.set()
                except:
                    logger.info("Rejected block")

            if len(data) == GET_BLOCKS_CHUNK:
                node.sync()

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
        s.sendall(message)
        if response:
            return read_message(s)


#######
# CLI #
#######

def lookup_private_key(name):
    exponent = {
        "alice": 1, "bob": 2, "node0": 3, "node1": 4, "node2": 5
    }[name]
    return SigningKey.from_secret_exponent(exponent, curve=SECP256k1)

def lookup_public_key(name):
    return lookup_private_key(name).get_verifying_key()

def main(args):
    if args["serve"]:
        threading.current_thread().name = "main"
        name = os.environ["NAME"]

        duration = 10 * ["node0", "node1", "node2"].index(name)
        time.sleep(duration)

        global node
        node = Node(address=(name, PORT))

        # Alice is Satoshi!
        mine_genesis_block(lookup_public_key("alice"))

        # Start server thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        # Join the network
        peers = [(p, PORT) for p in os.environ['PEERS'].split(',')]
        for peer in peers:
            node.connect(peer)

        # Wait for peer connections
        time.sleep(1)

        # Do initial block download
        node.sync()

        # Wait for IBD to finish
        time.sleep(1)

        # Start miner thread
        miner_public_key = lookup_public_key(name)
        miner_thread = threading.Thread(target=mine_forever, 
                args=[miner_public_key], name="miner")
        miner_thread.start()

    elif args["ping"]:
        address = external_address(args["--node"])
        send_message(address, "ping", "")
    elif args["balance"]:
        public_key = lookup_public_key(args["<name>"])
        address = external_address(args["--node"])
        response = send_message(address, "balance", public_key, response=True)
        print(response["data"])
    elif args["tx"]:
        # Grab parameters
        sender_private_key = lookup_private_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = lookup_private_key(args["<to>"])
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
