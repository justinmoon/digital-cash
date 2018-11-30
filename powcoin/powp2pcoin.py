"""
POW Syndacoin

Usage:
  powp2pcoin.py.py serve
  powp2pcoin.py.py ping [--node <node>]
  powp2pcoin.py.py tx <from> <to> <amount> [--node <node>]
  powp2pcoin.py.py balance <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import socketserver, socket, sys, argparse, time, os, logging, threading, hashlib, random, pickle

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1

GET_BLOCKS_CHUNK = 10
BLOCK_SUBSIDY = 50
PORT = 10000
node = None

logging.basicConfig(level="INFO", format='%(threadName)-6s | %(message)s')
logger = logging.getLogger(__name__)

def print_exc():
    import traceback
    logger.info(traceback.format_exc())

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
    def id(self):
        return hashlib.sha256(serialize(self)).hexdigest()

    @property
    def is_coinbase(self):
        return isinstance(self.tx_ins[0].signature, int)

class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def outpoint(self):
        return (self.tx_id, self.index)

class TxOut:

    def __init__(self, amount, public_key):
        self.amount = amount
        self.public_key = public_key

class UnspentTxOut:
    # "TxIn.outpoint" + "TxOut"

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
        self.address = address

    def connect(self, peer):
        if peer not in self.peers and peer != self.address:
            send_message(peer, "connect", None)

    def handle_peer(self, peer):
        if peer not in self.peers and peer != self.address:
            node.peers.append(peer)

    def sync(self):
        for peer in self.peers:
            block_ids = [block.id for block in self.blocks][-50:]  # HACK
            send_message(peer, "sync", block_ids)

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx.tx_ins]

    def fetch_utxos(self, public_key):
        return [UnspentTxOut(tx_id=tx_id, index=index, 
                    amount=tx_out.amount, public_key=tx_out.public_key)
                for ((tx_id, index), tx_out) in self.utxo_set.values() 
                if tx_out.public_key == public_key]

    def update_utxo_set(self, tx):
        # Remove utxos that were just spent (coinbases don't spend utxos)
        for tx_in in tx.tx_ins[1:]:
            del self.utxo_set[tx_in.outpoint]

        # Save utxos which were just created
        for index, tx_out in enumerate(tx.tx_outs):
            self.utxo_set[(tx.id, index)] = tx_out

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

    def validate_coinbase(self, tx):
        assert len(tx.tx_ins) == 1
        assert len(tx.tx_outs) == 1
        assert tx.tx_outs[0].amount == BLOCK_SUBSIDY

    def handle_tx(self, tx):
        self.validate_tx(tx)
        self.mempool.append(tx)

    def validate_block(self, block):
        assert block.proof < POW_TARGET, "Insufficient Proof-of-Work"
        assert block.prev_id == self.blocks[-1].id

    def handle_block(self, block):
        # Check work, chain ordering
        self.validate_block(block)

        # Validate coinbase transaction separately
        self.validate_coinbase(block.txns[0])

        # Check the transactions are valid
        for tx in block.txns[1:]:
            self.validate_tx(tx)

        # If they're all good, update self.blocks and self.utxo_set
        for tx in block.txns[1:]:
            self.update_utxo_set(tx)
        
        # Add the block to our chain
        self.blocks.append(block)

        logger.info(f"Block accepted: height={len(self.blocks) - 1}")

        # Block propogation
        for peer_address in self.peers:
            send_message(peer_address, "blocks", [block])
            # disrupt(send_message, [peer_address, "blocks", [block]])

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

def prepare_coinbase(public_key, height):
    # Put height in signature field so coinbase hashes don't collide
    return Tx(
        tx_ins=[
            TxIn(None, None, height)
        ],
        tx_outs=[
            TxOut(amount=BLOCK_SUBSIDY, public_key=public_key), 
        ],
    )

##########
# Mining #
##########

DIFFICULTY_BITS = 16
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
        coinbase = prepare_coinbase(public_key, len(node.blocks) -1)
        unmined_block = Block(
            txns=[coinbase] + deepcopy(node.mempool),
            prev_id=node.blocks[-1].id,
            nonce=random.randint(0, 1000000000),
        )
        mined_block = mine_block(unmined_block)

        if mined_block:
            logger.info("")
            logger.info("Mined a block")
            node.handle_block(mined_block)

def mine_genesis_block(node, public_key):
    coinbase = prepare_coinbase(public_key=public_key, height=0)
    unmined_block = Block(txns=[], prev_id=None, nonce=0)
    mined_block = mine_block(unmined_block)
    node.blocks.append(mined_block)
    node.update_utxo_set(coinbase)


##############
# Networking #
##############

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
        chunk = s.recv(1024)
        data += chunk
        msg_len -= len(chunk)

    return deserialize(data)

def prepare_message(command, data):
    d = {
        "command": command,
        "data": data,
    }
    ser = serialize(d)
    length = len(ser).to_bytes(4, "big")
    # return length + ser
    return ser

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
        return self.request.sendall(serialize(response))

    def handle(self):
        peer = self.get_canonical_peer_address()
        # FIXME: let's add message length here ...
        # message = read_message(self.request)
        message_bytes = self.request.recv(4*1024).strip()
        message = deserialize(message_bytes)

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

        if command == "sync":
            # Find our most recent block they don't know about, 
            # But which builds off a block they know about
            peer_block_ids = data
            for block in node.blocks[::-1]:
                if block.id not in peer_block_ids \
                        and block.prev_id in peer_block_ids:
                    height = node.blocks.index(block)
                    blocks = node.blocks[height:height+GET_BLOCKS_CHUNK]
                    logger.info("served sync request")
                    send_message(peer, command="blocks", data=blocks)
                    return

            logger.info("couldn't serve sync request")
            send_message(peer, command="blocks", data=[])

        if command == "blocks":
            for block in data:
                try:
                    node.handle_block(block)
                    mining_interrupt.set()
                except Exception as e:
                    # print_exc()
                    # logger.info([[node.prev_id, node.id] for node in data])
                    # logger.info(data)
                    # return
                    logger.info(f"Rejected block {str(e)}")
                    return

            # If we received a full sync response, continue syncing
            if len(data) > 1:
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
    # server = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def disrupt(func, args):
    # Adds latency. Drops packets.
    if random.randint(1, 10) != 5:
        threading.Timer(random.random(), func, args).start()

def send_message(address, command, data, response=False):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(message)
        if response:
            return deserialize(s.recv(5000))


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

        # Hack to start nodes at different times
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

        # Start server thread
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
        
        time.sleep(1)

        # Do initial block download
        logger.info("Starting initial block download")
        node.sync()

        time.sleep(1)  # let sync happen
        # Start miner thread
        miner_thread = threading.Thread(
                target=mine_forever, args=[lookup_public_key(node_id)], name="miner")
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
        sender_private_key = lookup_public_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = lookup_public_key(args["<to>"])
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
