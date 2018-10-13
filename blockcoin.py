"""
BlockCoin

Usage:
  blockcoin.py server
  blockcoin.py ping [--node <node>]
  blockcoin.py tx <from> <to> <amount> [--node <node>]
  blockcoin.py balance <name> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import user_private_key, user_public_key, bank_private_key, bank_public_key, airdrop_tx


NUM_BANKS = 3
BLOCK_TIME = 5   # in seconds
PORT = 10000
bank = None

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)-15s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Tx:

    def __init__(self, id, tx_ins, tx_outs):
        self.id = id
        self.tx_ins = tx_ins
        self.tx_outs = tx_outs

    def sign_input(self, index, private_key):
        signature = private_key.sign(self.tx_ins[index].message)
        self.tx_ins[index].signature = signature

class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def message(self):
        # FIXME: we need something about the recipient here ...
        return f"{self.tx_id}:{self.index}".encode()

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

    def __init__(self, height, timestamp, txns, signature=None):
        self.height = height
        self.timestamp = timestamp
        self.signature = signature
        self.txns = txns

    @property
    def message(self):
        return serialize([self.height, self.timestamp, self.txns])

    def sign(self, private_key):
        self.signature = private_key.sign(self.message)

class Bank:

    def __init__(self, id, private_key):
        self.id = id
        self.blocks = []
        self.utxo = {}
        self.mempool = []
        self.private_key = private_key
        self.peer_addresses = {(p, PORT) for p in os.environ.get('PEERS', '').split(',') if p}

    @property
    def next_id(self):
        return len(self.blocks) % NUM_BANKS

    @property
    def our_turn(self):
        return self.id == self.next_id

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx.tx_ins]

    def fetch_utxo(self, public_key):
        return [utxo for utxo in self.utxo.values() 
                if utxo.public_key.to_string() == public_key.to_string()]

    def update_utxo(self, tx):
        # Remove utxos that were just spent
        for tx_in in tx.tx_ins:
            del self.utxo[tx_in.outpoint]
        # Save utxos which were just created
        for tx_out in tx.tx_outs:
            self.utxo[tx_out.outpoint] = tx_out

    def fetch_balance(self, public_key):
        # Fetch utxo associated with this public key
        unspents = self.fetch_utxo(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in unspents])

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0
        for tx_in in tx.tx_ins:
            # TxIn spending an unspent output
            assert tx_in.outpoint in self.utxo

            # No pending transactions spending this same output
            assert tx_in.outpoint not in self.mempool_outpoints

            tx_out = self.utxo[tx_in.outpoint]
            # Verify signature using public key of TxOut we're spending
            public_key = tx_out.public_key
            public_key.verify(tx_in.signature, tx_in.message)

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

    def handle_block(self, block):
        assert block.height == len(self.blocks)

        # Genesis block has no signature
        if block.height > 0:
            public_key = bank_public_key(self.next_id)
            public_key.verify(block.signature, block.message)

        # Check the transactions are valid
        for tx in block.txns:
            self.validate_tx(tx)

        # If they're all good, update self.blocks and self.utxo
        for tx in block.txns:
            self.update_utxo(tx)
        
        # Add the block and increment the id of bank who will report next block
        self.blocks.append(block)

        # Schedule submisison of next block
        self.schedule_next_block()

    def make_block(self):
        # Reset mempool
        txns = deepcopy(self.mempool)
        self.mempool = []
        block = Block(
            height=len(self.blocks),
            timestamp=time.time(),
            txns=txns
        )
        block.sign(self.private_key)
        return block

    def submit_block(self):
        # Make the block
        block = self.make_block()

        # Save locally
        self.handle_block(block)

        # Tell peers
        for address in self.peer_addresses:
            send_message(address, "block", block)

    def schedule_next_block(self):
        if self.our_turn:
            threading.Timer(5, self.submit_block, []).start()

    def airdrop(self, tx):
        assert len(self.blocks) == 0

        # Update utxos
        self.update_utxo(tx)

        # Update blockchain
        block = Block(height=0, timestamp=time.time(), signature=None, txns=[tx])
        self.blocks.append(block)

def send_value(utxo, sender_private_key, recipient_public_key, amount):
    sender_public_key = sender_private_key.get_verifying_key()

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxo:
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
            bank.handle_block(data)

        if command == "tx":
            bank.handle_tx(data)

        if command == "balance":
            balance = bank.fetch_balance(data)
            self.respond(command="balance-response", data=balance)

        if command == "utxo":
            utxo = bank.fetch_utxo(data)
            self.respond(command="utxo-response", data=utxo)

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ('localhost', port)

def server():
    server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def send_message(address, command, data, response=False):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(serialize(message))
        if response:
            return deserialize(s.recv(5000))

def main(args):
    if args["server"]:
        global bank
        bank_id = int(os.environ["BANK_ID"])
        bank = Bank(
            id=bank_id,
            private_key=bank_private_key(bank_id)
        )
        # Airdrop starting balances
        bank.airdrop(airdrop_tx())
        # Start producing blocks
        bank.schedule_next_block()
        server()
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
        response = send_message(address, "utxo", sender_public_key, response=True)
        utxo = response["data"]

        # Prepare transaction
        tx = send_value(utxo, sender_private_key, recipient_public_key, amount)

        # send to bank
        send_message(address, "tx", tx)
    else:
        print("Invalid command")


if __name__ == '__main__':
    main(docopt(__doc__))