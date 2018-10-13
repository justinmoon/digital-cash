"""
BlockCoin

Usage:
  blockcoin.py server
  blockcoin.py ping [--node <node>]
  blockcoin.py tx <from> <to> <amount> [--node <node>]
  blockcoin.py balance <name> [--node <node>]
  blockcoin.py block <height> [--node <node>]

Options:
  -h --help      Show this screen.
  --node=<node>  Hostname of node [default: node0]
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import lookup_key, bank_private_key, bank_public_key


NUM_BANKS = 3

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
        signature = private_key.sign(self.tx_ins[index].spend_message)
        self.tx_ins[index].signature = signature

class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def spend_message(self):
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
        # FIXME improve variable name
        return serialize([self.height, self.timestamp, self.txns])

    def sign(self, private_key):
        # message just omits the signature
        self.signature = private_key.sign(self.message)

class Bank:

    def __init__(self, id, private_key):
        self.next_id = 0
        self.id = id
        self.blocks = []
        self.utxo = {}
        self.mempool = []
        self.private_key = private_key

    def update_utxo(self, tx):
        for tx_out in tx.tx_outs:
            self.utxo[tx_out.outpoint] = tx_out
        for tx_in in tx.tx_ins:
            del self.utxo[tx_in.outpoint]

    def issue(self, amount, public_key):
        id_ = str(uuid.uuid4())
        tx_ins = []
        tx_outs = [TxOut(tx_id=id_, index=0, amount=amount, public_key=public_key)]
        tx = Tx(id=id_, tx_ins=tx_ins, tx_outs=tx_outs)

        self.update_utxo(tx)

        return tx

    @property
    def mempool_outpoints(self):
        return [tx_in.outpoint for tx in self.mempool for tx_in in tx]

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
            public_key.verify(tx_in.signature, tx_in.spend_message)

            # Sum up the total inputs
            amount = tx_out.amount
            in_sum += amount

        for tx_out in tx.tx_outs:
            out_sum += tx_out.amount

        assert in_sum == out_sum

    def fetch_utxo(self, public_key):
        return [utxo for utxo in self.utxo.values() 
                if utxo.public_key.to_string() == public_key.to_string()]

    def fetch_balance(self, public_key):
        # Fetch utxo associated with this public key
        unspents = self.fetch_utxo(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in unspents])

    def schedule_next_block(self):
        # submit a block `Constants.block_interval` seconds
        # handle_block calls this if it is our turn
        pass

    def make_block(self):
        # Reset mempool
        txns = deepcopy(self.mempool)  # FIXME???
        self.mempool = []
        block = Block(
            height=len(self.blocks),
            timestamp=time.time(),
            txns=txns
        )
        block.sign(self.private_key)
        return block

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
        self.next_id = (self.next_id + 1) % NUM_BANKS

    def airdrop(self, tx):
        """Special logic to execute an airdrop transaction"""
        assert len(self.blocks) == 0

        # Update utxos
        self.update_utxo(tx)

        # Update blockchain
        block = Block(height=0, timestamp=time.time(), signature=None, txns=[tx])
        self.blocks.append(block)


def send_value(bank, sender_private_key, recipient_public_key, amount):
    # Fetch utxos
    sender_public_key = sender_private_key.get_verifying_key()
    utxo = bank.fetch_utxo(sender_public_key)
    utxo_sorted = sorted(utxo, key=lambda tx_out: tx_out.amount)
    utxo_sum = sum([u.amount for u in utxo])

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxo_sorted:
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

def cron():
    peer_hostnames = {p for p in os.environ.get('PEERS', '').split(',') if p}

    def ping_peers():
        if bank.id == bank.next_id:
            for hostname in peer_hostnames:
                address = (hostname, 10000)
                send_message(address, "ping", "")
        # TODO make a helper to advance next_id
        bank.next_id = (bank.next_id + 1) % 3
        threading.Timer(1, ping_peers, []).start()

    # New blocks every 10 seconds
    threading.Timer(1, ping_peers, []).start()

class TCPHandler(socketserver.BaseRequestHandler):

    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(serialize(response))

    # def propogate(self, command, data):
        # """Send a message to all other banks. Good for Tx propogation."""
        # for peer_id in range(NUM_BANKS):
            # if peer_id != bank.id:
                # connect(command, data, get_address(peer_id))

    def handle(self):
        message_bytes = self.request.recv(1024*4).strip()
        message = deserialize(message_bytes)
        command = message["command"]
        data = message["data"]

        logger.info(f"received {command}")

        if command == "ping":
            self.respond(command="pong", data="")

        if command == "tx":
            try:
                # FIXME: this method no longer exists. Logic needs complete reboot.
                bank.handle_tx(data)
                self.respond(command="tx-response", data="accepted")
            except:
                self.respond(command="tx-response", data="rejected")

        if command == "balance":
            balance = bank.fetch_balance(data)
            self.respond(command="balance-response", data=balance)


PORT = 10000
bank = Bank(
    id=int(os.environ["BANK_ID"]),
    private_key=bank_private_key(0)
)


def network_address_for_bank(bank):
    hostname = f"node{bank.id}"
    return (hostname, PORT)

def address_from_host(host):
    return (host, PORT)

def server():
    cron()
    server = socketserver.TCPServer(("0.0.0.0", PORT), TCPHandler)
    server.serve_forever()

def send_message(address, command, data):
    # FIXME: add "address" parameter
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(serialize(message))
        # _data = s.recv(1024*4)
        # data = deserialize(_data)

    # print('Received', data)
    # return data


def main(args):
    if args["server"]:
        from identities import alice_public_key
        bank.issue(1000, alice_public_key)
        server()
    elif args["ping"]:
        address = address_from_host(args["--node"])
        print(address)
        send_message(address, "ping", "")
    elif args["balance"]:
        name = args["<name>"]
        private_key = lookup_key(name)
        public_key = private_key.get_verifying_key()
        address = address_from_host(args["--node"])
        connect(address, "balance", public_key)
    elif args["tx"]:
        sender_private_key = lookup_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = lookup_key(args["<to>"])
        recipient_public_key = recipient_private_key.get_verifying_key()
        amount = int(args["<amount>"])

        utxo = bank.fetch_utxo(sender_public_key)
        utxo_sum = sum([u.amount for u in utxo])

        tx_ins = [
            TxIn(tx_id=tx_out.tx_id, index=tx_out.index, signature=None)
            for tx_out in utxo
        ]
        tx_id = uuid.uuid4()
        tx_outs = [
            TxOut(tx_id=tx_id, index=0, amount=amount, public_key=recipient_public_key), 
            TxOut(tx_id=tx_id, index=1, amount=utxo_sum-amount, public_key=sender_public_key),
        ]
        tx = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
        for i in range(len(tx.tx_ins)):
            tx.sign_input(i, sender_public_key)
        address = address_from_host(args["--node"])
        connect(address, "tx", tx)
    else:
        print("Invalid commands")


if __name__ == '__main__':
    main(docopt(__doc__))
