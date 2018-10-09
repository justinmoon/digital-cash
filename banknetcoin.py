import uuid, socketserver, socket, sys, argparse
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize


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

class Bank:

    def __init__(self):
        self.utxo = {}

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

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0
        for tx_in in tx.tx_ins:
            assert tx_in.outpoint in self.utxo

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

    def handle_tx(self, tx):
        # Save to self.utxo if it's valid
        self.validate_tx(tx)
        self.update_utxo(tx)

    def fetch_utxo(self, public_key):
        return [utxo for utxo in self.utxo.values() 
                if utxo.public_key.to_string() == public_key.to_string()]

    def fetch_balance(self, public_key):
        # Fetch utxo associated with this public key
        unspents = self.fetch_utxo(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in unspents])


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
        data = self.request.recv(1024*4).strip()
        message = deserialize(data)
        command = message["command"]

        if command == "ping":
            self.respond(command="pong", data="")


HOST, PORT = 'localhost', 9002


def server():
    server = socketserver.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()


def connect(command, data):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(serialize(message))
        _data = s.recv(1024*4)
        data = deserialize(_data)

    print('Received', data)
    return data


def cli():
    parser = argparse.ArgumentParser(description='BankNetCoin')
    parser.add_argument('role', type=str, nargs="?",
            help='client or server?')
    parser.add_argument('-c', '--command', type=str, 
            help='What command to send? (client role only)')
    parser.add_argument('d', '--data', type=str, 
            default="",
            help='Data to send alongside command? (client role only)')
    return parser.parse_args()


def main():
    args = cli()
    if args.role == "client":
        connect(args.command, args.data)
    elif args.role == "server":
        server()
    else:
        print(f"Invalid role: {args.role}")


if __name__ == '__main__':
    main()
