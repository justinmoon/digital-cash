import time, logging, threading, hashlib

logger = logging.getLogger(__name__)



mining_interrupt = threading.Event()
chain_lock = threading.Lock()

chain = []

bits = 21
target = 1 << (256 - bits)

def message_generator():
    """Arbitrary message for blocks"""
    number = 1
    while True:
        yield int.to_bytes(number, 'little')
        number += 1


class Block:

    def __init__(self, previous, nonce=None):
        self.previous = previous
        self.nonce = nonce

    @property
    def id(self):
        return mining_hash(self.header(self.nonce))

    def header(self, nonce):
        return f"{self.previous}{nonce}"

    def __repr__(self):
        return (f"Block(previous={self.previous}, nonce={self.nonce}, "
                f"id={self.id})")


def mining_hash(s):
    if not isinstance(s, bytes):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def mine_block(block):
    nonce = 0
    while int(mining_hash(block.header(nonce)), 16) >= target:
        nonce += 1
        if mining_interrupt.is_set():
            print("Mining interrupted")
            mining_interrupt.clear()
            return
    block.nonce = nonce
    print(f"Nonce found {block}")
    return block


def mine_forever():
    while True:
        unmined_block = Block(previous=chain[-1].id)
        mined_block = mine_block(unmined_block)
        
        # This is False if mining was interrupted
        # Perhaps an exception would be wiser ...
        if mined_block:
            with chain_lock:
                chain.append(mined_block)


def chain_is_valid():
    current_block = chain[0]
    for block in chain[1:]:
        assert block.previous == current_block.id
        assert int(block.id, 16) < target
        current_block = block


def main():
    global chain

    thread = threading.Thread(target=mine_forever)
    thread.start()

    while True:
        if len(chain) == 2:
            mining_interrupt.set()
            print("Set mining interrupt")
            with chain_lock:
                block = Block(
                    previous="0000070cfe252d71f30a7d66e652174fce5bb6dc90cb7c52997871ffbc731433", 
                    nonce=62706,
                )
                chain.append(block)

        chain_is_valid()

        time.sleep(.1)




if __name__ == "__main__":
    genesis = Block(
        previous="0" * 64, 
        nonce=0
    )
    print("Setting genesis block: {genesis}")
    chain.append(genesis)
    main()
