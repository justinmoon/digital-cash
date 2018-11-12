import time
import hashlib


def get_proof(header, nonce):
    preimage = f"{header}:{nonce}".encode()
    proof_hex = hashlib.sha256(preimage).hexdigest()
    return int(proof_hex, 16)


def mine(header, target):
    nonce = 0
    while get_proof(header, nonce) >= target:
        nonce += 1
    return nonce

def mining_demo(header):
    previous_nonce = nonce = 0
    # "proof" need this many leading zero's in the 256-bit representation of 
    # sha256(header + nonce)
    for difficulty_bits in range(2, 30):
        # "proof" needs to be less than this "target"
        # Happens if "target" binary representation has "difficulty_bits" or more leading zeros
        target = 2 ** (256 - difficulty_bits)

        start_time = time.time()
        nonce = mine(header, target)
        elapsed_time = time.time() - start_time

        proof = get_proof(header, nonce)

        target_str = f"{target:.0e}"
        elapsed_time_str = f"{elapsed_time:.0e}" if nonce != previous_nonce else ""
        bin_proof_str = f"{proof:0256b}"[:50]

        print(f"bits: {difficulty_bits:>3} target: {target_str:>7} secs: {elapsed_time_str:>7} nonce: {nonce:>10} proof: {bin_proof_str}...")

        previous_nonce = nonce

if __name__ == "__main__":
    header = "hi"
    mining_demo(header)


    # Binary representation of "target":
    # 256 chars (unless difficulty_bits is 0, in which case 257)
    # difficulty_bits-1 0's, 1, 256-difficulty_bit 0's
    # therefore, if candidate proof has 256-difficulty_bits leading zeros, it's less
    # because it has a zero where "target" has a 1, and zeros to the left

