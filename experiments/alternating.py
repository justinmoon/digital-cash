import socketserver, socket, random, os, threading, logging, re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


ID = int(os.environ["ID"])
PORT = 10000
current = 2
peer_hostnames = {p for p in os.environ['PEERS'].split(',')}


def ping(hostname):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((hostname, 10000))
        s.sendall(b"ping")
        logger.info(f'Sent "ping" to "{hostname}"')
        data = s.recv(1000)
        logger.info(f'Received {str(data)} from "{hostname}"')


class TCPHandler(socketserver.BaseRequestHandler):

    def peer(self):
        address = self.client_address[0]
        host_info = socket.gethostbyaddr(address)
        return re.search(r"_(.*?)_", host_info[0]).group(1)

    def handle(self):
        message_bytes = self.request.recv(10)
        logger.info(f'Received {str(message_bytes)} from "{self.peer()}"')
        self.request.sendall(b"pong")
        logger.info(f'Sent b"pong" to "{self.peer()}"')

        schedule_ping()


def ping_peers():
    for hostname in peer_hostnames:
        ping(hostname)
    schedule_ping()

def schedule_ping():
    global current
    current = (current + 1) % 3
    if ID == current:
        threading.Timer(1, ping_peers, []).start()

def main():
    schedule_ping()
    server = socketserver.TCPServer(('0.0.0.0', 10000), TCPHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
