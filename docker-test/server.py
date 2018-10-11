import socketserver, socket, random, os, threading, logging, re


logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)-15s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def ping(hostname):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((hostname, 9999))
        s.sendall(b"ping")
        logger.info(f'Sent "ping" to "{hostname}"')
        data = s.recv(1024*4)
        logger.info(f'Received {str(data)} from "{hostname}"')

class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        message_bytes = self.request.recv(1024*4).strip()
        host_info = socket.gethostbyaddr(self.client_address[0])
        pattern = r"_(.*?)_"
        host = re.search(pattern, host_info[0]).group(1)
        logger.info(f'Received {str(message_bytes)} from "{host}"')
        self.request.sendall(b"pong")
        logger.info(f'Sent b"pong" to "{host}"')


def cron(peer_hostnames):
    def duration():
        return 10 * random.random()

    def ping_peers():
        for hostname in peer_hostnames:
            ping(hostname)
        threading.Timer(duration(), ping_peers, []).start()

    # New blocks every 10 seconds
    threading.Timer(duration(), ping_peers, []).start()

def main():
    peer_hostnames = {p for p in os.environ.get('PEERS', '').split(',') if p}

    cron(peer_hostnames)

    server = socketserver.TCPServer(('0.0.0.0', 9999), TCPHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
