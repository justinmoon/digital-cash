import socket, socketserver, sys, logging

host = "0.0.0.0"
port = 10000
address = (host, port)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        message_bytes = self.request.recv(10).strip()
        logger.info(f'Received {str(message_bytes)}')
        if message_bytes == b"ping":
            self.request.sendall(b"pong")
            logger.info(f'Sent b"pong"')

def serve():
    server = socketserver.TCPServer(address, TCPHandler)
    server.serve_forever()

def ping():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(b"ping")
        data = s.recv(10)
        logger.info(f'Received {str(data)}')


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "serve":
        serve()
    elif command == "ping":
        ping()
    else:
        print("python ping_pong.py <serve|ping>")
