import socket, socketserver, sys

server_host = "0.0.0.0"
client_host = "127.0.0.1"
port = 10000
server_address = (server_host, port)
client_address = (client_host, port)

class MyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        while True:  
            message_bytes = self.request.recv(10).strip()
            print(f'Received {str(message_bytes)}')
            if message_bytes == b"ping":
                self.request.sendall(b"pong")
                print(f'Sent b"pong"')
            if message_bytes == b"":
                print("Closing connection, bye!")
                break

def serve():
    server = MyTCPServer(server_address, TCPHandler)
    server.serve_forever()

def ping():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(client_address)
        s.sendall(b"ping")
        data = s.recv(10)
        print(f'Received {str(data)}')


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "serve":
        serve()
    elif command == "ping":
        ping()
    else:
        print("python ping_pong.py <serve|ping>")
