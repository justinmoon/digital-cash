# Needed to import utils from parent dir:
import sys 
sys.path.append('..')

from utils import serialize, deserialize
import socket, socketserver, sys

server_host = "0.0.0.0"
client_host = "127.0.0.1"
port = 10000
server_address = (server_host, port)
client_address = (client_host, port)

def prepare_message(command, data):
    return {
        "command": command,
        "data": data,
    }

class MyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class TCPHandler(socketserver.BaseRequestHandler):

    def respond(self, command, data):
        response = prepare_message(command, data)
        serialized_response = serialize(response)
        self.request.sendall(serialized_response)   
        print(f'Sent {response}')     

    def handle(self):
        while True: 

            message_data = self.request.recv(5000).strip()

            # Need to break early if we got empty bytes,
            # otherwise we get a deserialization error
            # trying to deserialize empty bytes
            if message_data == b"":
                print("Closing connection, bye!")
                break
            
            message = deserialize(message_data)

            print(f'Received {message}')

            if message["command"] == "ping":
                self.respond("pong", "")

def serve():
    server = MyTCPServer(server_address, TCPHandler)
    server.serve_forever()

def send_message(command, data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(client_address)
        message = prepare_message(command, data)
        serialized_message = serialize(message)
        s.sendall(serialized_message)
        message_data = s.recv(5000)
        message = deserialize(message_data)
        print(f'Received {message}')


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "serve":
        serve()
    elif command == "ping":
        send_message("ping", "")        
    else:
        print("python serialized_ping_pong.py <serve|ping>")
