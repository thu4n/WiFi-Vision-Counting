import socket

def setup_server(ip, port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the address and port
    server_socket.bind((ip, port))

    # Listen for incoming connections
    server_socket.listen(1)
    print("Server listening on {}:{}".format(ip, port))

    # Accept a connection
    conn, addr = server_socket.accept()
    print("Connected by", addr)
    
    return conn, addr

def setup_client(server_ip, server_port):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((server_ip, server_port))
    print("Connected to server at {}:{}".format(server_ip, server_port))
    
    return client_socket

# Example usage
# server_ip = '192.168.1.10'
# server_port = 12345
# conn, addr = setup_server(server_ip, server_port)