import socket
import sys
import time

HOST, PORT = "localhost", 9999


class MPQueueClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((host, port))
            self.connected = True
            print("Connected!")
        except ConnectionRefusedError:
            self.connected = False
            print("Failed to connect to MPQueueServer at address {} on port {}".format(host, port))

    def start(self, floors_file, walls_file):
        self.sock.sendall(bytes("floor_file {}\n".format(floors_file), "utf-8"))
        self.sock.sendall(bytes("wall_file {}\n".format(walls_file), "utf-8"))
        self.sock.sendall(bytes("start\n", "utf-8"))

    def close(self):
        self.command_shutdown()
        self.sock.close()

    def command_move(self, pos):
        self.sock.sendall(bytes("move {} {}\n".format(pos[0], pos[1]), "utf-8"))

    def command_turn(self, orn):
        self.sock.sendall(bytes("turn {}\n".format(orn), "utf-8"))

    def command_scale(self, scale):
        self.sock.sendall(bytes("scale {} {}\n".format(scale[0], scale[1]), "utf-8"))

    def command_shutdown(self):
        self.sock.sendall(bytes("shutdown\n", "utf-8"))

    def read_variable(self, var):
        self.sock.sendall(bytes("read {}\n".format(var), "utf-8"))
        return self.sock.recv(1024)


if __name__ == "__main__":
    srv = MPQueueClient(HOST, PORT)
    srv.start("/Users/otgaard/Development/dbm/sim/output/test2_floors.obj",
              "/Users/otgaard/Development/dbm/sim/output/test2_walls.obj")
    time.sleep(1)
    data = sys.argv[1:]

    for i in range(10, 300):
        srv.command_move([i, i])

    #srv.command_move(data)

