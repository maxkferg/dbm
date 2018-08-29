# A basic socket server used to provide a connection for the pybullet trainer to push data containing car orientation
# position, etc and use this data to provide additional displays for debugging.

import sys
import socketserver
from tools.MPQueue import MPQueue


class MPQueueServer(socketserver.BaseRequestHandler):
    def command_move(self, input):
        """Format: move x y"""
        command, x, y = input.split(b" ")
        if command != b"move":
            return

        pos = [int(x), int(y)]
        self.server.queue.command_move(pos)

    def command_turn(self, input):
        """Format: turn theta"""
        command, angle = input.split(b" ")
        if command != b"turn":
            return

        orn = float(angle)
        self.server.queue.command_turn(orn)

    def command_scale(self, input):
        """Format: scale x y"""
        command, x, y = input.split(b" ")
        if command != b"scale":
            return

        scale = [int(x), int(y)]
        self.server.queue.command_scale(scale)

    def read_variable(self, input):
        """Format: read var"""
        command, var = input.split(b" ")
        if command != b"read":
            return

        result = self.server.queue.read_variable(var)
        return result

    def handle(self):
        data = self.request.recv(128).strip()

        if data[0:4] == b"move":
            self.command_move(data)
        elif data[0:4] == b"turn":
            self.command_turn(data)

        self.request.sendall(data.upper())


def start_server(host, port, floors_file, walls_file):
    mp = MPQueue()
    mp.run(floors_file, walls_file)

    with socketserver.TCPServer((host, port), MPQueueServer) as server:
        server.queue = mp
        server.serve_forever()


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    floors_file = "../output/test2_floors.obj"
    walls_file = "../output/test2_walls.obj"

    start_server(HOST, PORT, floors_file, walls_file)