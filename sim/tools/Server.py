# A basic socket server used to provide a connection for the pybullet trainer to push data containing car orientation
# position, etc and use this data to provide additional displays for debugging.

import socketserver
from tools.MPQueue import MPQueue

class MyTCPHandler(socketserver.BaseRequestHandler):
    def command_move(self, input):
        """Format: move 25 10"""
        command, x, y = input.split(b" ")
        if command != b"move":
            return

        pos = [int(x), int(y)]
        self.server.queue.command_move(pos)

    def command_turn(self, input):
        """Format: turn 38.8"""
        command, angle = input.split(b" ")
        if command != b"turn":
            return

        orn = float(angle)
        print("ORN:", orn)
        self.server.queue.command_turn(orn)

    def handle(self):
        print("mp:", self.server.queue)
        self.data = self.request.recv(128).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data, self.data[0:4])

        if self.data[0:4] == b"move":
            print("Move")
            self.command_move(self.data)
        elif self.data[0:4] == b"turn":
            print("Turn")
            self.command_turn(self.data)

        self.request.sendall(self.data.upper())


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    floors_file = "../output/test2_floors.obj"
    walls_file = "../output/test2_walls.obj"

    mp = MPQueue()
    mp.run(floors_file, walls_file)

    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        server.queue = mp
        server.serve_forever()
