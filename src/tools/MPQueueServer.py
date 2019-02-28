# A basic socket server used to provide a connection for the pybullet trainer to push data containing car orientation
# position, etc and use this data to provide additional displays for debugging.

import sys
import socketserver
from tools.MPQueue import MPQueue

MOVE_COMMAND = b"move"
TURN_COMMAND = b"turn"
SCALE_COMMAND = b"scale"
START_COMMAND = b"start"
SHUTDOWN_COMMAND = b"shutdown"
READ_COMMAND = b"read"
FLOOR_FILE = b"floor_file"
WALL_FILE = b"wall_file"


class MPQueueServer(socketserver.BaseRequestHandler):
    def set_floor_file(self, input):
        command, file = input.split(b" ", 1)
        if command != FLOOR_FILE:
            return

        file = file.decode("utf-8")
        print("Setting Floor File:" + file)
        self.server.floor_file = file

    def set_wall_file(self, input):
        command, file = input.split(b" ", 1)
        if command != WALL_FILE:
            return

        file = file.decode("utf-8")
        print("Setting Wall File:" + file)
        self.server.wall_file = file

    def command_start(self, input):
        print("Starting...", input[:len(START_COMMAND)])
        if self.server.started:
            print("Shutdown server before restarting")
            return

        if input[:len(START_COMMAND)] != START_COMMAND:
            return

        if not self.server.floor_file or not self.server.wall_file:
            print("Set floor and wall file first")
            return

        print("Starting Server for:" + self.server.floor_file)
        self.server.queue.run(self.server.floor_file, self.server.wall_file)
        self.server.started = True

    def command_move(self, input):
        """Format: move x y"""
        command, x, y = input.split(b" ")
        if command != MOVE_COMMAND:
            return

        pos = [int(x), int(y)]
        print("Moving:", pos)

        self.server.queue.command_move(pos)

    def command_turn(self, input):
        """Format: turn theta"""
        command, angle = input.split(b" ")
        if command != TURN_COMMAND:
            return

        orn = float(angle)
        self.server.queue.command_turn(orn)

    def command_scale(self, input):
        """Format: scale x y"""
        command, x, y = input.split(b" ")
        if command != SCALE_COMMAND:
            return

        scale = [int(x), int(y)]
        self.server.queue.command_scale(scale)

    def command_shutdown(self):
        if not self.server.started:
            print("Server not started, cannot shutdown")
            return

        self.server.queue.command_shutdown()
        self.server.started = False

    def read_variable(self, input):
        """Format: read var"""
        command, var = input.split(b" ")
        if command != READ_COMMAND:
            return None

        result = self.server.queue.read_variable(var)
        return result

    def handle(self):
        # This function should spawn a handler that operates until disconnection by the client
        data = self.request.recv(1024).strip()

        while len(data) > 0:
            lines = data.split(b"\n")

            for line in lines:
                print(line)
                if line[:len(MOVE_COMMAND)] == MOVE_COMMAND:
                    self.command_move(line)
                elif line[:len(TURN_COMMAND)] == TURN_COMMAND:
                    self.command_turn(line)
                elif line[:len(FLOOR_FILE)] == FLOOR_FILE:
                    self.set_floor_file(line)
                elif line[:len(WALL_FILE)] == WALL_FILE:
                    self.set_wall_file(line)
                elif line[:len(START_COMMAND)] == START_COMMAND:
                    self.command_start(line)
                elif line[:len(SHUTDOWN_COMMAND)] == SHUTDOWN_COMMAND:
                    self.command_shutdown(line)
                    break

            data = self.request.recv(1024).strip()


def start_server(host, port):
    mp = MPQueue()

    with socketserver.TCPServer((host, port), MPQueueServer) as server:
        server.queue = mp
        server.data = None
        server.floor_file = None
        server.wall_file = None
        server.started = False
        server.serve_forever()


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    floors_file = "../output/test2_floors.obj"
    walls_file = "../output/test2_walls.obj"

    start_server(HOST, PORT)