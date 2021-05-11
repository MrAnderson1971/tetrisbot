import copy
import time
import sys
import pygame
import simplePygame
import statistics
import numpy as np
from pygame.locals import *

from pieces import *

pygame.init()

WIDTH = 640
HEIGHT = 480
BOX_SIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

TICKS_BEFORE_LOCKING = 9
TICS_BEFORE_MOVING = 1

LEFT_MARGIN = int((WIDTH - BOARD_WIDTH * BOX_SIZE) / 2)
TOP_MARGIN = HEIGHT - (BOARD_HEIGHT * BOX_SIZE) - 5

SEED = 1
POPULATION_SIZE = 64
MAX_TIME_BEFORE_LOCK = 30  # Amount of time the player has to lock a piece before game over.

FONT = pygame.font.SysFont(None, 36)

window = pygame.display.set_mode((WIDTH, HEIGHT))


def main():
    # global window

    pygame.display.set_caption('tetris')

    player = Player(None)
    game = Game(player, pieces)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    game.moveLeft = True
                    game.moveRight = False
                if event.key == K_RIGHT:
                    game.moveRight = True
                    game.moveLeft = False
                if event.key == K_UP:
                    game.rotateDirection = 1
                if event.key == K_DOWN:
                    game.rotateDirection = -1
                if event.key == ord('w'):
                    game.rotateDirection = 2
                if event.key == ord('s'):
                    game.rotateDirection = -2
                if event.key == K_SPACE:
                    game.swap()
            if event.type == KEYUP:
                if event.key == K_LEFT:
                    game.moveLeft = False
                if event.key == K_RIGHT:
                    game.moveRight = False
                if event.key == K_UP or event.key == K_DOWN or event.key == ord('w') or event.key == ord('s'):
                    game.rotateDirection = 0

        game.update()
        if game.over:
            terminate()
        time.sleep(0.1)


def terminate():
    pygame.quit()
    sys.exit()


def convertToPixelCoords(x, y):
    """Converts board coords to window coords."""
    return (LEFT_MARGIN + (x * BOX_SIZE)), (TOP_MARGIN + (y * BOX_SIZE))


class Player:
    def __init__(self, neuralNet):
        self.neuralNet = neuralNet
        self.fitness = 0
        self.score = 0

        self.moveRight = self.moveLeft = self.rotateUp = self.rotateDown = False

    def clone(self):
        return Player(self.neuralNet.clone())


class Game:
    def __init__(self, player, pieces_, draw=True):
        self.player = player
        self.board = np.zeros(shape=(BOARD_WIDTH, BOARD_HEIGHT))  # .tolist()
        self.over = False  # game over

        # Take tetriminos from a list so that all games are the same.
        self.refill = pieces_
        self.piecesQueue = self.refill[:]
        self.currentPiece = Tetrimino(self.piecesQueue.pop(0), self.board)
        self.nextPiece = Tetrimino(self.piecesQueue.pop(0), self.board)
        self.swapPiece = Tetrimino(self.piecesQueue.pop(0), self.board)
        # self.currentPiece = Tetrimino(random.choice(list(PIECES)), self.board)
        # self.nextPiece = Tetrimino(random.choice(list(PIECES)), self.board)
        # self.swapPiece = Tetrimino(random.choice(list(PIECES)), self.board)

        self.lockedPieces = []

        self.draw_ = draw

        self.moveRight = self.moveLeft = self.rotateUp = self.rotateDown = False
        self.ticksToAutoLock = 0

        self.heights = []
        self.emptySpaces = []
        self.rotateDirection = 0

    def think(self):
        inputs = []
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                inputs.append(self.board[x][y])
        inputs.append(ord(self.currentPiece.char) - 97)
        inputs.append(self.currentPiece.x)
        inputs.append(self.currentPiece.y)
        inputs.append(self.currentPiece.rotation)
        inputs.append(ord(self.nextPiece.char) - 97)
        inputs.append(ord(self.swapPiece.char) - 97)
        inputs.append(int(MAX_TIME_BEFORE_LOCK - self.ticksToAutoLock))

        outputs = self.player.neuralNet.output(inputs)

        self.moveRight = bool(round(outputs[0][0]))
        self.moveLeft = bool(round(outputs[1][0]))
        self.rotateDown = bool(round(outputs[2][0]))
        self.rotateUp = bool(round(outputs[3][0]))

        _ = max(outputs[2][0], outputs[3][0], outputs[4][0], outputs[5][0], outputs[6][0])
        self.rotateDirection = 2

        if _ == outputs[2][0]:
            self.rotateDirection = -2
        elif _ == outputs[3][0]:
            self.rotateDirection = -1
        elif _ == outputs[4][0]:
            self.rotateDirection = 0
        elif _ == outputs[5][0]:
            self.rotateDirection = 1
        # if round(outputs[4][0]) == 1:
        #     self.swap()

    def move(self, left, right, direction):
        self.currentPiece.moveLeft = left
        self.currentPiece.moveRight = right
        self.currentPiece.rotate(direction)

    def attemptMoveDown(self):
        overlapping = False
        while not overlapping:
            for c in self.currentPiece.currentCoords:
                try:
                    if self.board[c[0]][c[1] + 1] != 0:
                        overlapping = True
                        break
                except IndexError:
                    overlapping = True
                    break
            if not overlapping:
                for c in self.currentPiece.currentCoords:
                    c[1] += 1
                    self.currentPiece.y += 1
        self.currentPiece.locked = True

    def update(self):
        self.move(self.moveLeft, self.moveRight, self.rotateDirection)

        self.currentPiece.update()

        # If too much time passed move it down and lock it.
        self.ticksToAutoLock += 1
        if self.ticksToAutoLock > MAX_TIME_BEFORE_LOCK:
            self.currentPiece.locked = True

        # get new piece
        if self.currentPiece.locked:
            self.ticksToAutoLock = 0
            self.attemptMoveDown()
            for c in self.currentPiece.currentCoords:
                if self.player.score > 64:
                    print(c, self.currentPiece.char)
                if c[1] < 0:
                    self.over = True
                    self.calculateFitness()
                self.board[c[0]][c[1]] = 1  # add locked piece to coordinate
                self.lockedPieces.append(LockedPiece(self.board, self.currentPiece.color, c))

            if len(self.piecesQueue) < 10:
                self.piecesQueue = self.refill[:]

            self.heights.extend([BOARD_HEIGHT - c[1] for c in self.currentPiece.currentCoords])

            # Calculate amount of empty spaces this piece covers.
            for c in self.currentPiece.currentCoords:
                y = c[1] + 1
                currentEmptySpaces = 0
                try:
                    while self.board[c[0]][y] == 0:
                        currentEmptySpaces += 1
                        y += 1
                except IndexError:
                    pass
                self.emptySpaces.append(currentEmptySpaces)

            self.currentPiece = self.nextPiece
            # for c in self.currentPiece.currentCoords:
            #     if self.board[c[0]][c[1]] != 0:
            #         self.over = True
            self.nextPiece = Tetrimino(self.piecesQueue.pop(0), self.board)

            # calculate max height of pieces
            # heights = []
            # for x in range(BOARD_WIDTH):
            #     for y in range(BOARD_HEIGHT):
            #         if self.board[x][y] != 0:
            #             heights.append(y)
            # self.heights.append(BOARD_HEIGHT - min(heights))

        self.player.score += self.clearRows()

        if self.draw_:
            self.draw()

    def calculateFitness(self):

        for y in range(BOARD_HEIGHT):
            currentRow = 0
            for x in range(BOARD_WIDTH):
                if self.board[x][y] != 0:
                    currentRow += 1
            self.player.fitness += (currentRow / BOARD_WIDTH) ** 2
        if len(self.heights) > 1:
            try:
                self.player.fitness /= statistics.mean(
                    self.heights) / 4  # apply height penalty to discourage ai from building too tall
            except ZeroDivisionError:
                pass
        if len(self.emptySpaces) > 1:
            try:
                self.player.fitness /= statistics.mean(self.emptySpaces) / 2  # penalize empty spaces
            except ZeroDivisionError:
                pass

        self.player.fitness += 100 * self.player.score

    def draw(self):
        window.fill(COLORS['black'])
        for c in self.currentPiece.currentCoords:
            x_, y_ = convertToPixelCoords(c[0], c[1])
            pygame.draw.rect(window, self.currentPiece.color, pygame.Rect(x_, y_, BOX_SIZE, BOX_SIZE))

        # Draw each locked piece.
        for piece in self.lockedPieces:
            x_, y_ = convertToPixelCoords(piece.x, piece.y)
            color_ = piece.color
            pygame.draw.rect(window, color_, pygame.Rect(x_, y_, BOX_SIZE, BOX_SIZE))

        simplePygame.printTextLeft('Score: %s' % self.player.score, FONT, window, 0, 0, COLORS['white'])
        simplePygame.printTextLeft('Next Piece: ' + self.nextPiece.char, FONT, window, 0, 50, COLORS['white'])
        simplePygame.printTextLeft('Swap: ' + self.swapPiece.char, FONT, window, 0, 100, COLORS['white'])
        simplePygame.printTextLeft('Time left: %s' % (MAX_TIME_BEFORE_LOCK - self.ticksToAutoLock),
                                   FONT, window, 0, 150, COLORS['white'])
        pygame.draw.rect(window, COLORS['white'],
                         pygame.Rect(LEFT_MARGIN, TOP_MARGIN, BOARD_WIDTH * BOX_SIZE, BOARD_HEIGHT * BOX_SIZE), 1)
        pygame.display.update()

    def clearRows(self):
        """Clears filled rows, then returns score."""

        filledRows = []
        for row in range(BOARD_HEIGHT):
            if all([self.board[col][row] == 1 for col in range(BOARD_WIDTH)]):
                filledRows.append(row)

        for y in range(BOARD_HEIGHT, 0, -1):
            if y in filledRows:

                for piece in self.lockedPieces[:]:
                    if piece.y == y:
                        self.lockedPieces.remove(piece)
                for piece in self.lockedPieces:
                    if piece.y < y:
                        piece.y += 1

                for y_ in range(y, 0, -1):
                    for x in range(BOARD_WIDTH):
                        self.board[x][y_] = self.board[x][y_ - 1]

                    for x in range(BOARD_WIDTH):
                        self.board[x][0] = 0

        return len(filledRows) ** 2

    def swap(self):
        """Swaps the current Tetrimino with the swap piece."""

        temp = self.swapPiece
        self.swapPiece = Tetrimino(self.currentPiece.char, self.board)
        self.currentPiece = Tetrimino(temp.char, self.board)


class Tetrimino:
    def __init__(self, shape, board):
        self.board = board
        self.shape = copy.deepcopy(PIECES[shape])
        self.char = shape
        self.rotation = 0

        self.currentShape = self.shape[self.rotation % len(self.shape)]
        self.x = int(BOARD_WIDTH / 2)
        self.y = 0
        self.color = TETRIMINO_COLORS[shape]

        self.locked = False  # if the piece is locked
        self.ticksBeforeLocking = TICKS_BEFORE_LOCKING  # ticks before the piece is locked if it hits something
        self.ticksBeforeMoving = TICS_BEFORE_MOVING

        self.currentCoords = []
        for c in self.currentShape:
            x = c[0] + self.x
            y = c[1] + self.y
            self.currentCoords.append([x, y])

        self.moveRight = self.moveLeft = False

    def move(self):
        delta = 0
        if self.moveRight:
            delta = 1
        if self.moveLeft:
            delta = -1

        if delta != 0:
            overlapping = False
            copy_ = copy.deepcopy(self.currentCoords)

            for c in copy_:  # prevent piece from overlapping or going out of bounds
                c[0] += delta
                if c[0] < 0 or c[0] >= BOARD_WIDTH:
                    overlapping = True
                    break
                if self.board[c[0]][c[1]] != 0:
                    overlapping = True
                    break

            if not overlapping:
                for c in self.currentCoords:
                    c[0] += delta  # move the piece
                    self.ticksBeforeLocking = TICKS_BEFORE_LOCKING
                self.x += delta

        # # Prevent piece from going out of bounds while moving horizontally.
        # while any([c[0] < 0 for c in self.currentCoords]):
        #     for c1 in self.currentCoords:
        #         c1[0] += 1
        #     self.x += 1
        # while any([c[0] >= BOARD_WIDTH for c in self.currentCoords]):
        #     for c1 in self.currentCoords:
        #         c1[0] -= 1
        #     self.x -= 1

        self.ticksBeforeMoving -= 1
        canMoveDown = True
        for c in self.currentCoords:
            try:  # lock the piece
                if c[1] + 1 == BOARD_HEIGHT or self.board[c[0]][c[1] + 1] != 0:
                    canMoveDown = False
                    break
            except IndexError as e:
                pass

        if self.ticksBeforeMoving <= 0:
            if canMoveDown:  # Move the piece down if there is an available space.
                for c in self.currentCoords:
                    c[1] += 1
                self.y += 1
            else:
                self.ticksBeforeLocking -= 1
            self.ticksBeforeMoving = TICS_BEFORE_MOVING

    def update(self):
        if self.ticksBeforeLocking <= 0:
            self.locked = True

        if not self.locked:
            self.move()

    def rotate(self, direction):
        if direction == 0:
            return
        self.rotation += direction

        currentShapeCopy = self.shape[self.rotation % len(self.shape)]
        currentCoordsCopy = []

        # Make a copy of the current Tetrimino, rotate it, and see that the rotation is valid
        # before applying it to the real thing.

        # currentShapeCopy = self.shape[self.rotation % len(self.shape)]
        # currentCoordsCopy = []
        # for c in currentShapeCopy:
        #     x = c[0] + self.x
        #     y = c[1] + self.y
        #     currentCoordsCopy.append([x, y])
        # for c in currentCoordsCopy:
        #     try:
        #         if self.board[c[0]][c[1]] != 0:
        #             return  # can't rotate
        #     except IndexError:
        #         return  # also can't rotate
        #
        # self.currentShape = currentShapeCopy
        # self.currentCoords = currentCoordsCopy
        for c in currentShapeCopy:
            x = c[0] + self.x
            y = c[1] + self.y
            currentCoordsCopy.append([x, y])

        # If tetrimino goes out of bounds, move it
        while any([c[1] >= BOARD_HEIGHT for c in currentCoordsCopy]):
            for c1 in currentCoordsCopy:
                c1[1] -= 1
            self.y -= 1

        while any([c[0] < 0 for c in currentCoordsCopy]):
            for c1 in currentCoordsCopy:
                c1[0] += 1
            self.x += 1

        while any([c[0] >= BOARD_WIDTH for c in currentCoordsCopy]):
            for c1 in currentCoordsCopy:
                c1[0] -= 1
            self.x -= 1

        # If tetrimino overlaps with locked piece, move it up.
        # Separate from previous while loop to prevent IndexError.
        if any([self.board[c[0]][c[1]] == 1 for c in currentCoordsCopy]):
            return False

        self.ticksBeforeLocking = TICKS_BEFORE_LOCKING
        self.currentShape = currentShapeCopy
        self.currentCoords = currentCoordsCopy
        return True
        # while any([self.board[c[0]][c[1]] == 1 for c in self.currentCoords]):
        #     out = False
        #     for c1 in self.currentCoords:
        #         if c1[1] - 1 < 0:
        #             out = True
        #             break
        #
        #     if not out:
        #         for c1 in self.currentCoords:
        #             c1[1] -= 1
        #         self.y -= 1
        #     else:
        #         break


class LockedPiece:
    def __init__(self, board, color_, coords):
        self.board = board
        self.color = color_
        self.coords = coords

        self.x = self.coords[0]
        self.y = self.coords[1]


if __name__ == '__main__':
    main()
