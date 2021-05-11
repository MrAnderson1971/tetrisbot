from tetris import *
from neuralNet import *
import pickle
import statistics
import time

GLOBAL_MUTATION_RATE = 0.1
POPULATION_SIZE = 128


def main():

    pop = Population(POPULATION_SIZE)

    while True:

        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

        pop.updateAlive()
        time.sleep(0.1)
        if pop.done():
            pop.naturalSelection()


class Population:
    def __init__(self, size):
        self.games = []
        for i in range(size):
            try:
                with open('player.txt', 'rb') as f:
                    n = pickle.load(f)
                self.games.append(Game(Player(n), pieces, False))
                self.naturalSelection()  # Don't want all players behaving the same way when loading a neural net, that's just useless.
            except:
                self.games.append(Game(Player(NeuralNet(BOARD_HEIGHT * BOARD_WIDTH + 7, 138, 7)), pieces, False))  # No of neurons in hidden layers = 2/3 neurons of input layer

        self.games[0].draw_ = True
        self.bestPlayer = self.games[0].player
        self.bestPlayerIndex = 0
        self.generation = 0
        self.bestFitness = 0

    def updateAlive(self):
        """Update all living species."""

        for game in self.games:
            if not game.over:
                try:
                    game.think()
                    game.update()
                except IndexError:
                    game.over = True

    def setBestPlayer(self):
        max_ = 0
        maxIndex = 0
        for i in range(len(self.games)):
            if self.games[i].player.fitness > max_:
                max_ = self.games[i].player.fitness
                maxIndex = i

        self.bestPlayerIndex = maxIndex
        if self.games[self.bestPlayerIndex].player.fitness > self.bestFitness:

            self.bestPlayer = self.games[maxIndex].player.clone()
            self.bestFitness = self.bestPlayer.fitness

            self.bestPlayer.neuralNet.export('player.txt')

    def done(self):
        """True if all specimen are dead."""

        for game in self.games:
            if not game.over:
                return False
        return True

    def printSummary(self):
        print("Gen %s Summary:" % self.generation)
        fitnessList = [game.player.score for game in self.games]
        max_ = max(fitnessList)
        med = statistics.median(fitnessList)
        min_ = min(fitnessList)
        print("Max: %s\t\tMedian: %s\t\tMin: %s" % (max_, med, min_))
        print("Best fitness: %s" % self.bestFitness)

    def naturalSelection(self):

        newGames = [None] * len(self.games)
        self.setBestPlayer()

        self.printSummary()

        for i in range(1, len(self.games)):
            if i < len(self.games) / 2:
                newGames[i] = Game(self.selectPlayer().clone(), pieces, False)
            else:
                newGames[i] = Game(Player(self.selectPlayer().neuralNet.clone().crossover(self.selectPlayer().neuralNet.clone())), pieces, False)
            newGames[i].player.neuralNet.mutate(GLOBAL_MUTATION_RATE)
        newGames[0] = Game(self.bestPlayer.clone(), pieces, True)  # put the best specimen into the next generation so that evolution never regresses

        self.games = newGames[:]
        for game in self.games:
            game.over = False
        self.generation += 1

    def selectPlayer(self):
        """Selects players randomly. Higher fitness means higher chance of being chosen."""

        # get all players into playersByFitness and sort them by fitness
        playersByFitness = [game.player for game in self.games]
        sorted_ = False
        while not sorted_:
            sorted_ = True
            for i in range(len(playersByFitness) - 2):
                if playersByFitness[i].fitness > playersByFitness[i + 1].fitness:
                    temp = playersByFitness[i]
                    playersByFitness[i] = playersByFitness[i + 1]
                    playersByFitness[i + 1] = temp
                    sorted_ = False
        del playersByFitness[:int(len(playersByFitness) / 2)]  # Thanos snaps his fingers, and the worst half of the population is deleted.

        fitnessSum = 0
        for player in playersByFitness:
            fitnessSum += player.fitness
        rand = random.randint(0, round(fitnessSum))
        runningSum = 0

        for player in playersByFitness:
            runningSum += player.fitness
            if runningSum > rand:
                return player
        return playersByFitness[-1]


if __name__ == '__main__':
    main()
