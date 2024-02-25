import pygame
from pong import Game
import neat
import os
import pickle

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left = True, up = True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left = True, up = False)
            
            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left = False, up = True)
            else:
                self.game.move_paddle(left = False, up = False)

            game_info = self.game.loop()
            #print(game_info.left_score, game_info.right_score)
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()
    def train_ai(self, genome1, config, var):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output1 = net1.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x))) 
            decision1 = output1.index(max(output1))
            
            if decision1 == 0:
                pass
            elif decision1 == 1:
                var += 0.000006
                self.game.move_paddle(left = False, up = True)
            else:
                var += 0.000006
                self.game.move_paddle(left = False, up = False)
            
            if self.game.ball.y < self.game.left_paddle.y:
                self.game.move_paddle(left = True, up = True)
            elif self.game.ball.y > self.game.left_paddle.y + self.game.left_paddle.HEIGHT:
                self.game.move_paddle(left = True, up = False)

            game_info = self.game.loop()
            self.game.draw(draw_score = False, draw_hits = True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.right_hits > 50:
                self.calculate_fitness(genome1, game_info, var)
                break
    def calculate_fitness(self, genome1, game_info, var):
        genome1.fitness -= var
        genome1.fitness += game_info.right_hits

def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        genome1.fitness = 0
        var = 0
        game = PongGame(window, width, height)
        game.train_ai(genome1, config, var)

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7')
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27') -> in cazul in care vrei sa te intorci la o generatie mai veche
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)
    with open("best1.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    game = PongGame(window, width, height)
    game.test_ai(winner, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # run_neat(config)
    test_ai(config)