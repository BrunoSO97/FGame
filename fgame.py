from audioop import add
import pygame
import random
import sys
import os
import neat
from sqlalchemy import false

pygame.init()
screen = pygame.display.set_mode((1280,400))
clock = pygame.time.Clock()
pygame.display.set_caption("FGame")

# Váriaveis
game_speed = 5

# Imagens
player_img = pygame.transform.scale(pygame.image.load("player.png"), (50,50))
rec_img = pygame.image.load("retangulo.png")

class Player():
    """
    Player class representing the green ball
    """

    player = player_img

    def __init__(self, x_pos, y_pos):
        """
        Initialize the object
        :param x_pos: starting x pos (int)
        :param y_pos: starting y pos (int)
        :return: None
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.img = self.player
        self.vel = 5

    def moveup(self):
        """
        make the ball move up
        :return: None
        """
        if self.y_pos > 0:
            self.y_pos -= self.vel
        else:
            pass

    def movedown(self):
        """
        make the ball move down
        :return: None
        """
        if self.y_pos < 400-self.img.get_height():
            self.y_pos += self.vel
        else:
            pass

    def moveright(self):
        """
        make the ball move right (not used in AI)
        :return: None
        """
        if self.x_pos < 1280-self.img.get_width():
            self.x_pos += self.vel
        else:
            pass

    def moveleft(self):
        """
        make the ball move left (not used in AI)
        :return: None
        """
        if self.x_pos > 0:
            self.x_pos -= self.vel
        else:
            pass

    def update(self):
        """
        update de player image on screen
        :return: None
        """
        screen.blit(self.img, (self.x_pos , self.y_pos))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Bar():

    """
    Bar class representing the red bars
    """

    rec = rec_img

    def __init__(self, x_pos, y_pos, vel=5):
        """
        Initialize the object
        :param x_pos: starting x pos (int)
        :param y_pos: starting y pos (int)
        :param vel: starting speed (int)
        :return: None
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.vel = vel
        self.img = self.rec
        self.passed = False
        self.set_y()

    def set_y(self):
        """
        set random y position for new bars
        :return: None
        """
        self.y_pos = random.randrange(-150+10,400-10)

    def move(self):
        """
        move and update the bar on screen
        :return: None
        """
        self.x_pos -= self.vel
        screen.blit(self.img, (self.x_pos , self.y_pos))

    def collide(self, player):
        """
        check the colision with the player
        :return: None
        """
        player_mask = player.get_mask()
        bar_mask = pygame.mask.from_surface(self.img)

        offset = (self.x_pos - player.x_pos, self.y_pos - player.y_pos)

        return player_mask.overlap(bar_mask, offset)

def main(genomes, config):
    
    """
    runs the simulation of the current population of
    players and sets their fitness based on the distance they
    reach in the game.
    """

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # player object that uses that network to play
    nets = []
    ge = []
    plrs = []
    #plr = Player(0,200)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        plrs.append(Player(0,200))
        g.fitness = 0
        ge.append(g)


    speed = 5
    bars = [Bar(1200,0, speed)]

    score = 0


    run = True
    while run:
        screen.fill((255, 255, 255))
        #key_pressed_is = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        bar_ind = 0
        if len(plrs) > 0:
            if len(bars) > 1 and plrs[0].x_pos > bars[0].x_pos: # determine whether to use the first or second bar on screen for neural network input
                bar_ind = 1
        else:
            run = False
            break

        for x, plr in enumerate(plrs): # give each bird a fitness of 0.01 for each frame it stays alive
            plr.update()
            ge[x].fitness += 0.01
            
            # send player location, top and bottom bar location and determine from network whether to move up or down
            output = nets[x].activate((plr.y_pos, abs(plr.y_pos - bars[bar_ind].y_pos), abs(plr.y_pos - bars[bar_ind].y_pos+150)))

            if output[0] > 0.5: # we use a tanh activation function so result will be between -1 and 1. if over 0.5 move up, if under -0.5 move down
                plr.moveup()
            elif output[0] < -0.5:
                plr.movedown()
            
        speed = 5 + score*0.1 # increase the speed of the bars when a bar pass through
        
        rem = []
        add_bar = False
        for bar in bars:
            
            bar.vel = speed
            bar.move()
            
            for x, plr in enumerate(plrs):
                if bar.collide(plr):
                    ge[x].fitness -= 1
                    plrs.pop(x)
                    nets.pop(x)
                    ge.pop(x)

            if bar.x_pos + bar.img.get_width() < 0:
                rem.append(bar)
                bar.passed = True
            elif bar.x_pos + bar.img.get_width() < 640 and len(bars) < 2:
                add_bar = True

        if add_bar:
            bars.append(Bar(1200,0))

        if bars[0].passed:
            score += 1
            for g in ge:
                g.fitness += 5

        for r in rem:
            bars.remove(r)
                
        screen.blit(pygame.font.SysFont("monospace", 16).render(f"Score: {score}", 1, (0,0,0)), (10,10))
        screen.blit(pygame.font.SysFont("monospace", 16).render(f"Speed: {bars[0].vel}", 1, (0,0,0)), (200,10))
        
        melhor_fit = 0
        for g in ge:
            if g.fitness > melhor_fit:
                melhor_fit = g.fitness
        
        screen.blit(pygame.font.SysFont("monospace", 16).render(f"Melhor Fit: {melhor_fit:.2f}", 1, (0,0,0)), (400,10))

        clock.tick(200)
        pygame.display.update()

#main()

def run(config_path):
    """
    runs the NEAT algorithm to train a neural network to play.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 50 generations.
    winner = p.run(main,50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)