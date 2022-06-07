from audioop import add
import pygame
import random
import sys
import os

pygame.init()
screen = pygame.display.set_mode((1280,400))
clock = pygame.time.Clock()
pygame.display.set_caption("FGame")

# Váriaveis
game_speed = 5

# Imagens
player_img = pygame.transform.scale(pygame.image.load("AI/FGame/player.png"), (50,50))
rec_img = pygame.image.load("AI/FGame/retangulo.png")

class Player():

    player = player_img

    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.img = self.player
        self.vel = 5

    def moveup(self):
        if self.y_pos >= 0:
            self.y_pos -= self.vel
        else:
            pass

    def movedown(self):
        if self.y_pos <= 400-self.img.get_height():
            self.y_pos += self.vel
        else:
            pass

    def moveright(self):
        if self.x_pos <= 1280-self.img.get_width():
            self.x_pos += self.vel
        else:
            pass

    def moveleft(self):
        if self.x_pos >= 0:
            self.x_pos -= self.vel
        else:
            pass

    def update(self):
        screen.blit(self.img, (self.x_pos , self.y_pos))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Bar():

    rec = rec_img

    def __init__(self, x_pos, y_pos, vel=5):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.vel = vel
        self.img = self.rec
        self.passed = False
        self.set_y()

    def set_y(self):
        self.y_pos = random.randrange(0,400-150)

    def move(self):
        self.x_pos -= self.vel
        screen.blit(self.img, (self.x_pos , self.y_pos))

    def collide(self, player):
        player_mask = player.get_mask()
        bar_mask = pygame.mask.from_surface(self.img)

        offset = (self.x_pos - player.x_pos, self.y_pos - player.y_pos)

        return player_mask.overlap(bar_mask, offset)

def main():
    plr = Player(0,200)

    speed = 5
    bars = [Bar(1200,0, speed)]

    score = 0

    game_over = False
    run = True
    while run:

        if game_over:
            screen.fill((255, 255, 255))
            screen.blit(pygame.font.SysFont("monospace", 16).render(f"GAME OVER", 1, (0,0,0)), (640,200))

        else:
            screen.fill((255, 255, 255))
            key_pressed_is = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if key_pressed_is[pygame.K_DOWN]:
                plr.movedown()

            if key_pressed_is[pygame.K_UP]:
                plr.moveup()
            
            if key_pressed_is[pygame.K_RIGHT]:
                plr.moveright()
            
            if key_pressed_is[pygame.K_LEFT]:
                plr.moveleft()
                
            speed = 5 + score*0.1
                    
            plr.update()
            
            rem = []
            add_bar = False
            for bar in bars:
                bar.vel = speed
                bar.move()
                if bar.collide(plr):
                    run = False

                if bar.x_pos + bar.img.get_width() < 0:
                    rem.append(bar)
                    bar.passed = True
                elif bar.x_pos + bar.img.get_width() < 640 and len(bars) < 2:
                    add_bar = True

            if add_bar:
                bars.append(Bar(1200,0))

            if bars[0].passed:
                score += 1

            for r in rem:
                bars.remove(r)
                
            screen.blit(pygame.font.SysFont("monospace", 16).render(f"Score: {score}", 1, (0,0,0)), (10,10))
            screen.blit(pygame.font.SysFont("monospace", 16).render(f"Speed: {bars[0].vel}", 1, (0,0,0)), (200,10))
        
        clock.tick(60)
        pygame.display.update()



#main()

def run():
    pass

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    print("LOcal", local_dir)