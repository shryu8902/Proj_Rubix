import pygame
import numpy as np
pygame.init()
pygame.display.init()
n_width = 12
n_height = 9
n_pixel = 100
WIDTH = n_width * n_pixel
HEIGHT = n_height * n_pixel
WINDOW =pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RubiXube")
clock = pygame.time.Clock()

RGBDict ={0: ('W', (255, 255, 255)),
                       1: ('O', (255, 165, 0)),
                       2: ('G', (0, 128, 0)),
                       3: ('R', (255, 0, 0)),
                       4: ('B', (0, 0, 255)),
                       5: ('Y', (255, 255, 0))}

cube = np.arange(6,dtype=np.int8)[:,np.newaxis,np.newaxis]*np.ones([6,3,3],dtype=np.int8)

canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill((230, 230, 230))
origins = {'U':(n_pixel*3,0),'L':(0,n_pixel*3),'F':(n_pixel*3, n_pixel*3), 'R': (n_pixel*6, n_pixel*3), 'B':(n_pixel*9, n_pixel*3),
'D':(n_pixel*3, n_pixel*6)}
for i, face in enumerate(['U','L','F','R','B','D']):
    for x in range(3):
        for y in range(3):
            xloc, yloc = origins[face]
            xloc += x*n_pixel
            yloc += y*n_pixel
            pygame.draw.rect(canvas,  RGBDict[cube[i][x,y]][1],
            [xloc, yloc, n_pixel, n_pixel]
            )
            pygame.draw.rect(canvas, 
            (0,0,0),
            [xloc, yloc, n_pixel, n_pixel],2
            )
WINDOW.blit(canvas, canvas.get_rect())
pygame.event.pump()
pygame.display.update()
clock.tick(20)

# pygame.quit()
