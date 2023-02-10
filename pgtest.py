import pygame

pygame.init()
pygame.display.init()
window_size = 512
window =pygame.display.set_mode((2*window_size, window_size))
clock = pygame.time.Clock()

canvas = pygame.Surface((2*window_size, window_size))
canvas.fill((255, 255, 255))
pix_square_size = (window_size / 12)
# pygame.draw.rect(canvas, ()
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )

# for x in range(13):
#     pygame.draw.line(canvas, 0, (0, pix_square_size * x), (window_size, pix_square_size * x), width = 1)
#     pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, window_size),width=1)
pygame.event.pump()
pygame.display.update()
clock.tick(4)

