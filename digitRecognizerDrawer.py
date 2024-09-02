import pygame
import sys
import numpy
import digitRecognizer

import pygame.draw
import helperFunctions


# Functyion to make drawing smoother
def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)







# Initialize Pygame
pygame.init()
imageHeight = 28
imageWidth = 28

# Colors
black  = (0, 0, 0)
white = (255, 255, 255)
lightGrey = (100, 100, 100)
medGrey = (150, 150, 150)
darkGrey = (200, 200, 200)
blue = (0, 0, 255)
red = (255, 0, 0)

network = digitRecognizer.network('mnist_test.csv')
  

# Set the width and height of the window (in pixels)
width, height = imageWidth*10 + 20, imageHeight*10 + 90

# Create a window
window = pygame.display.set_mode((width, height))
window.fill(lightGrey)
pygame.draw.rect(window, black, (10, 10, 280, 280), 0)
pygame.draw.rect(window, black, (0, 300, 300, 60), 0)

# Set the title of the window
pygame.display.set_caption("Digit Recognizer")

# Set up font
font = pygame.font.SysFont(None, 36)  # None uses the default font, 36 is the size

# Create text surface
text_surface = font.render("Draw a digit.", True, red)  # Render the text with anti-aliasing and color
window.blit(text_surface, (20, 320))


# Button properties
button_width = 50
button_height = 20
button_color = darkGrey
button_hover_color = lightGrey
button_text_color = blue
clear_text = "Clear"
submit_text = "Submit"


# Function to draw text on the button
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = pygame.Rect(x, y, button_width, button_height)
    surface.blit(text_obj, text_rect)

# Set up a boolean variable to track if the mouse button is pressed
drawing = False







# Main program loop
while True:

    pos = pygame.mouse.get_pos()
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN: 
                # on enter, submit image
                pixArray = pygame.PixelArray(window)
                temp = []
                for j in range(10, (imageHeight + 1)*10, 10):
                    for i in range(10, (imageHeight + 1)*10, 10):
                        if pixArray[i, j] == 0:
                            temp.append(0)
                        else:
                            temp.append(1)
                
                del pixArray
                guess, z = network.guess(numpy.array(temp))
                guess = helperFunctions.oneHotInverse(guess[-1])
                # Create text surface
                pygame.draw.rect(window, black, (0, 300, 300, 60), 0)
                text_surface = font.render("I see a " + str(guess) + ".", True, red)  # Render the text with anti-aliasing and color
                
                window.blit(text_surface, (20, 320))


            elif event.key == pygame.K_c:
                pygame.draw.rect(window, black, (10, 10, 280, 280), 0)
                pygame.draw.rect(window, black, (0, 300, 300, 60), 0)
                text_surface = font.render("Draw a digit.", True, red)  # Render the text with anti-aliasing and color
                
                window.blit(text_surface, (20, 320))
                

        
        if drawing:
            # Get the current position of the mouse
            posRound = (pygame.mouse.get_pos()[0]//10 * 10, pygame.mouse.get_pos()[1]//10 * 10)
            # Round position
            if 10 < posRound[0] < 280 and 10 < posRound[1] < 280:
                # Draw a small white rectangle at the mouse position
                pygame.draw.rect(window, white, (posRound[0]-10 , posRound[1]-10, 30, 30), 0)

    


    
                
    


                
    

    # Refresh the display
    
    pygame.display.flip()