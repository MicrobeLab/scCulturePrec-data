from PIL import Image
import pygame
import os
import sys

image_path = sys.argv[1]
output_file = sys.argv[2]

image_for_size = Image.open(image_path)
screen_width, screen_height = image_for_size.size
print(f"Image size: {screen_width} x {screen_height}")

pygame.init()

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drag the left mouse button to frame the required cells in order")

if not os.path.exists(image_path):
    print("Image file does not exist!")
    exit()
image = pygame.image.load(image_path)
image_width, image_height = image.get_rect().size

index = 0

click_coordinates = []

screen.fill((255, 255, 255))  
screen.blit(image, ((screen_width - image_width) // 2, (screen_height - image_height) // 2)) 

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
          
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                index += 1
                print(f"Skipped cell: {index}")

        elif event.type == pygame.MOUSEBUTTONDOWN:  
            if event.button == 1:  
                mouse_x_start, mouse_y_start = pygame.mouse.get_pos()
            elif event.button == 3: 
                mouse_x, mouse_y = pygame.mouse.get_pos()
                index += 1
                manual_info = input("Please enter the information to be noted for this click: ")
                click_coordinates.append((manual_info, mouse_x, mouse_y, None, None))
                print(f"Information: {manual_info} , ", f"Click coordinates: ({mouse_x}, {mouse_y})") 

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_x_end, mouse_y_end = pygame.mouse.get_pos()
                index += 1
                click_coordinates.append((index, mouse_x_start, mouse_y_start, mouse_x_end, mouse_y_end))
                print(f"Cell number: {index} , ", f"Start coordinates: ({mouse_x_start}, {mouse_y_start})  End coordinates:({mouse_x_end}, {mouse_y_end})")
                pygame.draw.rect(screen, (0, 0, 255), (mouse_x_start, mouse_y_start, mouse_x_end - mouse_x_start, mouse_y_end - mouse_y_start), 2)
                pygame.draw.circle(screen, (0, 0, 100), (mouse_x_start, mouse_y_start), 3)
                pygame.draw.circle(screen, (0, 0, 100), (mouse_x_end, mouse_y_end), 3)
                pygame.display.flip()
           
    pygame.display.flip()


with open(output_file, 'w') as file:
    for coord in click_coordinates:
        file.write(f"{coord[0]}\t{coord[1]}\t{coord[2]}\t{coord[3]}\t{coord[4]}\n")

print("Box coordinates have been saved to file:", output_file)

pygame.quit()
