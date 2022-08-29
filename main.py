"""
A Pygame drawing program to detect which base 10 digits (0-9) are being drawn
using a neural network model trained with TensorFlow.
"""

__author__ = "Firdavs Nasriddinov"
__version__ = 1.0

# Import modules
from button import Button
import numpy as np
import pickle
import pygame as pg
import sys
import warnings

# Disable warnings caused by the knn classifier
warnings.simplefilter(action='ignore', category=FutureWarning)

# CONSTANTS
# Directory for classifiers and icons used in GUI
DIRS = {'tf_mnist': 'tf_models_mnist/tf_model_final.pkl',
        'tf_nist': 'tf_models_nist/tf_model_final.pkl',
        'dark_icon': 'icons/dark.png',
        'light_icon': 'icons/light.png'}

# Dark mode icon
DARK_ICON = pg.image.load(DIRS['dark_icon'])
DARK_ICON = pg.transform.scale(DARK_ICON, (40, 40))

print(type(DARK_ICON))

# Light mode icon
LIGHT_ICON = pg.image.load(DIRS['light_icon'])
LIGHT_ICON = pg.transform.scale(LIGHT_ICON, (40, 40))

# Dictionary of all colors used in program
COLORS = {'white': (255, 255, 255), 'black': (0, 0, 0), 
          'gray': (128, 128, 128), 'light-gray': (240, 240, 240),
          'dark-gray': (60, 60, 60)}

# Grid line width
LINE_WIDTH = 1

FPS = 300

class OHDR:
    """Class to represent the Pygame program."""

    def __init__(self, dataset_type):
        """Initialization method.
        
        Arguments:
            dataset_type
                which dataset the model was trained on; nist or mnist
        """

        self.dataset_type = dataset_type
        
        # Setting the number of pixels based on the dataset type
        if dataset_type == 'nist':
            self.num_pixels = 128
        elif dataset_type == 'mnist':
            self.num_pixels = 28

        # Size of each grid box in pixels
        self.box_size = int(800/self.num_pixels)

        # Screen dimensions
        self.height, self.width = 1000, self.box_size*self.num_pixels 

        pg.init()
        pg.font.init()
        pg.display.set_caption('Optical Handwritten Digits Recognition')

        # Pygame screen to display contents
        self.screen = pg.display.set_mode([self.width, self.height])

        # Pygame clock to keep a constant fps
        self.clock = pg.time.Clock()

        # Size 20 font
        self.font_20 = pg.font.SysFont("Helvetica", 20)

        # Size 40 font
        self.font_40 = pg.font.SysFont("Helvetica", 40)

        # Mumpy array to represent each pixel (grid box) value
        self.grid = np.zeros([self.num_pixels, self.num_pixels], dtype=np.uint8)

        # Clear button
        self.clear_button = Button(self.screen,
                                   pos=(400, 830),
                                   dims=(200, 50),
                                   font=self.font_40,
                                   text='CLEAR',
                                   bg_color=COLORS['gray'],
                                   text_color=COLORS['white'])
        
        # Toggle grid button
        self.toggle_grid_button = Button(self.screen,
                                         pos=(200, 830),
                                         dims=(150, 30),
                                         font=self.font_20,
                                         text='TOGGLE GRID',
                                         bg_color=COLORS['gray'],
                                         text_color=COLORS['white'])
        
        # Toggle dark mode button
        self.toggle_dark_button = Button(self.screen,
                                         pos=(600, 830),
                                         dims=(50, 50),
                                         font=self.font_20,
                                         text=None,
                                         img=DARK_ICON,
                                         bg_color=COLORS['gray'],
                                         text_color=COLORS['white'])

        self.guess = None           # which digit has been guessed
        self.is_reset = True        # whether or not grid has been reset
        self.show_grid = False      # whether to show grid lines
        self.mode = 'light'         # light or dark mode

        # Initializing the classifier
        self.classifier = None

        # Loading in requested classifier based on dataset type
        file_dir = DIRS[f'tf_{dataset_type}']
        self.classifier = self.get_pickle_file(file_dir)

    def get_pickle_file(self, file_dir):
        """Retrieve pickle file from directory.
        
        Returns:
            file
                pickle file
        """

        with open(file_dir, "rb") as f:
            return pickle.load(f)

    def draw_grid_lines(self):
        """Draw the grid lines on the screen."""

        # Draw a white background
        if self.mode == 'light':
            self.screen.fill(COLORS['white'])
        elif self.mode == 'dark':
            self.screen.fill(COLORS['dark-gray'])

        if self.show_grid:
            # Display each grid line
            for x in range(0, self.num_pixels*self.box_size + 1, self.box_size):
                pg.draw.line(self.screen, COLORS['light-gray'], (x, 0), 
                            (x, self.num_pixels*self.box_size), LINE_WIDTH)
            
            for y in range(0, self.num_pixels*self.box_size + 1, self.box_size):
                pg.draw.line(self.screen, COLORS['light-gray'], (0, y), 
                            (self.num_pixels*self.box_size, y), LINE_WIDTH)
            
        # Draw a divisor line at the bottom of the grid 
        pg.draw.line(self.screen, COLORS['light-gray'], (0, self.width), 
                            (self.width, self.width), LINE_WIDTH)

    def is_mouse_pressed(self) -> bool:
        """Return whether or not left click has been pressed.
        
        Returns:
            mouse_pressed
                whether or not left click has been pressed
        """

        return pg.mouse.get_pressed()[0]

    def get_mouse_pos(self) -> list[int]:
        """Return mouse position on screen.
        
        Returns:
            mouse_pos
                mouse position coordinates; [pos_x, pos_y]
        """

        return pg.mouse.get_pos()

    def is_mouse_in_borders(self) -> bool:
        """Return whether or not the mouse is in the borders of the grid."""

        mouse_pressed = self.is_mouse_pressed()
        mouse_pos = self.get_mouse_pos()

        return mouse_pressed and \
            mouse_pos[0] <= self.num_pixels*self.box_size and \
                mouse_pos[1] <= self.num_pixels*self.box_size
    
    def color_new_pixels(self):
        """Color new pixels in the grid."""

        # Make sure grid is being colored on
        if self.is_mouse_in_borders():
            mouse_pos = self.get_mouse_pos()

            # Position of the center of the pixels
            center_x = int(mouse_pos[0] / self.box_size)
            center_y = int(mouse_pos[1] / self.box_size)

            self.is_reset = False
            
            # Set the diameter of brush circle based on dataset type
            if self.dataset_type == 'mnist':
                r = 2
            elif self.dataset_type == 'nist':
                r = 7

            for i in range(-int(r/2), int(r/2)+1):
                for j in range(-int(r/2), int(r/2)+1):
                    val = 255
                    if (i**2 + j**2) > int(r/2)**2 + 1:
                        val = 0

                    try:
                        self.grid[center_y + i, center_x + j] += val
                        if self.grid[center_y + i, center_x + j] > 255:
                            self.grid[center_y + i, center_x + j] = 255
                    # Incase the 3x3 square is not in the boundaries of grid
                    except IndexError:
                        pass

    def draw_pixels(self):
        """Draw all pixels on the grid."""

        for y in range(self.num_pixels):
            for x in range(self.num_pixels):
                # Value of pixel at (x, y)
                val = self.grid[y, x]

                # Only display pixels that are black
                # NOTE: gray-scale values inverted: 0 is white and 255 is black
                if val > 0:
                    # Invert gray-scale values if on light-mode
                    if self.mode == 'light':
                        val = 255 - val
                    
                    # Display pixel as a pygame rectangle
                    pg.draw.rect(self.screen, (val, val, val), 
                            (x * self.box_size, y * self.box_size, 
                                self.box_size, self.box_size))
    
    def reset_grid(self):
        """Reset grid."""

        # Reset all grid pixel values to 0 (white)
        self.grid = np.zeros([self.num_pixels, self.num_pixels])

        self.guess = None
        self.is_reset = True

    def make_guess(self):
        """Make guess using TF model."""

        # Copy grid to new array
        new_grid = self.grid.copy()
        # Swap gray-scale values if using nist dataset
        # NOTE: values in mnist and nist are swapped (idek why)
        if self.dataset_type == 'nist':
            new_grid = 255 - new_grid

        # Reshape grid to be compatible with model
        new_grid = new_grid.reshape(1, self.num_pixels, self.num_pixels)
        
        # Make prediction
        # NOTE: verbose=0 means don't print prediction process to terminal
        pred = self.classifier.predict(new_grid, verbose=0)

        # Only set guess if there are any pixels on screen and if there is 
        # even a valid prediction 
        if not self.is_reset and 1 in pred[0]:
            self.guess = list(pred[0]).index(1)

    def display_guess(self):
        """Display guessed digit on screen."""

        if self.guess is not None:
            guess_pos = (400, 900)

            # Render text to display
            guess_text = self.font_40.render(f'Guess: {self.guess}', False,
                                        COLORS['gray'])
            
            self.screen.blit(guess_text, 
                            (guess_pos[0] - guess_text.get_size()[0]/2,
                             guess_pos[1] - guess_text.get_size()[1]/2))

    def run(self):
        """Main run loop."""

        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            self.draw_grid_lines()
            self.draw_pixels()
            self.color_new_pixels()
            self.display_guess()
            self.make_guess()

            self.clear_button.draw()
            self.toggle_grid_button.draw()
            self.toggle_dark_button.draw()

            # Reset grid once clear button is pressed
            if self.clear_button.is_pressed():
                self.reset_grid()

            # Toggle grid 
            if self.toggle_grid_button.is_pressed():
                if self.show_grid:
                    self.show_grid = False
                else:
                    self.show_grid = True
            
            # Toggle dark mode
            if self.toggle_dark_button.is_pressed():
                if self.mode == 'light':
                    self.mode = 'dark'
                    self.toggle_dark_button.change_img(LIGHT_ICON)
                elif self.mode == 'dark':
                    self.mode = 'light'
                    self.toggle_dark_button.change_img(DARK_ICON)

            pg.display.update()
            self.clock.tick(FPS)

        pg.quit()

def check_command_line_args(args):
    if len(args) == 1:
        sys.exit('ERROR: Please enter dataset type in command line arguments.')
    elif len(args) > 2:
        sys.exit('ERROR: Please only enter dataset type in command line \
arguments.')

    if args[1] not in ['mnist', 'nist']:
        sys.exit("ERROR: Please enter dataset type as 'mnist' or 'nist'.")


def main():
    """Main function to run program."""

    # Parse command-line arguments
    args = sys.argv

    # Check for any errors in args
    check_command_line_args(args)
    
    # Index dataset type from args
    dataset_type = args[1]

    ohdr = OHDR(dataset_type)
    ohdr.run()

if __name__ == '__main__':
    main()