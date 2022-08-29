"""
A Button class to represent any buttons in the program.
"""

# Import modules
import pygame as pg
from time import sleep

class Button:
    "Button class."

    def __init__(self, 
                screen: pg.surface, 
                pos: tuple[int], 
                dims: tuple[int], 
                font: pg.font,
                text: str,
                img=None,
                bg_color=(0, 0, 0), 
                text_color=(255, 255, 255), 
                border_radius=5):
        """Initialization method.
        
        Arugments:
            screen
                pygame screen to display contents
            pos
                button position with the position in the center of the button;
                [pos_x, pos_y]
            dims
                button dimensions; [width, height]
            font
                font to render the text for the button
            text
                button text
            img : pg.surface
                button image
            bg_color : tuple[int]
                background color in RGB
            text_color : tuple[int]
                text color in RGB
            border_radius : int
                button border_radius
        """

        self.screen = screen
        self.pos = pos
        self.dims = dims
        self.font = font
        self.text = text
        self.img = img
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_radius = border_radius

        # Whether or not the button is disabled, meaning pressing it won't do
        # anything
        self.disabled = False
        
        # Whether or not the button is pressed
        self.pressed = False

    def draw(self):
        """Draw the button, the text, and or img in it."""

        self.draw_bg()
        if self.text is not None:
            self.draw_text()
        if self.img is not None:
            self.draw_img()

    def draw_bg(self):
        """Draw the button background."""

        # pos_x, pos_y, width, height
        rect = [self.pos[0] - self.dims[0]/2, self.pos[1] - self.dims[1]/2,
                self.dims[0], self.dims[1]]

        pg.draw.rect(self.screen, self.bg_color, 
                     rect, border_radius=self.border_radius)

    def draw_text(self):
        """Draw the button text."""

        # Render the text as a pygame surface
        label = self.font.render(self.text, False, self.text_color)
        label_size = label.get_size()

        # Convert the position from the center to the top left
        label_top_left_pos = (self.pos[0] - label_size[0]/2, 
                              self.pos[1] - label_size[1]/2 + 3)
        
        self.screen.blit(label, label_top_left_pos)

    def draw_img(self):
        """Draw button image."""

        pos = [self.pos[0] - self.img.get_size()[0]/2,
               self.pos[1] - self.img.get_size()[1]/2]
        
        self.screen.blit(self.img, pos)

    def change_img(self, img):
        """Change button image."""

        self.img = img

    def press(self):
        """Press the button."""

        self.pressed = True
    
    def unpress(self):
        """Unpress the button."""

        self.pressed = False

    def is_pressed(self) -> bool:
        """Return whether or not the button is pressed"""

        if not self.disabled:
            # If mouse is left clicked
            left_click = pg.mouse.get_pressed()[0]
            # The mouse position
            mouse_pos = pg.mouse.get_pos()

            # If the mouse is in the button borders
            mouse_in_border = self.pos[0] - self.dims[0]/2 < \
                              mouse_pos[0] < \
                              self.pos[0] + self.dims[0]/2 and \
                              self.pos[1] - self.dims[1]/2 < \
                              mouse_pos[1] < \
                              self.pos[1] + self.dims[1]/2
            
            if left_click and mouse_in_border:
                # Disable for 100 ms
                self.disable(.1)
                return True

        return self.pressed
    
    def disable(self, time: float):
        """Disable the button for a given amount of time.
        
        Arguments:
            time
                amount of time to disable the button for in seconds
        """

        self.disabled = True
        sleep(time)
        self.disabled = False