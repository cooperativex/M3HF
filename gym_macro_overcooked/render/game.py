import os
import sys

import pygame
import numpy as np
from .utils import *
from ..items import Tomato, Lettuce, Plate, Knife, Delivery, Agent, Food

graphics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'graphics'))
_image_library = {}

ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}

def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        # Load image with alpha to detect transparent areas
        raw_image = pygame.image.load(canonicalized_path).convert_alpha()
        
        # Create a new surface without alpha 
        image = pygame.Surface(raw_image.get_size()).convert()
        
        # Fill with a unique transparent color (bright magenta)
        transparent_color = (255, 0, 255)
        image.fill(transparent_color)
        
        # Blit the original image, but only non-transparent pixels
        for x in range(raw_image.get_width()):
            for y in range(raw_image.get_height()):
                pixel = raw_image.get_at((x, y))
                if pixel[3] > 128:  # If alpha > 128, consider it opaque
                    image.set_at((x, y), pixel[:3])  # Copy RGB only
        
        # Set magenta as transparent
        image.set_colorkey(transparent_color)
        _image_library[path] = image
    return image


class Game:
    def __init__(self, env):
        self._running = True
        self.env = env

        # Visual parameters
        self.scale = 80   # num pixels per tile
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.width = self.scale * self.env.xlen
        self.height = self.scale * self.env.ylen
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple((self.holding_scale * np.asarray(self.tile_size)).astype(int))
        self.container_size = tuple((self.container_scale * np.asarray(self.tile_size)).astype(int))
        self.holding_container_size = tuple((self.container_scale * np.asarray(self.holding_size)).astype(int))

        pygame.init()

    def on_init(self):
        pygame.init()
        if self.play:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.width, self.height))
        self._running = True


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_render(self):
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill(Color.FLOOR)
        for x in range(self.env.xlen):
            for y in range(self.env.ylen):
                sl = self.scaled_location((y, x))
                if self.env.map[x][y] == ITEMIDX["counter"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                elif self.env.map[x][y] == ITEMIDX["delivery"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)                    
                    pygame.draw.rect(self.screen, Color.DELIVERY, fill)
                    self.draw('delivery', self.tile_size, sl)
                    for k in self.env.delivery:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["knife"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)    
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('cutboard', self.tile_size, sl)
                    for k in self.env.knife:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["tomato"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for t in self.env.tomato:
                        if t.x == x and t.y == y:
                            self.draw(t.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["lettuce"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.lettuce:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["onion"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.onion:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["plate"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('plate', self.tile_size, sl)
                    for p in self.env.plate:
                        if p.x == x and p.y == y:
                            if p.containing:
                                self.draw(p.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["agent"]:
                    for agent in self.env.agent:
                        if agent.x == x and agent.y == y:
                            self.draw('agent-{}'.format(agent.color), self.tile_size, sl)
                            if agent.holding:
                                if isinstance(agent.holding, Plate):
                                    self.draw('plate', self.holding_size, self.holding_location((y, x)))
                                    if agent.holding.containing:
                                        self.draw(agent.holding.containedName, self.holding_container_size, self.holding_container_location((y, x)))
                                else:
                                    self.draw(agent.holding.name, self.holding_size, self.holding_location((y, x)))
        pygame.display.flip()
        pygame.display.update()

        # Use surfarray for better performance and correct colors
        try:
            # Method 1: surfarray (fastest and most reliable)
            img_array = pygame.surfarray.array3d(self.screen)
            img_rgb = np.transpose(img_array, (1, 0, 2))  # Convert from (x,y,3) to (y,x,3)
            return img_rgb
        except Exception as e:
            print(f"surfarray failed, using manual method: {e}")
            # Fallback: manual pixel access (slower but reliable)
            img_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    color = self.screen.get_at((x, y))
                    img_rgb[y, x, 0] = color[0]  # Red
                    img_rgb[y, x, 1] = color[1]  # Green  
                    img_rgb[y, x, 2] = color[2]  # Blue
            return img_rgb

    def draw(self, path, size, location):
        image_path = '{}/{}.png'.format(graphics_dir, path)
        try:
            raw_image = get_image(image_path)
            image = pygame.transform.scale(raw_image, size)
            self.screen.blit(image, location)
        except Exception as e:
            print(f"Failed to draw {path}: {e}")

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.holding_scale)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.container_scale)/2).astype(int))

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1-self.holding_scale) + (1-self.container_scale)/2*self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale*factor).astype(int))

    def on_cleanup(self):
        pygame.display.quit()
        pygame.quit()

    def get_image_obs(self):
        # Use existing screen or create one if needed
        if not hasattr(self, 'screen') or self.screen is None:
            # Create regular surface for compatibility with dummy driver
            self.screen = pygame.Surface((self.width, self.height)).convert()
        
        # Clear and fill background
        self.screen.fill(Color.FLOOR)
        
        # Render the environment
        for x in range(self.env.xlen):
            for y in range(self.env.ylen):
                sl = self.scaled_location((y, x))
                if self.env.map[x][y] == ITEMIDX["counter"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                elif self.env.map[x][y] == ITEMIDX["delivery"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)                    
                    pygame.draw.rect(self.screen, Color.DELIVERY, fill)
                    self.draw('delivery', self.tile_size, sl)
                    for k in self.env.delivery:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["knife"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)    
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('cutboard', self.tile_size, sl)
                    for k in self.env.knife:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["tomato"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for t in self.env.tomato:
                        if t.x == x and t.y == y:
                            self.draw(t.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["lettuce"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.lettuce:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["onion"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.onion:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["plate"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('plate', self.tile_size, sl)
                    for p in self.env.plate:
                        if p.x == x and p.y == y:
                            if p.containing:
                                self.draw(p.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["agent"]:
                    for agent in self.env.agent:
                        if agent.x == x and agent.y == y:
                            self.draw('agent-{}'.format(agent.color), self.tile_size, sl)
                            if agent.holding:
                                if isinstance(agent.holding, Plate):
                                    self.draw('plate', self.holding_size, self.holding_location((y, x)))
                                    if agent.holding.containing:
                                        self.draw(agent.holding.containedName, self.holding_container_size, self.holding_container_location((y, x)))
                                else:
                                    self.draw(agent.holding.name, self.holding_size, self.holding_location((y, x)))

        # Use surfarray for better performance and correct colors
        try:
            # Method 1: surfarray (fastest and most reliable)
            img_array = pygame.surfarray.array3d(self.screen)
            img_rgb = np.transpose(img_array, (1, 0, 2))  # Convert from (x,y,3) to (y,x,3)
            return img_rgb
        except Exception as e:
            print(f"surfarray failed, using manual method: {e}")
            # Fallback: manual pixel access (slower but reliable)
            img_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    color = self.screen.get_at((x, y))
                    img_rgb[y, x, 0] = color[0]  # Red
                    img_rgb[y, x, 1] = color[1]  # Green  
                    img_rgb[y, x, 2] = color[2]  # Blue
            return img_rgb