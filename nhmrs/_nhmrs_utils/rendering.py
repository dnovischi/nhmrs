"""Pygame-based renderer following MPE2 pattern."""
import numpy as np
import pygame
import math


class Transform:
    """Transform for positioning and rotating geometries."""
    def __init__(self):
        self.translation = (0.0, 0.0)
        self.rotation = 0.0

    def set_translation(self, x, y):
        self.translation = (float(x), float(y))

    def set_rotation(self, angle):
        self.rotation = float(angle)


class Geom:
    """Base geometry class."""
    def __init__(self):
        self.color = (127, 127, 127)  # RGB 0-255
        self.attrs = []

    def set_color(self, r, g, b, a=1.0):
        """Set color with values in [0, 1] range."""
        self.color = (int(r * 255), int(g * 255), int(b * 255))

    def add_attr(self, attr):
        self.attrs.append(attr)

    def render(self, screen, transform_func):
        """Render geometry with transformations."""
        raise NotImplementedError


class FilledPolygon(Geom):
    """Filled polygon geometry."""
    def __init__(self, vertices):
        super().__init__()
        self.vertices = [(float(x), float(y)) for x, y in vertices]

    def render(self, screen, transform_func):
        """Render polygon with rotation and translation."""
        # Get transform
        translation = (0.0, 0.0)
        rotation = 0.0
        for attr in self.attrs:
            if isinstance(attr, Transform):
                translation = attr.translation
                rotation = attr.rotation
        
        # Apply rotation and translation to vertices
        transformed_vertices = []
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        for x, y in self.vertices:
            # Rotate
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            # Translate
            x_final = x_rot + translation[0]
            y_final = y_rot + translation[1]
            # Transform to screen coordinates
            screen_pos = transform_func(x_final, y_final)
            transformed_vertices.append(screen_pos)
        
        # Draw polygon
        if len(transformed_vertices) >= 3:
            pygame.draw.polygon(screen, self.color, transformed_vertices)
            # Draw border
            pygame.draw.polygon(screen, (0, 0, 0), transformed_vertices, 1)


class Circle(Geom):
    """Circle geometry."""
    def __init__(self, radius, num_points=30):
        super().__init__()
        self.radius = float(radius)
        self.num_points = int(num_points)

    def render(self, screen, transform_func):
        """Render circle."""
        # Get transform
        translation = (0.0, 0.0)
        for attr in self.attrs:
            if isinstance(attr, Transform):
                translation = attr.translation
        
        # Transform center to screen coordinates
        center = transform_func(translation[0], translation[1])
        
        # Calculate radius in screen space (approximate)
        radius_screen = abs(transform_func(self.radius, 0)[0] - transform_func(0, 0)[0])
        radius_screen = max(1, int(radius_screen))
        
        # Draw circle
        pygame.draw.circle(screen, self.color, center, radius_screen)
        # Draw border
        pygame.draw.circle(screen, (0, 0, 0), center, radius_screen, 1)


def make_polygon(vertices):
    """Create a filled polygon geometry."""
    return FilledPolygon(vertices)


def make_circle(radius, num_points=30):
    """Create a circle geometry."""
    return Circle(radius, num_points)


class Viewer:
    """Pygame-based viewer for rendering."""
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NHMRS Environment")
        self.clock = pygame.time.Clock()
        
        self.geoms = []
        self.onetime_geoms = []
        self.bounds = None
        self.isopen = True

    def set_bounds(self, left, right, bottom, top):
        """Set coordinate bounds for the view."""
        self.bounds = (float(left), float(right), float(bottom), float(top))

    def add_geom(self, geom):
        """Add persistent geometry."""
        self.geoms.append(geom)

    def add_onetime(self, geom):
        """Add one-time geometry (cleared after render)."""
        self.onetime_geoms.append(geom)

    def _world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        if self.bounds:
            left, right, bottom, top = self.bounds
        else:
            left, right, bottom, top = -1, 1, -1, 1
        
        # Normalize to [0, 1]
        x_norm = (x - left) / (right - left)
        y_norm = (y - bottom) / (top - bottom)
        
        # Convert to screen coordinates (flip y-axis)
        screen_x = int(x_norm * self.width)
        screen_y = int((1.0 - y_norm) * self.height)
        
        return (screen_x, screen_y)

    def render(self, return_rgb_array=False):
        """Render the scene."""
        if not self.isopen:
            return None
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False
                return None
        
        # Clear screen (white background)
        self.screen.fill((255, 255, 255))
        
        # Render all geometries
        for geom in self.geoms:
            geom.render(self.screen, self._world_to_screen)
        
        for geom in self.onetime_geoms:
            geom.render(self.screen, self._world_to_screen)
        self.onetime_geoms = []
        
        # Get RGB array if requested
        arr = None
        if return_rgb_array:
            arr = np.array(pygame.surfarray.pixels3d(self.screen))
            arr = np.transpose(arr, axes=(1, 0, 2))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
        return arr

    def close(self):
        """Close the viewer."""
        if self.isopen:
            pygame.quit()
            self.isopen = False
