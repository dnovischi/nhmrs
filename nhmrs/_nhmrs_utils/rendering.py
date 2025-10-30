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
    def __init__(self, radius, num_points=30, filled=True):
        super().__init__()
        self.radius = float(radius)
        self.num_points = int(num_points)
        self.filled = filled

    def render(self, screen, transform_func):
        """Render circle with proper transform chain."""
        # Collect all transforms and apply them in order
        local_translation = (0.0, 0.0)
        rotation = 0.0
        final_translation = (0.0, 0.0)
        
        for attr in self.attrs:
            if isinstance(attr, Transform):
                # First transform is local offset, second is agent position/rotation
                if local_translation == (0.0, 0.0) and final_translation == (0.0, 0.0):
                    local_translation = attr.translation
                else:
                    rotation = attr.rotation
                    final_translation = attr.translation
        
        # Apply rotation to local offset, then add final translation
        if rotation != 0.0:
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            x_rot = local_translation[0] * cos_r - local_translation[1] * sin_r
            y_rot = local_translation[0] * sin_r + local_translation[1] * cos_r
            center_x = x_rot + final_translation[0]
            center_y = y_rot + final_translation[1]
        else:
            center_x = local_translation[0] + final_translation[0]
            center_y = local_translation[1] + final_translation[1]
        
        # Transform center to screen coordinates
        center = transform_func(center_x, center_y)
        
        # Calculate radius in screen space (approximate)
        radius_screen = abs(transform_func(self.radius, 0)[0] - transform_func(0, 0)[0])
        radius_screen = max(1, int(radius_screen))
        
        if self.filled:
            # Draw filled circle
            pygame.draw.circle(screen, self.color, center, radius_screen)
            # Draw border
            pygame.draw.circle(screen, (0, 0, 0), center, radius_screen, 1)
        else:
            # Draw unfilled circle (outline only)
            pygame.draw.circle(screen, self.color, center, radius_screen, 2)


def make_polygon(vertices):
    """Create a filled polygon geometry."""
    return FilledPolygon(vertices)


def make_circle(radius, num_points=30, filled=True):
    """Create a circle geometry.
    
    Args:
        radius: Circle radius in world coordinates
        num_points: Number of points for approximation (unused in pygame implementation)
        filled: If True, draw filled circle; if False, draw outline only
    """
    return Circle(radius, num_points, filled)


def make_agent_geom(agent_size, agent_color):
    """Create agent geometry with arrowhead and footprint circle.
    
    This creates a standard agent representation with:
    - A footprint circle centered at the robot origin
    - An arrowhead polygon inscribed in the circle (3 pointy corners nearly touch edge)
    
    The arrowhead vertices are positioned at specific angles (0°, 140°, 220°) 
    at 95% of the circle radius to be inscribed within the circle.
    
    Args:
        agent_size: Base size of the agent
        agent_color: RGB tuple (0-1 range) for agent color
        
    Returns:
        tuple: (collision_circle, arrow, transform)
            - collision_circle: Circle geometry (drawn first, behind arrow)
            - arrow: Polygon geometry (drawn second, on top)
            - transform: Shared transform for both geometries
    """
    import numpy as np
    
    # Define footprint circle radius
    radius = agent_size * 2.0
    
    # Position arrowhead vertices at specific angles, at 95% of radius
    vertex_radius = radius * 0.95
    
    # Define angles for the three pointy corners (in radians)
    angle_tip = 0  # Front tip points forward (0 degrees)
    angle_back_left = np.deg2rad(140)  # Back left corner
    angle_back_right = np.deg2rad(220)  # Back right corner
    
    # Calculate vertex positions using polar coordinates
    tip = (vertex_radius * np.cos(angle_tip), vertex_radius * np.sin(angle_tip))
    back_left = (vertex_radius * np.cos(angle_back_left), vertex_radius * np.sin(angle_back_left))
    back_right = (vertex_radius * np.cos(angle_back_right), vertex_radius * np.sin(angle_back_right))
    
    # All vertices for visual polygon (including center point for filled appearance)
    all_vertices = [
        tip,  # Front tip
        back_left,  # Back left
        (0, 0),  # Center (creates inward-extending bottom)
        back_right,  # Back right
    ]
    
    # Create shared transform for both geometries
    transform = Transform()
    
    # Create collision circle centered at robot origin (added first to draw behind)
    collision_circle = make_circle(radius, filled=False)
    collision_circle.set_color(0.5, 0.5, 0.5)  # Gray circle
    collision_circle.add_attr(transform)
    
    # Create arrow (added second to draw on top)
    arrow = make_polygon(all_vertices)
    arrow.set_color(*agent_color)
    arrow.add_attr(transform)
    
    return collision_circle, arrow, transform


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
