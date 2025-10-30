"""Simple pyglet-based renderer similar to MPE2 approach."""
import numpy as np
import pyglet
from pyglet.gl import *


class Transform:
    def __init__(self):
        self.translation = (0.0, 0.0)
        self.rotation = 0.0

    def set_translation(self, x, y):
        self.translation = (float(x), float(y))

    def set_rotation(self, angle):
        self.rotation = float(angle)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(np.degrees(self.rotation), 0, 0, 1.0)

    def disable(self):
        glPopMatrix()


class Geom:
    def __init__(self):
        self.color = (0.5, 0.5, 0.5, 1.0)
        self.attrs = []

    def set_color(self, r, g, b, a=1.0):
        self.color = (float(r), float(g), float(b), float(a))

    def add_attr(self, attr):
        self.attrs.append(attr)

    def render(self):
        for attr in self.attrs:
            attr.enable()
        self.render1()
        for attr in reversed(self.attrs):
            attr.disable()

    def render1(self):
        raise NotImplementedError


class FilledPolygon(Geom):
    def __init__(self, vertices):
        super().__init__()
        self.vertices = [(float(x), float(y)) for x, y in vertices]

    def render1(self):
        glColor4f(*self.color)
        glBegin(GL_POLYGON)
        for x, y in self.vertices:
            glVertex2f(x, y)
        glEnd()


class Circle(Geom):
    def __init__(self, radius, num_points=30):
        super().__init__()
        self.radius = float(radius)
        self.num_points = int(num_points)

    def render1(self):
        glColor4f(*self.color)
        glBegin(GL_POLYGON)
        for i in range(self.num_points):
            angle = 2.0 * np.pi * i / self.num_points
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()


def make_polygon(vertices):
    return FilledPolygon(vertices)


def make_circle(radius, num_points=30):
    return Circle(radius, num_points)


class Viewer:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.bounds = None
        self.isopen = True

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        self.bounds = (float(left), float(right), float(bottom), float(top))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        if not self.isopen:
            return None

        glClearColor(1.0, 1.0, 1.0, 1.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Setup projection
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.bounds:
            left, right, bottom, top = self.bounds
        else:
            left, right, bottom, top = -1, 1, -1, 1
        
        glOrtho(left, right, bottom, top, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Render all geoms
        for geom in self.geoms:
            geom.render()
        
        for geom in self.onetime_geoms:
            geom.render()
        self.onetime_geoms = []

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(self.height, self.width, 4)
            arr = arr[::-1, :, :3]

        self.window.flip()
        return arr

    def close(self):
        if self.window:
            self.window.close()
            self.isopen = False
