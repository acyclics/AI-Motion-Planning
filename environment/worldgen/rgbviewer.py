import copy
import glfw
import imageio
import numpy as np
import time
import sys

from mujoco_py.builder import cymj
from mujoco_py.generated import const
from mujoco_py.utils import rec_copy, rec_assign
from multiprocessing import Process, Queue
from threading import Lock


class RgbViewerBasic(cymj.MjRenderContextWindow):
    """
    A simple display GUI showing the scene of an :class:`.MjSim` as an RGB array.

    Parameters
    ----------
    sim : :class:`.MjSim`
        The simulator to display.
    """

    def __init__(self, sim):
        sim.model.stat.extent = 15.0
        super().__init__(sim)

        self._gui_lock = Lock()
        framebuffer_width, _ = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """
        if self.window is None:
            return
        elif glfw.window_should_close(self.window):
            glfw.terminate()
            sys.exit(0)

        with self._gui_lock:
            super().render()

        glfw.poll_events()

        frame = self._read_pixels_as_in_window()

        return frame
    
    def _read_pixels_as_in_window(self, resolution=None):
        if resolution is None:
            resolution = glfw.get_framebuffer_size(self.sim._render_context_window.window)

        resolution = np.array(resolution)
        resolution = np.array([640, 480]) #resolution * min(1000 / np.min(resolution), 1)[2560 1559]
        resolution = resolution.astype(np.int32)
        if self.sim._render_context_offscreen is None:
            self.sim.render(resolution[0], resolution[1])
        img = self.sim.render(*resolution)
        img = img[::-1, :, :] # Rendered images are upside-down.
        return img
