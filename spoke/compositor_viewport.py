"""Fixed display-local capture geometry for native compositor experiments."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CaptureViewport:
    x: float
    y: float
    width: float
    height: float
    scale: float
    screen_x: float
    screen_y: float
    screen_height: float

    @classmethod
    def for_screen(cls, screen, rect):
        x, y, width, height = map(float, rect)
        frame = screen.frame()
        scale = float(screen.backingScaleFactor())
        values = (x, y, width, height, scale)
        if (not all(math.isfinite(v) for v in values) or scale <= 0
                or width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > frame.size.width or y + height > frame.size.height
                or any(not math.isclose(v * scale, round(v * scale), abs_tol=1e-7,
                                        rel_tol=0) for v in values[:4])):
            raise ValueError("capture viewport must fit the display on backing-pixel boundaries")
        return cls(x, y, width, height, scale, float(frame.origin.x),
                   float(frame.origin.y), float(frame.size.height))

    @property
    def source_rect(self):
        return ((self.x, self.y), (self.width, self.height))

    @property
    def window_rect(self):
        return ((self.screen_x + self.x,
                 self.screen_y + self.screen_height - self.y - self.height),
                (self.width, self.height))

    @property
    def pixel_size(self):
        return (round(self.width * self.scale), round(self.height * self.scale))

    def local_config(self, config):
        return {**config, "center_x": float(config["center_x"]) - self.x * self.scale,
                "center_y": float(config["center_y"]) - self.y * self.scale}
