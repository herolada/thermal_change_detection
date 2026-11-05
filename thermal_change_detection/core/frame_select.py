"""Choice of the frames that get processed (``select.*``).

'periodic': the first frame after ``period``.

'closest': every incoming frame is ranked by ``KeyframeStore.query`` (score =
distance + angle_weight * angle to its best keyframe), so the frame that is
processed for a keyframe is the one taken closest to it. The decision for a
keyframe is made once we have driven past it, i.e. when

  - 'passed':   the score rose ``rise_margin_m`` above its minimum,
  - 'switched': the best match moved to another keyframe,
  - 'lost':     no keyframe matches any more (left the route),
  - 'timeout':  ``max_period`` elapsed (e.g. robot standing still).

Each keyframe is processed once per pass (until ``max_period``). Without any
matching keyframe (first lap, off the route) frames are taken every ``period``
('no_ref'), which also keeps the keyframe map growing at the usual spacing.

Frames must be pushed in stamp order. The decision lags the chosen frame by
the time it takes the score to rise by ``rise_margin_m``.
"""
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np


@dataclass
class Selected:
    stamp: float
    T_map_cam: np.ndarray
    item: Any                       # caller's payload (e.g. the image message)
    reason: str                     # periodic | no_ref | passed | switched | lost | timeout
    kf_id: Optional[int] = None
    score: Optional[float] = None


class FrameSelector:
    def __init__(self, cfg, store):
        if cfg.mode not in ('periodic', 'closest'):
            raise ValueError(f'unknown select.mode {cfg.mode!r}')
        self.cfg, self.store = cfg, store
        self.reset()

    def reset(self):
        self.last_t = None          # stamp of the last selected frame
        self.last_kf = None         # keyframe it was selected for
        self.best = None            # best candidate (Selected) for best.kf_id

    def _emit(self, sel):
        self.last_t, self.last_kf, self.best = sel.stamp, sel.kf_id, None
        return sel

    def push(self, stamp, T_map_cam, item) -> Optional[Selected]:
        """Feed the next frame; returns the frame to process now (possibly an
        earlier one), or None."""
        c = self.cfg
        if self.last_t is not None and stamp < self.last_t:
            self.reset()                                    # time jumped back
        since = np.inf if self.last_t is None else stamp - self.last_t

        if c.mode == 'periodic':
            if since >= c.period:
                return self._emit(Selected(stamp, T_map_cam, item, 'periodic'))
            return None

        if since < c.min_period:
            return None
        m = self.store.query(T_map_cam, stamp)
        if m is None:
            if self.best is not None:
                return self._emit(replace(self.best, reason='lost'))
            if since >= c.period:
                return self._emit(Selected(stamp, T_map_cam, item, 'no_ref'))
            return None

        cand = Selected(stamp, T_map_cam, item, '', m.keyframe.id, m.score)
        b = self.best
        if b is None:
            if m.keyframe.id != self.last_kf or since >= c.max_period:
                self.best = cand
            return None
        if m.keyframe.id != b.kf_id:
            out = self._emit(replace(b, reason='switched'))
            if stamp - out.stamp >= c.min_period:
                self.best = cand
            return out
        if m.score < b.score:
            self.best = cand
        elif m.score > b.score + c.rise_margin_m:
            return self._emit(replace(b, reason='passed'))
        held = stamp - (self.last_t if self.last_t is not None else self.best.stamp)
        if held >= c.max_period:
            return self._emit(replace(self.best, reason='timeout'))
        return None
