""" CLARIFY ambiguity gate for SafeVLA / Praxis Guard. """
import time
from typing import List, Optional

class PraxisVisionGate:
    """
    Bounding-box COUNT heuristic used to gate the CLARIFY taxonomy.
    This purely counts how many plausible candidate bounding boxes are 
    present in the most recently stored frame to detect referent ambiguity.
    """
    def __init__(self):
        self._current_bboxes: List = []
        self._last_update_time: Optional[float] = None

    def update_frame(self, bboxes: list) -> None:
        """Store the current frame's bounding boxes as internal state."""
        self._current_bboxes = list(bboxes) if bboxes is not None else []
        self._last_update_time = time.monotonic()

    def frame_age_s(self) -> Optional[float]:
        """Seconds since the last update_frame() call."""
        if self._last_update_time is None:
            return None
        return time.monotonic() - self._last_update_time

    def check_ambiguity(self, max_age_s: Optional[float] = None) -> bool:
        """
        Returns True if more than one candidate bounding box is present 
        in the most recently stored frame.
        """
        if max_age_s is not None:
            age = self.frame_age_s()
            if age is None or age > max_age_s:
                return False
        return len(self._current_bboxes) > 1