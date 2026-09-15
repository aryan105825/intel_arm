""" CLARIFY ambiguity gate for Praxis Guard. """
import time
from typing import List, Optional

class PraxisVisionGate:
    def __init__(self):
        self._current_bboxes: List = []
        self._last_update_time: Optional[float] = None

    def update_frame(self, bboxes: list) -> None:
        self._current_bboxes = list(bboxes) if bboxes is not None else []
        self._last_update_time = time.monotonic()

    def frame_age_s(self) -> Optional[float]:
        if self._last_update_time is None:
            return None
        return time.monotonic() - self._last_update_time

    def check_ambiguity(self, max_age_s: Optional[float] = None) -> bool:
        if max_age_s is not None:
            age = self.frame_age_s()
            if age is None or age > max_age_s:
                return False
        return len(self._current_bboxes) > 1

class PraxisSemanticGate:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PraxisSemanticGate is not implemented in this build.")
        
    def check_ambiguity(self, *args, **kwargs) -> bool:
        raise NotImplementedError("PraxisSemanticGate.check_ambiguity is not implemented.")