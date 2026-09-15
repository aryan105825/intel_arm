""" Deterministic dispatch layer between classified voice/manual interrupts and the motor policy. """
from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Dict, List, Optional
import numpy as np
from src.audit_logger import AppendOnlyLedger, sign_event  

log = logging.getLogger(__name__)

PRAXIS_GUARD_JOINT_COUNT = 14  # Updated to 14 to match your new bimanual 7-DOF arms

try:
    from openvino import AsyncInferQueue, Core
except ImportError:
    try:
        from openvino.runtime import AsyncInferQueue, Core
    except ImportError:
        AsyncInferQueue = None
        Core = None

@dataclass
class ActionChunk:
    actions: List[np.ndarray]
    index: int = 0
    @property
    def horizon(self) -> int:
        return len(self.actions)
    @property
    def remaining(self) -> int:
        return max(0, self.horizon - self.index)
    def current(self) -> Optional[np.ndarray]:
        if 0 <= self.index < self.horizon:
            return self.actions[self.index]
        return None
    def advance(self) -> None:
        self.index += 1

class PraxisGuardArbiter:
    def __init__(self, openvino_model_path: str, device: str = "AUTO"):
        self._model_path = openvino_model_path
        self._device = device
        self._compiled_model = None
        self._infer_queue = None
        self._pending_infer_result: Optional[np.ndarray] = None
        self._load_motor_policy(openvino_model_path, device)
        self.emergency_stop_flag: bool = False
        self._active_chunk: Optional[ActionChunk] = None
        self._velocity_scale: float = 1.0
        self._last_commanded_velocity: Optional[np.ndarray] = None
        self._last_n_joints: int = 0
        self._pending_replan_prompt: Optional[str] = None
        self._active_chunk_frozen: bool = False
        self._pending_clarification_query: Optional[str] = None
        self._ledger = AppendOnlyLedger(path="benchmarks/raw/compliance_ledger.jsonl")
        self._state_lock = threading.Lock()
        self.on_interrupt_dispatched_hook = None
        self._valid_resume_tokens = {"resume", "resume operation", "manual-override-ack"}

    def _load_motor_policy(self, openvino_model_path: str, device: str) -> None:
        if Core is None or AsyncInferQueue is None or openvino_model_path is None:
            self._compiled_model = None
            self._infer_queue = None
            return
        core = Core()
        model = core.read_model(openvino_model_path)
        self._compiled_model = core.compile_model(model, device)
        self._infer_queue = AsyncInferQueue(self._compiled_model, jobs=1)
        self._infer_queue.set_callback(self._on_infer_complete)

    def _on_infer_complete(self, infer_request, userdata) -> None:
        raw_output = infer_request.get_output_tensor(0).data
        self._pending_infer_result = np.asarray(raw_output, dtype=np.float64)

    def dispatch_voice_interrupt(self, event: Dict[str, Any]) -> None:
        taxonomy = event.get("taxonomy")
        known = taxonomy in ("HALT", "REDIRECT", "MODIFY", "CLARIFY")
        with self._state_lock:
            if taxonomy == "HALT":
                self.emergency_stop_flag = True
                self._active_chunk = None
                self._issue_zero_velocity_command()
            elif taxonomy == "REDIRECT":
                self._active_chunk = None
                self._trigger_replan(event["new_prompt"])
            elif taxonomy == "MODIFY":
                self._patch_velocity_scale(event["parameters"])
            elif taxonomy == "CLARIFY":
                self._freeze_and_query(event["clarification_query"])
        
        self._ledger.append(event)
        
        if known and self.on_interrupt_dispatched_hook is not None:
            self.on_interrupt_dispatched_hook(event)
        if not known:
            log.error("Unknown taxonomy in event (audited, then raising): %r", taxonomy)
            raise ValueError(f"Unknown taxonomy in event: {taxonomy!r}")

    def _issue_zero_velocity_command(self) -> None:
        self._last_commanded_velocity = np.zeros(self._last_n_joints, dtype=np.float64)

    def _trigger_replan(self, new_prompt: str) -> None:
        self._pending_replan_prompt = new_prompt

    def _patch_velocity_scale(self, parameters: Optional[Dict[str, Any]]) -> None:
        if not parameters:
            return
        if "velocity_factor" in parameters and parameters["velocity_factor"] is not None:
            try:
                factor = float(parameters["velocity_factor"])
            except (TypeError, ValueError):
                return
            self._velocity_scale = min(2.0, max(0.05, factor))
            return
        direction = parameters.get("direction")
        step = 0.15
        if direction == "increase":
            self._velocity_scale = min(2.0, self._velocity_scale + step)
        elif direction == "decrease":
            self._velocity_scale = max(0.05, self._velocity_scale - step)

    def _freeze_and_query(self, clarification_query: str) -> None:
        self._active_chunk_frozen = True
        self._pending_clarification_query = clarification_query

    def clear_estop(self, confirmation_token: str) -> bool:
        with self._state_lock:
            if confirmation_token in self._valid_resume_tokens:
                self.emergency_stop_flag = False
                return True
            return False

    def load_action_chunk(self, actions: List[np.ndarray]) -> None:
        with self._state_lock:
            self._active_chunk = ActionChunk(actions=list(actions))

    @property
    def active_action_chunk_index(self) -> Optional[int]:
        with self._state_lock:
            return self._active_chunk.index if self._active_chunk is not None else None

    @property
    def remaining_chunk_horizon(self) -> Optional[int]:
        with self._state_lock:
            return self._active_chunk.remaining if self._active_chunk is not None else None

    def step_control_loop(self, vision_frame: Any, state_vector: np.ndarray) -> np.ndarray:
        state_vector = np.asarray(state_vector)
        n_joints = state_vector.shape[0] if state_vector.ndim >= 1 else 0
        self._last_n_joints = n_joints
        
        with self._state_lock:
            if self.emergency_stop_flag:
                return np.zeros(n_joints, dtype=np.float64)
            if self._active_chunk is not None and not self._active_chunk_frozen:
                chunk_action = self._active_chunk.current()
                if chunk_action is not None:
                    self._active_chunk.advance()
                    return np.asarray(chunk_action, dtype=np.float64) * self._velocity_scale
            velocity_scale = self._velocity_scale
            
        if self._infer_queue is None or self._compiled_model is None:
            return np.zeros(n_joints, dtype=np.float64)
            
        self._infer_queue.wait_all()
        self._infer_queue.start_async({0: state_vector})
        self._infer_queue.wait_all()
        if self._pending_infer_result is None:
            return np.zeros(n_joints, dtype=np.float64)
        return self._pending_infer_result * velocity_scale