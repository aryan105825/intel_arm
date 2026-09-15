""" Speechmatics Real-Time client and negation-aware, priority-ordered intent router. """
import asyncio
import functools
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

try:
    import websockets
except ImportError:
    websockets = None

log = logging.getLogger(__name__)

SPEECHMATICS_RT_URL = "wss://eu2.rt.speechmatics.com/v2"
TAXONOMY_HALT = "HALT"
TAXONOMY_REDIRECT = "REDIRECT"
TAXONOMY_MODIFY = "MODIFY"
TAXONOMY_CLARIFY = "CLARIFY"

_HALT_KEYWORDS = ["stop", "halt", "freeze", "drop it", "drop", "hold", "wait"]
_CLEAR_IDIOM_EXCEPTIONS = ["all clear", "we're clear", "were clear", "clear to proceed"]
_REDIRECT_KEYWORDS = ["clear", "instead", "go to", "move to", "over there", "put it in"]
_MODIFY_KEYWORDS = ["slower", "slow down", "faster", "speed up", "gentler", "gentle", "softer", "harder", "more force", "less force"]
_CLARIFY_TRIGGER_PHRASES = ["that one", "this one", "the other one", "grab it", "pick that up", "put it there", "get that"]
_NEGATION_MARKERS = ["don't", "do not", "dont", "not", "never mind", "no need to"]
_HALT_PHRASAL_NONHAZARD_FOLLOWERS = ["worrying", "worry", "rushing", "rushing me", "panicking", "freaking out"]

_TOKEN_RE = re.compile(r"[a-z']+")

def _tokenize(transcript: str) -> List[str]:
    return _TOKEN_RE.findall(transcript.lower())

@functools.lru_cache(maxsize=None)
def _compile_phrase_pattern(phrase: str) -> "re.Pattern":
    return re.compile(r"\b" + re.escape(phrase) + r"\b")

def _contains_phrase(lower_transcript: str, phrase: str) -> bool:
    return _compile_phrase_pattern(phrase).search(lower_transcript) is not None

def _find_phrase_span(lower_transcript: str, phrase: str) -> Optional[Tuple[int, int]]:
    match = _compile_phrase_pattern(phrase).search(lower_transcript)
    if match is None:
        return None
    return (match.start(), match.end())

class PraxisVoiceSupervisor:
    def __init__(self, api_key: str, interrupt_callback: Callable[[Dict[str, Any]], None], vision_gate):
        self._api_key = api_key
        self._interrupt_callback = interrupt_callback
        self._vision_gate = vision_gate

    def _halt_keyword_is_suppressed(self, lower: str, keyword: str, span: Tuple[int, int]) -> bool:
        start, end = span
        
        following = lower[end:end + 24].lstrip()
        for follower in _HALT_PHRASAL_NONHAZARD_FOLLOWERS:
            if _contains_phrase(following, follower):
                return True
                
        preceding = lower[:start][-16:]
        if any(_contains_phrase(preceding, marker) for marker in _NEGATION_MARKERS):
            return True
            
        return False

    def _classify_utterance(self, transcript: str, raw_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lower = transcript.lower()
        confidence = float(raw_meta.get("confidence", 0.0))
        
        for keyword in _HALT_KEYWORDS:
            span = _find_phrase_span(lower, keyword)
            if span is not None:
                if not self._halt_keyword_is_suppressed(lower, keyword, span):
                    return self._make_event(TAXONOMY_HALT, transcript, confidence)

        for phrase in _CLARIFY_TRIGGER_PHRASES:
            if _contains_phrase(lower, phrase):
                if self._vision_gate and self._vision_gate.check_ambiguity():
                    return self._make_event(TAXONOMY_CLARIFY, transcript, confidence, clarification_query=f"Which object do you mean by '{phrase}'?")

        for keyword in _REDIRECT_KEYWORDS:
            if _contains_phrase(lower, keyword):
                return self._make_event(TAXONOMY_REDIRECT, transcript, confidence, new_prompt=lower.strip())

        for keyword in _MODIFY_KEYWORDS:
            if _contains_phrase(lower, keyword):
                direction = "decrease" if keyword in ("slower", "slow down", "gentler", "gentle", "softer", "less force") else "increase"
                return self._make_event(TAXONOMY_MODIFY, transcript, confidence, parameters={"keyword": keyword, "direction": direction})

        return None

    @staticmethod
    def _make_event(taxonomy: str, raw_transcript: str, confidence: float, parameters: Optional[Dict[str, Any]] = None, new_prompt: Optional[str] = None, clarification_query: Optional[str] = None) -> Dict[str, Any]:
        return {
            "taxonomy": taxonomy,
            "raw_transcript": raw_transcript,
            "confidence": confidence,
            "parameters": parameters,
            "new_prompt": new_prompt,
            "clarification_query": clarification_query,
        }

    async def run(self, audio_stream_generator):
        if websockets is None:
            log.warning("websockets package missing, skipping live audio ingestion.")
            return

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with websockets.connect(SPEECHMATICS_RT_URL, extra_headers=headers) as ws:
                start_req = {
                    "message": "StartRecognition",
                    "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
                    "transcription_config": {"language": "en", "enable_partials": False, "max_delay": 1.0}
                }
                await ws.send(json.dumps(start_req))

                async def _sender():
                    async for chunk in audio_stream_generator:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))

                async def _receiver():
                    async for raw_message in ws:
                        msg = json.loads(raw_message)
                        if msg.get("message") == "AddTranscript":
                            transcript = msg.get("metadata", {}).get("transcript", "")
                            if transcript:
                                event = self._classify_utterance(transcript, {"confidence": 0.99})
                                if event:
                                    log.info(f"VOICE INTERCEPT: {event}")
                                    self._interrupt_callback(event)

                await asyncio.gather(_sender(), _receiver())
        except Exception as e:
            log.error(f"Speechmatics connection failed: {e}")