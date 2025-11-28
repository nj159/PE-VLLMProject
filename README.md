# PE-VLLMProject
## Installation
- cd PE-VLLMProject
- pip install -r requirements.txt

## Training
### Model Finetuing
cd finetuing
please refer to README.md
### Whisper
cd whisper
please refer to README.md
### Emotion Analysis
cd Emotionanalysis
please refer to README.md

## Agents
### core code
```python
"""
nonverbal_assessment_crewai.py

Multi-agent CrewAI example for automated nonverbal behavior assessment in instructional videos.

Agents:
  - TeachingSegmenter
  - KeyframeExtractor
  - SpeechEmotionAnalyzer
  - AuxiliaryContentGenerator
  - BehaviorAnalyzer
  - AgentQualityEvaluator

Usage:
  # (1) pip install crewai  (and your ML libs: librosa, transformers, opencv-python, moviepy...)
  # (2) Replace placeholder functions with real implementations/models
  # (3) Run: python nonverbal_assessment_crewai.py
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ---- CrewAI imports (assumed) ----
# If your version exposes different names, adapt accordingly.
try:
    from crewai import Crew, Agent, Task, Process
except Exception:
    # Fallback shim: simple local classes so file is still runnable for testing without crewai
    class Agent:
        def __init__(self, name: str, **kwargs):
            self.name = name
        def run(self, *args, **kwargs):
            raise NotImplementedError

    class Crew:
        def __init__(self, agents: List[Agent], process=None, verbose=False):
            self.agents = agents
            self.process = process
            self.verbose = verbose
        def kickoff(self, inputs: Dict[str, Any]):
            # simple sequential execution mimic
            ctx = inputs.copy()
            for a in self.agents:
                if hasattr(a, "execute"):
                    ctx = a.execute(ctx) or ctx
            return ctx

# ---- Setup logging ----
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "crew_run.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NonverbalCrew")

# ---- Data containers ----
@dataclass
class Segment:
    start: float
    end: float
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Keyframe:
    timestamp: float
    frame_path: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionFrame:
    timestamp: float
    speech_emotion: Dict[str, float]  # e.g., {happy: 0.7, neutral:0.3}
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---- Placeholder tools / helper functions (replace with real implementations) ----
def load_video(path: str):
    # Placeholder: return video metadata / duration
    return {"path": path, "duration": 600.0}  # seconds

def run_segmentation_model(video_path: str) -> List[Segment]:
    """
    Replace this stub with your teaching segmenter implementation.
    Should return list of Segment(start, end, reason).
    """
    # Example: coarse segmentation every 60 seconds
    meta = load_video(video_path)
    dur = meta["duration"]
    segs = []
    for s in range(0, int(dur), 60):
        segs.append(Segment(start=float(s), end=float(min(s+60, dur)), reason="auto_chunk"))
    logger.info(f"[Segmentation] produced {len(segs)} segments")
    return segs

def extract_keyframes_method(video_path: str, segment: Segment) -> List[Keyframe]:
    """
    Replace with your 'proposed method' for keyframe selection.
    Return a few Keyframe(timestamp, frame_path).
    """
    # Example: pick three evenly spaced timestamps inside the segment
    step = (segment.end - segment.start) / 4
    kms = []
    for i in range(1, 4):
        ts = segment.start + i * step
        # frame_path would refer to extracted image file (user should implement extractor)
        kms.append(Keyframe(timestamp=ts, frame_path=f"/tmp/frames/{int(ts)}.jpg"))
    return kms

def speech_emotion_recognition(audio_chunk_path: str) -> Dict[str, float]:
    """
    Replace with model inference. This stub returns a dummy distribution.
    """
    # Dummy: neutral 0.6, happy 0.3, sad 0.1
    return {"neutral": 0.6, "happy": 0.3, "sad": 0.1}

def transcribe_audio_at(timestamp: float, audio_path: str) -> str:
    """Stub: return transcript snippet for given timestamp"""
    return f"transcript at {timestamp:.1f}s"

def vllm_prompt_builder(frames: List[Keyframe], transcripts: List[str], emotions: List[EmotionFrame]) -> str:
    """
    Build a structured prompt that helps VLLMs understand context.
    Keep it structured: JSON-like or bullet lists.
    """
    prompt = {
        "keyframes": [{ "t": k.timestamp, "path": k.frame_path } for k in frames],
        "transcripts": transcripts,
        "emotions": [{ "t": e.timestamp, "dist": e.speech_emotion } for e in emotions]
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)

def score_nonverbal_behavior(multimodal_struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given integrated multimodal features, produce assessment scores, rationale, suggestions.
    Replace with your real scoring model or rule-based system.
    """
    # Example scoring stub:
    score = 0.75  # normalized
    report = {
        "score": score,
        "rationale": "Balanced eye contact & gestures detected; occasional monotone speech.",
        "suggestions": [
            "Increase vocal prosody during explanations (2-3 instances).",
            "Use more open gestures when transitioning topics."
        ]
    }
    return report

# ---- Agent Implementations ----
class TeachingSegmenterAgent(Agent):
    def __init__(self, name="TeachingSegmenter"):
        super().__init__(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        video_path = ctx["video_path"]
        try:
            segments = run_segmentation_model(video_path)
            ctx["segments"] = segments
            self.logger.info(f"Segmenter -> {len(segments)} segments")
        except Exception as e:
            self.logger.exception("Segmentation failed")
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})
        return ctx

class KeyframeExtractorAgent(Agent):
    def __init__(self, name="KeyframeExtractor"):
        super().__init__(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        video_path = ctx["video_path"]
        segments: List[Segment] = ctx.get("segments", [])
        all_keyframes: Dict[int, List[Keyframe]] = {}
        try:
            for i, seg in enumerate(segments):
                kfs = extract_keyframes_method(video_path, seg)
                all_keyframes[i] = kfs
                self.logger.info(f"Extracted {len(kfs)} keyframes for segment {i}")
            ctx["keyframes"] = all_keyframes
        except Exception as e:
            self.logger.exception("Keyframe extraction failed")
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})
        return ctx

class SpeechEmotionAnalyzerAgent(Agent):
    def __init__(self, audio_path_key="audio_path", name="SpeechEmotionAnalyzer"):
        super().__init__(name=name)
        self.audio_key = audio_path_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        keyframes_map: Dict[int, List[Keyframe]] = ctx.get("keyframes", {})
        audio_path = ctx.get(self.audio_key)
        emotion_map: Dict[int, List[EmotionFrame]] = {}
        try:
            for seg_idx, kfs in keyframes_map.items():
                ef_list = []
                for kf in kfs:
                    # Here, align short audio window around kf.timestamp and run emotion model
                    audio_chunk_path = f"/tmp/audio_chunks/{int(kf.timestamp)}.wav"  # stub path
                    dist = speech_emotion_recognition(audio_chunk_path)
                    ef_list.append(EmotionFrame(timestamp=kf.timestamp, speech_emotion=dist))
                emotion_map[seg_idx] = ef_list
                self.logger.info(f"Analyzed emotions for {len(ef_list)} keyframes in segment {seg_idx}")
            ctx["emotions"] = emotion_map
        except Exception as e:
            self.logger.exception("Speech emotion analysis failed")
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})
        return ctx

class AuxiliaryContentGeneratorAgent(Agent):
    def __init__(self, name="AuxiliaryContentGenerator"):
        super().__init__(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        keyframes_map = ctx.get("keyframes", {})
        emotions_map = ctx.get("emotions", {})
        audio_path = ctx.get("audio_path")
        content_prompts: Dict[int, str] = {}
        try:
            for seg_idx, kfs in keyframes_map.items():
                transcripts = [transcribe_audio_at(kf.timestamp, audio_path) for kf in kfs]
                emotions = emotions_map.get(seg_idx, [])
                prompt = vllm_prompt_builder(kfs, transcripts, emotions)
                content_prompts[seg_idx] = prompt
                self.logger.info(f"Built auxiliary prompt for segment {seg_idx}")
            ctx["aux_prompts"] = content_prompts
        except Exception as e:
            self.logger.exception("Aux content gen failed")
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})
        return ctx

class BehaviorAnalyzerAgent(Agent):
    def __init__(self, name="BehaviorAnalyzer"):
        super().__init__(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        keyframes_map = ctx.get("keyframes", {})
        aux_prompts = ctx.get("aux_prompts", {})
        final_reports: Dict[int, Dict[str, Any]] = {}
        try:
            for seg_idx, prompt in aux_prompts.items():
                # Here you would call a VLLM or other reasoning engine with the prompt
                # to get semantic features, then aggregate with keyframes/emotions to score
                multimodal_struct = {
                    "prompt": prompt,
                    "keyframes": [kf.__dict__ for kf in keyframes_map.get(seg_idx, [])],
                    "emotions": [ef.__dict__ for ef in ctx.get("emotions", {}).get(seg_idx, [])]
                }
                report = score_nonverbal_behavior(multimodal_struct)
                final_reports[seg_idx] = report
                self.logger.info(f"Generated behavior report for segment {seg_idx}: score={report.get('score')}")
            ctx["reports"] = final_reports
        except Exception as e:
            self.logger.exception("Behavior analysis failed")
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})
        return ctx

class AgentQualityEvaluatorAgent(Agent):
    """
    Monitors outputs and execution health. Ensures reports are complete & parsable,
    logs exceptions and summary metadata.
    """
    def __init__(self, name="AgentQualityEvaluator"):
        super().__init__(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audit_log = []

    def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        summary = {"time": timestamp, "checks": []}

        # Check presence/format of segments
        segments = ctx.get("segments")
        if not segments or not isinstance(segments, list):
            summary["checks"].append(("segments", "missing_or_bad_format"))
        else:
            summary["checks"].append(("segments", f"ok:{len(segments)}"))

        # Check keyframes
        kfs = ctx.get("keyframes")
        if not kfs:
            summary["checks"].append(("keyframes", "missing"))
        else:
            counts = {k: len(v) for k, v in kfs.items()}
            summary["checks"].append(("keyframes", f"counts:{counts}"))

        # Check reports parsability
        reports = ctx.get("reports", {})
        try:
            json.dumps(reports)
            summary["checks"].append(("reports", f"ok:{len(reports)}"))
        except Exception as e:
            summary["checks"].append(("reports", f"unserializable:{str(e)}"))
            ctx.setdefault("errors", []).append({"agent": self.__class__.__name__, "err": str(e)})

        # Record exceptions if any
        errors = ctx.get("errors", [])
        if errors:
            summary["checks"].append(("errors", errors))

        # Save audit
        self.audit_log.append(summary)
        self.logger.info(f"QualityEval: {summary}")
        ctx["quality_audit"] = summary
        ctx["audit_log"] = self.audit_log
        return ctx

# ---- Crew assembly & execution example ----
def build_and_run_crew(video_path: str, audio_path: Optional[str] = None):
    agents = [
        TeachingSegmenterAgent(),
        KeyframeExtractorAgent(),
        SpeechEmotionAnalyzerAgent(),
        AuxiliaryContentGeneratorAgent(),
        BehaviorAnalyzerAgent(),
        AgentQualityEvaluatorAgent()
    ]
    # If using real CrewAI, you would create Crew(...). Here we use shim or real Crew if installed.
    crew = Crew(agents=agents, process=None, verbose=True)
    inputs = {"video_path": video_path, "audio_path": audio_path}

    logger.info("Kicking off crew run...")
    result_ctx = crew.kickoff(inputs=inputs) if hasattr(crew, "kickoff") else crew.kickoff(inputs)
    logger.info("Crew finished. Summary:")
    logger.info(json.dumps({
        "reports": result_ctx.get("reports"),
        "quality": result_ctx.get("quality_audit"),
        "errors": result_ctx.get("errors", [])
    }, ensure_ascii=False, indent=2))
    return result_ctx

# ---- Example run (for testing) ----
if __name__ == "__main__":
    # Replace with actual paths
    VIDEO_PATH = "/path/to/instructional_video.mp4"
    AUDIO_PATH = "/path/to/instructional_audio.wav"
    # Run crew
    ctx = build_and_run_crew(VIDEO_PATH, AUDIO_PATH)
