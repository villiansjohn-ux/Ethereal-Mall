#!/usr/bin/env python3
"""ETHEREAL MALL - an endless liminal mall (Ursina) scored live by the Ethereal Orchestration Engine.

One file.  Launching it opens a native 3D window with an infinite brutalist mall corridor ("The
Path") and, concurrently, runs the real-time ethereal orchestra built from YOUR music folder.
What you do in the world changes how the orchestra sounds, and the album art of the songs it is
playing hangs on the walls as glowing ads.

Install:  pip install numpy scipy librosa sounddevice ursina pillow
          optional: pip install mutagen    (embedded album art for the light-boxes)
          (.m4a needs ffmpeg on PATH; Linux also needs PortAudio: apt install libportaudio2)

Run:      python ethereal_mall.py --library ~/Music
          python ethereal_mall.py --library ~/Music --tracks 2,11,100 --seed 7
          python ethereal_mall.py --silent                 # just walk (no audio stack needed)
          python ethereal_mall.py --audio-only --library ~/Music     # original headless engine
          python ethereal_mall.py --no-shader              # if your GPU rejects the custom GLSL

Controls: W/A/S/D move (forward/left/back/right), mouse look (or arrow keys L/R), Shift run,
          Space jump, E interact (sit, vending machines, treadmills), V first/third person,
          mouse wheel camera distance, Tab free/lock mouse, F3 audio/debug overlay, Esc quit.
          I inventory/album, F torch, R refuel go-kart, W/S/A/D + Space(drift) drive, E also talks to people
          (people speak only when you face them and press E).  Progress is saved in ~/.ethereal_mall/ (--no-save).
          Phone (sold in the Backrooms cafe): P open (map / garage / music), M play-pause downloaded music, N next track.
          v5: K summon selected kart, Z mute mall ambience (phone), Q scan a kart (hold phone), H/O kart light/neon, B spoiler passenger, J Eco-Kart dashboard/music,
          T beside a Listening Post / Eco-Kart with the audio jack held; phone GPS page: T teleports ($1,000); E at Sakura-ya's silver table washes dishes for tips.
          Rare room, 1 chance in 30 per block: SAKURA-YA, a Japanese izakaya (sit at a middle table: a server comes;
          Y/N answers, R rings the table bell; at the belt side E takes a plate).

How the game drives the sound (AudioBridge -> LiveParams -> EtherealDSPProcessor):
  * The cold infinite corridor : dark, very wet, long granular smear, two lanes.
  * Cafe (warm)                : open and bright, short reverb, little smear, three lanes.
  * Gym (retro-futuristic)     : dry, dense, every lane; running a treadmill adds more.
  * Empty restaurant           : drowned, heavily smeared, a single lane.
  * Walking faster             : brighter, drier, denser.  Standing still: dissolves into smear.
  * Approaching an ad          : sound comes close (drier, brighter, clearer); linger ~2 s and
                                 that ad's song is weighted to become the orchestra's next layer.
  * Ads glow brighter while their song is actually sounding in the mix.
  wet/tone/gain change within a block (+ the render-ahead buffer, --buffer); stretch, smear,
  warmth and lane count take effect as each lane prepares its next layer (--max-chunk shortens that).

--- original engine notes -------------------------------------------------------------------
Plays a never-ending, slowly evolving "ethereal orchestra" LIVE from the music files in a folder.
For every layer the engine decides how much of a track to play (a few seconds up to the entire
track, aligned to the song's own sections), streams it through gentle processing, and blends
several such layers through one shared cathedral reverb. Nothing is rendered to disk.

Glitch-free design: the audio callback only copies samples from a pre-filled ring buffer; a
dedicated render thread mixes, reverberates and masters ahead of time; track analysis runs in
low-priority worker PROCESSES; reverb is one shared partitioned-FFT convolution on the mix bus.
Track indices are 0-based positions in the case-insensitively sorted list of audio files.
"""
from __future__ import annotations

import argparse
import colorsys
import dataclasses
import gc
import hashlib
import io
import logging
import math
import multiprocessing
import os
import pickle
import random
import sys
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional, Sequence, Union

import numpy as np
import scipy.fft as sfft
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

# Optional-at-import so the file can be imported/tested anywhere; main() checks them.
_MISSING: list = []
try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]
    _MISSING.append("librosa")
try:
    import sounddevice as sd
except (ImportError, OSError):   # OSError = PortAudio library not found
    sd = None  # type: ignore[assignment]
    _MISSING.append("sounddevice (and the PortAudio system library)")

log = logging.getLogger("ethereal")


# ============================================================================
# UTILITIES
# ============================================================================
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class EtherealError(Exception):
    """Base class for engine errors."""


class DependencyError(EtherealError):
    pass


class LibraryError(EtherealError):
    pass


class AnalysisError(EtherealError):
    pass


class TrackNotFoundError(EtherealError):
    pass


def require_dependencies() -> None:
    if _MISSING:
        raise DependencyError("Missing: " + ", ".join(_MISSING)
                              + ".  Install with:  pip install numpy scipy librosa sounddevice")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s",
                        datefmt="%H:%M:%S")


@contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s took %.2fs", label, time.perf_counter() - t0)


def pitch_class_from_name(name: str) -> int:
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    key = name.strip().upper()
    key = flats.get(key, key)
    for i, p in enumerate(PITCH_NAMES):
        if p.upper() == key:
            return i
    raise ValueError(f"Unknown pitch class: {name!r}")


# ============================================================================
# CONFIGURATION
# ============================================================================
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


@dataclass(frozen=True)
class AnalysisConfig:
    analysis_sr: int = 22_050
    hop_length: int = 512
    max_analysis_duration_s: Optional[float] = 600.0
    min_track_duration_s: float = 20.0
    sections_per_s: float = 1 / 25.0


@dataclass(frozen=True)
class SliceConfig:
    min_chunk_s: float = 5.0                 # shortest chunk of a track
    max_chunk_s: Optional[float] = 120.0     # longest chunk; None = whole track allowed
    whole_track_ratio: float = 0.9           # chunks this close to full length use the entire track
    boundary_snap_s: float = 3.0             # pull chunk edges onto song-section boundaries
    window_hop_s: float = 1.0
    snap_window_s: float = 0.75
    zero_cross_window_ms: float = 15.0
    prefer_ambient: bool = True
    w_tonal: float = 0.30
    w_stability: float = 0.20
    w_energy: float = 0.15
    w_sparse: float = 0.20
    w_section: float = 0.15
    boundary_penalty: float = 0.08           # only applied to short chunks
    w_harmony: float = 0.8                   # weight of harmonic fit in --harmonize-with mode
    onset_clearance_s: float = 0.12


@dataclass(frozen=True)
class DSPConfig:
    attack_s: float = 6.0                    # fade-in of the first layer of each lane
    max_fade_s: float = 12.0                 # longest crossfade (chunks scale their fades up to this)
    envelope_curve: str = "equal_power"      # equal_power | cosine | exponential
    voice_target_rms: float = 0.1            # per-layer loudness matching (~ -20 dBFS)
    reverb_rt60_s: float = 4.5
    predelay_ms: float = 35.0
    damping_hz: float = 3_500.0
    reverb_wet: float = 0.55
    ir_seed: int = 7
    smear_enabled: bool = True
    smear_prob: float = 0.4                  # share of layers that become granular textures
    smear_stretch: float = 1.25
    grain_ms: float = 180.0
    grain_overlap: int = 4
    grain_jitter_ms: float = 40.0
    harmonize_target_key: Optional[str] = None   # e.g. "A": varispeed layers toward this key
    max_harmonize_shift: int = 3
    warmth_hz: float = 9_000.0
    stereo_spread: float = 0.35


@dataclass(frozen=True)
class MasterConfig:
    highpass_hz: float = 25.0
    target_rms_dbfs: float = -18.0
    agc_tau_s: float = 6.0
    agc_up_db_per_s: float = 2.0
    agc_down_db_per_s: float = 10.0
    agc_max_gain_db: float = 24.0
    ceiling_dbfs: float = -1.0
    knee_db: float = 6.0
    lookahead_ms: float = 50.0
    fade_out_s: float = 3.0


@dataclass(frozen=True)
class EngineConfig:
    library_dir: Path
    cache_dir: Path = Path(".ethereal_cache")
    use_cache: bool = True
    sample_rate: int = 44_100
    blocksize: int = 0                       # audio device block (0 = let PortAudio choose)
    device: Optional[Union[int, str]] = None
    engine_block: int = 2048                 # render-thread block (reverb partition size)
    buffer_s: float = 2.0                    # how far ahead the render thread stays (anti-glitch)
    max_workers: int = field(default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) - 1)))
    lanes: int = 2                           # polyphony: independent endless layer streams
    overlap_s: float = 4.0                   # minimum crossfade
    prefetch_s: float = 15.0                 # start decoding the next chunk this long before it is due
    memory_mb: int = 512                     # budget for decoded chunks held in RAM
    seed: int = 69
    strict_indices: bool = True
    follow_dynamics: bool = True             # --harmonize-with: orchestra swells with the guide song
    dynamics_depth: float = 0.5              # 0 = ignore guide loudness, 0.5 = up to -6 dB in quiet parts
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    slicing: SliceConfig = field(default_factory=SliceConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    master: MasterConfig = field(default_factory=MasterConfig)


# ============================================================================
# ANALYSIS CACHE
# ============================================================================
def file_signature(path: Path) -> tuple:
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)


class DiskCache:
    """Atomic pickle cache for track analyses (restarts are instant)."""

    def __init__(self, root: Path, namespace: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.dir = Path(root) / namespace
        if enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning("Cache disabled (%s)", exc)
                self.enabled = False

    @staticmethod
    def key(*parts) -> str:
        h = hashlib.sha1()
        for p in parts:
            h.update(repr(p).encode())
            h.update(b"\x1f")
        return h.hexdigest()

    def load(self, key: str):
        if not self.enabled:
            return None
        try:
            with open(self.dir / f"{key}.pkl", "rb") as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.debug("Ignoring bad cache entry %s (%s)", key[:10], exc)
            return None

    def store(self, key: str, obj) -> None:
        if not self.enabled:
            return
        path = self.dir / f"{key}.pkl"
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except OSError as exc:
            log.debug("Cache write failed (%s)", exc)
            tmp.unlink(missing_ok=True)


# ============================================================================
# MODULE 1 - LIBRARY INDEXER & SMART SLICER
# ============================================================================
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_SECTION_WEIGHT = {"ambient_breakdown": 1.0, "chorus": 0.6, "verse": 0.4, "intro": 0.3, "outro": 0.3}


@dataclass
class Section:
    start_s: float
    end_s: float
    label: str            # intro | verse | chorus | ambient_breakdown | outro
    energy: float
    onset_density: float


@dataclass
class TrackAnalysis:
    duration_s: float
    hop_s: float
    tempo_bpm: float
    key_pc: int
    key_mode: str
    chroma_mean: np.ndarray
    onset_times_s: np.ndarray
    strong_beats_s: np.ndarray
    boundary_frames: np.ndarray
    sections: list
    rms_n: np.ndarray
    flat_n: np.ndarray
    onset_n: np.ndarray
    section_weight: np.ndarray
    chroma_pooled: Optional[np.ndarray] = None   # (n, 12) ~1 s resolution, for harmony matching
    rms_pooled: Optional[np.ndarray] = None      # (n,) ~1 s loudness curve
    pool_s: float = 1.0

    @property
    def n_frames(self) -> int:
        return int(self.rms_n.size)

    @property
    def key_name(self) -> str:
        return f"{PITCH_NAMES[self.key_pc]} {self.key_mode}"


@dataclass
class TrackInfo:
    index: int
    path: Path
    analysis: Optional[TrackAnalysis] = None


@dataclass(frozen=True)
class SlicePlan:
    track_index: int
    start_s: float
    end_s: float
    shift: float = 0.0        # semitones to transpose by (harmonize-with mode)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class AudioSegment:
    track_index: int
    audio: np.ndarray     # (2, n) float32
    sample_rate: int
    start_s: float
    end_s: float
    key_pc: int
    key_name: str


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _estimate_key(chroma_mean: np.ndarray) -> tuple:
    best, best_r = (0, "major"), -2.0
    for mode, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for pc in range(12):
            r = np.corrcoef(chroma_mean, np.roll(prof, pc))[0, 1]
            if r > best_r:
                best, best_r = (pc, mode), r
    return best


def _label_sections(boundaries: np.ndarray, n: int, hop_s: float,
                    rms_n: np.ndarray, onset_n: np.ndarray):
    edges = np.concatenate([[0], boundaries, [n]]).astype(int)
    spans = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    energy = np.array([rms_n[a:b].mean() for a, b in spans])
    density = np.array([onset_n[a:b].mean() for a, b in spans])
    lo, hi = np.quantile(energy, 0.33), np.quantile(energy, 0.67)
    med_e, med_d = np.median(energy), np.median(density)
    sections, weights = [], np.zeros(n, dtype=np.float32)
    for i, ((a, b), e, d) in enumerate(zip(spans, energy, density)):
        if i == 0 and e <= med_e:
            label = "intro"
        elif i == len(spans) - 1 and e <= med_e:
            label = "outro"
        elif e <= lo and d <= med_d:
            label = "ambient_breakdown"
        elif e >= hi:
            label = "chorus"
        else:
            label = "verse"
        sections.append(Section(a * hop_s, b * hop_s, label, float(e), float(d)))
        weights[a:b] = _SECTION_WEIGHT[label]
    return sections, weights


def analyze_track(path: Path, cfg: AnalysisConfig) -> TrackAnalysis:
    """Onsets, beats, key, structural sections and per-frame descriptors.
    Runs inside a worker PROCESS during playback, so it never competes for the GIL."""
    try:
        y, sr = librosa.load(str(path), sr=cfg.analysis_sr, mono=True,
                             duration=cfg.max_analysis_duration_s)
    except Exception as exc:
        raise AnalysisError(f"Cannot decode {path.name}: {exc}") from exc
    if y.size < cfg.min_track_duration_s * sr:
        raise AnalysisError(f"{path.name} is shorter than {cfg.min_track_duration_s:.0f}s")

    hop = cfg.hop_length
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=hop)
    n = min(rms.size, flat.size, onset_env.size, chroma.shape[1])
    rms, flat, onset_env, chroma = rms[:n], flat[:n], onset_env[:n], chroma[:, :n]

    rms_n, flat_n = _norm01(rms), _norm01(np.log10(flat + 1e-6))
    onset_n = _norm01(onset_env)
    hop_s, duration = hop / sr, y.size / sr

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    beats = np.asarray(beats)[np.asarray(beats) < n]
    strong = beats[onset_n[beats] >= np.percentile(onset_n[beats], 75)] if beats.size else beats

    w = rms_n + 1e-6
    chroma_mean = (chroma * w).sum(axis=1) / w.sum()
    key_pc, key_mode = _estimate_key(chroma_mean)

    pool = max(1, int(round(1.0 / hop_s)))
    n_cols = n // pool
    chroma_pooled = rms_pooled = None
    if n_cols >= 1:
        chroma_pooled = chroma[:, : n_cols * pool].reshape(12, n_cols, pool).mean(axis=2).T.astype(np.float32)
        rms_pooled = rms_n[: n_cols * pool].reshape(n_cols, pool).mean(axis=1).astype(np.float32)
    k = int(np.clip(round(duration * cfg.sections_per_s), 2, 12))
    boundary_frames = np.array([], dtype=int)
    if n_cols >= 4:
        feat = np.vstack([chroma, rms_n[None], 0.5 * onset_n[None]])[:, : n_cols * pool]
        pooled = feat.reshape(feat.shape[0], n_cols, pool).mean(axis=2)
        try:
            bounds = librosa.segment.agglomerative(pooled, min(k, n_cols))
            boundary_frames = (np.asarray(bounds[1:]) * pool).astype(int)
        except Exception:
            boundary_frames = np.linspace(0, n, k + 1, dtype=int)[1:-1]
    boundary_frames = np.unique(boundary_frames[(boundary_frames > 0) & (boundary_frames < n)])

    sections, weights = _label_sections(boundary_frames, n, hop_s, rms_n, onset_n)
    return TrackAnalysis(
        duration_s=duration, hop_s=hop_s, tempo_bpm=float(np.atleast_1d(tempo)[0]),
        key_pc=key_pc, key_mode=key_mode, chroma_mean=chroma_mean.astype(np.float32),
        onset_times_s=np.sort(onset_frames) * hop_s, strong_beats_s=strong * hop_s,
        boundary_frames=boundary_frames, sections=sections,
        rms_n=rms_n, flat_n=flat_n, onset_n=onset_n, section_weight=weights,
        chroma_pooled=chroma_pooled, rms_pooled=rms_pooled, pool_s=pool * hop_s)


def _worker_init() -> None:
    """Analysis workers run at low priority so audio threads always win the CPU."""
    try:
        if hasattr(os, "nice"):
            os.nice(10)
        elif sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
    except Exception:
        pass


def _analyze_job(path_str: str, cfg: AnalysisConfig) -> TrackAnalysis:
    return analyze_track(Path(path_str), cfg)


def keys_compatible(a: TrackAnalysis, b: TrackAnalysis) -> bool:
    """Same or adjacent on the circle of fifths (relative major/minor = same position)."""
    def pos(t: TrackAnalysis) -> int:
        major_pc = t.key_pc if t.key_mode == "major" else (t.key_pc + 3) % 12
        return (major_pc * 7) % 12
    d = abs(pos(a) - pos(b))
    return min(d, 12 - d) <= 1


def best_shift(track_chroma: np.ndarray, target_chroma: np.ndarray, max_shift: int = 3) -> tuple:
    """(fit, semitones): transposition within +/-max_shift that best matches the target chroma."""
    tc = target_chroma - target_chroma.mean()
    best = (-2.0, 0.0)
    for s in range(-max_shift, max_shift + 1):
        r = np.roll(track_chroma, s)
        rc = r - r.mean()
        fit = float(rc @ tc / (np.linalg.norm(rc) * np.linalg.norm(tc) + 1e-9)) - 0.02 * abs(s)
        if fit > best[0]:
            best = (fit, float(s))
    return best


class LibraryIndexer:
    """Concurrent scanner + process-pool analyser.

    Indices are positions in a case-insensitively sorted file list: stable across
    runs and unaffected by files that fail to decode.
    """

    def __init__(self, library_dir: Path, config: AnalysisConfig,
                 cache: Optional[DiskCache] = None, max_workers: int = 2) -> None:
        self.library_dir = Path(library_dir)
        self.config, self.cache, self.max_workers = config, cache, max_workers
        self.tracks: dict = {}
        self.failures: dict = {}
        self._scanned = False
        self._scan_lock = threading.Lock()
        self._pool: Optional[ProcessPoolExecutor] = None
        self._pool_lock = threading.Lock()

    # -- scanning
    @staticmethod
    def _walk(directory: Path) -> list:
        found = []
        for root, _d, files in os.walk(directory, followlinks=True):
            found += [Path(root) / f for f in files if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        return found

    def scan(self) -> list:
        with self._scan_lock:
            if self._scanned:
                return [t.path for t in self.tracks.values()]
            if not self.library_dir.is_dir():
                raise LibraryError(f"Library directory not found: {self.library_dir}")
            with timed("Library scan"):
                top = list(self.library_dir.iterdir())
                files = [p for p in top if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                from concurrent.futures import ThreadPoolExecutor
                subdirs = [p for p in top if p.is_dir()]
                with ThreadPoolExecutor(max(1, self.max_workers), thread_name_prefix="scan") as pool:
                    for res in pool.map(self._walk, subdirs):
                        files.extend(res)
            files.sort(key=lambda p: str(p).lower())
            if not files:
                raise LibraryError(f"No audio files {SUPPORTED_EXTENSIONS} in {self.library_dir}")
            self.tracks = {i: TrackInfo(i, p) for i, p in enumerate(files)}
            self._scanned = True
            log.info("Found %d audio files", len(files))
            for i, p in enumerate(files):
                log.debug("  [%d] %s", i, p)
            return files

    # -- analysis
    def _cache_key(self, info: TrackInfo) -> str:
        return DiskCache.key("analysis-v4", file_signature(info.path), self.config)

    def _get_pool(self) -> ProcessPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                ctx = multiprocessing.get_context("spawn")      # safe with threads / PortAudio
                self._pool = ProcessPoolExecutor(max_workers=max(1, self.max_workers),
                                                 mp_context=ctx, initializer=_worker_init)
            return self._pool

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None

    def _store_result(self, info: TrackInfo, analysis: TrackAnalysis) -> None:
        info.analysis = analysis
        if self.cache and self.cache.enabled:
            self.cache.store(self._cache_key(info), analysis)
        log.info("[%d] %s: %s, %.0f BPM, %d sections", info.index, info.path.name,
                 analysis.key_name, analysis.tempo_bpm, len(analysis.sections))

    def _fail(self, info: TrackInfo, exc: Exception) -> None:
        self.failures[info.index] = str(exc)
        log.warning("[%d] %s skipped: %s", info.index, info.path.name, exc)

    def ensure_analyzed(self, indices: Iterable[int], parallel: bool = True) -> None:
        """Analyse tracks (cache first, then worker processes). parallel=False runs them
        one at a time, which keeps CPU use minimal while music is playing."""
        self.scan()
        todo = []
        for i in sorted(set(indices)):
            if i not in self.tracks:
                raise TrackNotFoundError(
                    f"Track index {i} out of range (library has {len(self.tracks)} tracks: 0..{len(self.tracks) - 1})")
            info = self.tracks[i]
            if info.analysis is not None or i in self.failures:
                continue
            if self.cache and self.cache.enabled:
                hit = self.cache.load(self._cache_key(info))
                if isinstance(hit, TrackAnalysis):
                    info.analysis = hit
                    continue
            todo.append(info)
        if not todo:
            return
        with timed(f"Analysis of {len(todo)} track(s)"):
            try:
                pool = self._get_pool()
                if parallel:
                    futs = {pool.submit(_analyze_job, str(t.path), self.config): t for t in todo}
                    for fut in as_completed(futs):
                        t = futs[fut]
                        try:
                            self._store_result(t, fut.result())
                        except BrokenProcessPool:
                            raise
                        except Exception as exc:
                            self._fail(t, exc)
                else:
                    for t in todo:
                        try:
                            self._store_result(t, pool.submit(_analyze_job, str(t.path),
                                                              self.config).result())
                        except BrokenProcessPool:
                            raise
                        except Exception as exc:
                            self._fail(t, exc)
            except (BrokenProcessPool, OSError, RuntimeError) as exc:
                log.warning("Worker processes unavailable (%s); analysing in-process", exc)
                self.close()
                for t in todo:
                    if t.analysis is None and t.index not in self.failures:
                        try:
                            self._store_result(t, analyze_track(t.path, self.config))
                        except Exception as exc2:
                            self._fail(t, exc2)

    def start_background_analysis(self, indices: Sequence[int], stop: threading.Event) -> threading.Thread:
        """Analyse the rest of the library, one track at a time, while music plays."""
        def worker() -> None:
            for i in list(indices):
                if stop.is_set():
                    return
                try:
                    self.ensure_analyzed([i], parallel=False)
                except EtherealError as exc:
                    log.warning("Background analysis: %s", exc)
            log.info("Background analysis complete: %d usable / %d failed",
                     len(self.analyzed_indices()), len(self.failures))
        th = threading.Thread(target=worker, name="bg-analyse", daemon=True)
        th.start()
        return th

    def resolve_track(self, spec: str) -> tuple:
        """Find the guide song: a track number, a file path, or part of a file name.
        Returns (path, index_in_library_or_None)."""
        self.scan()
        spec = str(spec).strip()
        if spec.isdigit():
            i = int(spec)
            if i not in self.tracks:
                raise TrackNotFoundError(f"--harmonize-with {i}: library has tracks 0..{len(self.tracks) - 1}")
            return self.tracks[i].path, i
        p = Path(spec).expanduser()
        if p.is_file():
            rp = p.resolve()
            for i, t in self.tracks.items():
                if t.path.resolve() == rp:
                    return t.path, i
            return p, None                      # a file outside the library: guide only
        hits = [i for i, t in self.tracks.items() if spec.lower() in t.path.name.lower()]
        if len(hits) == 1:
            return self.tracks[hits[0]].path, hits[0]
        if not hits:
            raise TrackNotFoundError(f"--harmonize-with {spec!r}: no such file or track name")
        names = "; ".join(f"[{i}] {self.tracks[i].path.name}" for i in hits[:5])
        raise TrackNotFoundError(f"--harmonize-with {spec!r} matches several tracks: {names} - use the number")

    def analyze_file(self, path: Path) -> TrackAnalysis:
        """Analyse one file completely (no length cap), via the worker pool and cache."""
        cfg = dataclasses.replace(self.config, max_analysis_duration_s=None, min_track_duration_s=10.0)
        key = DiskCache.key("analysis-v4-guide", file_signature(path), cfg)
        if self.cache and self.cache.enabled:
            hit = self.cache.load(key)
            if isinstance(hit, TrackAnalysis):
                return hit
        with timed(f"Analysis of guide song {path.name}"):
            try:
                result = self._get_pool().submit(_analyze_job, str(path), cfg).result()
            except (BrokenProcessPool, OSError, RuntimeError):
                self.close()
                result = analyze_track(path, cfg)
        if self.cache and self.cache.enabled:
            self.cache.store(key, result)
        return result

    def is_ready(self, index: int) -> bool:
        t = self.tracks.get(index)
        return t is not None and t.analysis is not None

    def analyzed_indices(self) -> list:
        return [i for i, t in list(self.tracks.items()) if t.analysis is not None]

    def get(self, index: int) -> TrackInfo:
        if index in self.failures:
            raise TrackNotFoundError(f"Track {index} failed analysis: {self.failures[index]}")
        info = self.tracks.get(index)
        if info is None or info.analysis is None:
            raise TrackNotFoundError(f"Track {index} is not indexed/analysed")
        return info

    def compatible_with(self, index: int) -> list:
        ref = self.get(index).analysis
        return [i for i in self.analyzed_indices()
                if i != index and keys_compatible(ref, self.tracks[i].analysis)]

    def group_by_key(self) -> dict:
        groups: dict = {}
        for i in self.analyzed_indices():
            groups.setdefault(self.tracks[i].analysis.key_name, []).append(i)
        return groups


class SmartSlicer:
    """Decides WHICH part of a track to play and HOW MUCH of it: any length from a few
    seconds to the entire track. Edges are pulled onto the song's own section
    boundaries, then onto quiet / onset-free frames, then onto zero crossings."""

    def __init__(self, config: SliceConfig, sample_rate: int) -> None:
        self.cfg, self.sr = config, sample_rate

    def plan(self, track: TrackInfo, duration_s: float, exclude: Sequence = (),
             target_chroma: Optional[np.ndarray] = None, max_shift: int = 3) -> SlicePlan:
        a = track.analysis
        if a is None:
            raise AnalysisError(f"Track {track.index} has no analysis")
        c = self.cfg
        duration_s = float(np.clip(duration_s, 1.0, a.duration_s))
        shift = 0.0
        if duration_s >= c.whole_track_ratio * a.duration_s:     # the entire track, start to end
            start_s, end_s = 0.0, a.duration_s
            if target_chroma is not None:
                shift = best_shift(a.chroma_mean, target_chroma, max_shift)[1]
        else:
            start_s, shift = self._select_window(a, duration_s, exclude, target_chroma, max_shift)
            end_s = min(start_s + duration_s, a.duration_s)
            bt = a.boundary_frames * a.hop_s
            if bt.size:
                def to_boundary(t: float) -> float:
                    j = int(np.argmin(np.abs(bt - t)))
                    return float(bt[j]) if abs(bt[j] - t) <= c.boundary_snap_s else t
                s1, e1 = to_boundary(start_s), to_boundary(end_s)
                if e1 - s1 >= 0.6 * duration_s:
                    start_s, end_s = s1, e1
        s2, e2 = self._snap(a, start_s), self._snap(a, end_s)
        if e2 - s2 >= 0.6 * (end_s - start_s):
            start_s, end_s = s2, e2
        return SlicePlan(track.index, start_s, end_s, float(shift))

    def _select_window(self, a: TrackAnalysis, dur: float, exclude,
                       target_chroma: Optional[np.ndarray] = None, max_shift: int = 3):
        c, hop, n = self.cfg, a.hop_s, a.n_frames
        w = max(2, int(round(dur / hop)))
        if n <= w:
            return 0.0, 0.0

        def wmean(x: np.ndarray) -> np.ndarray:
            cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
            return (cs[w:] - cs[:-w]) / w

        mean_rms = wmean(a.rms_n)
        std_rms = np.sqrt(np.maximum(wmean(a.rms_n ** 2) - mean_rms ** 2, 0.0))
        tonal = 1.0 - wmean(a.flat_n)
        sparse = 1.0 - wmean(a.onset_n)
        energy = 1.0 - np.clip(np.abs(mean_rms - 0.55) * 2.0, 0.0, 1.0)
        stability = 1.0 - np.clip(std_rms * 3.0, 0.0, 1.0)
        section = wmean(a.section_weight) if c.prefer_ambient else np.zeros_like(tonal)
        starts = np.arange(n - w + 1)
        crossings = (np.searchsorted(a.boundary_frames, starts + w, side="left")
                     - np.searchsorted(a.boundary_frames, starts, side="right"))
        bp = c.boundary_penalty if dur < 30.0 else 0.0     # long chunks are meant to span sections
        score = (c.w_tonal * tonal + c.w_stability * stability + c.w_energy * energy
                 + c.w_sparse * sparse + c.w_section * section - bp * crossings)

        shifts = np.zeros(starts.size)
        if target_chroma is not None and a.chroma_pooled is not None and a.chroma_pooled.shape[0] >= 2:
            fit, shifts = self._harmony_fit(a, starts, dur, target_chroma, max_shift)
            score = score + c.w_harmony * (fit + 1.0) / 2.0

        step = max(1, int(round(c.window_hop_s / hop)))
        mask = np.zeros(starts.size, dtype=bool)
        mask[::step] = True
        t0 = starts * hop
        for es, ee in exclude:               # reusing a track picks a fresh region
            mask &= ~((t0 < ee) & (t0 + dur > es))
        if not mask.any():
            mask[::step] = True
        score = score + 0.01 * np.random.default_rng().random(score.size)   # avoid exact repeats
        best = int(np.argmax(np.where(mask, score, -np.inf)))
        return float(t0[best]), float(shifts[best])

    @staticmethod
    def _harmony_fit(a: TrackAnalysis, starts: np.ndarray, dur: float,
                     target_chroma: np.ndarray, max_shift: int):
        """For every candidate window: best correlation between its average chroma (allowing
        transposition by up to +/-max_shift semitones) and the guide song's chroma."""
        cp = a.chroma_pooled.astype(np.float64)
        nc = cp.shape[0]
        dcols = int(np.clip(round(dur / a.pool_s), 1, nc))
        cs = np.vstack([np.zeros((1, 12)), np.cumsum(cp, axis=0)])
        col = np.clip((starts * a.hop_s / a.pool_s).astype(int), 0, nc - dcols)
        means = (cs[col + dcols] - cs[col]) / dcols                       # (S, 12)
        tc = target_chroma - target_chroma.mean()
        tn = np.linalg.norm(tc) + 1e-9
        best_fit = np.full(starts.size, -2.0)
        best_s = np.zeros(starts.size)
        for s in range(-max_shift, max_shift + 1):
            r = np.roll(means, s, axis=1)
            rc = r - r.mean(axis=1, keepdims=True)
            fit = (rc @ tc) / (np.linalg.norm(rc, axis=1) * tn + 1e-9) - 0.02 * abs(s)
            better = fit > best_fit
            best_fit = np.where(better, fit, best_fit)
            best_s = np.where(better, float(s), best_s)
        return best_fit, best_s

    def _snap(self, a: TrackAnalysis, t: float) -> float:
        hop = a.hop_s
        centre = int(round(t / hop))
        r = max(1, int(round(self.cfg.snap_window_s / hop)))
        lo, hi = max(0, centre - r), min(a.n_frames, centre + r + 1)
        if hi <= lo:
            return t
        idx = np.arange(lo, hi)
        cost = a.rms_n[lo:hi] + a.onset_n[lo:hi] + 0.05 * np.abs(idx - centre) / r
        ot = a.onset_times_s
        if ot.size:
            tt = idx * hop
            j = np.searchsorted(ot, tt)
            d = np.minimum(np.abs(tt - ot[np.clip(j - 1, 0, ot.size - 1)]),
                           np.abs(tt - ot[np.clip(j, 0, ot.size - 1)]))
            cost = cost + 0.5 * np.clip(1.0 - d / max(self.cfg.onset_clearance_s, 1e-3), 0.0, 1.0)
        return float(np.clip(idx[np.argmin(cost)] * hop, 0.0, a.duration_s))

    def extract(self, track: TrackInfo, plan: SlicePlan) -> AudioSegment:
        a = track.analysis
        margin = 0.5
        off = max(0.0, plan.start_s - margin)
        try:
            y, _ = librosa.load(str(track.path), sr=self.sr, mono=False, offset=off,
                                duration=plan.duration_s + 2 * margin)
        except Exception as exc:
            raise AnalysisError(f"Cannot read chunk from {track.path.name}: {exc}") from exc
        if y.ndim == 1:
            y = np.stack([y, y])
        elif y.shape[0] == 1:
            y = np.repeat(y, 2, axis=0)
        y = y[:2].astype(np.float32, copy=False)
        mono = y.mean(axis=0)
        i0 = self._zero_cross(mono, int(round((plan.start_s - off) * self.sr)))
        i1 = min(self._zero_cross(mono, int(round((plan.end_s - off) * self.sr))), y.shape[1])
        if i1 - i0 < self.sr:
            raise AnalysisError(f"Chunk of {track.path.name} too short after alignment")
        return AudioSegment(track.index, np.ascontiguousarray(y[:, i0:i1]), self.sr,
                            plan.start_s, plan.end_s, a.key_pc, a.key_name)

    def _zero_cross(self, x: np.ndarray, target: int) -> int:
        win = int(self.cfg.zero_cross_window_ms * self.sr / 1000)
        lo, hi = max(0, target - win), min(x.size - 1, target + win)
        if hi <= lo:
            return int(np.clip(target, 0, x.size))
        sign = np.signbit(x[lo:hi + 1])
        cross = np.flatnonzero(sign[:-1] != sign[1:]) + lo
        if cross.size == 0:
            return int(np.clip(target, 0, x.size))
        return int(cross[np.argmin(np.abs(cross - target))])


# ============================================================================
# MODULE 2 - ETHEREAL DSP: streaming voices, granular smear, shared cathedral reverb
# ============================================================================
def fade_curve(kind: str, x: np.ndarray, rising: bool) -> np.ndarray:
    """Cloud-like fade shapes for x in [0, 1]."""
    if kind == "equal_power":          # sin/cos pair: overlapping fades keep constant power
        return np.sin(0.5 * np.pi * x) if rising else np.cos(0.5 * np.pi * x)
    if kind == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * x) if rising else 0.5 + 0.5 * np.cos(np.pi * x)
    if kind == "exponential":
        y = (1 - np.exp(-5 * x)) / (1 - np.exp(-5))
        return y if rising else 1 - y
    raise ValueError(f"Unknown envelope curve: {kind}")


def interp_cubic(src: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Catmull-Rom interpolation of (C, N) `src` at fractional positions `pos` (any shape)."""
    n = src.shape[1]
    i = np.floor(pos).astype(np.int64)
    f = (pos - i).astype(np.float32)
    y0, y1 = src[:, np.clip(i - 1, 0, n - 1)], src[:, np.clip(i, 0, n - 1)]
    y2, y3 = src[:, np.clip(i + 1, 0, n - 1)], src[:, np.clip(i + 2, 0, n - 1)]
    c1 = 0.5 * (y2 - y0)
    c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
    return ((c3 * f + c2) * f + c1) * f + y1


class StreamVoice:
    """One layer: plays a decoded chunk block by block. Either plain playback (optionally
    varispeed to pull toward a key) or a granular 'smear' texture. Holds only the dry
    chunk in memory; reverb lives on the shared bus, so memory stays small."""

    def __init__(self, src: np.ndarray, sr: int, cfg: DSPConfig, *, rate: float,
                 granular: bool, gain: float, seed: int, label: str) -> None:
        self.src, self.sr, self.cfg, self.label = src, sr, cfg, label
        self.rate = float(rate)
        n = src.shape[1]
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.granular = False
        if granular:
            K = max(int(cfg.grain_overlap), 2)
            self.K = K
            self.hop = max(1, int(cfg.grain_ms * sr / 1000) // K)
            self.G = self.hop * K
            self.J = int(cfg.grain_jitter_ms * sr / 1000)
            self.stretch = cfg.smear_stretch
            length = int(n / self.rate * self.stretch) - self.G
            if length > sr:
                self.granular = True
                self.length = length
                self.win = signal.windows.hann(self.G, sym=False).astype(np.float32)
                self.carry = np.zeros((2, self.G), dtype=np.float32)
                self.next_grain = 0
        if not self.granular:
            self.length = int((n - 4) / self.rate)
        pan = rng.uniform(-cfg.stereo_spread, cfg.stereo_spread)       # equal-power balance
        th = (pan + 1.0) * np.pi / 4.0
        self.gains = (np.array([[np.cos(th)], [np.sin(th)]], dtype=np.float32)
                      * np.float32(np.sqrt(2.0) * gain))
        self.sos = signal.butter(2, min(cfg.warmth_hz, sr * 0.45), "low", fs=sr, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, 2))
        self.fade_in = self.fade_out = 0
        self._fi_s = self._fo_s = 0.0
        self.start = 0
        self.pos = 0
        self.dead = False

    @property
    def end(self) -> int:
        return self.start + self.length

    def set_fades(self, fade_in_s: float, fade_out_s: float) -> None:
        self._fi_s, self._fo_s = fade_in_s, fade_out_s
        cap = int(0.45 * self.length)
        self.fade_in = int(min(max(fade_in_s, 0.0) * self.sr, cap))
        self.fade_out = int(min(max(fade_out_s, 0.0) * self.sr, cap))

    def truncate(self, max_len: int, fade_out_s: Optional[float] = None) -> None:
        """Shorten this layer (used so the orchestra ends exactly with the guide song)."""
        if max_len < self.length:
            self.length = max(int(max_len), 1)
            self.set_fades(self._fi_s, self._fo_s if fade_out_s is None else fade_out_s)

    def _envelope(self, idx: np.ndarray) -> np.ndarray:
        kind = self.cfg.envelope_curve
        env = np.ones(idx.size, dtype=np.float32)
        if self.fade_in > 0:
            m = idx < self.fade_in
            if m.any():
                env[m] = fade_curve(kind, idx[m] / self.fade_in, True)
        if self.fade_out > 0:
            j = idx - (self.length - self.fade_out)
            m = j > 0
            if m.any():
                env[m] *= fade_curve(kind, np.minimum(j[m] / self.fade_out, 1.0), False)
        return env

    def _plain(self, m: int) -> np.ndarray:
        p0 = self.pos
        if self.rate == 1.0:
            x = self.src[:, p0:p0 + m]
            if x.shape[1] < m:
                x = np.pad(x, ((0, 0), (0, m - x.shape[1])))
            return x
        return interp_cubic(self.src, (p0 + np.arange(m)) * self.rate)

    def _granular(self, m: int) -> np.ndarray:
        p0, G, n = self.pos, self.G, self.src.shape[1]
        acc = np.zeros((2, m + G), dtype=np.float32)
        acc[:, :G] += self.carry
        ar = np.arange(G) * self.rate
        while self.next_grain < p0 + m:
            s = self.next_grain
            base = s * self.rate / self.stretch + int(self.rng.integers(-self.J, self.J + 1))
            grain = interp_cubic(self.src, np.clip(base + ar, 1, n - 3)) * self.win
            acc[:, s - p0:s - p0 + G] += grain
            self.next_grain += self.hop
        self.carry = acc[:, m:m + G].copy()
        return acc[:, :m] / (self.K / 2.0)             # periodic Hann, K-fold overlap sums to K/2

    def render(self, m: int):
        """Next m output samples -> (audio (2, m), envelope (m,))."""
        x = self._granular(m) if self.granular else self._plain(m)
        x, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        env = self._envelope(np.arange(self.pos, self.pos + m))
        self.pos += m
        return (x * env * self.gains).astype(np.float32, copy=False), env


class PartitionedConvolver:
    """Uniformly-partitioned FFT convolution (overlap-save), block by block. Cost per
    block is a few hundred thousand complex MACs, so a 5 s cathedral IR is trivial in
    real time (no giant FFTs, no per-layer reverb)."""

    def __init__(self, ir: np.ndarray, block: int) -> None:
        self.B = block
        m = ir.shape[1]
        self.P = P = int(np.ceil(m / block))
        padded = np.zeros((2, P * block), dtype=np.float32)
        padded[:, :m] = ir
        H = sfft.rfft(padded.reshape(2, P, block), n=2 * block, axis=-1)       # (2, P, B+1)
        self.Hrev = np.ascontiguousarray(H[:, ::-1, :])
        self.X = np.zeros((2, 2 * P, block + 1), dtype=np.complex64)
        self.i = 0
        self.prev = np.zeros((2, block), dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        B, P, i = self.B, self.P, self.i
        spec = sfft.rfft(np.concatenate([self.prev, x], axis=1), axis=-1)
        self.prev = x.astype(np.float32, copy=True)
        self.X[:, i] = spec
        self.X[:, i + P] = spec                 # duplicated so the last P spectra are contiguous
        acc = np.einsum("cpk,cpk->ck", self.X[:, i + 1:i + 1 + P], self.Hrev)
        self.i = (i + 1) % P
        return sfft.irfft(acc, n=2 * B, axis=-1)[:, B:].astype(np.float32, copy=False)


class EtherealDSPProcessor:
    def __init__(self, sample_rate: int, config: DSPConfig, block: int = 2048) -> None:
        self.sr, self.cfg, self.block = sample_rate, config, block
        self.live: Optional[LiveParams] = None      # set by the game bridge
        self.ir = self.synthesize_ir()
        log.info("Cathedral IR ready: %.1fs tail (RT60 %.1fs)", self.ir.shape[1] / sample_rate,
                 config.reverb_rt60_s)

    def synthesize_ir(self) -> np.ndarray:
        """Decorrelated stereo noise IR: exponential decay (-60 dB at RT60), darker late
        tail, diffuse build-up, pre-delay, early reflections. Unit energy per channel."""
        c, sr = self.cfg, self.sr
        rng = np.random.default_rng(c.ir_seed)
        n = int(c.reverb_rt60_s * 1.15 * sr)
        t = np.arange(n) / sr
        noise = rng.standard_normal((2, n))
        lp = signal.butter(2, min(c.damping_hz, sr * 0.45), "low", fs=sr, output="sos")
        dark = signal.sosfilt(lp, noise, axis=-1)
        w = np.clip(t / c.reverb_rt60_s, 0.0, 1.0) ** 0.6
        ir = ((1 - w) * noise + w * dark) * 10.0 ** (-3.0 * t / c.reverb_rt60_s)
        ir *= 1.0 - np.exp(-t / 0.03)
        for tap in (0.011, 0.019, 0.027, 0.041, 0.057):
            ir[:, int(tap * sr)] += rng.choice([-1.0, 1.0], size=2) * 3.2 * 10.0 ** (-3.0 * tap / c.reverb_rt60_s)
        hp = signal.butter(2, 90.0, "high", fs=sr, output="sos")
        ir = signal.sosfilt(hp, ir, axis=-1)
        ir = np.pad(ir, ((0, 0), (int(c.predelay_ms * sr / 1000), 0)))
        ir /= np.sqrt(np.sum(ir ** 2, axis=-1, keepdims=True))
        return ir.astype(np.float32)

    def new_convolver(self) -> PartitionedConvolver:
        return PartitionedConvolver(self.ir, self.block)

    def harmonize_semitones(self, key_pc: Optional[int]) -> float:
        target = self.cfg.harmonize_target_key
        if target is None or key_pc is None:
            return 0.0
        diff = (pitch_class_from_name(target) - key_pc + 6) % 12 - 6
        return float(diff) if abs(diff) <= self.cfg.max_harmonize_shift else 0.0

    def make_voice(self, audio: np.ndarray, *, key_pc: Optional[int], seed: int,
                   label: str, semitones: Optional[float] = None) -> StreamVoice:
        c = self.cfg
        if self.live is not None:                                     # game-driven texture controls
            c = dataclasses.replace(c, smear_stretch=float(self.live.stretch),
                                    smear_prob=float(self.live.smear_prob),
                                    warmth_hz=float(self.live.warmth_hz))
        rng = np.random.default_rng(seed)
        semis = semitones if semitones is not None else self.harmonize_semitones(key_pc)
        rate = 2.0 ** (semis / 12.0)                                  # varispeed toward the key
        granular = c.smear_enabled and c.smear_stretch > 1.0 and rng.random() < c.smear_prob
        rms = float(np.sqrt(np.mean(audio[:, ::8].astype(np.float64) ** 2))) + 1e-9
        gain = float(np.clip(c.voice_target_rms / rms, 0.25, 4.0))    # match layer loudness
        if granular:
            gain *= 1.5            # grains overlap incoherently (~ -4 dB); compensate
        return StreamVoice(audio, self.sr, c, rate=rate, granular=granular, gain=gain,
                           seed=seed + 1, label=label)


# ============================================================================
# MIXER, MASTER, RING BUFFER, RENDER THREAD
# ============================================================================
class Mixer:
    """Sample-clock mixer. Layers are scheduled at absolute sample positions and summed
    block by block; stacked layers are power-normalised from their envelopes so
    equal-power crossfades stay transparent while piles of layers cannot build up."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self.clock = 0
        self._lock = threading.Lock()
        self._voices: list = []

    def schedule(self, voice: StreamVoice, start: int) -> int:
        with self._lock:
            voice.start = max(int(start), self.clock)
            self._voices.append(voice)
            return voice.start

    def playing_tracks(self) -> dict:
        """{track_index: audibility 0..1} for layers sounding right now (used by the visuals)."""
        with self._lock:
            voices, t = list(self._voices), self.clock
        out: dict = {}
        for v in voices:
            ti = getattr(v, "track_index", None)
            if ti is None or v.dead or not (v.start <= t < v.end):
                continue
            pos, lvl = t - v.start, 1.0
            if v.fade_in and pos < v.fade_in:
                lvl = pos / v.fade_in
            rem = v.length - pos
            if v.fade_out and rem < v.fade_out:
                lvl = min(lvl, rem / v.fade_out)
            out[ti] = max(out.get(ti, 0.0), float(lvl))
        return out

    def active_voices(self) -> int:
        with self._lock:
            return sum(1 for v in self._voices if v.start <= self.clock < v.end)

    def render(self, frames: int) -> np.ndarray:
        with self._lock:
            t0 = self.clock
            self.clock = t1 = t0 + frames
            voices = list(self._voices)
        out = np.zeros((2, frames), dtype=np.float32)
        power = np.zeros(frames, dtype=np.float32)
        for v in voices:
            a, b = max(t0, v.start), min(t1, v.end)
            if b <= a or v.dead:
                continue
            try:
                audio, env = v.render(b - a)
            except Exception as exc:                   # one bad layer must never silence the rest
                v.dead = True
                log.warning("Layer dropped (%s: %s)", type(exc).__name__, exc)
                continue
            out[:, a - t0:b - t0] += audio
            power[a - t0:b - t0] += env * env
        with self._lock:
            self._voices = [v for v in self._voices if v.end > t1 and not v.dead]
        out /= np.sqrt(np.maximum(power, 1.0))
        return out


class StreamingMasterer:
    """Block-based master chain with persistent state: rumble high-pass -> slow RMS AGC
    (steady lush level) -> soft-knee look-ahead peak limiter (never exceeds the ceiling)."""

    def __init__(self, config: MasterConfig, sample_rate: int) -> None:
        self.cfg, self.sr = config, sample_rate
        self.sos = signal.butter(2, config.highpass_hz, "high", fs=sample_rate, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, 2))
        self.L = max(3, int(config.lookahead_ms * sample_rate / 1000)) | 1
        self.hist_a = np.zeros((2, 2 * self.L), dtype=np.float32)
        self.hist_g = np.zeros(2 * self.L, dtype=np.float32)
        self.ms: Optional[float] = None
        self.agc_db = 0.0
        self.peak_db = -120.0
        self.gr_db = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        F = block.shape[1]
        y, self.zi = signal.sosfilt(self.sos, block, axis=-1, zi=self.zi)
        y = self._agc(y.astype(np.float32, copy=False), F)
        y = self._limit(y, F)
        self.peak_db = float(20 * np.log10(max(float(np.max(np.abs(y))), 1e-9)))
        return y

    def _agc(self, y: np.ndarray, F: int) -> np.ndarray:
        c = self.cfg
        m = float(np.mean(y.astype(np.float64) ** 2))
        if m > 1e-9:
            alpha = 1.0 - np.exp(-F / (c.agc_tau_s * self.sr))
            self.ms = m if self.ms is None else self.ms + alpha * (m - self.ms)
        new_db = self.agc_db
        if self.ms is not None:
            target = float(np.clip(c.target_rms_dbfs - 10 * np.log10(self.ms + 1e-12),
                                   -c.agc_max_gain_db, c.agc_max_gain_db))
            step = (c.agc_up_db_per_s if target > self.agc_db else c.agc_down_db_per_s) * F / self.sr
            new_db = self.agc_db + float(np.clip(target - self.agc_db, -step, step))
        g = (10.0 ** (np.linspace(self.agc_db, new_db, F, endpoint=False) / 20.0)).astype(np.float32)
        self.agc_db = new_db
        return y * g

    def _limit(self, y: np.ndarray, F: int) -> np.ndarray:
        c, L = self.cfg, self.L
        peak = np.max(np.abs(y), axis=0)
        over = 20.0 * np.log10(np.maximum(peak, 1e-9)) - c.ceiling_dbfs
        k = max(c.knee_db, 1e-3)
        gr = np.where(over < -k / 2, 0.0,
                      np.where(over <= k / 2, -((over + k / 2) ** 2) / (2 * k), -over)).astype(np.float32)
        ha = np.concatenate([self.hist_a, y], axis=1)
        hg = np.concatenate([self.hist_g, gr])
        sm = uniform_filter1d(minimum_filter1d(hg, size=L, mode="nearest"), size=L, mode="nearest")
        out = ha[:, L:L + F] * (10.0 ** (sm[L:L + F] / 20.0)).astype(np.float32)
        self.hist_a, self.hist_g = ha[:, -2 * L:].copy(), hg[-2 * L:].copy()
        self.gr_db = float(-sm[L:L + F].min())
        ceil = 10.0 ** (c.ceiling_dbfs / 20.0)
        return np.clip(out, -ceil, ceil)


class LiveParams:
    """Thread-safe-by-convention (plain float reads/writes) bridge from the game to the DSP.
    wet/tone/gain act within one render block (plus the render-ahead buffer); stretch,
    smear_prob, warmth_hz and lanes are picked up by the next layer each lane prepares."""

    def __init__(self, cfg: "EngineConfig") -> None:
        d = cfg.dsp
        self.wet = d.reverb_wet
        self.tone_hz = 16_000.0
        self.gain = 1.0
        self.stretch = d.smear_stretch
        self.smear_prob = d.smear_prob
        self.warmth_hz = d.warmth_hz
        self.lanes = cfg.lanes

    def snapshot(self) -> dict:
        return dict(wet=self.wet, tone_hz=self.tone_hz, gain=self.gain, stretch=self.stretch,
                    smear_prob=self.smear_prob, warmth_hz=self.warmth_hz, lanes=self.lanes)


class RingBuffer:
    """Single-producer / single-consumer audio FIFO. The audio callback only ever calls
    read_into(), which is a couple of memcopies - it cannot be starved by heavy work."""

    def __init__(self, capacity: int, channels: int = 2) -> None:
        self.buf = np.zeros((channels, capacity), dtype=np.float32)
        self.cap = capacity
        self.w = 0          # total frames written (monotonic)
        self.r = 0          # total frames read (monotonic)

    def available_read(self) -> int:
        return self.w - self.r

    def available_write(self) -> int:
        return self.cap - (self.w - self.r)

    def write(self, data: np.ndarray) -> None:
        n = data.shape[1]
        i = self.w % self.cap
        first = min(n, self.cap - i)
        self.buf[:, i:i + first] = data[:, :first]
        if n > first:
            self.buf[:, :n - first] = data[:, first:]
        self.w += n                                  # publish only after the copy

    def read_into(self, out: np.ndarray) -> int:
        """out: (frames, channels). Returns frames actually available."""
        n = min(out.shape[0], self.available_read())
        i = self.r % self.cap
        first = min(n, self.cap - i)
        out[:first] = self.buf[:, i:i + first].T
        if n > first:
            out[first:n] = self.buf[:, :n - first].T
        self.r += n
        return n


class RenderThread(threading.Thread):
    """Keeps the ring buffer topped up: mix -> shared reverb -> master. Because it runs
    seconds ahead of playback, momentary CPU hiccups are absorbed instead of heard."""

    def __init__(self, cfg: EngineConfig, mixer: Mixer, convolver: PartitionedConvolver,
                 masterer: StreamingMasterer, ring: RingBuffer,
                 dynamics: Optional[tuple] = None, live: Optional[LiveParams] = None) -> None:
        super().__init__(name="render", daemon=True)
        self.live = live
        self._wet_s = cfg.dsp.reverb_wet
        self._gain_s = 1.0
        self._tone_zi = [np.zeros((2, 1)), np.zeros((2, 1))]
        self.dynamics = dynamics          # (curve 0..1, seconds per point, depth) or None
        self.cfg, self.mixer, self.conv, self.master, self.ring = cfg, mixer, convolver, masterer, ring
        self.stop_flag = threading.Event()
        self.max_block_ms = 0.0
        self.errors = 0
        self._fade_total = 0
        self._fade_pos = 0

    def start_fade(self, seconds: float) -> None:
        self._fade_pos = 0
        self._fade_total = max(1, int(seconds * self.cfg.sample_rate))

    def run(self) -> None:
        B = self.cfg.engine_block
        target = int(self.cfg.buffer_s * self.cfg.sample_rate)
        last_err = 0.0
        while not self.stop_flag.is_set():
            if self.ring.available_read() < target and self.ring.available_write() >= B:
                t0 = time.perf_counter()
                try:
                    block = self._make_block(B)
                except Exception as exc:
                    self.errors += 1
                    block = np.zeros((2, B), dtype=np.float32)
                    if time.time() - last_err > 5:
                        log.error("Render error: %s: %s", type(exc).__name__, exc)
                        last_err = time.time()
                self.ring.write(block)
                self.max_block_ms = max(self.max_block_ms, (time.perf_counter() - t0) * 1000)
            else:
                self.stop_flag.wait(0.004)

    def _tone(self, y: np.ndarray, fc: float) -> np.ndarray:
        """Two cascaded one-pole low-passes (12 dB/oct) whose cutoff the game moves live."""
        sr = self.cfg.sample_rate
        if fc >= 0.42 * sr:
            for zi in self._tone_zi:                      # keep states tidy while bypassed
                zi[:] = 0.0
            return y
        a = float(np.exp(-2.0 * np.pi * max(fc, 300.0) / sr))
        b, den = [1.0 - a], [1.0, -a]
        for k in range(2):
            y, self._tone_zi[k] = signal.lfilter(b, den, y, axis=-1, zi=self._tone_zi[k])
        return y.astype(np.float32, copy=False)

    def _dynamics_gain(self, t0: float, B: int) -> np.ndarray:
        curve, pool_s, depth = self.dynamics

        def at(t: float) -> float:
            return float(np.interp(t / pool_s - 0.5, np.arange(curve.size), curve))
        g0 = 1.0 - depth * (1.0 - at(t0))
        g1 = 1.0 - depth * (1.0 - at(t0 + B / self.cfg.sample_rate))
        return np.linspace(g0, g1, B, endpoint=False).astype(np.float32)

    def _make_block(self, B: int) -> np.ndarray:
        wet = self.cfg.dsp.reverb_wet
        live = self.live
        t_block = self.mixer.clock / self.cfg.sample_rate
        dry = self.mixer.render(B)
        if live is not None:                               # smooth per-block slew: no zipper noise
            self._wet_s += 0.2 * (float(min(max(live.wet, 0.0), 1.0)) - self._wet_s)
            wet = self._wet_s
        y = (1.0 - wet) * dry + wet * self.conv.process(dry)
        if live is not None:
            y = self._tone(y, float(live.tone_hz))
        y = self.master.process(y)
        if live is not None:                               # post-master gain (the AGC would undo it before)
            g0 = self._gain_s
            self._gain_s += 0.2 * (float(min(max(live.gain, 0.0), 1.0)) - g0)
            y = y * np.linspace(g0, self._gain_s, B, endpoint=False, dtype=np.float32)
        if self.dynamics is not None:                       # orchestra breathes with the guide song
            y = y * self._dynamics_gain(t_block, B)
        if self._fade_total:                               # fade AFTER reverb so the tail fades too
            idx = np.arange(B) + self._fade_pos
            g = np.where(idx < self._fade_total,
                         np.cos(0.5 * np.pi * np.minimum(idx / self._fade_total, 1.0)), 0.0)
            y = y * g.astype(np.float32)
            self._fade_pos += B
        return y


# ============================================================================
# MODULE 3 - ORCHESTRAL CONDUCTOR (endless sequencing, polyphonic lanes)
# ============================================================================
@dataclass
class HarmonyTarget:
    """The guide song for --harmonize-with. Only its analysis is used (key, chord movement
    as a ~1 s chroma timeline, loudness curve); its audio is never played or mixed."""
    path: Path
    duration_s: float
    key_name: str
    chroma: np.ndarray          # (n, 12)
    pool_s: float
    energy_curve: np.ndarray    # (n,) smoothed loudness 0..1

    @classmethod
    def from_analysis(cls, path: Path, a: TrackAnalysis) -> "HarmonyTarget":
        if a.chroma_pooled is None or a.chroma_pooled.shape[0] < 2:
            raise AnalysisError(f"{path.name}: too short to use as a harmony guide")
        e = a.rms_pooled if a.rms_pooled is not None else np.ones(a.chroma_pooled.shape[0])
        energy = _norm01(uniform_filter1d(e.astype(np.float64), size=6, mode="nearest"))
        return cls(path, a.duration_s, a.key_name, a.chroma_pooled, a.pool_s, energy)

    def span_chroma(self, t0: float, t1: float) -> np.ndarray:
        """Loudness-weighted average chroma of the guide song between t0 and t1 seconds."""
        n = self.chroma.shape[0]
        i0 = int(np.clip(t0 / self.pool_s, 0, n - 1))
        i1 = int(np.clip(np.ceil(t1 / self.pool_s), i0 + 1, n))
        w = self.energy_curve[i0:i1] + 0.1
        return (self.chroma[i0:i1] * w[:, None]).sum(axis=0) / w.sum()


@dataclass(frozen=True)
class SequenceItem:
    track_index: int
    duration_s: Optional[float] = None   # chunk length in SOURCE seconds; None = engine decides
    overlap_s: Optional[float] = None    # crossfade at the END of this layer; None = adaptive

    @classmethod
    def coerce(cls, spec) -> "SequenceItem":
        if isinstance(spec, SequenceItem):
            return spec
        if isinstance(spec, int):
            return cls(spec)
        if len(spec) not in (1, 2, 3):
            raise ValueError(f"Bad sequence entry {spec!r}; use (index[, seconds[, overlap]])")
        dur = spec[1] if len(spec) > 1 else None
        ov = spec[2] if len(spec) > 2 else None
        return cls(int(spec[0]), None if dur is None else float(dur), None if ov is None else float(ov))


class SequenceStrategy:
    """Explicit score, looped forever. Lane k starts at a rotated position so the same
    material is heard overlapping with itself from different places."""

    def __init__(self, items: Sequence, lane_id: int, lanes: int) -> None:
        self.items = list(items)
        self.pos = (lane_id * len(self.items)) // max(lanes, 1) % len(self.items)

    def next_item(self, prev_track: Optional[int]) -> Optional[SequenceItem]:
        item = self.items[self.pos]
        self.pos = (self.pos + 1) % len(self.items)
        return item


class GenerativeStrategy:
    """Endless generative choice from a pool: prefers harmonically compatible tracks,
    avoids recent ones, only draws from tracks that finished analysis. The chunk length
    is left to the conductor."""

    def __init__(self, indexer: LibraryIndexer, pool: Sequence, shared_rng: np.random.Generator,
                 lock: threading.Lock, recent: deque, harmony: Optional[HarmonyTarget] = None,
                 time_fn=None, max_shift: int = 3) -> None:
        self.indexer, self.pool = indexer, list(pool)
        self.rng, self.lock, self.recent = shared_rng, lock, recent
        self.harmony, self.time_fn, self.max_shift = harmony, time_fn, max_shift
        self.boost: dict = {}          # {track_index: weight multiplier}, set by the game

    def next_item(self, prev_track: Optional[int]) -> Optional[SequenceItem]:
        ready = [i for i in self.pool if self.indexer.is_ready(i)]
        if not ready:
            return None
        prev = (self.indexer.tracks[prev_track].analysis
                if prev_track is not None and self.indexer.is_ready(prev_track) else None)
        span = None
        if self.harmony is not None and self.time_fn is not None:     # what the guide song needs now
            t = self.time_fn()
            span = self.harmony.span_chroma(t, t + 30.0)
        with self.lock:
            w = np.ones(len(ready))
            for j, i in enumerate(ready):
                a = self.indexer.tracks[i].analysis
                if i in self.recent and len(ready) > 1:
                    w[j] *= 0.05
                w[j] *= float(self.boost.get(i, 1.0))
                if span is not None:
                    w[j] *= float(np.exp(2.5 * best_shift(a.chroma_mean, span, self.max_shift)[0]))
                elif prev is not None and not keys_compatible(prev, a):
                    w[j] *= 0.2
            choice = ready[int(self.rng.choice(len(ready), p=w / w.sum()))]
            self.recent.append(choice)
        return SequenceItem(choice)


class Lane(threading.Thread):
    """One endless stream of layers. While the current layer plays it prepares the next
    one (plan -> decode -> voice), then schedules it to start as the crossfade begins."""

    def __init__(self, lane_id: int, conductor: "OrchestralConductor", strategy,
                 stagger_s: float) -> None:
        super().__init__(name=f"lane-{lane_id}", daemon=True)
        self.id, self.c, self.strategy, self.stagger_s = lane_id, conductor, strategy, stagger_s
        self.prev_track: Optional[int] = None
        self.gated = False
        self.prefetch_s = conductor.cfg.prefetch_s

    def _next(self) -> Optional[SequenceItem]:
        while not self.c.stop_event.is_set():
            if self.id >= self.c.active_lanes():              # game says: fewer voices right now
                self.gated = True
                self.c.stop_event.wait(0.4)
                continue
            item = self.strategy.next_item(self.prev_track)
            if item is not None:
                return item
            self.c.stop_event.wait(0.5)       # waiting for background analysis
        return None

    def run(self) -> None:
        c, sr, mixer = self.c, self.c.cfg.sample_rate, self.c.mixer
        guard, end = int(0.25 * sr), self.c.end_clock
        first, counter, next_start, fade_in_s = True, 0, 0, 0.0
        while not c.stop_event.is_set():
            item = self._next()
            if item is None:
                break
            if self.gated:                                   # re-enter like a fresh lane (soft attack)
                first, self.gated = True, False
            try:
                t0 = time.perf_counter()
                attack = c.dsp.cfg.attack_s if first else fade_in_s
                est = (mixer.clock + guard + int(self.stagger_s * sr) + 2 * sr) if first else next_start
                if end is not None and est >= end - int(2.5 * sr):
                    log.info("%s: the guide song is about to end - no more layers", self.name)
                    break
                voice = c.prepare_voice(item, attack, c.cfg.seed + self.id * 100_003 + counter,
                                        t_start_s=est / sr)
                took = time.perf_counter() - t0
                if first:
                    start = mixer.clock + guard + int(self.stagger_s * sr)
                else:
                    start = next_start
                    if start < mixer.clock + guard:
                        log.warning("%s: preparing the chunk took %.1fs and finished late; "
                                    "starting earlier next time", self.name, took)
                        self.prefetch_s = min(self.prefetch_s * 1.5, 60.0)
                if end is not None:                  # make the orchestra end with the guide song
                    room = end - max(start, mixer.clock)
                    if room < int(2 * sr):
                        break
                    voice.truncate(room, fade_out_s=min(c.dsp.cfg.max_fade_s, 0.45 * room / sr))
                start = mixer.schedule(voice, start)
                log.info("%s +%.1fs  %s  %.0fs%s (prepared in %.1fs)", self.name,
                         (start - mixer.clock) / sr, voice.label, voice.length / sr,
                         " smear" if voice.granular else "", took)
                next_start = start + voice.length - voice.fade_out
                fade_in_s = voice.fade_out / sr
                self.prefetch_s = max(c.cfg.prefetch_s, 2.5 * took + 1.0)
                self.prev_track = item.track_index
                first, counter = False, counter + 1
                if end is not None and voice.end >= end - 1:
                    break
            except Exception as exc:         # endless show: log, skip this item, carry on
                log.warning("%s: layer failed (%s: %s); skipping", self.name, type(exc).__name__, exc)
                counter += 1
                c.stop_event.wait(1.0)
                continue
            wake = next_start - int(self.prefetch_s * sr)
            while not c.stop_event.is_set() and mixer.clock < wake:
                c.stop_event.wait(0.05)


class OrchestralConductor:
    def __init__(self, cfg: EngineConfig, indexer: LibraryIndexer, slicer: SmartSlicer,
                 dsp: EtherealDSPProcessor, mixer: Mixer,
                 sequence_spec: Optional[Sequence] = None, pool: Optional[Sequence] = None,
                 harmony: Optional[HarmonyTarget] = None, exclude_tracks: Sequence = ()) -> None:
        self.cfg, self.indexer, self.slicer, self.dsp, self.mixer = cfg, indexer, slicer, dsp, mixer
        self.sequence_spec, self.pool = sequence_spec, pool
        self.harmony, self.exclude = harmony, set(exclude_tracks)
        # timeline == mixer clock (starts at 0 when playback starts); the orchestra ends with the guide
        self.end_clock: Optional[int] = int(harmony.duration_s * cfg.sample_rate) if harmony else None
        self.stop_event = threading.Event()
        self.live: Optional[LiveParams] = None
        self.boost: dict = {}
        self.lanes: list = []
        self._used: dict = {}
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(cfg.seed + 7)
        # RAM budget: decoded chunks (stereo float32) of lanes+1 concurrent layers
        per_voice = cfg.memory_mb * 1e6 / (cfg.lanes + 1)
        self.mem_cap_s = per_voice / (cfg.sample_rate * 2 * 4)

    def start(self) -> None:
        cfg = self.cfg
        self.indexer.scan()
        n_total = len(self.indexer.tracks)
        mean_chunk = min(60.0, 0.5 * (cfg.slicing.min_chunk_s + (cfg.slicing.max_chunk_s or 120.0)))
        stagger = max(mean_chunk - cfg.overlap_s, 1.0) / max(cfg.lanes, 1)
        stagger = min(stagger, 20.0)

        if self.sequence_spec:
            items = [SequenceItem.coerce(s) for s in self.sequence_spec]
            dropped = [i.track_index for i in items if i.track_index in self.exclude]
            if dropped:
                log.warning("Track(s) %s are the guide song and are not used as material", sorted(set(dropped)))
                items = [i for i in items if i.track_index not in self.exclude]
            if not items:
                raise TrackNotFoundError("The sequence only contains the guide song")
            self.indexer.ensure_analyzed({i.track_index for i in items})
            good = [i for i in items if self.indexer.is_ready(i.track_index)]
            if len(good) != len(items) and cfg.strict_indices:
                bad = sorted({i.track_index for i in items} - {i.track_index for i in good})
                raise TrackNotFoundError(f"Could not use track indices {bad}: "
                                         + "; ".join(self.indexer.failures.get(b, "unknown") for b in bad))
            if not good:
                raise TrackNotFoundError("No playable items in the sequence")
            strategies = [SequenceStrategy(good, k, cfg.lanes) for k in range(cfg.lanes)]
            log.info("Looping an explicit score of %d items across %d lane(s)", len(good), cfg.lanes)
        else:
            pool = list(self.pool) if self.pool else list(range(n_total))
            pool = [i for i in pool if i not in self.exclude]        # guide song is never material
            if not pool:
                raise TrackNotFoundError("No tracks left to build the orchestra from")
            bad = [i for i in pool if i not in self.indexer.tracks]
            if bad:
                raise TrackNotFoundError(f"Track indices out of range: {bad} (library has {n_total})")
            order = [int(i) for i in np.random.default_rng(cfg.seed).permutation(pool)]
            first_batch = order[:max(cfg.lanes + 2, 4)]
            self.indexer.ensure_analyzed(first_batch)               # just enough to begin
            if not self.indexer.analyzed_indices():
                raise LibraryError("None of the first tracks could be analysed; check "
                                   "the files / ffmpeg installation")
            self.indexer.start_background_analysis(order[len(first_batch):], self.stop_event)
            rng, lock, recent = np.random.default_rng(cfg.seed), threading.Lock(), deque(maxlen=6)
            strategies = [GenerativeStrategy(self.indexer, pool, rng, lock, recent, self.harmony,
                                             self.timeline_now_s, self.dsp.cfg.max_harmonize_shift)
                          for _ in range(cfg.lanes)]
            log.info("Generative mode over %d candidate tracks, %d lane(s)", len(pool), cfg.lanes)

        for s_ in strategies:
            if hasattr(s_, "boost"):
                s_.boost = self.boost            # shared dict: the game mutates it in place
        self.lanes = [Lane(k, self, strategies[k], stagger * k) for k in range(cfg.lanes)]
        for lane in self.lanes:
            lane.start()

    def active_lanes(self) -> int:
        n = self.cfg.lanes if self.live is None else int(self.live.lanes)
        return max(1, min(self.cfg.lanes, n))

    def timeline_now_s(self) -> float:
        """Where on the guide song's timeline the next layer will start (roughly)."""
        return (self.mixer.clock + int(self.cfg.prefetch_s * self.cfg.sample_rate)) / self.cfg.sample_rate

    def stop(self) -> None:
        self.stop_event.set()
        for lane in self.lanes:
            lane.join(timeout=3.0)

    def choose_length(self, track: TrackInfo) -> float:
        """How much of this track to play: log-uniform between the minimum chunk and the
        longest allowed (up to the entire track), capped by the memory budget."""
        s, dur = self.cfg.slicing, track.analysis.duration_s
        hi = min(s.max_chunk_s or dur, dur, self.mem_cap_s)
        lo = min(s.min_chunk_s, hi)
        with self._lock:
            d = float(np.exp(self._rng.uniform(np.log(lo), np.log(hi)))) if hi > lo * 1.01 else hi
        return dur if d >= s.whole_track_ratio * dur and dur <= self.mem_cap_s else d

    def prepare_voice(self, item: SequenceItem, fade_in_s: float, seed: int,
                      t_start_s: float = 0.0) -> StreamVoice:
        track = self.indexer.get(item.track_index)
        want = item.duration_s if item.duration_s is not None else self.choose_length(track)
        if want > self.mem_cap_s:
            log.warning("Chunk of %.0fs exceeds the memory budget; using %.0fs (raise --memory-mb)",
                        want, self.mem_cap_s)
            want = self.mem_cap_s
        target = None
        if self.harmony is not None:                     # what the guide song needs at this moment
            want = min(want, max(self.harmony.duration_s - t_start_s, 3.0))
            target = self.harmony.span_chroma(t_start_s, t_start_s + want)
        with self._lock:
            used = self._used.setdefault(item.track_index, deque(maxlen=4))
            plan = self.slicer.plan(track, want, list(used), target_chroma=target,
                                    max_shift=self.dsp.cfg.max_harmonize_shift)
            used.append((plan.start_s, plan.end_s))
        seg = self.slicer.extract(track, plan)           # decode: C code, releases the GIL
        label = (f"[{item.track_index}] {track.path.name[:36]} "
                 f"{plan.start_s:.0f}-{plan.end_s:.0f}s ({seg.key_name}"
                 + (f" {plan.shift:+.0f}st" if self.harmony is not None else "") + ")")
        voice = self.dsp.make_voice(seg.audio, key_pc=seg.key_pc, seed=seed, label=label,
                                    semitones=plan.shift if self.harmony is not None else None)
        voice.track_index = item.track_index
        length_s = voice.length / self.cfg.sample_rate
        want_fade = item.overlap_s if item.overlap_s is not None else float(
            np.clip(0.12 * length_s, self.cfg.overlap_s, self.dsp.cfg.max_fade_s))
        voice.set_fades(fade_in_s, min(want_fade, 0.45 * length_s))
        return voice


# ============================================================================
# AUDIO OUTPUT (real-time callback = ring-buffer copy only)
# ============================================================================
class AudioOutput:
    def __init__(self, cfg: EngineConfig, ring: RingBuffer) -> None:
        self.cfg, self.ring = cfg, ring
        self.stream = None
        self.primed = False
        # first start waits for a comfortable cushion; after a dropout it resumes sooner
        self.need = int(min(cfg.buffer_s * 0.75, 1.5) * cfg.sample_rate)
        self.resume = int(0.5 * cfg.sample_rate)
        self.underruns = 0

    def _callback(self, outdata, frames, _time, status) -> None:
        try:
            if not self.primed:
                if self.ring.available_read() >= self.need:
                    self.primed = True
                    self.need = self.resume
                else:
                    outdata.fill(0)
                    return
            n = self.ring.read_into(outdata)
            if n < frames:                       # buffer ran dry: silence, refill, then resume
                outdata[n:] = 0
                self.underruns += 1
                self.primed = False
        except Exception:
            outdata.fill(0)

    def start(self) -> None:
        try:
            self.stream = sd.OutputStream(
                samplerate=self.cfg.sample_rate, channels=2, dtype="float32",
                blocksize=self.cfg.blocksize, latency="high", device=self.cfg.device,
                callback=self._callback)
            self.stream.start()
        except Exception as exc:
            raise EtherealError(f"Could not open audio output: {exc}") from exc

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ============================================================================
# MAIN / CLI
# ============================================================================
def parse_sequence(text: str) -> list:
    """'2:10,11:auto,100' -> [(2,10.0),(11,None),(100,None)]"""
    out = []
    try:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            idx, _, rest = part.partition(":")
            rest = rest.strip().lower()
            out.append((int(idx), None if rest in ("", "auto") else float(rest)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use INDEX[:SECONDS|auto] pairs, e.g. 2:10,11:auto,100") from exc
    return out


def parse_indices(text: str) -> list:
    try:
        return [int(p) for p in text.split(",") if p.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers, e.g. 2,11,100") from exc


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Endless real-time ethereal orchestra from your music folder.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--library", type=Path, help="folder with .wav/.mp3/.flac/.m4a files")
    p.add_argument("--tracks", type=parse_indices, help="restrict the endless mix to these track indices")
    p.add_argument("--sequence", type=parse_sequence,
                   help="explicit looping score INDEX[:SECONDS|auto],... (instead of generative choice)")
    p.add_argument("--min-chunk", type=float, default=6.0, help="shortest piece of a track to play (s)")
    p.add_argument("--max-chunk", type=float, default=60.0,
                   help="longest piece of a track to play (s); 0 = up to the entire track")
    p.add_argument("--lanes", type=int, default=3, help="polyphony: simultaneous layer streams")
    p.add_argument("--overlap", type=float, default=4.0, help="minimum crossfade (s)")
    p.add_argument("--max-fade", type=float, default=12.0, help="longest crossfade for long chunks (s)")
    p.add_argument("--rt60", type=float, default=4.5, help="reverb tail length (s)")
    p.add_argument("--wet", type=float, default=0.55, help="reverb amount 0..1")
    p.add_argument("--stretch", type=float, default=1.25, help="granular smear stretch (1 = no smear)")
    p.add_argument("--smear-prob", type=float, default=0.4, help="share of layers that become smeared textures")
    p.add_argument("--harmonize-key", help="pull layers toward this key, e.g. A or F# (varispeed, max 3 semitones)")
    p.add_argument("--harmonize-with", metavar="SONG",
                   help="build an orchestra that follows this song's key, chords and loudness for its whole "
                        "duration. SONG = track number, file path, or part of a file name. The song itself is "
                        "never used as material")
    p.add_argument("--no-dynamics", action="store_true",
                   help="with --harmonize-with: do not follow the guide song's loudness")
    p.add_argument("--max-shift", type=int, default=3,
                   help="largest pitch shift (semitones) used to fit layers to a key or guide song")
    p.add_argument("--level", type=float, default=-18.0, help="target RMS loudness (dBFS)")
    p.add_argument("--buffer", type=float, default=1.5,
                   help="seconds of audio rendered ahead; raise to 4-6 if you ever hear glitches")
    p.add_argument("--memory-mb", type=int, default=512, help="RAM budget for decoded chunks")
    p.add_argument("--device", help="output device index or name (see --list-devices)")
    p.add_argument("--samplerate", type=int, help="default: the output device's native rate")
    p.add_argument("--blocksize", type=int, default=0, help="audio device block size (0 = automatic)")
    p.add_argument("--workers", type=int, help="analysis worker processes")
    p.add_argument("--cache-dir", type=Path, default=Path(".ethereal_cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--lenient", action="store_true", help="skip unusable indices instead of failing")
    p.add_argument("--for", dest="for_s", type=float, help="stop after this many seconds")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    add_game_args(p)
    return p.parse_args(argv)


def build_config(a: argparse.Namespace, sample_rate: int) -> EngineConfig:
    base = EngineConfig(library_dir=a.library or Path("."))
    sl = dataclasses.replace(base.slicing, min_chunk_s=max(1.0, a.min_chunk),
                             max_chunk_s=(a.max_chunk if a.max_chunk > 0 else None))
    dsp = dataclasses.replace(
        base.dsp, reverb_rt60_s=a.rt60, reverb_wet=min(max(a.wet, 0.0), 1.0),
        smear_stretch=a.stretch, smear_enabled=a.stretch > 1.0, smear_prob=a.smear_prob,
        harmonize_target_key=None if a.harmonize_with else a.harmonize_key, max_fade_s=a.max_fade,
        max_harmonize_shift=max(0, a.max_shift))
    device: Optional[Union[int, str]] = a.device
    if device is not None and str(device).isdigit():
        device = int(device)
    return dataclasses.replace(
        base, cache_dir=a.cache_dir, use_cache=not a.no_cache, sample_rate=sample_rate,
        blocksize=a.blocksize, device=device, lanes=max(1, a.lanes), overlap_s=a.overlap,
        buffer_s=max(0.5, a.buffer), memory_mb=a.memory_mb, strict_indices=not a.lenient,
        follow_dynamics=not a.no_dynamics,
        max_workers=a.workers or base.max_workers, slicing=sl, dsp=dsp,
        master=dataclasses.replace(base.master, target_rms_dbfs=a.level))


def run_audio_only(args: argparse.Namespace) -> int:
    indexer: Optional[LibraryIndexer] = None
    try:
        require_dependencies()
        if args.list_devices:
            print(sd.query_devices())
            return 0
        if not args.library:
            raise LibraryError("--library is required")
        device = int(args.device) if args.device and str(args.device).isdigit() else args.device
        sr = args.samplerate
        if sr is None:
            try:
                sr = int(sd.query_devices(device, "output")["default_samplerate"])
            except Exception:
                sr = 44_100
        cfg = build_config(args, sr)
        log.info("Output %d Hz | %d lane(s) | %.1fs render-ahead buffer", cfg.sample_rate,
                 cfg.lanes, cfg.buffer_s)

        indexer = LibraryIndexer(cfg.library_dir, cfg.analysis,
                                 DiskCache(cfg.cache_dir, "analysis", cfg.use_cache), cfg.max_workers)
        harmony, exclude, dynamics = None, [], None
        if args.harmonize_with:
            guide_path, guide_idx = indexer.resolve_track(args.harmonize_with)
            harmony = HarmonyTarget.from_analysis(guide_path, indexer.analyze_file(guide_path))
            exclude = [guide_idx] if guide_idx is not None else []
            if args.harmonize_key:
                log.warning("--harmonize-key is ignored because --harmonize-with is set")
            m, s_ = divmod(int(harmony.duration_s), 60)
            log.info("Guide song: %s (%s, %d:%02d). The orchestra will follow it and play for its duration; "
                     "the song itself is NOT used as material.", guide_path.name, harmony.key_name, m, s_)
            if cfg.follow_dynamics:
                dynamics = (harmony.energy_curve, harmony.pool_s, cfg.dynamics_depth)
        mixer = Mixer(cfg.sample_rate)
        dsp = EtherealDSPProcessor(cfg.sample_rate, cfg.dsp, cfg.engine_block)
        conductor = OrchestralConductor(cfg, indexer, SmartSlicer(cfg.slicing, cfg.sample_rate),
                                        dsp, mixer, sequence_spec=args.sequence, pool=args.tracks,
                                        harmony=harmony, exclude_tracks=exclude)
        masterer = StreamingMasterer(cfg.master, cfg.sample_rate)
        ring = RingBuffer(int((cfg.buffer_s + 2.0) * cfg.sample_rate))
        renderer = RenderThread(cfg, mixer, dsp.new_convolver(), masterer, ring, dynamics)
        output = AudioOutput(cfg, ring)

        conductor.start()          # analyse first tracks, then lanes prepare layers in the background
        renderer.start()
        output.start()
        log.info("Playing (first sound in a few seconds). Ctrl+C to fade out and quit.")
        t_end = time.time() + args.for_s if args.for_s else None
        tail = int(min(cfg.dsp.reverb_rt60_s, 6.0) * 0.6 * cfg.sample_rate)
        tick = 0
        try:
            while t_end is None or time.time() < t_end:
                time.sleep(1.0)
                if conductor.end_clock is not None and mixer.clock >= conductor.end_clock + tail:
                    log.info("The guide song has ended - orchestra complete.")
                    break
                tick += 1
                if tick % 5:
                    continue
                where = ""
                if conductor.end_clock is not None:
                    t, tot = mixer.clock / cfg.sample_rate, conductor.end_clock / cfg.sample_rate
                    where = f"orchestra {int(t) // 60}:{int(t) % 60:02d}/{int(tot) // 60}:{int(tot) % 60:02d} | "
                log.info("status: %s%d layer(s) | peak %.1f dBFS | AGC %+.1f dB | limiter %.1f dB | "
                         "buffer %.1fs | worst render %.0f ms | underruns %d",
                         where, mixer.active_voices(), masterer.peak_db, masterer.agc_db, masterer.gr_db,
                         ring.available_read() / cfg.sample_rate, renderer.max_block_ms, output.underruns)
                renderer.max_block_ms = 0.0
        except KeyboardInterrupt:
            pass
        log.info("Fading out...")
        conductor.stop()
        renderer.start_fade(cfg.master.fade_out_s)
        time.sleep(cfg.master.fade_out_s + cfg.buffer_s + 0.5)
        output.stop()
        renderer.stop_flag.set()
        return 0
    except EtherealError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if indexer is not None:
            indexer.close()



# ============================================================================
# ============================================================================
#  GAME LAYER  -  an endless liminal mall in Ursina, wired to the orchestra
# ============================================================================
# ============================================================================
# Ursina is imported lazily (in _load_ursina) so that the analysis worker processes, which
# re-import this file under "spawn", never pay for - or open - a graphics stack.
ur = None            # the ursina module, once loaded
Image = ImageDraw = ImageFilter = ImageFont = None   # Pillow, once loaded

CH = 24.0                    # chunk length along the Path (multiple of 8: light-strip spacing)
CORR_HW, ROOM_HW = 5.0, 9.0  # half widths: corridor / sanctuary
WALL_H = 6.0
ZONE_PERIOD, ZONE_SLOT = 7, 3     # every 7th chunk is a sanctuary
SANCTUARIES = ("cafe", "gym", "restaurant")

ZONES = {   # fog / ambient / lamp (r, g, b, intensity) per zone, and its HUD name
    "corridor":   dict(title="THE PATH", fog=(0.72, 0.79, 0.92), dens=0.034, amb=(0.060, 0.068, 0.085),
                       lamp=(0.80, 0.90, 1.00, 1.70)),
    "cafe":       dict(title="CAFE LUMEN", fog=(0.95, 0.78, 0.58), dens=0.050, amb=(0.20, 0.14, 0.09),
                       lamp=(1.00, 0.72, 0.42, 1.35)),
    "gym":        dict(title="AEON ATHLETIC", fog=(0.70, 0.62, 0.95), dens=0.045, amb=(0.10, 0.09, 0.16),
                       lamp=(0.75, 0.80, 1.00, 1.45)),
    "restaurant": dict(title="RISTORANTE (CLOSED?)", fog=(0.58, 0.40, 0.34), dens=0.055, amb=(0.12, 0.07, 0.06),
                       lamp=(1.00, 0.55, 0.28, 0.90)),
}

# Audio personality of each place.  The bridge eases toward these (and modulates them by
# speed, ads, sitting...).  wet/tone/gain are live; stretch/smear/warmth/lanes apply per layer.
AUDIO_PRESETS = {
    "corridor":   dict(wet=0.74, tone_hz=4200.0, gain=1.00, stretch=1.60, smear_prob=0.65, warmth_hz=5500.0, lanes=2),
    "cafe":       dict(wet=0.38, tone_hz=11000.0, gain=0.92, stretch=1.10, smear_prob=0.15, warmth_hz=11000.0, lanes=3),
    "gym":        dict(wet=0.22, tone_hz=15000.0, gain=1.00, stretch=1.00, smear_prob=0.08, warmth_hz=14000.0, lanes=9),
    "restaurant": dict(wet=0.82, tone_hz=3200.0, gain=0.85, stretch=1.90, smear_prob=0.80, warmth_hz=4000.0, lanes=1),
}

# ---------------------------------------------------------------------------------------------
# v2 WORLD TUNING: new zones, per-kind geometry, winding path, sub-levels
# ---------------------------------------------------------------------------------------------
ZONES.update({
    "anchor":    dict(title="ANCHOR STORE  -  THE VOID", fog=(0.012, 0.012, 0.018), dens=0.095, amb=(0.005, 0.005, 0.007),
                      lamp=(0.55, 0.60, 0.80, 0.10)),
    "atrium":    dict(title="THE ABANDONED ATRIUM", fog=(0.62, 0.70, 0.64), dens=0.026, amb=(0.14, 0.16, 0.14),
                      lamp=(0.90, 0.95, 0.80, 1.10)),
    "skywindow": dict(title="SKY-WINDOW CORRIDOR", fog=(0.70, 0.80, 0.95), dens=0.028, amb=(0.10, 0.11, 0.14),
                      lamp=(0.80, 0.90, 1.00, 1.00)),
    "cinema":    dict(title="CINEMA CORRIDOR", fog=(0.20, 0.09, 0.14), dens=0.058, amb=(0.050, 0.030, 0.050),
                      lamp=(1.00, 0.50, 0.60, 0.75)),
    "restroom":  dict(title="RESTROOMS", fog=(0.82, 0.90, 0.90), dens=0.050, amb=(0.20, 0.22, 0.22),
                      lamp=(0.90, 1.00, 1.00, 1.45)),
    "security":  dict(title="SECURITY  -  STAFF ONLY", fog=(0.30, 0.40, 0.34), dens=0.050, amb=(0.08, 0.10, 0.09),
                      lamp=(0.60, 1.00, 0.70, 1.00)),
    "garage":    dict(title="LEVEL P-INFINITY  PARKING", fog=(0.34, 0.31, 0.23), dens=0.040, amb=(0.07, 0.065, 0.05),
                      lamp=(1.00, 0.75, 0.35, 1.25)),
    "pool":      dict(title="THE POOLROOMS", fog=(0.55, 0.85, 0.90), dens=0.034, amb=(0.30, 0.36, 0.38),
                      lamp=(0.90, 1.00, 1.00, 1.55)),
})
AUDIO_PRESETS.update({
    "anchor":    dict(wet=0.88, tone_hz=2400.0, gain=0.95, stretch=2.10, smear_prob=0.85, warmth_hz=3200.0, lanes=1),
    "atrium":    dict(wet=0.80, tone_hz=7000.0, gain=0.95, stretch=1.60, smear_prob=0.50, warmth_hz=7000.0, lanes=2),
    "skywindow": dict(wet=0.52, tone_hz=9500.0, gain=0.95, stretch=1.30, smear_prob=0.30, warmth_hz=9000.0, lanes=3),
    "cinema":    dict(wet=0.60, tone_hz=5000.0, gain=0.90, stretch=1.50, smear_prob=0.45, warmth_hz=5500.0, lanes=2),
    "restroom":  dict(wet=0.90, tone_hz=8000.0, gain=0.85, stretch=1.80, smear_prob=0.60, warmth_hz=8000.0, lanes=1),
    "security":  dict(wet=0.34, tone_hz=6500.0, gain=0.90, stretch=1.10, smear_prob=0.12, warmth_hz=7000.0, lanes=2),
    "garage":    dict(wet=0.92, tone_hz=2600.0, gain=0.90, stretch=2.00, smear_prob=0.80, warmth_hz=3000.0, lanes=1),
    "pool":      dict(wet=0.95, tone_hz=6200.0, gain=0.90, stretch=1.90, smear_prob=0.75, warmth_hz=6500.0, lanes=2),
})
# Convolution-reverb "rooms" (impulse responses swapped live on the shared reverb bus)
REVERB_PROFILES = {   # name: (rt60 s, damping Hz, predelay ms, ir seed)
    "tile":   (2.0, 9500.0, 6.0, 21),     # ceramic bathroom / pool tile: short, bright, flutter-ish
    "garage": (7.5, 2300.0, 55.0, 33),    # bare concrete cavern: very long, dark
    "hall":   (5.5, 5200.0, 40.0, 45),    # tall atrium
}
REVERB_OF = {"restroom": "tile", "pool": "tile", "garage": "garage", "anchor": "garage", "atrium": "hall"}
DRIP_RATE = {"restroom": 0.75, "pool": 0.55, "garage": 0.16, "atrium": 0.10, "anchor": 0.05}   # drips/s

ATRIUM_HW, ANCHOR_HW = 13.0, 15.0
KIND_HW = dict(corridor=CORR_HW, cafe=ROOM_HW, gym=ROOM_HW, restaurant=ROOM_HW, atrium=ATRIUM_HW, anchor=ANCHOR_HW,
               skywindow=7.0, cinema=8.0, restroom=8.0, security=9.0, garage=11.0, pool=11.0)
KIND_H = dict(atrium=11.0, anchor=8.0, cinema=7.0, garage=3.4, pool=4.8, skywindow=6.5)      # default WALL_H
LM0_KINDS = ("corridor", "skywindow", "cinema", "garage", "anchor")   # lit by the corridor-style strip rows
MULTI = {"anchor": 3, "cinema": 2}                 # zones that span several consecutive chunks
MAIN_ZONES = (("cafe", 1.0), ("gym", 1.0), ("restaurant", 1.0), ("atrium", 1.3), ("anchor", 0.9),
              ("skywindow", 1.3), ("cinema", 1.0))
MINOR_ZONES = (("restroom", 1.0), ("security", 0.8), (None, 1.6))
MINOR_SLOT = 1
SUBLEVELS = ("garage", "pool")
FORCED_BAYS = {(0, -1, 12): "kiosk", (0, 1, 20): "booth", (2, 1, 12): "elevator", (1, -1, 12): "payphone"}
FORCED_LOOT = {(0, "cassette"): (-2.6, 17.0), (0, "fuel"): (3.4, 6.0)}


def path_x(seed: int, idx: int) -> float:
    """Gentle sine-wave staggering of the Path's centre line (one value per chunk index)."""
    ph = (seed % 628) / 100.0
    return round(5.2 * math.sin(idx * 0.43 + ph) + 2.4 * math.sin(idx * 0.17 + 2.0 * ph), 2)


RARE_ODDS = 30          # a rare room replaces the sanctuary slot of a block with chance 1 / RARE_ODDS


def zone_info(seed: int, idx: int) -> tuple:
    """Deterministic (kind, part) for a chunk index; `part` counts chunks inside multi-chunk zones."""
    block, slot = divmod(idx, ZONE_PERIOD)
    if abs(idx) > 1:
        if ZONE_SLOT <= slot <= ZONE_SLOT + 2 and seeded(seed, "rare", block).randrange(max(1, RARE_ODDS)) == 0:
            return {ZONE_SLOT: ("jpcourt", 0), ZONE_SLOT + 1: ("japan", 1), ZONE_SLOT + 2: ("jpcourt", 2)}[slot]
        zk = seeded(seed, "zone", block)
        main = zk.choices([k for k, _ in MAIN_ZONES], [w for _, w in MAIN_ZONES])[0]
        n = MULTI.get(main, 1)
        if ZONE_SLOT <= slot < ZONE_SLOT + n:
            return main, slot - ZONE_SLOT
        if slot == MINOR_SLOT:
            m = seeded(seed, "minor", block).choices([k for k, _ in MINOR_ZONES], [w for _, w in MINOR_ZONES])[0]
            if m:
                return m, 0
    return "corridor", 0


def chunk_kind(seed: int, idx: int) -> str:
    return zone_info(seed, idx)[0]


def chunk_geom(seed: int, idx: int, level: str = "mall") -> tuple:
    """(kind, part, x offset, half width, ceiling height) - pure function, so neighbours can be asked."""
    if level != "mall":
        hw = KIND_HW[level]
        if level in BIOMES and idx % 5 == 3 and idx != 0:      # branching junction: left / right wings + straight on
            hw = JUNCTION_HW
        return level, 0, 0.0, hw, KIND_H[level]
    kind, part = zone_info(seed, idx)
    return kind, part, path_x(seed, idx - part), KIND_HW[kind], KIND_H.get(kind, WALL_H)


GLSL_VERT = """#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform vec2 uv_scale;
uniform vec2 uv_off;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
out vec3 wpos;
out float vdist;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0 * uv_scale + uv_off;
    wpos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    vdist = length((p3d_ModelViewMatrix * p3d_Vertex).xyz);
}
"""
GLSL_FRAG = """#version 130
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec3 fog_col;
uniform float fog_dens;
uniform vec3 amb;
uniform vec3 pl_pos;
uniform float pulse;
uniform float emissive;
uniform float lm;
uniform vec4 lamp;
uniform float xc;
uniform float emis_scale;
uniform float lights_k;
uniform float pl_k;
uniform vec3 fl_pos;
uniform vec3 fl_dir;
uniform float fl_k;
in vec2 uv;
in vec3 wpos;
in float vdist;
out vec4 fragColor;

vec3 plight(vec3 lp, vec3 col, vec3 n, float k) {
    vec3 L = lp - wpos;
    float d = length(L);
    float ndl = max(dot(n, L / d), 0.0) * 0.8 + 0.2;
    return col * (k * ndl / (1.0 + 0.045 * d * d));
}
void main() {
    vec4 base = texture(p3d_Texture0, uv) * p3d_ColorScale;
    vec3 n = normalize(cross(dFdx(wpos), dFdy(wpos)));
    // fluorescent strips (lm=0, two per row at x=+-2) or warm lamps (lm=1, three per row)
    float fz0 = floor((wpos.z - 4.0) / 8.0) * 8.0 + 4.0;
    vec3 cx = (lm < 0.5) ? vec3(-2.0, 2.0, 0.0) : vec3(-6.0, 0.0, 6.0);
    cx += vec3(xc);
    float w3 = (lm < 0.5) ? 0.0 : 1.0;
    float ly = (lm < 0.5) ? 5.7 : 4.3;
    vec3 lsum = vec3(0.0);
    for (int i = 0; i < 2; i++) {
        float fz = fz0 + 8.0 * float(i);
        lsum += plight(vec3(cx.x, ly, fz), lamp.rgb, n, lamp.a);
        lsum += plight(vec3(cx.y, ly, fz), lamp.rgb, n, lamp.a);
        lsum += plight(vec3(cx.z, ly, fz), lamp.rgb, n, lamp.a * w3);
    }
    vec3 pl = plight(pl_pos, vec3(0.30, 0.34, 0.42), n, 0.5 * pl_k);
    vec3 fl = vec3(0.0);
    if (fl_k > 0.0) {
        vec3 Lf = fl_pos - wpos;
        float df = length(Lf) + 1e-4;
        float spot = smoothstep(0.84, 0.95, dot(-Lf / df, fl_dir));
        float ndf = abs(dot(n, Lf / df)) * 0.7 + 0.3;
        fl = vec3(1.0, 0.95, 0.80) * (fl_k * spot * ndf * 3.4 / (1.0 + 0.05 * df * df));
    }
    vec3 lit = base.rgb * (amb + lsum * pulse * lights_k + pl + fl) + base.rgb * emissive * emis_scale;
    lit = 1.0 - exp(-lit * 1.35);
    float f = clamp(1.0 - exp(-vdist * fog_dens), 0.0, 1.0);
    fragColor = vec4(mix(lit, fog_col, f), base.a);
}
"""


def _load_ursina() -> None:
    """Import Ursina + Pillow into module globals (raises DependencyError if missing)."""
    global ur, Image, ImageDraw, ImageFilter, ImageFont
    if ur is not None:
        return
    try:
        import ursina as _ur
        from PIL import Image as _I, ImageDraw as _D, ImageFilter as _F, ImageFont as _T
    except ImportError as exc:
        raise DependencyError(f"Game mode needs ursina and Pillow ({exc}).  Install with:  "
                              "pip install ursina pillow   (optional: mutagen for embedded cover art)") from exc
    ur, Image, ImageDraw, ImageFilter, ImageFont = _ur, _I, _D, _F, _T


def C(r: float, g: float, b: float, a: float = 1.0):
    return ur.color.Color(float(r), float(g), float(b), float(a))


def seeded(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def stable_hash(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ============================================================================
# PROCEDURAL TEXTURES (concrete, marble, glass, brushed metal, wood, ...)
# ============================================================================
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


class TextureBank:
    """Generates every surface texture in numpy (spectral noise is tileable by construction),
    uploads them once, and caches them.  No texture files are needed."""
    SZ = 256

    def __init__(self, seed: int, flip: bool = False) -> None:
        self.seed, self.flip = seed, flip
        self._tex: dict = {}
        self._k = 0
        self._yy, self._xx = np.mgrid[0:self.SZ, 0:self.SZ]

    # -- helpers
    def to_texture(self, img, mip: bool = True):
        if self.flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        try:
            t = ur.Texture(img, filtering="mipmap") if mip else ur.Texture(img)
        except TypeError:
            t = ur.Texture(img)
        try:
            t._texture.setAnisotropicDegree(8)
        except Exception:
            pass
        return t

    def _fbm(self, beta: float) -> np.ndarray:
        n, self._k = self.SZ, self._k + 1
        rng = np.random.default_rng(self.seed * 7919 + self._k)
        f = np.fft.fftfreq(n)
        r2 = f[:, None] ** 2 + f[None, :] ** 2
        r2[0, 0] = 1.0
        spec = r2 ** (-beta / 4.0) * np.exp(2j * np.pi * rng.random((n, n)))
        spec[0, 0] = 0
        a = np.fft.ifft2(spec).real
        return (a - a.min()) / (np.ptp(a) + 1e-9)

    def _pack(self, rgb: np.ndarray, alpha=1.0) -> np.ndarray:
        a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), rgb.shape[:2])
        out = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])
        return (out * 255).astype(np.uint8)

    # -- generators
    def _gen_white(self):
        return np.full((8, 8, 4), 255, np.uint8)

    def _gen_concrete(self):
        yy, xx, n = self._yy, self._xx, self.SZ
        b = 0.34 + 0.30 * self._fbm(2.3) + 0.08 * (self._fbm(1.2) - 0.5)
        b = b - 0.14 * ((yy % 64) < 2) - 0.07 * ((xx % 128) < 1)             # formwork seams
        b = b - 0.28 * np.exp(-(((xx % 64) - 32) ** 2 + ((yy % 64) - 32) ** 2) / 9.0)   # tie holes
        return self._pack(b[..., None] * np.array([0.93, 0.95, 1.0]))

    def _gen_marble(self):
        n, f, g = self.SZ, self._fbm(2.6), self._fbm(1.8)
        x = np.linspace(0, 1, n, endpoint=False)
        v = np.sin(2 * np.pi * (2 * x[None, :] + 1 * x[:, None] + 2.2 * f))
        vein = np.exp(-(v * v) * 28.0) * (0.35 + 0.65 * f)
        b = 0.80 + 0.14 * g - 0.34 * vein
        edge = (self._xx < 3) | (self._yy < 3)
        b = np.where(edge, b * 0.55, b)
        return self._pack(b[..., None] * np.array([0.95, 0.95, 0.98]))

    def _gen_glass(self):
        n, f = self.SZ, self._fbm(2.0)
        x = np.linspace(0, 1, n, endpoint=False)
        s = np.mod(x[None, :] + 0.7 * x[:, None], 1.0)
        streak = np.exp(-((s - 0.35) ** 2) / 0.0016) + 0.5 * np.exp(-((s - 0.72) ** 2) / 0.0008)
        rgb = np.dstack([0.28 + 0.5 * streak, 0.42 + 0.5 * streak, 0.48 + 0.5 * streak])
        return self._pack(rgb, 0.16 + 0.10 * f + 0.55 * np.clip(streak, 0, 1))

    def _gen_metal(self):
        rng = np.random.default_rng(self.seed + 31)
        from scipy.ndimage import uniform_filter1d
        m = uniform_filter1d(rng.random((self.SZ, self.SZ)), size=48, axis=1, mode="wrap")
        m = (m - m.mean()) / (m.std() + 1e-9)
        b = 0.58 + 0.09 * m + 0.04 * np.sin(2 * np.pi * 3 * self._yy / self.SZ)
        return self._pack(b[..., None] * np.array([0.92, 0.95, 1.0]))

    def _gen_shutter(self):
        base = self._gen_metal()[..., :3] / 255.0
        rib = 0.78 + 0.22 * np.sin(2 * np.pi * 16 * self._yy / self.SZ)
        seam = 1.0 - 0.45 * ((self._yy % 16) < 1)
        return self._pack(base * (rib * seam)[..., None] * 0.8)

    def _gen_wood(self):
        n, rng = self.SZ, np.random.default_rng(self.seed + 57)
        from scipy.ndimage import uniform_filter1d
        grain = uniform_filter1d(rng.random((n, n)), size=40, axis=1, mode="wrap")
        grain = (grain - grain.min()) / (np.ptp(grain) + 1e-9)
        img = np.zeros((n, n, 3))
        for r in range(n // 32):
            for sgm in range(n // 128):
                ys, xs = slice(r * 32, r * 32 + 32), slice(sgm * 128, sgm * 128 + 128)
                tint = rng.uniform(0.78, 1.12)
                g = np.roll(grain, int(rng.integers(0, n)), axis=1)[ys, xs]
                img[ys, xs] = np.array([0.50, 0.32, 0.18]) * tint * (0.82 + 0.36 * g)[..., None]
        img[(self._yy % 32) < 1] *= 0.45
        img[(self._xx % 128) < 1] *= 0.55
        return self._pack(img)

    def _gen_terrazzo(self):
        n, rng = self.SZ, np.random.default_rng(self.seed + 91)
        img = (np.array([0.88, 0.84, 0.78]) + 0.06 * (self._fbm(1.6) - 0.5)[..., None]).copy()
        pal = [(0.55, 0.50, 0.45), (0.30, 0.30, 0.33), (0.76, 0.60, 0.50), (0.96, 0.96, 0.92)]
        for _ in range(520):
            s, y0, x0 = int(rng.integers(2, 6)), int(rng.integers(0, n)), int(rng.integers(0, n))
            img[np.ix_((y0 + np.arange(s)) % n, (x0 + np.arange(s)) % n)] = pal[int(rng.integers(0, 4))]
        return self._pack(img)

    def _gen_rubber(self):
        yy, xx = self._yy, self._xx
        b = 0.10 + 0.04 * self._fbm(1.0) + 0.035 * (((xx % 16) - 8) ** 2 + ((yy % 16) - 8) ** 2 < 9)
        return self._pack(b[..., None] * np.array([1.0, 1.0, 1.08]))

    def _gen_carpet(self):
        yy, xx = self._yy, self._xx
        d = np.abs((xx % 64) - 32) + np.abs((yy % 64) - 32)
        lat = ((d % 16) < 2) * 0.22
        base = np.array([0.30, 0.07, 0.10]) * (0.85 + 0.3 * self._fbm(0.8))[..., None]
        return self._pack(base + lat[..., None] * np.array([0.55, 0.42, 0.20]))

    def _gen_plaster(self):
        b = 0.84 + 0.10 * (self._fbm(2.0) - 0.5) - 0.06 * ((self._yy % 128) < 2)
        return self._pack(b[..., None] * np.array([1.0, 0.96, 0.88]))

    def _gen_panel(self):
        yy, xx = self._yy, self._xx
        b = 0.22 + 0.10 * self._fbm(2.2) + 0.05 * np.sin(2 * np.pi * 6 * xx / self.SZ)
        b = b - 0.10 * ((xx % 64) < 2) - 0.05 * ((yy % 128) < 3)
        return self._pack(b[..., None] * np.array([0.9, 0.55, 0.32]) * 1.6)

    def _gen_poster_blank(self):
        yy, xx = np.mgrid[0:64, 0:64]
        g = 0.15 + 0.5 * (1 - yy / 64.0)
        return self._pack(np.dstack([g * 0.8, g * 0.9, g]))

    def _gen_vending(self):
        rng = random.Random(self.seed + 5)
        im = Image.new("RGBA", (256, 384), (14, 16, 22, 255))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 255, 28], fill=(230, 60, 70, 255))
        d.text((12, 4), "REFRESH", fill=(255, 255, 255, 255), font=_font(18))
        pal = [(240, 90, 70), (70, 160, 240), (250, 200, 60), (90, 210, 140), (190, 110, 240), (240, 240, 240)]
        for r in range(5):
            for c in range(4):
                x, y = 16 + c * 52, 44 + r * 62
                d.rectangle([x, y, x + 40, y + 52], fill=rng.choice(pal) + (255,))
                d.rectangle([x + 4, y + 18, x + 36, y + 30], fill=(255, 255, 255, 200))
        d.rectangle([0, 360, 255, 383], fill=(0, 0, 0, 255))
        d.line([0, 40, 255, 220], fill=(255, 255, 255, 40), width=26)
        return np.asarray(im)

    # ---- v2 generators: tile, asphalt, cracked marble, star carpet, static, sky, clouds, aurora, glow, stars
    def _gen_tile(self):
        yy, xx = self._yy, self._xx
        b = 0.86 + 0.08 * self._fbm(2.0) - 0.03 * (((yy // 64) + (xx // 64)) % 2)
        b = np.where(((yy % 64) < 3) | ((xx % 64) < 3), 0.50, b)
        return self._pack(b[..., None] * np.array([0.90, 0.97, 0.98]))

    def _gen_asphalt(self):
        yy, xx = self._yy, self._xx
        b = 0.20 + 0.15 * self._fbm(2.2) + 0.04 * self._fbm(1.4)
        rgb = b[..., None] * np.array([0.95, 0.93, 0.88])
        line = ((xx % 128) < 4) | (((yy % 128) < 3) & ((xx % 128) < 64))
        rgb = np.where(line[..., None], np.array([0.74, 0.66, 0.22]) * (0.8 + 0.4 * b[..., None]), rgb)
        return self._pack(rgb)

    def _gen_cracked(self):
        im = Image.fromarray(self._gen_marble().copy())
        d, rng, n = ImageDraw.Draw(im), np.random.default_rng(self.seed + 77), self.SZ
        for _ in range(8):
            x, y, a = float(rng.integers(0, n)), float(rng.integers(0, n)), float(rng.uniform(0, 6.28))
            pts = [(x, y)]
            for _ in range(int(rng.integers(6, 13))):
                a += float(rng.normal(0, 0.6))
                x, y = x + math.cos(a) * float(rng.uniform(8, 24)), y + math.sin(a) * float(rng.uniform(8, 24))
                pts.append((x, y))
            d.line(pts, fill=(26, 26, 34, 255), width=2)
        return np.asarray(im)

    def _gen_starcarpet(self):
        yy, xx, n = self._yy, self._xx, self.SZ
        rng = np.random.default_rng(self.seed + 123)
        base = np.array([0.075, 0.030, 0.120]) * (0.8 + 0.5 * self._fbm(0.9))[..., None]
        lat = ((np.abs(((xx + yy) % 64) - 32) < 1) | (np.abs(((xx - yy) % 64) - 32) < 1)) * 0.20
        img = base + lat[..., None] * np.array([0.8, 0.62, 0.2])
        cols = [(0.2, 0.95, 1.0), (1.0, 0.3, 0.7), (1.0, 0.85, 0.3), (0.7, 0.5, 1.0)]
        for _ in range(46):
            cx, cy, c = int(rng.integers(4, n - 4)), int(rng.integers(4, n - 4)), np.array(cols[int(rng.integers(0, 4))])
            r = int(rng.integers(2, 6))
            for k in range(-r, r + 1):
                w = 1.0 - abs(k) / (r + 1)
                img[cy + k, cx] = np.maximum(img[cy + k, cx], c * w)
                img[cy, cx + k] = np.maximum(img[cy, cx + k], c * w)
            img[cy, cx] = (1.0, 1.0, 1.0)
        return self._pack(img)

    def _gen_static(self):
        rng = np.random.default_rng(self.seed + 9)
        r = rng.random((self.SZ, self.SZ))
        g = (0.18 + 0.82 * r) * (0.78 + 0.22 * np.sin(self._yy * np.pi))
        return self._pack(g[..., None] * np.array([0.95, 1.0, 0.97]))

    def _gen_sky(self):
        t = np.linspace(0.0, 1.0, 256)[:, None]                  # row 0 = top (flip-safe: symmetric use)
        g = 0.62 + 0.38 * np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1) ** 0.7
        return self._pack(np.dstack([g, g, g]).repeat(4, axis=1))

    def _gen_clouds(self):
        f = self._fbm(2.7)
        a = np.clip((f - 0.42) * 3.2, 0.0, 1.0)
        return self._pack(np.dstack([0.9 + 0.1 * f, 0.92 + 0.08 * f, 0.97 + 0.0 * f]), a * 0.92 + 0.08)

    def _gen_aurora(self):
        yy, xx, n = self._yy, self._xx, self.SZ
        v = 1.0 - yy / n
        wav = 0.5 + 0.5 * np.sin(xx * 0.09 + 6.0 * self._fbm(2.4))
        a = np.clip(v ** 1.6 * (0.35 + 0.65 * wav), 0, 1)
        rgb = np.dstack([0.25 + 0.5 * (1 - v), 1.0 - 0.45 * (1 - v), 0.55 + 0.45 * (1 - v)])
        return self._pack(rgb, a)

    def _gen_glow(self):
        yy, xx = np.mgrid[0:128, 0:128]
        d = np.hypot(yy - 63.5, xx - 63.5) / 64.0
        core = np.clip((0.34 - d) * 40.0, 0, 1)
        halo = np.clip(1.0 - d, 0, 1) ** 3
        a = np.maximum(core, halo * 0.55)
        return self._pack(np.dstack([np.ones_like(d), np.ones_like(d), np.ones_like(d)]), a)

    def _gen_stars(self):
        rng = np.random.default_rng(self.seed + 5)
        img = np.zeros((256, 512, 4), np.float64)
        for _ in range(420):
            y, x = int(rng.integers(0, 256)), int(rng.integers(0, 512))
            b = float(rng.uniform(0.35, 1.0)) ** 2
            img[y, x] = (1, 1, 1, b)
            if b > 0.6:
                img[y, (x + 1) % 512] = (1, 1, 1, b * 0.5)
        return (img * 255).astype(np.uint8)

    def _gen_vignette(self):
        yy, xx = np.mgrid[0:128, 0:128]
        d = np.hypot((yy - 63.5) / 64.0, (xx - 63.5) / 64.0)
        a = np.clip((d - 0.45) / 0.75, 0, 1) ** 1.6
        return self._pack(np.zeros((128, 128, 3)), a)

    def get(self, name: str):
        if name not in self._tex:
            arr = getattr(self, "_gen_" + name)()
            self._tex[name] = self.to_texture(Image.fromarray(arr), mip=name != "white")
        return self._tex[name]

    def sign(self, text: str, fg=(255, 235, 200), bg=(20, 14, 12), w: int = 512, h: int = 128, sub: str = ""):
        key = f"sign:{text}:{fg}:{bg}:{sub}"
        if key not in self._tex:
            im = Image.new("RGBA", (w, h), bg + (255,))
            d = ImageDraw.Draw(im)
            d.rectangle([3, 3, w - 4, h - 4], outline=fg + (255,), width=3)
            f = _font(int(h * (0.42 if sub else 0.5)))
            tw = d.textlength(text, font=f)
            d.text(((w - tw) / 2, h * (0.10 if sub else 0.24)), text, fill=fg + (255,), font=f)
            if sub:
                f2 = _font(int(h * 0.18))
                d.text(((w - d.textlength(sub, font=f2)) / 2, h * 0.68), sub, fill=fg + (200,), font=f2)
            self._tex[key] = self.to_texture(im)
        return self._tex[key]

    def board(self, title: str, lines: list):
        key = "board:" + title
        if key not in self._tex:
            im = Image.new("RGBA", (256, 320), (24, 28, 26, 255))
            d = ImageDraw.Draw(im)
            d.text((16, 12), title, fill=(240, 240, 230, 255), font=_font(26))
            for i, ln in enumerate(lines):
                d.text((16, 62 + i * 38), ln, fill=(220, 225, 215, 255), font=_font(17))
            self._tex[key] = self.to_texture(im)
        return self._tex[key]


# ============================================================================
# COVER ART  (LibraryIndexer -> light-box posters)
# ============================================================================
def read_embedded_cover(path: Path) -> Optional[bytes]:
    """Embedded album art via mutagen (optional) - mp3 APIC, flac pictures, m4a covr."""
    try:
        ext = path.suffix.lower()
        if ext == ".mp3":
            from mutagen.id3 import ID3
            for fr in ID3(str(path)).getall("APIC"):
                return bytes(fr.data)
        elif ext == ".flac":
            from mutagen.flac import FLAC
            pics = FLAC(str(path)).pictures
            if pics:
                return bytes(pics[0].data)
        elif ext == ".m4a":
            from mutagen.mp4 import MP4
            covr = (MP4(str(path)).tags or {}).get("covr")
            if covr:
                return bytes(covr[0])
    except Exception:
        pass
    return None


def read_folder_cover(folder: Path) -> Optional[bytes]:
    names = ("cover", "folder", "front", "album", "albumart", "artwork")
    try:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.stem.lower() in names and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return f.read_bytes()
    except Exception:
        pass
    return None


class CoverArtService:
    """Builds poster textures for the mall's ad boxes from the tracks the LibraryIndexer found.
    Decoding/composition happens on worker threads; GPU upload happens on the main thread."""
    W, H = 384, 512

    def __init__(self, indexer: Optional[LibraryIndexer], tex: TextureBank, seed: int) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self.indexer, self.tex, self.seed = indexer, tex, seed
        self._pool = ThreadPoolExecutor(2, thread_name_prefix="cover")
        self._futs: dict = {}
        self._imgs: dict = {}
        self._gpu: dict = {}

    def ids(self) -> list:
        if self.indexer is not None and self.indexer.tracks:
            return sorted(self.indexer.tracks)
        return [-(i + 1) for i in range(8)]          # procedural stand-ins until the library is scanned

    def title(self, tid: int) -> str:
        if self.indexer is not None and tid in self.indexer.tracks:
            return self.indexer.tracks[tid].path.stem
        return "ETHEREAL"

    def request(self, tid: int) -> None:
        if tid not in self._imgs and tid not in self._futs:
            self._futs[tid] = self._pool.submit(self._build, tid)

    def poll(self, budget: int = 2) -> None:
        for tid, fut in list(self._futs.items()):
            if fut.done() and budget > 0:
                del self._futs[tid]
                budget -= 1
                try:
                    self._imgs[tid] = fut.result()
                except Exception as exc:
                    log.debug("cover %s failed: %s", tid, exc)
                    self._imgs[tid] = self._compose(self._procedural(tid, None), tid, None)

    def texture(self, tid: int):
        if tid in self._gpu:
            return self._gpu[tid]
        if tid in self._imgs:
            if len(self._gpu) > 80:                  # keep VRAM bounded
                self._gpu.pop(next(iter(self._gpu)))
            self._gpu[tid] = self.tex.to_texture(self._imgs[tid])
            return self._gpu[tid]
        self.request(tid)
        return None

    # -- worker-thread side
    def _build(self, tid: int):
        info = self.indexer.tracks.get(tid) if (self.indexer is not None and tid >= 0) else None
        art = None
        if info is not None:
            raw = read_embedded_cover(info.path) or read_folder_cover(info.path.parent)
            if raw:
                try:
                    art = Image.open(io.BytesIO(raw)).convert("RGB")
                    s = min(art.size)
                    l, t = (art.width - s) // 2, (art.height - s) // 2
                    art = art.crop((l, t, l + s, t + s)).resize((self.W, self.W))
                except Exception:
                    art = None
        if art is None:
            art = self._procedural(tid, info.analysis if info is not None else None)
        return self._compose(art, tid, info)

    def _procedural(self, tid: int, a):
        """Ethereal generative cover: soft glowing orbs tinted by the song's key, spaced by its tempo."""
        rng = np.random.default_rng(stable_hash(self.seed, "art", tid))
        hue = (a.key_pc / 12.0) if a is not None else rng.random()
        n = self.W
        yy, xx = np.mgrid[0:n, 0:n] / float(n)
        img = np.zeros((n, n, 3))
        img += np.array(colorsys.hsv_to_rgb(hue, 0.55, 0.22)) * (1.0 - yy)[..., None]
        orbs = 5 + int((a.tempo_bpm if a is not None else 90) // 40)
        for k in range(orbs):
            cx, cy, r = rng.random(), rng.random() * 0.9, rng.uniform(0.08, 0.30)
            col = colorsys.hsv_to_rgb((hue + rng.uniform(-0.12, 0.12)) % 1.0, rng.uniform(0.25, 0.7), 1.0)
            img += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None] * np.array(col) * 0.7
        img += rng.normal(0, 0.012, img.shape)
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

    def _compose(self, art, tid: int, info):
        W, H = self.W, self.H
        poster = Image.new("RGB", (W, H), (10, 10, 16))
        glow = art.resize((W, W)).filter(ImageFilter.GaussianBlur(10))
        poster.paste(Image.blend(art.resize((W, W)), glow, 0.28), (0, 0))
        d = ImageDraw.Draw(poster)
        d.rectangle([0, W, W, H], fill=(8, 9, 14))
        d.line([0, W, W, W], fill=(190, 200, 235), width=2)
        title = self.title(tid).replace("_", " ")
        title = title if len(title) <= 26 else title[:25] + "..."
        d.text((16, W + 14), title, fill=(236, 238, 250), font=_font(24))
        a = info.analysis if info is not None else None
        sub = f"{a.key_name}  |  {a.tempo_bpm:.0f} bpm" if a is not None else "ETHEREAL ORCHESTRA"
        d.text((16, W + 56), sub, fill=(150, 160, 200), font=_font(16))
        d.text((16, W + 84), "NOW PLAYING IN THE MALL", fill=(110, 120, 170), font=_font(12))
        return poster.convert("RGBA")


class Ad:
    """One light-box / banner.  `faces` are textured quads, `bar` a status strip that lights
    when the ad's track is actually sounding in the orchestra."""
    def __init__(self, faces, bar, x, y, z, key, chunk) -> None:
        self.faces, self.bar, self.x, self.y, self.z, self.key, self.chunk = faces, bar, x, y, z, key, chunk
        self.track = None
        self.next_change = 0.0
        self.level = 0.0
        self.last_em = -1.0


class AdManager:
    def __init__(self, covers: CoverArtService, seed: int) -> None:
        self.covers, self.seed = covers, seed
        self.ads: list = []
        self._n_ids = len(covers.ids())
        self.skip: set = set()

    def pick(self, key: str, t: float) -> int:
        ids = self.covers.ids()
        return ids[stable_hash(self.seed, key, int(t // 120)) % len(ids)]

    def register(self, ad: Ad, t: float) -> None:
        self.ads.append(ad)
        self.assign(ad, self.pick(ad.key, t), t)

    def assign(self, ad: Ad, tid: int, t: float) -> None:
        ad.track = tid
        ad.next_change = t + 40 + (stable_hash(ad.key, tid) % 50)
        self._apply(ad)

    def _apply(self, ad: Ad) -> None:
        tex = self.covers.texture(ad.track) or self.covers.tex.get("poster_blank")
        ad.pending = self.covers._gpu.get(ad.track) is None
        for f in ad.faces:
            f.texture = tex

    def remove_chunk(self, chunk_idx: int) -> None:
        self.ads = [a for a in self.ads if a.chunk != chunk_idx]

    def update(self, dt: float, t: float, playing: dict, px: float, pz: float) -> None:
        self.covers.poll()
        n = len(self.covers.ids())
        refresh = n != self._n_ids                       # library finished scanning -> re-bind every ad
        self._n_ids = n
        for ad in self.ads:
            if ad.chunk in self.skip and not refresh:
                continue
            if refresh:
                self.assign(ad, self.pick(ad.key, t), t)
            elif getattr(ad, "pending", False):
                self._apply(ad)                           # art arrived from the worker thread
            elif t > ad.next_change and abs(ad.z - pz) < 60:
                self.assign(ad, self.pick(ad.key + f":{int(t)}", t), t)
            lvl = playing.get(ad.track, 0.0)
            ad.level += (lvl - ad.level) * min(1.0, 3.0 * dt)
            em = 0.55 + 0.95 * ad.level
            if abs(em - ad.last_em) > 0.02:
                ad.last_em = em
                for f in ad.faces:
                    f.set_shader_input("emissive", em)
                if ad.bar is not None:
                    ad.bar.set_shader_input("emissive", 0.2 + 2.2 * ad.level)

    def nearest(self, px: float, py: float, pz: float, maxd: float = 7.0):
        best, bd = None, maxd
        for ad in self.ads:
            d = math.sqrt((ad.x - px) ** 2 + (ad.y - py) ** 2 * 0.15 + (ad.z - pz) ** 2)
            if d < bd:
                best, bd = ad, d
        return best, bd


# ============================================================================
# SCENE PRIMITIVES, CHUNKS, FURNITURE
# ============================================================================
FANCY = True        # custom fog/lighting shader (False = Ursina's default shader + built-in fog)
SHADER = None
BANK: Optional[TextureBank] = None
FACE_ROT = {"+x": -90.0, "-x": 90.0, "-z": 0.0, "+z": 180.0}     # quad front faces this way
FACE_YAW = {"+x": 90.0, "-x": -90.0, "+z": 0.0, "-z": 180.0}     # a sitter / machine facing this way


def mk(parent, model="cube", pos=(0, 0, 0), scale=(1, 1, 1), tex=None, tile=(1, 1), color=None,
       emissive=0.0, lm=0.0, lamp=None, rot=(0, 0, 0), double=False, alpha=False, off=(0, 0), xc=0.0):
    """The one place entities get made: texture, tint, shader inputs, transparency."""
    e = ur.Entity(parent=parent, model=model, position=pos, scale=scale, rotation=rot)
    e.ignore = True            # decor has no update()/input(): ursina's per-frame entity scan can skip it cheaply
    e.texture = BANK.get(tex or "white") if (tex is None or isinstance(tex, str)) else tex
    if color is not None:
        e.color = color
    if double:
        e.double_sided = True
    if alpha:
        e.setTransparency(True)
    if FANCY:
        e.shader = SHADER
        lp = lamp or ZONES["corridor"]["lamp"]
        e.set_shader_input("uv_scale", ur.Vec2(float(tile[0]), float(tile[1])))
        e.set_shader_input("uv_off", ur.Vec2(float(off[0]), float(off[1])))
        e.set_shader_input("emissive", float(emissive))
        e.set_shader_input("xc", float(xc))
        e.set_shader_input("lm", float(lm))
        e.set_shader_input("lamp", ur.Vec4(*[float(v) for v in lp]))
    else:
        e.texture_scale = tile
        if emissive > 0:
            e.unlit = True
    return e


def set_light_mode(entities, lm: float, lamp) -> None:
    if not FANCY:
        return
    for e in entities:
        e.set_shader_input("lm", float(lm))
        e.set_shader_input("lamp", ur.Vec4(*[float(v) for v in lamp]))


# ---- cheap entity teardown -------------------------------------------------------------------------------------------
# ursina's destroy() recurses through every child in Python, one list.remove() per entity.  A chunk holds hundreds of
# entities, so destroying one inside a frame was a visible stall.  fast_destroy() drops a whole subtree from
# scene.entities in ONE pass and lets Panda free the node tree with a single removeNode().
def fast_destroy(*roots) -> None:
    dead: list = []
    for r in roots:
        if r is None:
            continue
        stack = [r]
        while stack:
            e = stack.pop()
            dead.append(e)
            kids = getattr(e, "children", None)
            if kids:
                stack.extend(kids)
    if not dead:
        return
    bulk_ok = False
    try:
        gone = {id(e) for e in dead}
        ents = ur.scene.entities
        ents[:] = [e for e in ents if id(e) not in gone]
        bulk_ok = True
    except Exception:
        pass
    for r in roots:
        if r is None:
            continue
        try:
            if bulk_ok:
                r.removeNode()
            else:
                ur.destroy(r)
        except Exception:
            try:
                ur.destroy(r)
            except Exception:
                pass


class Reaper:
    """Entities that left the world are hidden at once and freed later, one tree per frame, never in a frame that
    also built a chunk - so leaving an area can no longer cause a hitch."""
    def __init__(self) -> None:
        self.q: deque = deque()

    def push(self, *roots) -> None:
        for r in roots:
            if r is None:
                continue
            try:
                r.hide()
            except Exception:
                pass
            self.q.append(r)

    def pump(self, budget_s: float = 0.004, force: bool = False) -> None:
        t0 = time.perf_counter()
        while self.q:
            fast_destroy(self.q.popleft())
            if not force and time.perf_counter() - t0 >= budget_s:
                break


REAPER = Reaper()


# ============================================================================
# v2 SYSTEMS: sound effects, inventory/save, vitals, power grid, sky, CCTV, screen FX
# ============================================================================
class OneShotVoice:
    """A pre-rendered effect that quacks like a StreamVoice, so the Mixer sends it through the
    shared convolution reverb (drips, buzz, thunder... get the room's acoustics for free)."""
    def __init__(self, audio: np.ndarray, gain: float = 1.0, label: str = "sfx") -> None:
        self.audio, self.gain, self.label = audio, float(gain), label
        self.length, self.start, self.pos, self.dead = int(audio.shape[1]), 0, 0, False
        self.fade_in = self.fade_out = 0
        self.track_index = None

    @property
    def end(self) -> int:
        return self.start + self.length

    def render(self, m: int):
        seg = self.audio[:, self.pos:self.pos + m]
        if seg.shape[1] < m:
            seg = np.pad(seg, ((0, 0), (0, m - seg.shape[1])))
        self.pos += m
        return (seg * self.gain).astype(np.float32, copy=False), np.zeros(m, dtype=np.float32)   # env 0: no power-norm


def _st(m: np.ndarray, pan: float = 0.0) -> np.ndarray:
    th = (float(np.clip(pan, -1, 1)) + 1.0) * np.pi / 4.0
    return np.stack([m * np.cos(th), m * np.sin(th)]).astype(np.float32) * np.float32(1.4142)


def _lp(x: np.ndarray, fc: float, sr: int, order: int = 2, kind: str = "low") -> np.ndarray:
    return signal.sosfilt(signal.butter(order, min(fc, sr * 0.45), kind, fs=sr, output="sos"), x)


def synth_sfx(name: str, sr: int, seed: int = 0) -> np.ndarray:
    """Tiny numpy synths for every sound the game makes (stereo float32, peak <= ~0.8)."""
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    t = lambda d: np.arange(int(d * sr)) / sr
    if name == "drip":
        x = t(0.5)
        f = 1100 + 1700 * (1 - np.exp(-x * 90))
        m = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-x * 34) * (1 - np.exp(-x * 1500))
        m += 0.25 * np.sin(2 * np.pi * 3100 * x) * np.exp(-x * 70) + 0.2 * rng.standard_normal(x.size) * np.exp(-x * 900)
        return _st(m * 0.55)
    if name == "buzz":
        x = t(0.7)
        m = sum(np.sin(2 * np.pi * 100.0 * k * x + k) / k for k in range(1, 9)) * 0.22
        gate = np.repeat(rng.random(x.size // 220 + 1) > 0.28, 220)[:x.size]
        crack = _lp(rng.standard_normal(x.size), 2600, sr, 2, "high") * 0.04 * gate
        env = np.minimum(1, np.minimum(x / 0.02, (0.7 - x) / 0.08))
        return _st((m * (0.6 + 0.4 * gate) + crack) * env * 0.5)
    if name in ("static", "static_long"):
        d = 3.2 if name == "static_long" else 1.1
        x = t(d)
        n = rng.standard_normal(x.size)
        m = _lp(n, 2200, sr) * (0.35 + 0.65 * (rng.random(x.size // 900 + 1) > 0.35).repeat(900)[:x.size])
        m += 0.12 * np.sin(2 * np.pi * 440 * x) * (np.sin(2 * np.pi * 5 * x) > 0.6)
        return _st(m * 0.5 * np.minimum(1, np.minimum(x / 0.05, (d - x) / 0.2)))
    if name == "thunder":
        x = t(4.2)
        n = _lp(rng.standard_normal(x.size), 190, sr, 3)
        env = (1 - np.exp(-x * 30)) * np.exp(-x * 0.9) * (0.7 + 0.3 * np.sin(x * 9 + rng.uniform(0, 6)))
        return _st(n * env * 5.0 * 0.5)
    if name == "rain":
        x = t(3.0)
        m = _lp(rng.standard_normal(x.size), 7000, sr) * 0.5 - _lp(rng.standard_normal(x.size), 600, sr) * 0.2
        return _st(m * 0.22 * np.minimum(1, np.minimum(x / 0.6, (3.0 - x) / 0.6)))
    if name == "ding":
        x = t(1.4)
        m = (np.sin(2 * np.pi * 880 * x) + 0.5 * np.sin(2 * np.pi * 1320 * x)) * np.exp(-x * 3.0)
        return _st(m * 0.3)
    if name == "thud":
        x = t(0.5)
        m = np.sin(2 * np.pi * np.cumsum(35 + 40 * np.exp(-x * 18)) / sr) * np.exp(-x * 7) + 0.3 * _lp(rng.standard_normal(x.size), 400, sr) * np.exp(-x * 20)
        return _st(m * 0.8)
    if name in ("click", "tick"):
        x = t(0.05)
        m = rng.standard_normal(x.size) * np.exp(-x * 260) * 0.5 + np.sin(2 * np.pi * (1800 if name == "click" else 1200) * x) * np.exp(-x * 120) * 0.4
        return _st(m)
    if name == "flash":
        x = t(0.9)
        f = 3500 + 6000 * (1 - np.exp(-x * 4))
        m = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-x * 6) * 0.25 + rng.standard_normal(x.size) * np.exp(-x * 120) * 0.45
        return _st(m)
    if name == "coin":
        x = t(0.45)
        m = (np.sin(2 * np.pi * 1568 * x) * np.exp(-x * 14) + np.sin(2 * np.pi * 2093 * np.maximum(x - 0.09, 0)) * (x > 0.09) * np.exp(-(x - 0.09) * 9)) * 0.35
        return _st(m)
    if name == "pickup":
        x = t(0.3)
        m = sum(np.sin(2 * np.pi * f * x) * ((x >= i * 0.07) * np.exp(-np.maximum(x - i * 0.07, 0) * 16)) for i, f in enumerate((660, 880, 1320)))
        return _st(m * 0.25)
    if name == "gulp":
        x = t(0.6)
        m = _lp(rng.standard_normal(x.size), 900, sr) * (np.sin(2 * np.pi * 3.2 * x) > 0.55) * np.exp(-x * 2.5)
        return _st(m * 0.5)
    if name == "power_down":
        x = t(1.3)
        m = np.sin(2 * np.pi * np.cumsum(420 * np.exp(-x * 3.2) + 40) / sr) * np.exp(-x * 2.2) * 0.4 + np.pad(synth_sfx("thud", sr, seed)[0], (0, max(0, x.size - int(0.5 * sr))))[:x.size] * 0.5
        return _st(m)
    if name == "power_up":
        x = t(1.0)
        m = (sum(np.sin(2 * np.pi * 100.0 * k * x) / k for k in range(1, 7)) * 0.2 * np.minimum(1, x / 0.5) * np.exp(-np.maximum(x - 0.6, 0) * 4)
             + rng.standard_normal(x.size) * np.exp(-x * 60) * 0.3)
        return _st(m * 0.6)
    if name == "rumble":
        x = t(2.6)
        m = _lp(rng.standard_normal(x.size), 120, sr, 3) * 3.0 + _lp(rng.standard_normal(x.size), 2500, sr) * 0.25
        return _st(m * np.minimum(1, np.minimum(x / 0.3, (2.6 - x) / 0.4)) * 0.5)
    if name == "phone":
        x = t(2.4)
        m = _lp(rng.standard_normal(x.size), 1300, sr) * 0.4 * (0.4 + 0.6 * (np.sin(2 * np.pi * 0.9 * x) > -0.2))
        for f1, f2, t0 in ((697, 1209, 0.4), (770, 1336, 0.75), (852, 1477, 1.1), (941, 1209, 1.45)):
            seg = (x > t0) & (x < t0 + 0.2)
            m += (np.sin(2 * np.pi * f1 * x) + np.sin(2 * np.pi * f2 * x)) * 0.12 * seg
        m = _lp(m, 1800, sr)
        return _st(m * 0.6 * np.minimum(1, np.minimum(x / 0.1, (2.4 - x) / 0.3)))
    if name == "creak":
        x = t(0.9)
        m = np.sin(2 * np.pi * np.cumsum(170 + 90 * np.sin(x * 11) * np.exp(-x * 2)) / sr) * np.exp(-x * 3) * 0.5
        return _st(_lp(m + 0.1 * rng.standard_normal(x.size) * np.exp(-x * 4), 1800, sr))
    raise KeyError(name)


class SfxBus:
    """Low-latency foley (direct output stream: UI sounds, torch, engine hum, kiosk previews)
    plus a 'wet' route into the orchestra's Mixer for ambience that should share the reverb."""
    def __init__(self, stack: "AudioStack", silent: bool) -> None:
        self.stack = stack
        self.sr = int(stack.cfg.sample_rate) if stack.cfg is not None else 44_100
        self._cache: dict = {}
        self._voices: list = []
        self._music: Optional[list] = None
        self._lock = threading.Lock()
        self.engine = dict(on=False, rpm=0.0, vol=0.0, skid=0.0, ph=0.0)
        self.stream = None
        self.enabled = False
        if not silent and sd is not None:
            try:
                self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", callback=self._cb, blocksize=0)
                self.stream.start()
                self.enabled = True
            except Exception as exc:
                log.warning("Direct SFX stream unavailable (%s); effects will use the orchestra bus only", exc)
                self.stream = None

    def _arr(self, name: str, sr: int) -> np.ndarray:
        k = (name, sr)
        if k not in self._cache:
            self._cache[k] = synth_sfx(name, sr)
        return self._cache[k]

    def play(self, name: str, gain: float = 1.0, pan: float = 0.0, wet: bool = False) -> None:
        try:
            st = self.stack
            if wet and st.active and st.mixer is not None:
                a = self._arr(name, st.mixer.sr)
                if abs(pan) > 0.02:
                    a = a * np.array([[1 - 0.5 * max(pan, 0)], [1 + 0.0 * pan - 0.5 * max(-pan, 0)]], np.float32)
                st.mixer.schedule(OneShotVoice(a, gain * 0.7, name), st.mixer.clock)
            elif self.enabled:
                with self._lock:
                    if len(self._voices) < 24:
                        self._voices.append([self._arr(name, self.sr), 0, float(gain)])
        except Exception as exc:
            log.debug("sfx %s failed: %s", name, exc)

    def play_music(self, audio: np.ndarray, gain: float = 0.8) -> None:
        if self.enabled:
            with self._lock:
                self._music = [np.ascontiguousarray(audio, dtype=np.float32), 0, gain]

    def stop_music(self) -> None:
        with self._lock:
            self._music = None

    @property
    def music_active(self) -> bool:
        return self._music is not None

    def _cb(self, outdata, frames, time_info, status) -> None:
        out = np.zeros((frames, 2), np.float32)
        with self._lock:
            keep = []
            for v in self._voices:
                arr, pos, g = v
                m = min(frames, arr.shape[1] - pos)
                if m > 0:
                    out[:m] += arr[:, pos:pos + m].T * g
                v[1] += m
                if v[1] < arr.shape[1]:
                    keep.append(v)
            self._voices = keep
            mu = self._music
            if mu is not None:
                m = min(frames, mu[0].shape[1] - mu[1])
                if m > 0:
                    out[:m] += mu[0][:, mu[1]:mu[1] + m].T * mu[2]
                mu[1] += m
                if mu[1] >= mu[0].shape[1]:
                    self._music = None
        e = self.engine
        if e["vol"] > 0.002:
            f = 34.0 + 120.0 * e["rpm"]
            ph = e["ph"] + 2 * np.pi * f * np.arange(frames) / self.sr
            e["ph"] = float((ph[-1] + 2 * np.pi * f / self.sr) % (2 * np.pi))
            eng = np.sin(ph) + 0.55 * np.sin(2 * ph + 0.7) + 0.3 * np.sin(3 * ph + 1.9) + 0.12 * np.random.randn(frames)
            out += (eng * 0.11 * e["vol"])[:, None].astype(np.float32)
        if e["skid"] > 0.02:
            out += (np.random.randn(frames) * 0.05 * e["skid"])[:, None].astype(np.float32)
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:] = out

    def close(self) -> None:
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass


# ---- items, inventory, persistence -------------------------------------------------------
ITEMS = {
    "water":  dict(name="Bottled Water", kind="drink", thirst=38, hunger=0, fx=None, desc="Tastes like a memory of rain.", col=(120, 180, 230)),
    "fizz":   dict(name="Citrus Fizz", kind="drink", thirst=22, hunger=2, fx="fizz", desc="Lemon-lime. Makes your knees bouncy.", col=(110, 220, 120)),
    "energy": dict(name="Neon Surge", kind="drink", thirst=10, hunger=0, fx="surge", desc="Loud pink. Ask your heart later.", col=(240, 70, 190)),
    "coffee": dict(name="Lumen Coffee", kind="drink", thirst=8, hunger=0, fx="caffeine", desc="Always almost three o'clock.", col=(150, 100, 60)),
    "tea":    dict(name="Midnight Tea", kind="drink", thirst=20, hunger=0, fx="calm", desc="Slow, blue and forgiving.", col=(90, 110, 200)),
    "food":   dict(name="Mall Sandwich", kind="food", thirst=-4, hunger=45, fx=None, desc="Cellophane and certainty.", col=(220, 190, 120)),
    "snack":  dict(name="Unbranded Crisps", kind="food", thirst=-6, hunger=18, fx=None, desc="Salt, mostly.", col=(230, 200, 70)),
    "fuel":   dict(name="Fuel Canister", kind="fuel", thirst=0, hunger=0, fx=None, desc="Red, heavy, smells like going somewhere.", col=(200, 50, 40)),
}
EFFECTS = {   # name: (seconds, speed x, jump x, regen x, tint rgba)
    "caffeine": (60.0, 1.22, 1.0, 1.6, (1.0, 0.82, 0.50, 0.08)),
    "fizz":     (30.0, 1.0, 1.28, 1.0, (0.45, 1.0, 0.55, 0.08)),
    "surge":    (45.0, 1.35, 1.40, 1.4, (1.0, 0.20, 0.80, 0.11)),
    "crash":    (20.0, 0.72, 0.85, 0.6, (0.20, 0.25, 0.50, 0.14)),
    "calm":     (60.0, 0.92, 1.0, 2.2, (0.50, 0.60, 1.0, 0.10)),
}
FRUITS = {"spring": ("Blush Plum", (210, 90, 140)), "summer": ("Golden Peach", (250, 190, 90)),
          "autumn": ("Ember Persimmon", (235, 110, 40))}
SHOP_STOCK = (("food", 4), ("snack", 2), ("coffee", 3), ("water", 2), ("tea", 3), ("fuel", 9))
VEND_STOCK = (("water", 2), ("fizz", 3), ("energy", 5), ("coffee", 4))


class Inventory:
    def __init__(self) -> None:
        self.coins = 3
        self.counts: dict = {}
        self.fruits: list = []
        self.cassettes: list = []
        self.photos: list = []
        self.unlocked: set = set()
        self.collected: set = set()
        self.dirty = False

    def n(self, key: str) -> int:
        return self.counts.get(key, 0)

    def add(self, key: str, k: int = 1) -> None:
        self.counts[key] = max(0, self.counts.get(key, 0) + k)
        self.dirty = True

    def to_json(self) -> dict:
        return dict(coins=self.coins, counts=self.counts, fruits=self.fruits, cassettes=self.cassettes,
                    photos=self.photos, unlocked=sorted(self.unlocked), collected=sorted(self.collected))

    def from_json(self, d: dict) -> None:
        self.coins = int(d.get("coins", self.coins))
        self.counts = {str(k): int(v) for k, v in d.get("counts", {}).items()}
        self.fruits, self.cassettes = list(d.get("fruits", [])), [int(c) for c in d.get("cassettes", [])]
        self.photos = [int(p) for p in d.get("photos", [])]
        self.unlocked = {int(u) for u in d.get("unlocked", [])}
        self.collected = {str(c) for c in d.get("collected", [])}


class SaveFile:
    def __init__(self, seed: int, enabled: bool) -> None:
        self.path = Path.home() / ".ethereal_mall" / f"save_{seed}.json"
        self.enabled, self.last = enabled, 0.0

    def load(self, inv: Inventory) -> None:
        if self.enabled and self.path.is_file():
            try:
                import json
                inv.from_json(json.loads(self.path.read_text("utf-8")))
            except Exception as exc:
                log.warning("Could not read save file (%s)", exc)

    def save(self, inv: Inventory, force: bool = False) -> None:
        if not self.enabled or not (inv.dirty or force):
            return
        try:
            import json
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(inv.to_json()), "utf-8")
            tmp.replace(self.path)
            inv.dirty = False
        except Exception as exc:
            log.warning("Could not write save file (%s)", exc)


def anomaly_photo(seed: int, S: int = 200):
    """A booth snapshot with something wrong in it (procedural, deterministic per seed)."""
    rng = random.Random(seed)
    x = np.linspace(0, 1, S)
    fold = 0.55 + 0.45 * np.sin(x * 34 + rng.uniform(0, 6))
    arr = np.zeros((S, S, 3))
    arr[..., 0], arr[..., 1], arr[..., 2] = (95 * fold)[None, :], (16 * fold)[None, :], (24 * fold)[None, :]
    im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([S * 0.08, S * 0.80, S * 0.92, S * 1.35], fill=(28, 30, 42))
    skin = (rng.randint(185, 238), rng.randint(160, 205), rng.randint(145, 195))
    fw, fh = S * rng.uniform(0.34, 0.42), S * rng.uniform(0.44, 0.54)
    cx, cy = S * (0.5 + rng.uniform(-0.05, 0.05)), S * 0.46
    d.rectangle([cx - fw * 0.16, cy + fh * 0.3, cx + fw * 0.16, cy + fh * 0.75], fill=skin)
    d.ellipse([cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2], fill=skin)
    kind = rng.choice(["eyes", "mouth", "melt", "double", "hollow", "behind"])

    def eye(ex, ey, r, pupil=True):
        d.ellipse([ex - r * 1.5, ey - r, ex + r * 1.5, ey + r], fill=(246, 246, 240))
        if pupil:
            d.ellipse([ex - r * 0.55, ey - r * 0.55, ex + r * 0.55, ey + r * 0.55], fill=(8, 8, 12))
    ey, r = cy - fh * 0.10, S * 0.035
    if kind == "hollow":
        for dx in (-1, 1):
            d.ellipse([cx + dx * fw * 0.22 - r * 1.5, ey - r * 1.6, cx + dx * fw * 0.22 + r * 1.5, ey + r * 1.6], fill=(4, 4, 6))
        d.ellipse([cx - fw * 0.12, cy + fh * 0.18, cx + fw * 0.12, cy + fh * 0.36], fill=(4, 4, 6))
    else:
        eye(cx - fw * 0.22, ey, r)
        eye(cx + fw * 0.22, ey + (S * 0.03 if kind == "double" else 0), r)
        d.line([cx, ey + r, cx - fw * 0.05, cy + fh * 0.12, cx + fw * 0.03, cy + fh * 0.14], fill=(170, 120, 105), width=2)
        if kind == "mouth":
            mw = fw * rng.uniform(0.9, 1.4)
            d.arc([cx - mw / 2, cy + fh * 0.12, cx + mw / 2, cy + fh * 0.52], 5, 175, fill=(120, 20, 30), width=4)
            for k in range(-5, 6):
                d.line([cx + k * mw / 12, cy + fh * 0.33, cx + k * mw / 12, cy + fh * 0.40], fill=(240, 240, 230), width=2)
        else:
            d.arc([cx - fw * 0.2, cy + fh * 0.18, cx + fw * 0.2, cy + fh * 0.38], 20, 160, fill=(130, 60, 60), width=3)
    if kind == "eyes":
        for _ in range(rng.randint(3, 6)):
            eye(cx + rng.uniform(-0.4, 0.4) * fw, cy + rng.uniform(-0.45, 0.35) * fh, S * rng.uniform(0.015, 0.03))
    a = np.asarray(im, dtype=np.float64)
    if kind == "melt":
        for c in range(S):
            sh = int(max(0, math.sin(c * 0.21 + seed) * 10 + (c - cx) ** 2 * 0.0015) + (S * 0.15 if abs(c - cx) < fw * 0.4 else 0) * rng.random())
            col = a[int(cy):, c].copy()
            a[int(cy) + sh:, c] = col[:max(0, S - int(cy) - sh)] if sh else col
    if kind == "double":
        sh = int(S * 0.09)
        a[:, sh:] = 0.55 * a[:, sh:] + 0.45 * np.roll(a, -sh, axis=1)[:, sh:]
    if kind == "behind":
        im2 = Image.fromarray(a.astype(np.uint8))
        d2 = ImageDraw.Draw(im2, "RGBA")
        bx, by, bw = S * 0.80, S * 0.30, S * 0.14
        d2.ellipse([bx - bw, by - bw * 1.3, bx + bw, by + bw * 1.3], fill=(225, 225, 232, 120))
        for dx in (-0.4, 0.4):
            d2.ellipse([bx + dx * bw - 4, by - 5, bx + dx * bw + 4, by + 9], fill=(0, 0, 0, 230))
        a = np.asarray(im2, dtype=np.float64)
    for r_ in range(0, S, 3):                                   # scan-row jitter + grain + warm cast
        a[r_:r_ + 3] = np.roll(a[r_:r_ + 3], int(rng.gauss(0, 1.4)), axis=1)
    a = a * np.array([1.05, 0.97, 0.86]) + np.random.default_rng(seed).normal(0, 9, a.shape)
    yy, xx = np.mgrid[0:S, 0:S]
    a *= (1.0 - 0.45 * (np.hypot(yy - S / 2, xx - S / 2) / (S * 0.75)) ** 2)[..., None]
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    return out.filter(ImageFilter.GaussianBlur(0.8))


def item_card(kind: str, data, title: str = "", W: int = 320, H: int = 400):
    """Close-up card for the album / preview (PIL image)."""
    im = Image.new("RGBA", (W, H), (22, 24, 30, 255))
    d = ImageDraw.Draw(im)
    if kind == "photo":
        d.rectangle([14, 14, W - 14, H - 14], fill=(238, 234, 222, 255))
        ph = anomaly_photo(int(data), 256).resize((W - 52, W - 52))
        im.paste(ph, (26, 26))
        d.text((28, W - 12), title or f"FRAME #{int(data) % 1000:03d}", fill=(60, 55, 50, 255), font=_font(20))
        d.text((28, W + 22), "taken in booth 3 (nobody else was in it)", fill=(110, 100, 95, 255), font=_font(13))
    elif kind == "cassette":
        d.rounded_rectangle([20, 80, W - 20, H - 120], 14, fill=(40, 40, 48, 255), outline=(180, 180, 190, 255), width=3)
        d.rectangle([44, 100, W - 44, 190], fill=(236, 226, 190, 255))
        f = _font(20)
        txt = (title or "UNTITLED")[:22]
        d.text((52, 108), txt, fill=(40, 30, 20, 255), font=f)
        d.text((52, 150), "SIDE A  -  C60", fill=(120, 80, 50, 255), font=_font(14))
        for cx_ in (96, W - 96):
            d.ellipse([cx_ - 28, 220, cx_ + 28, 276], fill=(230, 230, 235, 255), outline=(20, 20, 24, 255), width=3)
            d.ellipse([cx_ - 8, 240, cx_ + 8, 256], fill=(20, 20, 24, 255))
        d.rectangle([W // 2 - 50, 290, W // 2 + 50, 308], fill=(18, 18, 22, 255))
    elif kind == "fruit":
        col = data if isinstance(data, tuple) else (220, 120, 60)
        for k in range(40):
            rr = 112 - k * 2
            d.ellipse([W / 2 - rr, 190 - rr, W / 2 + rr, 190 + rr],
                      fill=tuple(int(min(255, c * (0.55 + 0.45 * k / 40))) for c in col) + (255,))
        d.ellipse([W / 2 + 10, 98, W / 2 + 62, 130], fill=(70, 150, 70, 255))
        d.line([W / 2, 80, W / 2 + 6, 110], fill=(80, 55, 30, 255), width=6)
    else:
        it = ITEMS.get(kind, dict(col=(180, 180, 180), desc=""))
        d.rounded_rectangle([W / 2 - 60, 90, W / 2 + 60, 300], 22, fill=it["col"] + (255,), outline=(240, 240, 245, 255), width=3)
        d.rectangle([W / 2 - 60, 160, W / 2 + 60, 220], fill=(250, 250, 250, 230))
        d.text((W / 2 - 52, 178), (title or kind).split()[0][:9].upper(), fill=(30, 30, 36, 255), font=_font(20))
    d.text((18, H - 98), (title or "")[:26], fill=(235, 238, 250, 255), font=_font(22))
    desc = data.get("desc", "") if isinstance(data, dict) else (ITEMS.get(kind, {}).get("desc", ""))
    if desc:
        d.text((18, H - 62), desc[:42], fill=(150, 160, 190, 255), font=_font(14))
    return im


class Vitals:
    """Hunger, thirst and stamina.  Gentle: nothing here kills you, it only slows you and dims the world."""
    def __init__(self) -> None:
        self.hunger, self.thirst, self.stamina = 85.0, 85.0, 100.0
        self.exhausted = False
        self.fx: dict = {}

    def apply(self, name: str) -> None:
        if name in EFFECTS:
            self.fx[name] = EFFECTS[name][0]

    def eat(self, key: str) -> None:
        it = ITEMS[key]
        self.hunger = min(100.0, max(0.0, self.hunger + it["hunger"]))
        self.thirst = min(100.0, max(0.0, self.thirst + it["thirst"]))
        if it.get("fx"):
            self.apply(it["fx"])

    def mult(self, idx: int) -> float:
        m = 1.0
        for n in self.fx:
            m *= EFFECTS[n][idx]
        return m

    def speed(self) -> float:
        pen = (0.88 if self.hunger < 15 else 1.0) * (0.88 if self.thirst < 10 else 1.0) * (0.8 if self.hunger <= 0 else 1.0)
        return self.mult(1) * pen

    def tint(self) -> tuple:
        r = g = b = a = 0.0
        for n in self.fx:
            c = EFFECTS[n][4]
            w = min(1.0, self.fx[n] / 5.0)
            r, g, b, a = r + c[0] * c[3] * w, g + c[1] * c[3] * w, b + c[2] * c[3] * w, a + c[3] * w
        return (r / a, g / a, b / a, min(0.3, a)) if a > 1e-4 else (0, 0, 0, 0.0)

    def distress(self) -> float:
        return float(np.clip(max((25 - self.hunger) / 25.0, (20 - self.thirst) / 20.0, 0.0), 0.0, 1.0))

    def update(self, dt: float, running: bool) -> None:
        self.hunger = max(0.0, self.hunger - 0.095 * dt)
        self.thirst = max(0.0, self.thirst - 0.14 * dt)
        for n in list(self.fx):
            self.fx[n] -= dt
            if self.fx[n] <= 0:
                del self.fx[n]
                if n == "surge":
                    self.apply("crash")
        regen = 15.0 * self.mult(3) * (0.5 if self.thirst < 15 else 1.0)
        self.stamina = max(0.0, self.stamina - 19.0 * dt) if running else min(100.0, self.stamina + regen * dt)
        if self.stamina <= 0.0:
            self.exhausted = True
        elif self.stamina > 30.0:
            self.exhausted = False


class PowerGrid:
    """Short and long blackouts.  level 1 = lights on, 0 = dead dark (ambient, strips, signs, ads)."""
    def __init__(self, seed: int, enabled: bool = True) -> None:
        self.rng = random.Random(seed * 17 + 3)
        self.enabled, self.level, self.state = enabled, 1.0, "on"
        self.t, self.dur, self.long = 0.0, 0.0, False
        self.next = 75.0 + self.rng.uniform(0, 60)
        self.cool = 0.0

    @property
    def out(self) -> bool:
        return self.state in ("off",) and self.level < 0.05

    def trigger(self, long: bool) -> list:
        if self.state != "on":
            return []
        self.state, self.t, self.long = "pre", 0.0, long
        self.dur = self.rng.uniform(55, 95) if long else self.rng.uniform(2.5, 6.0)
        self.cool = 25.0 + self.dur
        return ["power_down"]

    def update(self, dt: float) -> list:
        ev: list = []
        self.cool = max(0.0, self.cool - dt)
        if self.state == "on":
            self.level = 1.0
            self.next -= dt
            if self.enabled and self.next <= 0:
                ev += self.trigger(self.rng.random() < 0.35)
                self.next = self.rng.uniform(110, 260)
            return ev
        self.t += dt
        if self.state == "pre":
            self.level = 0.15 if self.rng.random() < 0.45 else 1.0
            if self.t > 0.7:
                self.state, self.t, self.level = "off", 0.0, 0.0
        elif self.state == "off":
            self.level = 0.0
            if self.t > self.dur:
                self.state, self.t = "restore", 0.0
                ev.append("power_up")
        elif self.state == "restore":
            self.level = 0.0 if self.rng.random() < 0.4 * (1 - self.t / 0.8) else 1.0
            if self.t > 0.8:
                self.state, self.level = "on", 1.0
        return ev


class SkySystem:
    DAY_LEN = 900.0
    KT = [0.0, 0.20, 0.27, 0.35, 0.50, 0.65, 0.73, 0.80, 1.0]
    COL = np.array([(0.010, 0.015, 0.050), (0.030, 0.050, 0.140), (0.95, 0.55, 0.45), (0.45, 0.70, 1.0), (0.40, 0.66, 1.0),
                    (0.45, 0.68, 1.0), (0.95, 0.50, 0.30), (0.050, 0.050, 0.160), (0.010, 0.015, 0.050)])
    LK = [0.05, 0.08, 0.45, 0.90, 1.0, 0.90, 0.45, 0.10, 0.05]
    RAIN_ZONES = ("skywindow",)          # weather particles: sky-window corridor only (the atrium has none)
    RAIN_SPAN = 28.0                     # height of one droplet tile; layers loop seamlessly over it
    TARGET = {"clear": (0.10, 0.0, 0.0), "cloudy": (0.75, 0.0, 0.0), "rain": (0.92, 0.75, 0.0),
              "storm": (1.0, 1.0, 0.0), "aurora": (0.05, 0.0, 1.0)}     # cloud, rain, aurora

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 404)
        self.t = 0.27
        self.weather, self.next_w = "clear", 40.0
        self.cloud, self.rain, self.aurora = 0.1, 0.0, 0.0
        self.flash, self.next_flash, self.thunder_in = 0.0, 6.0, -1.0
        self.built = False
        self.shown = False
        self.rain_box = (0.0, 0.0, 0.0)
        self.rain_on = False                     # droplets are only drawn in the zones listed in RAIN_ZONES
        self.rain_layers: list = []
        self.light = 1.0
        self.col = (0.4, 0.66, 1.0)
        self.drop = None
        self.sound_t = 0.0
        self.events: list = []

    # -- model (cheap; runs every frame even when invisible)
    def tick(self, dt: float) -> None:
        self.t = (self.t + dt / self.DAY_LEN) % 1.0
        self.next_w -= dt
        night = self.sun_alt < -0.15
        if self.next_w <= 0:
            opts = ["clear", "cloudy", "rain", "storm"] + (["aurora", "aurora"] if night else [])
            self.weather = self.rng.choice([o for o in opts if o != self.weather])
            self.next_w = self.rng.uniform(70, 170)
        elif self.weather == "aurora" and not night:
            self.weather, self.next_w = "clear", 30.0
        tc, tr, ta = self.TARGET[self.weather]
        a = 1.0 - math.exp(-dt / 6.0)
        self.cloud += (tc - self.cloud) * a
        self.rain += (tr - self.rain) * a
        self.aurora += (ta - self.aurora) * a
        self.light = float(np.interp(self.t, self.KT, self.LK)) * (1.0 - 0.5 * self.cloud)
        c = [float(np.interp(self.t, self.KT, self.COL[:, k])) for k in range(3)]
        g = 0.35 + 0.65 * self.light
        self.col = tuple(ci * (1 - 0.6 * self.cloud) + g * 0.55 * 0.6 * self.cloud for ci in c)
        self.flash = max(0.0, self.flash - dt * 5.0)
        if self.weather == "storm" and self.rain > 0.5:
            self.next_flash -= dt
            if self.next_flash <= 0:
                self.flash, self.next_flash = 1.0, self.rng.uniform(4, 13)
                self.thunder_in = self.rng.uniform(0.4, 2.8)
        if self.thunder_in >= 0:
            self.thunder_in -= dt
            if self.thunder_in < 0:
                self.events.append("thunder")

    @property
    def sun_alt(self) -> float:
        return math.sin((self.t - 0.25) * 2 * math.pi)

    @property
    def night(self) -> float:
        return float(np.clip((-self.sun_alt + 0.1) * 3.0, 0.0, 1.0))

    @property
    def clock_text(self) -> str:
        m = int(self.t * 1440)
        return f"{m // 60:02d}:{m % 60:02d}"

    def amb_k(self) -> float:
        return float(np.clip(self.light + 0.5 * self.flash, 0.0, 1.5))

    # -- visuals
    @staticmethod
    def _rain_mesh(rg, n: int, xr: tuple, length: float, width: float):
        """ONE static mesh of n droplets (two crossed thin quads each), stacked in two tiles so it can scroll forever."""
        span = SkySystem.RAIN_SPAN
        pts = np.column_stack([rg.uniform(xr[0], xr[1], n), rg.uniform(0.0, span, n), rg.uniform(-16.0, 16.0, n)])
        hw, slant = width * 0.5, 0.055 * length
        verts, tris = [], []
        for tile in (0.0, span):
            for x, y, z in pts:
                x, y, z = float(x), float(y) + tile, float(z)
                for quad in (((x - slant, y + length, z - hw), (x - slant, y + length, z + hw), (x, y, z + hw), (x, y, z - hw)),
                             ((x - slant - hw, y + length, z), (x - slant + hw, y + length, z), (x + hw, y, z), (x - hw, y, z))):
                    b = len(verts)
                    verts.extend(quad)
                    tris += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 2, b + 1), (b, b + 3, b + 2)]   # both windings
        return ur.Mesh(vertices=verts, triangles=tris)

    def _build(self) -> None:
        def sky_ent(**kw):
            e = ur.Entity(**kw)
            for fn in ("setLightOff", "setFogOff"):
                try:
                    getattr(e, fn)()
                except Exception:
                    pass
            try:
                e.setDepthWrite(False)
            except Exception:
                pass
            return e
        self._mk = sky_ent
        self.dome = sky_ent(model="sphere", scale=300, texture=BANK.get("sky"), double_sided=True)
        self.stars = sky_ent(model="sphere", scale=290, texture=BANK.get("stars"), double_sided=True, color=C(1, 1, 1, 0))
        self.sun = sky_ent(model="quad", scale=34, texture=BANK.get("glow"), color=C(1, 0.9, 0.6, 1), billboard=True)
        self.moon = sky_ent(model="quad", scale=24, texture=BANK.get("glow"), color=C(0.8, 0.85, 1.0, 1), billboard=True)
        self.sea = sky_ent(model="quad", scale=(290, 290, 1), rotation_x=90, texture=BANK.get("clouds"), double_sided=True)
        self.sea.texture_scale = (6, 6)
        self.puffs = [sky_ent(model="quad", scale=(self.rng.uniform(40, 70), self.rng.uniform(14, 24), 1), texture=BANK.get("clouds"),
                              billboard=True, color=C(1, 1, 1, 0)) for _ in range(9)]
        self.pufpos = [(self.rng.uniform(-130, 130), self.rng.uniform(-20, 40), self.rng.uniform(90, 140)) for _ in self.puffs]
        self.curtains = [sky_ent(model="quad", scale=(150, 60, 1), texture=BANK.get("aurora"), color=C(1, 1, 1, 0), billboard=True)
                         for _ in range(3)]
        for i, e in enumerate((self.dome, self.stars, self.sun, self.moon, self.sea, *self.puffs, *self.curtains)):
            try:
                e.setBin("background", i)
            except Exception:
                pass
        # Falling droplets: three STATIC meshes built once (no per-frame mesh rebuild, no zig-zag line strip).  A layer is
        # animated by sliding its entity down one tile height and wrapping, so the whole effect costs 3 position writes.
        rg = np.random.default_rng(3)
        self.rain_layers = []
        for n, xr, length, width, speed, alpha, thr in (
                (36, (0.3, 7.0), 0.20, 0.050, 8.5, 0.80, 0.05),      # fat, slow droplets close to the glass
                (60, (0.5, 22.0), 0.85, 0.030, 21.0, 0.50, 0.30),    # rain streaks
                (70, (4.0, 22.0), 1.00, 0.034, 26.0, 0.38, 0.60)):   # extra streaks only in heavy rain
            ent = sky_ent(model=self._rain_mesh(rg, n, xr, length, width), color=C(0.78, 0.86, 1.0, alpha), double_sided=True)
            try:
                ent.setTransparency(True)
            except Exception:
                pass
            ent.enabled = False
            self.rain_layers.append(dict(ent=ent, speed=speed, alpha=alpha, thr=thr, last_a=-1.0))
        self.all = [self.dome, self.stars, self.sun, self.moon, self.sea, *self.puffs, *self.curtains,
                    *[L["ent"] for L in self.rain_layers]]
        self.built = True

    def show(self, flag: bool) -> None:
        if flag and not self.built:
            self._build()
        if self.built and flag != self.shown:
            for e in self.all:
                e.enabled = flag
        self.shown = flag

    def apply(self, px: float, py: float, pz: float, dt: float, clock: float) -> None:
        if not (self.built and self.shown):
            return
        lk, f = self.light, self.flash
        c = self.col
        self.dome.position = (px, py, pz)
        self.dome.color = C(min(1, c[0] + f * 0.7), min(1, c[1] + f * 0.7), min(1, c[2] + f * 0.8))
        self.stars.position = (px, py, pz)
        self.stars.color = C(1, 1, 1, float(np.clip(self.night * (1.0 - 0.9 * self.cloud), 0, 1)))
        for body, off, col, alt in ((self.sun, 0.0, (1.0, 0.85 - 0.2 * (1 - abs(self.sun_alt)), 0.55), self.sun_alt),
                                    (self.moon, math.pi, (0.82, 0.86, 1.0), -self.sun_alt)):
            b = (self.t - 0.25) * 2 * math.pi + off
            v = np.array([1.0, math.sin(b) * 0.95, -math.cos(b) * 1.25])
            v = v / np.linalg.norm(v) * 140.0
            body.position = (px + float(v[0]), py + float(v[1]) - 6.0, pz + float(v[2]))
            body.color = C(col[0], col[1], col[2], float(np.clip((alt + 0.06) * 6.0, 0, 1)) * (1.0 - 0.8 * self.cloud))
        self.sea.position = (px, py - 38.0, pz)
        self.sea.texture_offset = ((clock * 0.004) % 1.0, (clock * 0.002) % 1.0)
        base = 0.28 + 0.72 * lk
        self.sea.color = C(min(1, base * (1 - 0.4 * self.cloud) + f * 0.5), min(1, base * (1 - 0.35 * self.cloud) + f * 0.5),
                           min(1, base * (1.0 - 0.15 * self.cloud) + 0.05 + f * 0.6), 1)
        for e, (ox, oy, oz) in zip(self.puffs, self.pufpos):
            e.position = (px + 100.0 + oz * 0.2, py - 14 + oy, pz + ox + math.sin(clock * 0.01 + ox) * 6)
            e.color = C(base, base, min(1, base + 0.05), float(np.clip(self.cloud * 0.85, 0, 1)))
        for i, e in enumerate(self.curtains):
            e.position = (px + 110.0 + i * 8.0, py + 42 + i * 5, pz + (i - 1) * 55 + math.sin(clock * 0.2 + i) * 6)
            e.texture_offset = ((clock * 0.02 + i * 0.3) % 1.0, 0)
            e.color = C(1, 1, 1, float(np.clip(self.aurora * self.night * 0.6 * (0.75 + 0.25 * math.sin(clock * 0.6 + i)), 0, 1)))
        bx, by, bz = self.rain_box
        live = self.rain_on and self.rain > 0.05
        for L in self.rain_layers:
            e = L["ent"]
            on = live and self.rain > L["thr"]
            if e.enabled != on:
                e.enabled = on
            if not on:
                continue
            e.position = (bx, by - 12.0 - (clock * L["speed"]) % self.RAIN_SPAN, bz)
            a = L["alpha"] * min(1.0, 0.45 + self.rain)
            if abs(a - L["last_a"]) > 0.02:
                L["last_a"] = a
                e.color = C(0.78, 0.86, 1.0, a)


# ---- CCTV: extra cameras that render the live scene into small textures -----------------------------------------------------
def set_raw_texture(ent, tex) -> None:
    try:
        ent.model.setTexture(tex, 1)
    except Exception:
        pass


class FeedCam:
    def __init__(self, name: str, w: int = 192, h: int = 144) -> None:
        self.ok, self.tex, self.label = False, None, ""
        try:
            from panda3d.core import Camera as PCamera, PerspectiveLens
            base = ur.application.base
            self.buf = base.win.makeTextureBuffer(name, w, h)
            self.buf.setSort(-10)
            lens = PerspectiveLens()
            lens.setFov(76)
            lens.setNearFar(0.1, 85)
            cam = PCamera(name)
            cam.setLens(lens)
            self.holder = ur.Entity(parent=ur.scene)
            self.cam_np = self.holder.attachNewNode(cam)
            self.buf.makeDisplayRegion().setCamera(self.cam_np)
            self.tex = self.buf.getTexture()
            self.buf.setActive(False)
            self.ok = True
        except Exception as exc:
            log.warning("CCTV feed camera unavailable (%s) - monitors will show static only", exc)

    def point(self, x: float, y: float, z: float, yaw: float, pitch: float, label: str = "") -> None:
        if self.ok:
            self.holder.position = (x, y, z)
            self.holder.rotation = (pitch, yaw, 0)
        self.label = label

    def active(self, flag: bool) -> None:
        if self.ok:
            try:
                self.buf.setActive(bool(flag))
            except Exception:
                pass


class ScreenFX:
    """Full-screen overlays on the UI layer: colour tint (consumable filters), vignette, white flash,
    fade-to-black and the animated static used by the elevator."""
    def __init__(self) -> None:
        def quad(col, tex=None, z=0.5):
            return ur.Entity(parent=ur.camera.ui, model="quad", scale=(3.4, 2.2), color=col, texture=tex, z=z)
        self.tint = quad(C(1, 1, 1, 0.0), z=0.50)
        self.vig = quad(C(1, 1, 1, 0.0), BANK.get("vignette"), 0.49)
        self.static_q = quad(C(1, 1, 1, 0.0), BANK.get("static"), 0.06)
        self.flash_q = quad(C(1, 1, 1, 0.0), z=0.05)
        self.fade_q = quad(C(0, 0, 0, 0.0), z=0.04)
        self.flash_a = 0.0
        self.static_a = 0.0
        self.fade_a = 0.0

    def update(self, dt: float, tint: tuple, vig: float) -> None:
        self.tint.color = C(tint[0], tint[1], tint[2], tint[3])
        self.vig.color = C(0.02, 0.0, 0.03, float(np.clip(vig, 0.0, 0.95)))
        self.flash_a = max(0.0, self.flash_a - dt * 1.6)
        self.flash_q.color = C(1, 1, 1, self.flash_a)
        self.static_q.color = C(1, 1, 1, self.static_a)
        if self.static_a > 0.01:
            self.static_q.texture_offset = (random.random(), random.random())
        self.fade_q.color = C(0, 0, 0, self.fade_a)



class Interactable:
    def __init__(self, kind: str, x: float, y: float, z: float, radius: float, label: str,
                 yaw: float = 0.0, data: Optional[dict] = None) -> None:
        self.kind, self.x, self.y, self.z, self.radius, self.label = kind, x, y, z, radius, label
        self.yaw, self.data = yaw, data if data is not None else {}
        self.occupied = False
        self.alive = True


class Litter:
    """Mall debris that nudges, flutters or rolls when you (or a go-kart) pass close.  x/z are WORLD
    coordinates; `xo` is the owning chunk's x offset (the entity lives in chunk-local space)."""
    PROFILE = {"receipt": (1.0, 2.6, 0.02), "paper": (0.9, 2.2, 0.01), "bag": (0.55, 0.9, 0.15),
               "cup": (0.8, 0.8, 0.05), "can": (1.3, 0.0, 0.06), "flyer": (1.1, 2.8, 0.015),
               "scrap": (1.2, 2.0, 0.01), "cart": (0.55, 0.0, 0.0)}        # (kick, hop, rest height)
    RAD = {"cart": 1.55}
    FRIC = {"cart": 0.9}
    FLUTTER = ("receipt", "paper", "flyer", "scrap")

    def __init__(self, ent, kind: str, x: float, z: float, hw: float, xo: float = 0.0) -> None:
        self.ent, self.kind, self.x, self.z, self.hw, self.xo = ent, kind, x, z, hw, xo
        self.kick, self.hop, self.rest = self.PROFILE[kind]
        self.y, self.vx, self.vy, self.vz, self.spin = self.rest, 0.0, 0.0, 0.0, 0.0
        self.yaw = ent.rotation_y if hasattr(ent, "rotation_y") else 0.0
        self.awake, self.t, self.ph = False, 0.0, random.random() * 6.28

    def update(self, dt: float, px: float, pz: float, pspeed: float, pvx: float, pvz: float) -> None:
        rad = self.RAD.get(self.kind, 1.3)
        dx, dz = self.x - px, self.z - pz
        d2 = dx * dx + dz * dz
        if d2 < rad * rad and pspeed > 0.8 and self.y <= self.rest + 0.02:
            d = math.sqrt(d2) + 1e-6
            power = (rad - d) * pspeed * 0.45 * self.kick
            self.vx += (dx / d * 0.7 + pvx / (pspeed + 1e-6) * 0.5) * power
            self.vz += (dz / d * 0.7 + pvz / (pspeed + 1e-6) * 0.5) * power
            self.vy = self.hop * min(power, 1.6)
            self.spin = (random.random() - 0.5) * (80.0 if self.kind == "cart" else 520.0) * min(power, 1.5)
            self.awake = True
        if not self.awake:
            return
        self.t += dt
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.vy -= (3.2 if self.kind in self.FLUTTER else 9.0) * dt          # paper floats down
        self.y += self.vy * dt
        if self.kind in self.FLUTTER and self.y > self.rest + 0.02:          # flutter on the way down
            self.x += math.sin(self.t * 7 + self.ph) * 0.55 * dt
            self.z += math.cos(self.t * 6 + self.ph) * 0.45 * dt
            self.ent.rotation_x = math.sin(self.t * 9 + self.ph) * 28.0
            self.ent.rotation_z = math.cos(self.t * 8 + self.ph) * 24.0
        if self.y <= self.rest:
            self.y, self.vy = self.rest, 0.0
            self.spin *= math.exp(-6.0 * dt)
            fr = math.exp(-self.FRIC.get(self.kind, 4.5) * dt)
            self.vx *= fr
            self.vz *= fr
            if self.kind in self.FLUTTER:
                self.ent.rotation_x = self.ent.rotation_z = 0.0
        if self.kind == "cart":
            if abs(self.vx) + abs(self.vz) > 0.4:
                self.yaw += wrap180(math.degrees(math.atan2(self.vx, self.vz)) - self.yaw) * min(1.0, 5.0 * dt)
        else:
            self.yaw += self.spin * dt
        lim = self.hw - (0.7 if self.kind == "cart" else 0.3)
        if abs(self.x - self.xo) > lim:
            self.x = self.xo + math.copysign(lim, self.x - self.xo)
            if self.kind == "cart":
                self.vx *= -0.45
        self.ent.position = (self.x - self.xo, self.y, self.z)
        self.ent.rotation_y = self.yaw
        if abs(self.vx) + abs(self.vz) < 0.03 and self.y <= self.rest + 1e-3:
            self.awake = False


class DustField:
    """Slowly drifting motes that wrap around the player (so the Path is dusty everywhere) and
    are stirred up by walking through them."""
    def __init__(self, n: int = 56) -> None:
        self.n = n
        rng = np.random.default_rng(5)
        self.p = np.column_stack([rng.uniform(-9, 9, n), rng.uniform(0.3, 5.5, n), rng.uniform(-14, 14, n)])
        self.v = rng.normal(0, 0.05, (n, 3))
        self.ph = rng.uniform(0, 6.28, n)
        self.ents = [mk(ur.scene, "sphere", (0, 0, 0), (0.028, 0.028, 0.028), color=C(1, 1, 1, 0.55), emissive=0.9,
                        alpha=True) for _ in range(n)]
        for e in self.ents:
            if hasattr(e, "set_depth_write"):
                e.set_depth_write(False)
        self.t = 0.0

    def update(self, dt: float, px: float, py: float, pz: float, pvx: float, pvz: float, hw: float, xo: float = 0.0) -> None:
        self.t += dt
        p, v = self.p, self.v
        p += v * dt
        p[:, 0] += np.sin(self.t * 0.31 + self.ph) * 0.05 * dt * 6
        p[:, 1] += np.cos(self.t * 0.23 + self.ph) * 0.03 * dt * 6
        rel = p - np.array([px, py, pz])
        d2 = (rel[:, 0] ** 2 + rel[:, 2] ** 2)
        near = d2 < 2.2 ** 2                                      # wake from the walker
        if near.any():
            v[near, 0] += pvx * 0.12 * dt * 8 + rel[near, 0] * 0.15 * dt
            v[near, 2] += pvz * 0.12 * dt * 8 + rel[near, 2] * 0.15 * dt
        v *= math.exp(-0.9 * dt)
        v += np.random.normal(0, 0.02, v.shape) * dt
        for ax, lim in ((0, 11.0), (2, 16.0)):                    # wrap into a box around the player
            r = p[:, ax] - (px if ax == 0 else pz)
            p[:, ax] -= np.where(r > lim, 2 * lim, 0) - np.where(r < -lim, 2 * lim, 0)
        p[:, 1] = np.clip(p[:, 1], 0.2, 5.6)
        p[:, 0] = np.clip(p[:, 0], xo - hw + 0.3, xo + hw - 0.3)
        for e, q in zip(self.ents, p):
            e.position = (float(q[0]), float(q[1]), float(q[2]))


class Chunk:
    def __init__(self, idx: int, kind: str, part: int = 0, xoff: float = 0.0, hw: Optional[float] = None,
                 h: Optional[float] = None) -> None:
        self.idx, self.kind, self.part, self.xoff = idx, kind, part, xoff
        self.z0 = idx * CH
        self.hw = hw if hw is not None else KIND_HW.get(kind, ROOM_HW)
        self.h = h if h is not None else KIND_H.get(kind, WALL_H)
        self.root = None
        self.detail = None             # small props: hidden when far away (LOD)
        self.colliders: list = []      # WORLD (x0, x1, z0, z1, top)
        self.occ: list = []            # tall colliders used as occluders by the culler
        self.npcs: list = []
        self.litter: list = []
        self.inters: list = []
        self.lm = 0.0 if kind in LM0_KINDS else 1.0
        self.lamp = ZONES[kind]["lamp"]
        self.orb = None
        self.belts: list = []          # (interactable, belt entity)
        self.strips: list = []         # [entity, base emissive, current k, next event, flicker-until, flick factor]
        self.doors: list = []          # swinging stall doors
        self.sliders: list = []        # sliding elevator doors
        self.drips: list = []
        self.monitors: list = []       # CRT screens that show CCTV feeds
        self.statics: list = []        # (entity) static-noise surfaces animated by the game
        self.twinkle: list = []        # (entity, phase) neon stars in the cinema carpet
        self.swing: list = []          # hanging payphone handsets
        self.tree: Optional[dict] = None
        self.floor_fn = None
        self.tickers: list = []        # v3: per-frame callbacks fn(dt, game) -> False to remove
        self.extra: dict = {}          # v3: per-chunk scratch (fountain, swim zone, windows, ...)
        self.tables: list = []         # v3: (x, z, top height, w, d) in chunk-local coordinates
        self.served: list = []         # v3: dishes/glasses the waiter put on tables
        self.shown = True
        self.detail_shown = True
        self.dist = 0.0


def merged_mesh(boxes: list):
    """One Mesh from many axis-aligned boxes (cx, cy, cz, sx, sy, sz) -> a single draw call."""
    verts, tris, uvs = [], [], []
    faces = ((0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7), (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0))
    quv = ((0, 0), (1, 0), (1, 1), (0, 1))
    for cx, cy, cz, sx, sy, sz in boxes:
        x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
        c = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
        for f in faces:
            b = len(verts)
            for k, i in enumerate(f):
                verts.append(c[i])
                uvs.append(quv[k])
            tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return ur.Mesh(vertices=verts, triangles=tris, uvs=uvs)


class ZoneBuilders:
    """The v2 zones and props.  Mixed into ChunkBuilder, so box / block / quad / inter / solid are all here."""
    SEASON_COL = ((0.55, 0.82, 0.42), (0.10, 0.42, 0.14), (0.86, 0.38, 0.08), (0.80, 0.88, 0.96))
    PROMOS = ("SALE -70%", "GRAND OPENING", "LOST: ONE HOUR", "FREE SAMPLES", "2 FOR 1 !!", "ASK ABOUT US")

    # ---- generic helpers
    def strip(self, ent, base: float, flick: float = 1.0) -> None:
        self.ch.strips.append([ent, base, 1.0, self.rng.uniform(1.0, 14.0), 0.0, flick])

    def merged(self, boxes: list, tex: str = "white", tile=(1, 1), **kw):
        return mk(self.par, merged_mesh(boxes), (0, 0, 0), (1, 1, 1), tex, tile, lm=self.ch.lm, lamp=self.ch.lamp,
                  xc=self.ch.xoff, double=True, **kw)

    def _clear(self, x: float, z: float, r: float = 0.7) -> bool:
        xo = self.ch.xoff
        return all(not (c[0] - r < x + xo < c[1] + r and c[2] - r < z < c[3] + r and c[4] > 0.25) for c in self.ch.colliders)

    def _jog(self) -> None:
        """Seal the lateral step between this chunk's walls and the previous chunk's (winding path)."""
        ch = self.ch
        pk, pp, pxo, phw, ph = self.w.geom(ch.idx - 1)
        for s in (-1, 1):
            a, b = pxo + s * phw, ch.xoff + s * ch.hw
            if abs(a - b) > 0.05:
                lo, hi = min(a, b), max(a, b)
                lo, hi = (lo - 0.5, hi) if s < 0 else (lo, hi + 0.5)
                hgt = max(ch.h, ph)
                self.block((lo + hi) / 2 - ch.xoff, hgt / 2, ch.z0, hi - lo, hgt, 0.5, "concrete", tile=((hi - lo) / 3, hgt / 4),
                           color=C(0.6, 0.6, 0.64))

    def loot(self, kind: str, x: float, z: float, tag: str) -> None:
        ch, g = self.ch, self.w.g
        key = f"{self.w.level}:{ch.idx}:{tag}"
        if key in g.inv.collected:
            return
        pivot = ur.Entity(parent=ch.detail, position=(x, 0, z), rotation=(0, self.rng.uniform(0, 360), 0))
        k = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        if kind == "fuel":
            mk(pivot, "cube", (0, 0.22, 0), (0.3, 0.42, 0.14), "metal", color=C(0.85, 0.12, 0.10), **k)
            mk(pivot, "cube", (0.08, 0.46, 0), (0.07, 0.06, 0.07), color=C(0.9, 0.85, 0.2), emissive=0.2, **k)
            mk(pivot, "cube", (-0.07, 0.46, 0), (0.1, 0.05, 0.03), color=C(0.1, 0.1, 0.1), **k)
            label = "Pick up fuel canister"
        else:
            mk(pivot, "cube", (0, 0.03, 0), (0.24, 0.035, 0.15), color=C(0.10, 0.10, 0.12), **k)
            mk(pivot, "cube", (0, 0.052, 0), (0.15, 0.004, 0.09), color=C(0.95, 0.92, 0.8), emissive=0.5, **k)
            mk(pivot, "sphere", (-0.05, 0.056, 0), (0.03, 0.01, 0.03), color=C(0.2, 0.2, 0.2), **k)
            mk(pivot, "sphere", (0.05, 0.056, 0), (0.03, 0.01, 0.03), color=C(0.2, 0.2, 0.2), **k)
            label = "Pick up cassette tape"
        self.inter(kind, x, 0.0, z, 1.4, label, 0.0, dict(key=key, ent=pivot, tag=tag))

    def escalator(self, cx: float, zs: float, top: float, w: float = 1.7, tex: str = "metal") -> None:
        """Steps climbing into a solid concrete soffit: the escalator to nowhere."""
        ch = self.ch
        rise, run = 0.3, 0.5
        n = int((top - 0.4) / rise)
        for i in range(n):
            y, z = (i + 1) * rise, zs + run * (i + 0.5)
            self.box(cx, y - 0.04, z, w, 0.08, run, tex, tile=(1, 0.5), color=C(0.55, 0.58, 0.62), emissive=0.0)
            self.box(cx, y - rise / 2, z - run / 2, w, rise, 0.04, color=C(0.12, 0.12, 0.14))
            self.solid(cx - w / 2 + ch.xoff * 0 + 0.0, cx + w / 2, z - run / 2, z + run / 2, y) if False else \
                self.solid(cx - w / 2, cx + w / 2, z - run / 2, z + run / 2, y)
        ang = math.degrees(math.atan2(rise, run))
        ln = math.hypot(n * rise, n * run)
        mid_z, mid_y = zs + n * run / 2, n * rise / 2
        for s in (-1, 1):
            self.box(cx + s * (w / 2 + 0.06), mid_y + 0.45, mid_z, 0.08, 0.9, ln, "glass", tile=(1, 6), alpha=True,
                     rot=(-ang, 0, 0), color=C(0.8, 0.9, 1.0, 0.5))
            self.box(cx + s * (w / 2 + 0.06), mid_y + 0.95, mid_z, 0.12, 0.07, ln, color=C(0.05, 0.05, 0.06), rot=(-ang, 0, 0))
        self.box(cx, mid_y - 0.25, mid_z, w, 0.3, ln, "metal", color=C(0.2, 0.2, 0.24), rot=(-ang, 0, 0))     # underside truss
        ze = zs + n * run
        self.block(cx, top + 0.1 - 0.7, ze + 1.6, w + 0.8, 1.4 if top > 3 else 1.0, 4.4, "concrete", tile=(1, 2), color=C(0.55, 0.55, 0.6))
        self.solid(cx - w / 2 - 0.4, cx + w / 2 + 0.4, ze - 1.3, ze + 3.8, 99.0)       # the soffit swallows the last steps

    def payphone(self, s: int, z: float) -> None:
        face = "-x" if s > 0 else "+x"
        self.box(s * 4.86, 1.45, z, 0.24, 0.6, 0.4, "metal", color=C(0.10, 0.34, 0.30))
        self.box(s * 4.74, 1.62, z, 0.03, 0.16, 0.26, color=C(0.03, 0.03, 0.04))
        self.quad(s * 4.73, 1.38, z, 0.26, 0.22, face, BANK.sign("1 2 3", (200, 255, 220), (10, 30, 24), 128, 96), emissive=0.5)
        pivot = ur.Entity(parent=self.par, position=(s * 4.7, 1.62, z - 0.2))
        mk(pivot, "cube", (0, -0.16, 0), (0.09, 0.30, 0.08), color=C(0.06, 0.07, 0.08), lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff)
        for k in range(6):                                    # dangling cord
            self.box(s * (4.7 - 0.02 * k), 1.45 - 0.07 * k + 0.0, z - 0.2 + 0.05 * math.sin(k), 0.02, 0.07, 0.02, color=C(0.04, 0.04, 0.05))
        it = self.inter("payphone", s * 4.0, 0.0, z, 1.7, "Inspect payphone", FACE_YAW[face], dict(pivot=pivot, kick=0.0, n=0))
        self.ch.swing.append(it)

    def kiosk(self, s: int, z: float) -> None:
        face = "-x" if s > 0 else "+x"
        self.block(s * 4.5, 0.8, z, 0.7, 1.6, 1.1, "metal", color=C(0.16, 0.17, 0.2))
        self.quad(s * 4.13, 1.35, z, 0.9, 0.5, face, BANK.sign("LISTENING POST", (120, 255, 205), (6, 24, 18), 512, 128, "insert tape / press E"), emissive=1.4)
        self.box(s * 4.13, 0.95, z, 0.06, 0.12, 0.5, color=C(0.02, 0.02, 0.02))
        self.box(s * 4.11, 0.95, z, 0.02, 0.025, 0.36, color=C(1.0, 0.6, 0.2), emissive=1.5)
        self.box(s * 4.2, 1.75, z - 0.35, 0.05, 0.2, 0.05, "metal", color=C(0.1, 0.1, 0.12))        # headphone hook
        self.ball(s * 4.2, 1.6, z - 0.35, 0.09, color=C(0.08, 0.08, 0.1))
        self.box(s * 3.6, 0.38, z, 0.42, 0.06, 0.42, "wood", color=C(0.5, 0.2, 0.2))             # stool
        self.box(s * 3.6, 0.18, z, 0.07, 0.36, 0.07, "metal")
        self.solid(s * 3.6 - 0.22, s * 3.6 + 0.22, z - 0.22, z + 0.22, 0.45)
        self.inter("kiosk", s * 3.6, 0.46, z, 1.8, "Use the listening post", FACE_YAW[face])

    def booth(self, s: int, z: float) -> None:
        face = "-x" if s > 0 else "+x"
        self.block(s * 4.8, 1.15, z, 0.1, 2.3, 1.6, color=C(0.55, 0.12, 0.16), tex="panel") if False else \
            self.block(s * 4.8, 1.15, z, 0.1, 2.3, 1.6, "panel", color=C(0.9, 0.4, 0.45))
        for dz in (-0.8, 0.8):
            self.block(s * 4.05, 1.15, z + dz, 1.5, 2.3, 0.08, "panel", color=C(0.9, 0.4, 0.45))
        self.box(s * 4.05, 2.34, z, 1.6, 0.1, 1.7, color=C(0.1, 0.1, 0.12))
        self.box(s * 3.36, 2.1, z, 0.1, 0.35, 1.7, color=C(0.95, 0.9, 0.3), emissive=1.1)                  # lit header
        self.quad(s * 3.3, 2.1, z, 1.4, 0.28, face, BANK.sign("PHOTO  x4", (60, 20, 10), (250, 225, 90), 384, 96), emissive=1.0)
        for dz, a in ((-0.42, 14), (0.42, -14)):                                                             # curtain halves
            self.box(s * 3.34, 1.1, z + dz, 0.05, 1.9, 0.78, "carpet", tile=(1, 1.5), color=C(1.3, 0.5, 0.5), rot=(0, a * s, 0))
        self.box(s * 4.2, 0.45, z, 0.5, 0.06, 0.5, color=C(0.2, 0.2, 0.22))                                # stool
        self.box(s * 4.2, 0.22, z, 0.06, 0.44, 0.06, "metal")
        self.box(s * 4.7, 1.45, z, 0.1, 0.3, 0.4, "metal", color=C(0.1, 0.1, 0.12))                         # camera + flash
        self.ball(s * 4.64, 1.45, z, 0.07, color=C(0.1, 0.12, 0.2), emissive=0.4)
        fl = self.box(s * 4.7, 1.85, z, 0.06, 0.2, 0.6, color=C(1, 1, 1), emissive=0.3)
        self.box(s * 3.2, 0.5, z + 1.05, 0.2, 0.05, 0.4, "metal", color=C(0.1, 0.1, 0.12))                  # output slot
        self.inter("booth", s * 4.05, 0.0, z, 1.0, "Take a photo", FACE_YAW[face],
                   dict(cool=0.0, slot=(s * 3.2 + self.ch.xoff, z + 1.25), flash=fl))

    def elevator(self, s: int, z: float) -> None:
        ch = self.ch
        ew = ch.hw - 0.2
        cx = s * (ew - 1.2)
        grey = C(0.34, 0.36, 0.32)
        for dz in (-1.5, 1.5):
            self.block(cx, 1.5, z + dz, 2.4, 3.0, 0.12, "metal", color=grey)
        self.block(s * (ew - 0.05), 1.5, z, 0.1, 3.0, 3.0, "metal", color=C(0.42, 0.44, 0.40))
        self.box(cx, 3.05, z, 2.4, 0.1, 3.1, "concrete", color=C(0.4, 0.4, 0.42))
        self.box(cx, 0.03, z, 2.4, 0.05, 2.9, "metal", tile=(3, 3), color=C(0.18, 0.19, 0.2))
        lamp = self.box(cx, 2.96, z, 1.0, 0.04, 1.4, color=C(1.0, 0.95, 0.8), emissive=1.5)
        self.strip(lamp, 1.5, 2.5)
        self.box(s * (ew - 0.14), 1.25, z + 1.2, 0.05, 0.4, 0.2, color=C(0.05, 0.05, 0.06))
        for k, col in enumerate(((1.0, 0.5, 0.1), (0.2, 1.0, 0.4))):
            self.box(s * (ew - 0.18), 1.35 - 0.2 * k, z + 1.2, 0.03, 0.07, 0.07, color=C(*col), emissive=1.6)
        fx = s * (ew - 2.4)
        self.box(fx, 2.85, z, 0.3, 0.3, 3.1, "concrete", color=C(0.5, 0.5, 0.52))
        self.quad(fx - s * 0.17, 2.85, z, 1.6, 0.26, "-x" if s > 0 else "+x", BANK.sign("FREIGHT LIFT", (255, 210, 110), (30, 24, 8), 384, 96), emissive=0.8)
        panels = []
        for k, sg in enumerate((-1, 1)):
            zc_ = z + sg * 0.45
            e = self.box(fx, 1.35, zc_, 0.08, 2.6, 0.9, "shutter", tile=(1, 1.5), color=C(0.55, 0.55, 0.5))
            panels.append((e, zc_ - z, sg * 1.25))
        ch.sliders.append(dict(panels=panels, x=cx + ch.xoff, z=z, open=0.0))
        self.inter("elevator", cx, 0.0, z, 1.35, "Press the button", 0.0, dict(tag=f"{ch.idx}:{s}"))

    def tree(self, cx: float, cz: float, trunk_h: float = 7.5) -> None:
        ch, rng = self.ch, self.rng
        self.block(cx, trunk_h / 2, cz, 0.9, trunk_h, 0.9, "wood", tile=(1, 4), color=C(0.55, 0.43, 0.34))
        for k in range(5):
            a = k * 72 + rng.uniform(0, 30)
            piv = ur.Entity(parent=self.par, position=(cx, 0.25, cz), rotation=(0, a, 0))
            mk(piv, "cube", (0, 0, 0.55), (0.28, 0.5, 1.1), "wood", color=C(0.5, 0.4, 0.32), lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        fruit = []
        for k in range(6):
            a = k * 60 + 22 + rng.uniform(-8, 8)
            piv = ur.Entity(parent=self.par, position=(cx, 2.15, cz), rotation=(0, a, 0))
            L = rng.uniform(1.9, 2.4)
            mk(piv, "cube", (0, 0, L / 2), (0.13, 0.13, L), "wood", color=C(0.45, 0.36, 0.28), lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            fx, fz = cx + math.sin(math.radians(a)) * L, cz + math.cos(math.radians(a)) * L
            fe = self.ball(fx, 1.92, fz, 0.15, color=C(0.9, 0.6, 0.3), emissive=0.15)
            it = self.inter("fruit", fx, 1.75, fz, 1.45, "Pick fruit", 0.0, dict(slot=k, ent=fe))
            fruit.append((it, fe))
        leaves = []
        for k in range(26):
            r, th = rng.uniform(0.6, 4.2), rng.uniform(0, 6.28)
            y = trunk_h - 0.4 + rng.uniform(-1.6, 2.6) * (1.0 - r / 7.0)
            rad = rng.uniform(1.0, 1.6)
            e = self.ball(cx + r * math.cos(th), y, cz + r * math.sin(th), rad, color=C(0.2, 0.5, 0.2))
            leaves.append((e, rad, rng.uniform(0.85, 1.15)))
        ch.tree = dict(leaves=leaves, fruit=fruit, cx=cx, cz=cz)

    def fountain(self, cx: float, cz: float, rim_y0: float) -> None:
        n, R = 20, 3.1
        for k in range(n):
            a = 2 * math.pi * k / n
            x, z = cx + R * math.cos(a), cz + R * math.sin(a)
            self.block(x, rim_y0 + 0.35, z, 1.15, 0.7, 0.5, "cracked", tile=(1, 0.5), rot=(0, -math.degrees(a) + 90, 0),
                       color=C(0.92, 0.9, 0.9)) if False else \
                self.block(x, rim_y0 + 0.35, z, 0.85, 0.7, 0.85, "cracked", tile=(1, 0.5), color=C(0.92, 0.9, 0.9))
        mk(self.par, "circle", (cx, rim_y0 + 0.25, cz), (R * 2 - 0.4, R * 2 - 0.4, 1), "glass", tile=(2, 2), color=C(0.25, 0.4, 0.42, 0.7),
           rot=(90, 0, 0), double=True, alpha=True, lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff)
        self.block(cx, rim_y0 + 0.9, cz, 0.7, 1.6, 0.7, "cracked", color=C(0.85, 0.85, 0.88))
        self.box(cx + 0.15, rim_y0 + 1.95, cz, 0.55, 0.5, 0.45, "cracked", rot=(0, 0, 18), color=C(0.8, 0.8, 0.84))   # broken top
        self.inter("fountain", cx, rim_y0 - 0.2, cz + R + 0.9, 1.5, "Toss a coin", 0.0, dict(n=0))

    def plant_dead(self, x: float, z: float) -> None:
        rng = self.rng
        self.block(x, 0.22, z, 0.5, 0.44, 0.5, "plaster", color=C(0.6, 0.55, 0.48))
        for k in range(5):
            a = rng.uniform(0, 6.28)
            self.box(x + 0.12 * math.cos(a), 0.65 + 0.1 * k, z + 0.12 * math.sin(a), 0.03, 0.55 + 0.1 * k, 0.03, color=C(0.32, 0.25, 0.14),
                     rot=(rng.uniform(-25, 25), 0, rng.uniform(-25, 25)))
            if k % 2:
                self.ball(x + 0.28 * math.cos(a), 1.0 + 0.1 * k, z + 0.28 * math.sin(a), 0.07, color=C(0.42, 0.34, 0.2))

    # ---- zone: Anchor Store (the void), three chunks long and three times as wide
    def _build_anchor(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("rubber", "concrete", C(0.25, 0.25, 0.3))
        if ch.part == 0:
            self.quad(0, H - 1.4, z0 + 0.3, 9.0, 1.3, "-z", BANK.sign("ANCHOR  -  ALL DEPARTMENTS", ZONE_FG["anchor"], ZONE_BG["anchor"], 768, 128,
                                                                      "closing soon (always)"), emissive=0.8, double=False)
        for lz in (4, 12, 20):
            for sx in (-9.0, 0.0, 9.0):
                st = self.box(sx, H - 0.06, z0 + lz, 0.5, 0.08, 2.2, color=C(0.85, 0.9, 1.0), emissive=0.9)
                self.strip(st, 0.9, 2.6 if rng.random() < 0.4 else 1.0)
        self.par = ch.detail
        for sx in (-10.0, 10.0):
            for lz in (6, 18):
                self.block(sx, H / 2, z0 + lz, 1.1, H, 1.1, "concrete", tile=(1, 2), color=C(0.3, 0.3, 0.34))
        racks, cloth = [], []
        skip_mid = 3.5 if ch.part == 1 else 1.4
        for cxr in (-12.0, -9.0, -6.0, -3.0, 3.0, 6.0, 9.0, 12.0):
            for k in range(7):
                czr = z0 + 3.2 + 3.0 * k
                if abs(cxr) < skip_mid or rng.random() < 0.12 or (abs(abs(cxr) - 10) < 1.6 and abs(czr - z0 - 6) < 1.6) or \
                        (abs(abs(cxr) - 10) < 1.6 and abs(czr - z0 - 18) < 1.6):
                    continue
                for px in (-0.85, 0.85):
                    racks.append((cxr + px, 0.85, czr, 0.04, 1.7, 0.04))
                racks.append((cxr, 1.68, czr, 1.8, 0.03, 0.03))
                racks.append((cxr, 0.32, czr, 1.8, 0.03, 0.45))
                for h_ in range(5):
                    hx = cxr - 0.66 + 0.33 * h_
                    racks.append((hx, 1.52, czr, 0.26, 0.02, 0.02))
                    racks.append((hx, 1.62, czr, 0.02, 0.12, 0.02))
                if rng.random() < 0.2:
                    cloth.append((cxr + rng.uniform(-0.5, 0.5), 1.1, czr, 0.5, 0.75, 0.06))
                self.solid(cxr - 0.95, cxr + 0.95, czr - 0.25, czr + 0.25, 1.75)
        if racks:
            self.merged(racks, "metal", color=C(0.30, 0.31, 0.35))
        if cloth:
            self.merged(cloth, "white", color=C(0.30, 0.10, 0.14))
        if ch.part == 1:
            self.escalator(0.0, z0 + 2.5, H - 0.3, 2.0)
        if ch.part == 2:                                                       # cash desks, the last light
            self.block(8.0, 0.5, z0 + 17.0, 5.0, 1.0, 1.4, "panel", color=C(0.7, 0.5, 0.4))
            self.box(8.0, 1.02, z0 + 17.0, 5.2, 0.05, 1.6, "marble", color=C(0.3, 0.3, 0.32))
            self.box(6.6, 1.25, z0 + 17.0, 0.5, 0.35, 0.4, color=C(0.1, 0.1, 0.12))
            self.ball(6.6, 1.5, z0 + 17.0, 0.1, color=C(0.6, 1.0, 0.7), emissive=1.2)
            self.quad(8.0, 3.2, z0 + 15.8, 3.0, 0.6, "-z", BANK.sign("RETURNS", (255, 120, 100), (30, 6, 6), 256, 64), emissive=0.9, double=False)

    # ---- zone: Abandoned Atrium with sunken fountain, skylight, seasonal tree and an escalator to nowhere
    def _build_atrium(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("marble", "plaster", C(0.8, 0.8, 0.78), ceil=False, floor=False)
        W2 = hw + 0.5
        px0, px1, pz0, pz1 = -7.0, 7.0, z0 + 3.0, z0 + 15.0
        self.box(0, -0.2, z0 + 1.5, 2 * W2, 0.4, 3.0, "marble", tile=(8, 1.2))
        self.box(0, -0.2, (pz1 + z0 + CH) / 2, 2 * W2, 0.4, z0 + CH - pz1, "marble", tile=(8, 3.5))
        for s in (-1, 1):
            xa, xb = (s * W2, s * 7.0)
            self.box((xa + xb) / 2, -0.2, (pz0 + pz1) / 2, abs(xa - xb), 0.4, pz1 - pz0, "marble", tile=(abs(xa - xb) / 2.5, 5))
        for lv in range(3):
            dep, sh = -0.3 * (lv + 1), 1.2 * lv
            self.box(0, dep - 0.2, (pz0 + pz1) / 2, 14.0 - 2 * sh, 0.4, 12.0 - 2 * sh, "marble", tile=((14 - 2 * sh) / 2.5, (12 - 2 * sh) / 2.5),
                     color=C(0.9 - 0.08 * lv, 0.9 - 0.08 * lv, 0.92 - 0.06 * lv))
        xo = ch.xoff

        def floor_fn(x, z, xo=xo, px0=px0, px1=px1, pz0=pz0, pz1=pz1):
            lx = x - xo
            if px0 <= lx <= px1 and pz0 <= z <= pz1:
                d = min(lx - px0, px1 - lx, z - pz0, pz1 - z)
                return -0.3 * min(3, int(d // 1.2) + 1)
            return 0.0
        ch.floor_fn = floor_fn
        for s in (-1, 1):                                                       # ceiling with an open skylight
            xa, xb = s * W2, s * 5.0
            self.box((xa + xb) / 2, H + 0.2, zc, abs(xa - xb), 0.4, CH, "concrete", tile=(3, 3), color=C(0.5, 0.5, 0.55))
        for zz, sz in ((zc - 8.5, 7.0), (zc + 8.5, 7.0)):
            self.box(0, H + 0.2, zz, 10.0, 0.4, sz, "concrete", tile=(3, 2), color=C(0.5, 0.5, 0.55))
        mk(self.par, "quad", (0, H + 0.05, zc), (10, 10, 1), "glass", (3, 3), color=C(1, 1, 1, 0.25), rot=(90, 0, 0), double=True, alpha=True,
           lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        for k in range(-2, 3):
            self.box(k * 2.5, H - 0.1, zc, 0.1, 0.2, 10.2, "metal", color=C(0.2, 0.2, 0.24))
        self.par = ch.detail
        self.fountain(0.0, z0 + 9.0, -0.9)
        self.tree(-3.0, z0 + 19.5)
        self.escalator(9.4, z0 + 3.0, 6.6, 1.6)
        for s in (-1, 1):
            self.lounge(s * 10.6, z0 + 20.5, 90 * -s, C(0.38, 0.33, 0.30))
        for (px, pz) in ((-11.5, 2.0), (11.5, 22.0), (-11.5, 12.0), (-7.5, 21.5), (8.0, 20.5), (-11.5, 22.0), (4.5, 1.8)):
            self.plant_dead(px, z0 + pz)
        for k, (cxr, czr) in enumerate(((-6.5, 3.0), (6.5, 21.0), (6.5, 3.0))):
            ring = self.box(cxr, 8.2, czr + z0, 1.6, 0.12, 1.6, color=C(1.0, 0.85, 0.6), emissive=1.4)
            self.strip(ring, 1.4, 1.5)
            self.box(cxr, 9.6, czr + z0, 0.03, 2.8, 0.03, color=C(0.1, 0.1, 0.1))
        self.ad(-hw + 0.22, 3.0, z0 + 7.0, 2.6, 3.4, "+x", f"{ch.idx}:at0")
        self.ad(-hw + 0.22, 3.0, z0 + 17.0, 2.6, 3.4, "+x", f"{ch.idx}:at1")

    # ---- zone: Sky-Window Corridor (a wall of glass over the clouds)
    def _build_skywindow(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("marble", "concrete", C(0.7, 0.72, 0.78), walls=(-1,))
        self.solid(hw, hw + 0.6, z0 - 0.1, z0 + CH + 0.1)
        self.quad(hw - 0.05, H / 2, zc, CH, H, "-x", "glass", tile=(4, 1), alpha=True, color=C(1.2, 1.25, 1.3, 0.55))
        for k in range(5):
            self.box(hw - 0.05, H / 2, z0 + 6.0 * k, 0.1, H, 0.14, "metal", color=C(0.22, 0.24, 0.28))
        for y in (0.05, H - 0.05):
            self.box(hw - 0.05, y, zc, 0.14, 0.12, CH, "metal", color=C(0.22, 0.24, 0.28))
        self.box(hw - 0.5, 0.45, zc, 0.5, 0.06, CH, color=C(0.1, 0.1, 0.12))
        for lz in (4, 12, 20):
            st = self.box(-3.0, H - 0.04, z0 + lz, 0.5, 0.08, 3.0, color=C(0.9, 0.95, 1.0), emissive=1.2)
            self.strip(st, 1.2, 0.6)
        self.par = ch.detail
        for lz in (6.0, 18.0):
            self.lounge(hw - 2.2, z0 + lz, 90, C(0.45, 0.5, 0.58))
            self.lounge(hw - 2.2, z0 + lz + 2.4, 90, C(0.45, 0.5, 0.58))
        self.plant(-hw + 1.0, z0 + 12.0, 1.2)
        self.quad(-hw + 0.3, 3.9, z0 + 12.0, 3.2, 0.6, "+x", BANK.sign("SKY LOUNGE", ZONE_FG["skywindow"], ZONE_BG["skywindow"], 512, 96), emissive=0.8)
        self.ad(-hw + 0.22, 2.2, z0 + 4.5, 1.9, 2.5, "+x", f"{ch.idx}:sk0")
        self.ad(-hw + 0.22, 2.2, z0 + 19.5, 1.9, 2.5, "+x", f"{ch.idx}:sk1")

    # ---- zone: Cinema Corridor (part 0) and Theater (part 1)
    def _build_cinema(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("starcarpet", "panel", C(0.75, 0.42, 0.45))
        gold = C(1.0, 0.78, 0.25)
        for s in (-1, 1):
            self.box(s * (hw - 0.15), 0.07, zc, 0.3, 0.12, CH, color=gold, emissive=0.45)
            self.box(s * (hw - 0.1), H - 0.25, zc, 0.2, 0.2, CH, color=gold, emissive=0.35)
        stars = [[], [], []]
        for _ in range(66):
            stars[rng.randrange(3)].append((rng.uniform(-hw + 0.8, hw - 0.8), 0.015, z0 + rng.uniform(0.4, CH - 0.4), 0.09, 0.012, 0.09))
        for k, grp in enumerate(stars):
            if grp:
                e = self.merged(grp, "white", color=C(*[(1.0, 0.95, 0.5), (0.4, 1.0, 1.0), (1.0, 0.45, 0.85)][k]), emissive=1.6)
                ch.twinkle.append((e, k * 2.1))
        for lz in (3, 9, 15, 21):                                              # flickering marquee bars
            for s in (-1, 1):
                st = self.box(s * 3.5, H - 0.12, z0 + lz, 0.35, 0.1, 2.2, color=C(1.0, 0.8, 0.5), emissive=2.0)
                self.strip(st, 2.0, 3.5)
        self.par = ch.detail
        if ch.part == 0:
            for s in (-1, 1):
                for lz in (5.0, 12.0, 19.0):
                    z, face = z0 + lz, "-x" if s > 0 else "+x"
                    self.ad(s * (hw - 0.45), 2.7, z, 1.5, 2.2, face, f"{ch.idx}:cin{s}:{lz}")
                    self.quad(s * (hw - 0.62), 2.7, z, 1.7, 2.4, face, "glass", alpha=True)
                    for dy in (-1.2, 1.2):
                        self.box(s * (hw - 0.6), 2.7 + dy, z, 0.06, 0.08, 1.8, color=gold, emissive=0.3)
                    for dz in (-0.85, 0.85):
                        self.box(s * (hw - 0.6), 2.7, z + dz, 0.06, 2.4, 0.08, color=gold, emissive=0.3)
                    self.box(s * (hw - 0.62), 1.45, z, 0.1, 0.16, 0.12, color=gold, emissive=0.5)          # padlock
                    self.inter("case", s * (hw - 1.4), 0.0, z, 1.7, "Inspect poster case", FACE_YAW[face], dict(n=0))
            self.block(5.0, 0.55, z0 + 20.5, 4.0, 1.1, 1.0, "panel", color=C(0.7, 0.3, 0.3))            # concession counter
            self.box(5.0, 1.12, z0 + 20.5, 4.2, 0.05, 1.2, "marble", color=C(0.9, 0.85, 0.8))
            self.box(3.4, 1.55, z0 + 20.5, 0.6, 0.8, 0.6, color=C(1.0, 0.9, 0.6), emissive=0.9)         # popcorn glow
            self.quad(0, H - 1.2, z0 + CH - 0.3, 7.0, 1.0, "+z", BANK.sign("THEATER 7  >>", ZONE_FG["cinema"], ZONE_BG["cinema"], 512, 96,
                                                                          "now showing: something you forgot"), emissive=1.2, double=False)
        else:
            scr = self.box(0, 3.7, z0 + CH - 0.7, 9.0, 5.0, 0.1, "static", tile=(2, 1.4), color=C(1.0, 1.0, 1.1), emissive=1.1)
            ch.statics.append(scr)
            for s in (-1, 1):
                self.box(s * 5.2, 3.4, z0 + CH - 1.0, 1.2, 6.0, 0.5, "carpet", tile=(1, 3), color=C(1.3, 0.35, 0.35))
            self.block(0, 0.25, z0 + CH - 1.9, 11.0, 0.5, 1.8, "carpet", color=C(0.4, 0.4, 0.5))
            seats = []
            for r in range(6):
                rz = z0 + 3.0 + 2.9 * r
                for side in (-1, 1):
                    for k in range(5):
                        sx = side * (1.9 + 0.9 * k)
                        seats.append((sx, 0.28, rz, 0.8, 0.46, 0.7))
                        seats.append((sx, 0.72, rz - 0.3, 0.8, 0.7, 0.12))
                        self.inter("sit", sx, 0.46, rz, 0.75, "Watch the screen", 0.0)
                    self.solid(side * 1.5 - 0.0 if False else min(side * 1.5, side * 6.0), max(side * 1.5, side * 6.0), rz - 0.4, rz + 0.4, 0.55)
            self.merged(seats, "carpet", tile=(1, 1), color=C(1.1, 0.3, 0.32))
            self.box(0, H - 1.0, z0 + 2.0, 0.6, 0.5, 0.8, color=C(0.1, 0.1, 0.12))
            self.box(0, H - 1.0, z0 + 2.45, 0.2, 0.2, 0.1, color=C(0.9, 0.9, 1.0), emissive=1.8)

    # ---- zone: Restrooms
    def _build_restroom(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("tile", "tile", C(0.95, 1.0, 1.0))
        self.box(0, H - 0.5, z0 + 0.25, 6.0, 1.0, 0.4, "concrete", color=C(0.6, 0.6, 0.64))
        self.quad(0, H - 0.5, z0 + 0.46, 4.6, 0.7, "-z", BANK.sign("RESTROOMS", ZONE_FG["restroom"], ZONE_BG["restroom"], 512, 96), emissive=1.1, double=False)
        for lz in (4, 12, 20):
            st = self.box(0, H - 0.04, z0 + lz, 0.6, 0.08, 3.4, color=C(0.95, 1.0, 1.0), emissive=1.8)
            self.strip(st, 1.8, 1.5)
        self.par = ch.detail
        stalls = 5
        for i in range(stalls + 1):                                              # partitions along the right wall
            zp = z0 + 4.0 + 2.6 * i - 1.3
            self.block(hw - 0.95, 1.2, zp, 1.9, 2.0, 0.06, "metal", tile=(1, 1), color=C(0.55, 0.75, 0.78))
        for i in range(stalls):
            zs = z0 + 4.0 + 2.6 * i
            self.box(hw - 0.5, 0.3, zs, 0.5, 0.6, 0.45, color=C(0.95, 0.96, 0.98))
            self.box(hw - 0.2, 0.75, zs, 0.2, 0.55, 0.45, color=C(0.95, 0.96, 0.98))
            piv = ur.Entity(parent=self.par, position=(hw - 1.9, 0, zs - 1.25))
            mk(piv, "cube", (0, 1.2, 1.25), (0.05, 2.0, 2.5), "metal", color=C(0.65, 0.82, 0.85), lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            ch.doors.append(dict(pivot=piv, x=hw - 1.9 + ch.xoff, z=zs - 0.0, ang=0.0, ajar=rng.choice((0, 0, 0, 20, 45, 70)), cr=0.0))
        self.block(-hw + 0.5, 0.45, zc, 1.0, 0.9, 12.0, "marble", tile=(1, 4), color=C(0.9, 0.9, 0.92))
        for k in range(4):
            zb = z0 + 5.5 + 4.2 * k
            self.box(-hw + 0.5, 0.92, zb, 0.5, 0.06, 0.6, color=C(0.03, 0.04, 0.05))
            self.box(-hw + 0.28, 1.1, zb, 0.06, 0.2, 0.06, "metal")
        self.box(-hw + 0.05, 2.1, zc, 0.06, 1.5, 12.0, "metal", tile=(4, 1), color=C(1.3, 1.35, 1.4))       # mirror strip
        self.quad(1.5, 0.012, zc - 2.0, 1.6, 1.0, "-z", "glass", alpha=True, color=C(0.3, 0.45, 0.5, 0.6), rot=(90, 0, 0)) if False else None
        for (px, pz, rr) in ((2.0, 8.0, 0.8), (-2.5, 15.0, 1.1), (3.5, 19.0, 0.6)):
            mk(self.par, "circle", (px, 0.012, z0 + pz), (rr, rr, 1), "glass", (1, 1), color=C(0.2, 0.3, 0.34, 0.65), rot=(90, 0, 0), double=True,
               alpha=True, lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            top = self.ball(px, 3.3, z0 + pz, 0.04, color=C(0.8, 0.9, 1.0), emissive=0.5)
            ch.drips.append(dict(x=px + ch.xoff, y=3.3, z=z0 + pz, t=rng.uniform(0.5, 4.0), fall=-1.0, ent=top))
            top.enabled = False
        for dz in (-0.3, 0.3):                                                    # wet floor sign
            self.box(-2.0, 0.3, z0 + 12.0 + dz, 0.5, 0.6, 0.03, color=C(1.0, 0.85, 0.1), emissive=0.2, rot=(0, 0, 0))
        self.box(0, H - 0.7, zc, 0.15, 0.15, CH - 1.0, "metal", color=C(0.6, 0.62, 0.66))
        self.inter("sit", -hw + 1.5, 0.45, z0 + 2.2, 1.4, "Sit on the floor, briefly", FACE_YAW["+x"]) if False else None

    # ---- zone: Security room with the CCTV hub
    def _build_security(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("rubber", "panel", C(0.7, 0.8, 0.72))
        for lz in (4, 12, 20):
            st = self.box(0, H - 0.04, z0 + lz, 0.5, 0.08, 4.0, color=C(0.7, 1.0, 0.8), emissive=1.2)
            self.strip(st, 1.2, 2.2)
        self.quad(0, H - 0.5, z0 + 0.3, 6.5, 0.7, "-z", BANK.sign("SECURITY - STAFF ONLY", ZONE_FG["security"], ZONE_BG["security"], 640, 96), emissive=1.0, double=False)
        self.par = ch.detail
        feeds = ch.monitors
        for row, y in enumerate((0.95, 1.9, 2.85)):
            for col in range(6):
                zz = z0 + 3.2 + 3.2 * col
                self.block(-hw + 0.55, y, zz, 0.9, 0.85, 1.05, "plaster", color=C(0.72, 0.68, 0.58))
                scr = self.quad(-hw + 1.03, y, zz, 0.86, 0.68, "+x", "white", color=C(0.01, 0.01, 0.012), emissive=0.0)       # blacked out: no live rendering
                feeds.append(dict(scr=scr, ov=None, feed=0))
        self.block(2.0, 0.4, z0 + 12.0, 1.2, 0.8, 6.0, "wood", color=C(0.6, 0.6, 0.62))
        for dz in (-2.0, 0.0, 2.0):
            self.box(1.8, 0.85, z0 + 12.0 + dz, 0.4, 0.05, 0.7, color=C(0.7, 0.7, 0.7))
        hub = self.box(1.7, 1.2, z0 + 12.0, 0.5, 0.7, 0.9, color=C(0.1, 0.12, 0.12))
        self.quad(1.43, 1.2, z0 + 12.0, 0.8, 0.55, "-x", "static", color=C(0.6, 1.0, 0.8), emissive=1.3)
        self.inter("cctv", 0.5, 0.0, z0 + 12.0, 1.9, "Operate the CCTV hub", FACE_YAW["+x"], dict(n=0))
        self.chair(3.2, z0 + 12.0, FACE_YAW["-x"], "metal", C(0.2, 0.2, 0.22))
        for k in range(3):
            self.block(hw - 0.6, 1.0, z0 + 4.0 + 1.0 * k, 0.9, 2.0, 0.9, "metal", color=C(0.3, 0.35, 0.4))
        led = self.ball(-hw + 1.0, H - 1.0, z0 + 2.0, 0.07, color=C(1, 0.1, 0.1), emissive=2.0)
        self.strip(led, 2.0, 4.0)

    # ---- sub-level: endless parking garage
    def _build_garage(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("asphalt", "concrete", C(0.55, 0.55, 0.52))
        for lz in (4, 12, 20):
            for sx in (-2.5, 2.5):
                st = self.box(sx, H - 0.05, z0 + lz, 0.3, 0.06, 3.0, color=C(1.0, 0.8, 0.45), emissive=1.5)
                self.strip(st, 1.5, 1.5 if rng.random() < 0.7 else 5.0)
        self.par = ch.detail
        for lz in (0, 8, 16):
            for sx in (-3.8, 3.8):
                self.block(sx, H / 2, z0 + lz, 0.8, H, 0.8, "concrete", tile=(1, 1), color=C(0.6, 0.6, 0.6))
            for sx in (-8.5, 8.5):
                self.block(sx, H / 2, z0 + lz + 4.0, 0.8, H, 0.8, "concrete", tile=(1, 1), color=C(0.6, 0.6, 0.6))
        for s in (-1, 1):
            self.quad(s * 3.4, 1.9, z0 + 8.0, 0.9, 0.5, "-x" if s > 0 else "+x", BANK.sign(f"P-{abs(ch.idx) % 9 + 1}{chr(65 + abs(ch.idx * 3) % 20)}",
                                                                                    (255, 210, 80), (20, 18, 6), 192, 96), emissive=0.5)
        bodies = []
        cab = []
        for s in (-1, 1):
            for k in range(8):
                if rng.random() < 0.55:
                    cz = z0 + 1.5 + 3.0 * k
                    cx = s * 6.2
                    bodies.append((cx, 0.55, cz, 4.4, 0.7, 1.9))
                    cab.append((cx, 1.15, cz, 2.4, 0.55, 1.7))
                    self.solid(cx - 2.2, cx + 2.2, cz - 0.95, cz + 0.95, 1.3)
        if bodies:
            col = rng.choice((C(0.55, 0.12, 0.12), C(0.12, 0.25, 0.45), C(0.7, 0.7, 0.72), C(0.15, 0.4, 0.3), C(0.1, 0.1, 0.12)))
            self.merged(bodies, "metal", color=col)
            self.merged(cab, "glass", color=col * 0.7)
        for k in range(2):
            px, pz = rng.uniform(-2.5, 2.5), z0 + rng.uniform(3, 21)
            top = self.ball(px, H - 0.4, pz, 0.04, color=C(0.8, 0.9, 1.0), emissive=0.4)
            top.enabled = False
            ch.drips.append(dict(x=px + ch.xoff, y=H - 0.4, z=pz, t=rng.uniform(1, 8), fall=-1.0, ent=top))
        if ch.idx % 12 == 0:
            self.elevator(1, z0 + 12.0)

    # ---- sub-level: flooded poolrooms
    def _build_pool(self):
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        W2 = hw + 0.5
        pit = (ch.idx % 2 == 0) or rng.random() < 0.3
        px0, px1, pz0, pz1 = -5.0, 5.0, z0 + 6.0, z0 + 18.0
        self._shell("tile", "tile", C(0.95, 1.0, 1.0), floor=not pit)
        xo = ch.xoff
        if pit:
            self.box(0, -0.2, z0 + 3.0, 2 * W2, 0.4, 6.0, "tile", tile=(8, 1))
            self.box(0, -0.2, z0 + 21.0, 2 * W2, 0.4, 6.0, "tile", tile=(8, 1))
            for s in (-1, 1):
                self.box(s * (W2 + 5.0) / 2, -0.2, zc, W2 - 5.0, 0.4, 12.0, "tile", tile=(3, 4))
            self.box(0, -0.2 - 0.4, zc, 10.0, 0.4, 12.0, "tile", tile=(4, 4), color=C(0.7, 0.9, 0.95))
            self.box(0, -0.2 - 0.0, zc, 8.0, 0.4, 10.0, "tile", tile=(3, 3), color=C(0.8, 0.95, 1.0)) if False else None

            def floor_fn(x, z, xo=xo):
                lx = x - xo
                if px0 <= lx <= px1 and pz0 <= z <= pz1:
                    return -0.4 if min(lx - px0, px1 - lx, z - pz0, pz1 - z) < 1.0 else -1.0
                return 0.0
            ch.floor_fn = floor_fn
            self.box(0, -0.8, zc, 8.0, 0.4, 10.0, "tile", tile=(3, 3), color=C(0.55, 0.85, 0.95))
            self.box(0, -1.2, zc, 6.0, 0.4, 8.0, "tile", tile=(3, 3), color=C(0.45, 0.8, 0.92))
            mk(self.par, "quad", (0, -0.28, zc), (10, 12, 1), "glass", (4, 4), color=C(0.4, 0.85, 1.0, 0.55), rot=(90, 0, 0), double=True, alpha=True,
               lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        mk(self.par, "quad", (0, 0.14, zc), (2 * hw, CH, 1), "glass", (8, 4), color=C(0.5, 0.9, 1.0, 0.35), rot=(90, 0, 0), double=True, alpha=True,
           lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        for lz in (4, 12, 20):
            for sx in (-6, 0, 6):
                st = self.box(sx, H - 0.05, z0 + lz, 1.2, 0.06, 1.2, color=C(1.0, 1.0, 0.95), emissive=1.3)
                self.strip(st, 1.3, 0.7)
        self.par = ch.detail
        for sx in (-8.5, 8.5):
            for pz in (6.0, 18.0):
                self.block(sx, H / 2, z0 + pz, 0.9, H, 0.9, "tile", tile=(1, 2), color=C(0.95, 1.0, 1.0))
        for k in range(3):
            rx, rz = rng.uniform(-6, 6), z0 + rng.uniform(3, 21)
            self.ball(rx, 0.18, rz, 0.5, color=rng.choice((C(1.0, 0.45, 0.6), C(1.0, 0.85, 0.3), C(0.3, 0.8, 1.0))), emissive=0.25)
        if rng.random() < 0.35:                                                   # a lonely slide
            self.block(-7.0, 1.4, z0 + 12.0, 1.4, 2.8, 1.4, "tile", color=C(0.4, 0.8, 1.0))
            self.box(-7.0, 1.4, z0 + 15.5, 1.2, 0.12, 6.0, color=C(1.0, 0.5, 0.7), emissive=0.2, rot=(25, 0, 0))
        for k in range(2):
            px, pz = rng.uniform(-6, 6), z0 + rng.uniform(3, 21)
            top = self.ball(px, H - 0.4, pz, 0.04, color=C(0.8, 0.95, 1.0), emissive=0.5)
            top.enabled = False
            ch.drips.append(dict(x=px + ch.xoff, y=H - 0.4, z=pz, t=rng.uniform(0.5, 5), fall=-1.0, ent=top))
        if ch.idx % 12 == 0:
            self.elevator(1, z0 + 12.0)



class ChunkBuilder(ZoneBuilders):
    """Deterministic chunk factory: the same index always gives the same room."""
    BAY_WEIGHTS = (("ad", 0.25), ("shutter", 0.15), ("glass", 0.15), ("vending", 0.09), ("bench", 0.08),
                   ("sign", 0.06), ("lounge", 0.03), ("blank", 0.06), ("payphone", 0.05), ("kiosk", 0.03),
                   ("booth", 0.025), ("elevator", 0.02))
    SIGNS = ("WAY OUT", "RESTROOMS", "PARKING P3", "CONCOURSE", "ESCALATOR ->", "INFORMATION", "LEVEL 0")

    def __init__(self, world: "World") -> None:
        self.w, self.seed = world, world.seed
        self.ch: Optional[Chunk] = None
        self.par = None
        self.rng = random.Random(0)

    # ---- primitives bound to the chunk being built
    def box(self, cx, cy, cz, sx, sy, sz, tex="white", tile=(1, 1), **kw):
        return mk(self.par, "cube", (cx, cy, cz), (sx, sy, sz), tex, tile, lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff, **kw)

    def ball(self, cx, cy, cz, r, tex="white", **kw):
        return mk(self.par, "sphere", (cx, cy, cz), (r * 2, r * 2, r * 2), tex, lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff, **kw)

    def quad(self, cx, cy, cz, w, h, face, tex, **kw):
        kw.setdefault("double", True)
        return mk(self.par, "quad", (cx, cy, cz), (w, h, 1), tex, lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff,
                  rot=(0, FACE_ROT[face], 0), **kw)

    def solid(self, x0, x1, z0, z1, top=99.0):
        xo = self.ch.xoff
        self.ch.colliders.append((min(x0, x1) + xo, max(x0, x1) + xo, min(z0, z1), max(z0, z1), top))

    def block(self, cx, cy, cz, sx, sy, sz, tex="white", collide=True, **kw):
        e = self.box(cx, cy, cz, sx, sy, sz, tex, **kw)
        if collide:
            self.solid(cx - sx / 2, cx + sx / 2, cz - sz / 2, cz + sz / 2, cy + sy / 2)
        return e

    def inter(self, kind, x, y, z, radius, label, yaw=0.0, data=None):
        it = Interactable(kind, x + self.ch.xoff, y, z, radius, label, yaw, data)
        self.ch.inters.append(it)
        return it

    def ad(self, cx, cy, cz, w, h, face, key, bar=True, thick=True):
        s = 1 if face == "-x" else -1
        faces, barent = [], None
        if face in ("-x", "+x"):
            if thick:
                self.box(cx + s * 0.13, cy, cz, 0.22, h + 0.3, w + 0.3, "metal", color=C(0.34, 0.37, 0.42))
            faces.append(self.quad(cx, cy, cz, w, h, face, "poster_blank", emissive=0.7))
            if bar:
                barent = self.box(cx, cy - h / 2 - 0.07, cz, 0.08, 0.06, w * 0.8, color=C(0.7, 0.85, 1.0), emissive=0.2)
        ad = Ad(faces, barent, cx + self.ch.xoff, cy, cz, key, self.ch.idx)
        self.w.ads.register(ad, self.w.g.clock)
        return ad

    # ---- furniture
    def chair(self, cx, cz, yaw, tex="wood", color=None, seat=0.46, sit=True, pulled=0.0):
        s, c = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        cx, cz = cx - s * pulled, cz - c * pulled
        self.box(cx, seat, cz, 0.46, 0.06, 0.46, tex, color=color, rot=(0, yaw, 0))
        self.box(cx - s * 0.21, seat + 0.3, cz - c * 0.21, 0.46, 0.5, 0.05, tex, color=color, rot=(0, yaw, 0))
        self.box(cx, seat / 2, cz, 0.4, seat, 0.05, "metal", color=C(0.2, 0.2, 0.22), rot=(0, yaw, 0))
        self.box(cx, seat / 2, cz, 0.05, seat, 0.4, "metal", color=C(0.2, 0.2, 0.22), rot=(0, yaw, 0))
        self.solid(cx - 0.25, cx + 0.25, cz - 0.25, cz + 0.25, seat + 0.02)
        if sit:
            self.inter("sit", cx, seat + 0.02, cz, 1.5, "Sit", yaw)

    def table(self, cx, cz, w=0.9, d=0.9, h=0.75, tex="wood", color=None):
        self.ch.tables.append((cx, cz, h, w, d))
        self.box(cx, h, cz, w, 0.06, d, tex, color=color)
        self.box(cx, h / 2, cz, 0.08, h, 0.08, "metal", color=C(0.25, 0.25, 0.28))
        self.box(cx, 0.03, cz, min(w, d) * 0.6, 0.05, min(w, d) * 0.6, "metal", color=C(0.25, 0.25, 0.28))
        self.solid(cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2, h + 0.03)

    def lounge(self, cx, cz, yaw, color):
        s, c = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        self.box(cx, 0.28, cz, 0.95, 0.34, 0.95, color=color, rot=(0, yaw, 0))
        self.box(cx - s * 0.4, 0.62, cz - c * 0.4, 0.95, 0.55, 0.18, color=color, rot=(0, yaw, 0))
        for k in (-1, 1):
            self.box(cx + c * k * 0.5, 0.45, cz - s * k * 0.5, 0.14, 0.3, 0.95, color=color * 0.8, rot=(0, yaw, 0))
        self.solid(cx - 0.55, cx + 0.55, cz - 0.55, cz + 0.55, 0.46)
        self.inter("sit", cx, 0.46, cz, 1.6, "Lounge", yaw)

    def bench_wall(self, s, z):
        yaw = FACE_YAW["-x" if s > 0 else "+x"]
        self.box(s * 4.55, 0.45, z, 0.5, 0.08, 1.8, "wood")
        self.box(s * 4.82, 0.8, z, 0.08, 0.6, 1.8, "wood")
        for dz in (-0.7, 0.7):
            self.box(s * 4.55, 0.22, z + dz, 0.4, 0.44, 0.06, "metal", color=C(0.2, 0.2, 0.22))
        self.solid(s * 4.3, s * 4.85, z - 0.9, z + 0.9, 0.5)
        self.inter("sit", s * 4.5, 0.5, z, 1.7, "Sit", yaw)

    def vending(self, s, z, tint):
        face = "-x" if s > 0 else "+x"
        self.block(s * 4.55, 1.0, z, 0.9, 2.0, 1.1, "metal", color=tint)
        f = self.quad(s * 4.09, 1.15, z, 0.95, 1.43, face, "vending", emissive=1.1)
        self.box(s * 4.12, 0.28, z, 0.05, 0.22, 0.6, color=C(0.02, 0.02, 0.03))
        self.inter("vend", s * 3.55, 0.0, z, 1.7, "Buy a drink", FACE_YAW[face],
                   {"face": f, "tray": (s * 3.95 + self.ch.xoff, 0.34, z), "cool": 0.0, "dir": -s,
                    "stock": VEND_STOCK[stable_hash(self.seed, self.ch.idx, s, int(z)) % len(VEND_STOCK)]})

    def plant(self, x, z, h=1.0):
        self.block(x, 0.25, z, 0.5, 0.5, 0.5, "plaster", color=C(0.85, 0.8, 0.7))
        for k in range(3):
            self.ball(x + (k - 1) * 0.12, 0.7 + k * h * 0.2, z + (k % 2 - 0.5) * 0.15, 0.28 + 0.06 * k, color=C(0.18, 0.45, 0.2))

    def pendant(self, x, y, z, k=1.5):
        self.box(x, (y + WALL_H) / 2 - 0.9, z, 0.02, WALL_H - y - 1.0, 0.02, color=C(0.1, 0.1, 0.1))
        self.ball(x, y, z, 0.18, color=C(*[min(1, v) for v in self.ch.lamp[:3]]), emissive=k)

    # ---- shells
    def _shell(self, floor_tex, wall_tex, wall_col=None, walls=(-1, 1), ceil=True, floor=True, ceil_tex="concrete"):
        ch, z0, hw, h = self.ch, self.ch.z0, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        if floor:
            self.box(0, -0.2, zc, 2 * hw + 1, 0.4, CH, floor_tex, tile=(2 * hw / 2.5, CH / 2.5))
        if ceil:
            self.box(0, h + 0.2, zc, 2 * hw + 1, 0.4, CH, ceil_tex, tile=(hw / 2, CH / 4), color=C(0.5, 0.5, 0.55))
        for s in walls:
            self.box(s * (hw + 0.25), h / 2, zc, 0.5, h, CH, wall_tex, tile=(CH / 4, h / 4), color=wall_col)
            self.solid(s * hw, s * (hw + 0.6), z0 - 0.1, z0 + CH + 0.1)

    def _build_corridor(self):
        ch, z0, rng = self.ch, self.ch.z0, self.rng
        zc = z0 + CH / 2
        self._shell("marble", "concrete")
        # the Path: an inlaid luminous guide line with brushed-metal borders
        self.box(0, 0.012, zc, 0.08, 0.02, CH, color=C(0.8, 0.9, 1.0), emissive=0.7)
        for s in (-1, 1):
            self.box(s * 0.45, 0.01, zc, 0.12, 0.02, CH, "metal", tile=(0.2, 6))
        for lz in (0, 8, 16):                                  # pillars
            for s in (-1, 1):
                self.block(s * 4.7, WALL_H / 2, z0 + lz, 0.6, WALL_H, 1.0, "concrete", tile=(0.5, 1.5))
        for lz in (4, 12, 20):                                 # stark overhead strips
            for s in (-2, 2):
                self.strip(self.box(s, WALL_H - 0.04, z0 + lz, 0.55, 0.08, 3.2, color=C(0.9, 0.95, 1.0), emissive=1.9), 1.9)
                self.box(s, WALL_H - 0.02, z0 + lz, 0.7, 0.04, 3.4, color=C(0.06, 0.06, 0.07))
        self.par = ch.detail
        kinds, wts = zip(*self.BAY_WEIGHTS)
        for s in (-1, 1):
            for lz in (4, 12, 20):
                kind, z = self._bay(kinds, wts, s, lz), z0 + lz
                face = "-x" if s > 0 else "+x"
                if kind == "ad":
                    self.ad(s * 4.77, 2.25, z, 2.1, 2.8, face, f"{ch.idx}:{s}:{lz}")
                elif kind == "shutter":
                    self.box(s * 4.9, 1.9, z, 0.2, 3.8, 6.4, "shutter", tile=(2, 1.2))
                    self.box(s * 4.75, 0.35, z, 0.12, 0.1, 0.5, "metal")
                elif kind == "glass":
                    self.box(s * 4.92, 1.9, z, 0.16, 3.6, 6.4, color=C(0.03, 0.035, 0.05), emissive=0.05)
                    if rng.random() < 0.55:
                        neon = rng.choice([(1.0, 0.3, 0.6), (0.3, 0.9, 1.0), (1.0, 0.7, 0.25)])
                        self.box(s * 4.84, 3.3, z, 0.05, 0.07, 4.5, color=C(*neon), emissive=1.6)
                    self.box(s * 4.85, 0.9, z + rng.uniform(-1.5, 1.5), 0.3, 1.6, 0.3, color=C(0.55, 0.57, 0.62), emissive=0.12)
                    self.ball(s * 4.85, 1.9, z + 0.0, 0.16, color=C(0.7, 0.7, 0.75), emissive=0.1)
                    self.quad(s * 4.78, 1.9, z, 6.4, 3.6, face, "glass", tile=(2, 1), alpha=True)
                    for dz in (-3.2, 0.0, 3.2):
                        self.box(s * 4.77, 1.9, z + dz, 0.05, 3.7, 0.08, "metal", color=C(0.25, 0.27, 0.3))
                elif kind == "vending":
                    self.vending(s, z, rng.choice([C(0.55, 0.12, 0.14), C(0.12, 0.28, 0.5), C(0.2, 0.2, 0.24)]))
                elif kind == "bench":
                    self.bench_wall(s, z)
                elif kind == "payphone":
                    self.payphone(s, z)
                elif kind == "kiosk":
                    self.kiosk(s, z)
                elif kind == "booth":
                    self.booth(s, z)
                elif kind == "elevator":
                    self.elevator(s, z)
                elif kind == "lounge":
                    self.lounge(s * 3.9, z - 1.0, FACE_YAW[face] + 20 * s, C(0.55, 0.52, 0.5))
                    self.lounge(s * 3.9, z + 1.0, FACE_YAW[face] - 20 * s, C(0.55, 0.52, 0.5))
                elif kind == "sign":
                    txt = rng.choice(self.SIGNS)
                    self.quad(s * 4.78, 3.6, z, 2.4, 0.6, face, BANK.sign(txt, (210, 255, 225), (8, 40, 24), 512, 128),
                              emissive=0.9)
                else:
                    self.box(s * 4.93, 2.4, z, 0.14, 2.4, 5.4, "concrete", tile=(1.4, 0.7), color=C(0.72, 0.72, 0.75))
        for lz in (0, 8, 16):                                  # hanging banners (ads) over the Path
            if rng.random() < 0.3:
                s = rng.choice((-1, 1))
                z = z0 + lz
                self.box(s * 3.2, 5.55, z, 0.04, 0.9, 0.04, color=C(0.1, 0.1, 0.1))
                f1 = self.quad(s * 3.2, 4.0, z - 0.02, 1.2, 1.6, "-z", "poster_blank", emissive=0.7, double=False)
                f2 = self.quad(s * 3.2, 4.0, z + 0.02, 1.2, 1.6, "+z", "poster_blank", emissive=0.7, double=False)
                ad = Ad([f1, f2], None, s * 3.2, 4.0, z, f"{ch.idx}:b{lz}", ch.idx)
                self.w.ads.register(ad, self.w.g.clock)

    def _room_shell(self, floor_tex, wall_tex, title, sub, wall_col=None):
        ch, z0, zc = self.ch, self.ch.z0, self.ch.z0 + CH / 2
        self._shell(floor_tex, wall_tex, wall_col)
        for zz, face in ((z0 + 0.25, "-z"), (z0 + CH - 0.25, "+z")):
            self.box(0, WALL_H - 0.7, zz, 10.2, 1.4, 0.5, "concrete", color=C(0.62, 0.62, 0.66))
            off = -0.27 if face == "-z" else 0.27
            self.quad(0, WALL_H - 0.7, zz + off, 6.0, 1.1, face, BANK.sign(title, ZONE_FG[ch.kind], ZONE_BG[ch.kind], 640, 128, sub),
                      emissive=1.3, double=False)
        self.box(0, 4.95, zc, 2 * ROOM_HW, 0.2, CH - 1.0, "plaster", tile=(4, 4), color=C(0.35, 0.33, 0.34))
        lc = C(*[min(1, v) for v in ch.lamp[:3]])
        for lz in (4, 12, 20):
            for x in (-6, 0, 6):
                self.box(x, 4.72, ch.z0 + lz, 0.1, 0.4, 0.1, color=C(0.08, 0.08, 0.08))
                self.strip(self.box(x, 4.5, ch.z0 + lz, 0.9, 0.1, 0.9, color=lc, emissive=1.7), 1.7, 0.8)
        self.par = ch.detail

    def _build_cafe(self):
        ch, z0, rng = self.ch, self.ch.z0, self.rng
        self._room_shell("terrazzo", "plaster", "CAFE LUMEN", "espresso | pastry | quiet")
        self.box(-5.6, 0.012, z0 + 12, 7.0, 0.02, 10, "wood", tile=(2.8, 4))
        self.block(7.1, 0.55, z0 + 15, 1.4, 1.1, 10, "marble", tile=(1, 3))                  # counter
        self.box(7.1, 1.13, z0 + 15, 1.6, 0.06, 10.2, "marble", color=C(1.0, 0.97, 0.92))
        self.box(7.3, 1.45, z0 + 12.5, 0.6, 0.55, 0.8, "metal", color=C(0.8, 0.82, 0.88))   # espresso machine
        for k in range(3):
            self.box(6.95, 1.35, z0 + 12.2 + k * 0.3, 0.1, 0.12, 0.1, color=C(0.1, 0.1, 0.1))
        self.box(7.2, 1.4, z0 + 17.5, 1.0, 0.5, 2.2, "glass", alpha=True, color=C(0.9, 1, 1))  # pastry case
        for k in range(5):
            self.box(7.2, 1.25, z0 + 16.6 + k * 0.4, 0.2, 0.12, 0.25, color=C(0.85, 0.6, 0.3), emissive=0.2)
        self.box(8.85, 2.1, z0 + 15, 0.2, 0.08, 9.0, "wood")                                  # back shelves
        for k in range(9):
            self.box(8.7, 2.27, z0 + 11.2 + k * 0.95, 0.12, 0.3, 0.16, color=C(*[rng.uniform(0.2, 0.9) for _ in range(3)]), emissive=0.25)
        for k in range(4):                                                                    # counter stools
            zz = z0 + 11.5 + k * 2.3
            self.box(5.75, 0.35, zz, 0.36, 0.06, 0.36, "wood")
            self.box(5.75, 0.17, zz, 0.06, 0.34, 0.06, "metal")
            self.solid(5.55, 5.95, zz - 0.2, zz + 0.2, 0.5)
            self.inter("sit", 5.75, 0.4, zz, 1.5, "Sit", FACE_YAW["+x"])
        self.quad(8.88, 2.7, z0 + 7.2, 1.5, 1.9, "-x", BANK.board("MENU", ["espresso  3.5", "flat white  4.5", "croissant  5", "oat? always.", "time  free"]), emissive=0.6)
        for (tx, tz) in ((-6.8, 5.5), (-3.0, 9.0), (-6.8, 15.0), (-3.0, 19.0)):               # tables + chairs
            self.table(tx, z0 + tz, 0.9, 0.9, 0.75, "wood")
            self.chair(tx - 0.85, z0 + tz, FACE_YAW["+x"])
            self.chair(tx + 0.85, z0 + tz, FACE_YAW["-x"])
            self.pendant(tx, 3.2, z0 + tz, 1.4)
            self.ball(tx, 0.84, z0 + tz, 0.04, color=C(1, 0.8, 0.5), emissive=1.5)
        self.lounge(1.6, z0 + 19.0, FACE_YAW["+x"] * 0 + 90, C(0.45, 0.30, 0.22))              # lounge corner
        self.lounge(4.0, z0 + 19.0, -90, C(0.45, 0.30, 0.22))
        self.table(2.8, z0 + 19.0, 0.7, 0.7, 0.4, "marble")
        for (px, pz) in ((-8.2, 1.5), (-8.2, 22.5), (8.2, 1.5)):
            self.plant(px, z0 + pz)
        self.ad(-8.78, 2.4, z0 + 12.0, 1.9, 2.5, "+x", f"{ch.idx}:cafe0")
        self.ad(-8.78, 2.4, z0 + 20.0, 1.9, 2.5, "+x", f"{ch.idx}:cafe1")

    def _build_gym(self):
        ch, z0, rng = self.ch, self.ch.z0, self.rng
        zc = z0 + CH / 2
        self._room_shell("rubber", "concrete", "AEON ATHLETIC", "open always", wall_col=C(0.45, 0.45, 0.6))
        self.box(-8.9, 2.2, zc, 0.2, 3.9, CH - 1, "metal", color=C(0.95, 0.97, 1.0))             # mirror wall
        self.quad(-8.78, 2.2, zc, CH - 1.2, 3.8, "+x", "glass", tile=(6, 1), alpha=True, color=C(1.2, 1.2, 1.3))
        for y, col in ((1.0, (1.0, 0.2, 0.75)), (3.4, (0.2, 0.9, 1.0))):                         # neon rails
            self.box(8.82, y, zc, 0.06, 0.09, CH - 1, color=C(*col), emissive=2.1)
        for lz in (4, 12, 20):
            self.box(0, 4.76, z0 + lz, 14, 0.05, 0.12, color=C(0.2, 0.9, 1.0), emissive=2.0)
        for k, tz in enumerate((4.5, 8.5, 12.5)):                                                # treadmills
            cx, cz = -6.0, z0 + tz
            self.block(cx, 0.17, cz, 2.2, 0.3, 0.85, "metal", color=C(0.25, 0.25, 0.32))
            belt = self.box(cx + 0.1, 0.33, cz, 1.7, 0.03, 0.62, "rubber", tile=(4, 1), color=C(1.4, 1.4, 1.5))
            for dz in (-0.4, 0.4):
                self.box(cx - 0.95, 0.75, cz + dz, 0.06, 1.2, 0.06, "metal")
            self.box(cx - 0.95, 1.3, cz, 0.25, 0.28, 0.75, "metal", color=C(0.15, 0.15, 0.2))
            self.quad(cx - 0.81, 1.3, cz, 0.6, 0.22, "+x", BANK.sign("00:00", (120, 255, 255), (4, 20, 30), 256, 96), emissive=1.5, double=False)
            it = self.inter("treadmill", cx + 0.3, 0.33, cz + 1.0, 1.7, "Start / stop treadmill", FACE_YAW["-x"], {"on": False, "ph": 0.0})
            ch.belts.append((it, belt))
        self.block(1.6, 0.25, z0 + 9.0, 0.5, 0.5, 1.5, color=C(0.1, 0.1, 0.12))                  # bench press
        self.inter("sit", 1.6, 0.5, z0 + 9.0, 1.6, "Sit on bench", FACE_YAW["+z"])
        for dz in (-0.6, 0.6):
            self.box(1.6, 1.0, z0 + 9.0 + dz, 0.08, 1.0, 0.08, "metal")
        self.box(1.6, 1.3, z0 + 9.0, 1.6, 0.05, 0.05, "metal", color=C(0.8, 0.8, 0.85))
        for dx in (-0.7, 0.7):
            self.box(1.6 + dx, 1.3, z0 + 9.0, 0.08, 0.5, 0.5, color=C(0.08, 0.08, 0.1), emissive=0.2)
        self.block(8.0, 0.8, z0 + 19.0, 1.2, 1.6, 3.6, "metal", color=C(0.22, 0.22, 0.3))         # weight rack
        for k in range(6):
            self.ball(7.3, 0.45 + (k % 3) * 0.45, z0 + 17.7 + (k // 3) * 1.6, 0.25, color=C(0.07, 0.07, 0.09), emissive=0.12)
        self.block(0, 0.4, zc + 1.0, 1.4, 0.8, 1.4, "metal", color=C(0.9, 0.92, 1.0))             # chrome orb pedestal
        ch.orb = self.ball(0, 1.9, zc + 1.0, 0.85, "metal", color=C(1.2, 0.4, 1.1), emissive=0.6)
        self.ad(8.78, 2.4, z0 + 9.0, 1.9, 2.5, "-x", f"{ch.idx}:gym0")
        self.ad(8.78, 2.4, z0 + 15.0, 1.9, 2.5, "-x", f"{ch.idx}:gym1")

    def _build_restaurant(self):
        ch, z0, rng = self.ch, self.ch.z0, self.rng
        self._room_shell("carpet", "panel", "RISTORANTE", "closed? reservations only")
        self.block(-8.1, 0.35, z0 + 12, 0.9, 0.7, 20, "panel", color=C(0.9, 0.7, 0.6))           # wall banquette
        self.box(-8.5, 1.1, z0 + 12, 0.15, 1.2, 20, "panel", color=C(0.9, 0.65, 0.5))
        for tz in (4.5, 9.5, 14.5, 19.5):                                                         # booths
            self.table(-6.0, z0 + tz, 0.9, 1.4, 0.75, "wood", C(0.7, 0.55, 0.45))
            self.chair(-4.9, z0 + tz, FACE_YAW["-x"], color=C(0.5, 0.2, 0.2))
            self.inter("sit", -7.7, 0.7, z0 + tz, 1.6, "Sit", FACE_YAW["+x"])
            for dz in (-0.35, 0.35):
                self.box(-6.0, 0.8, z0 + tz + dz, 0.22, 0.02, 0.22, color=C(0.95, 0.95, 0.92))
            self.ball(-6.0, 0.84, z0 + tz, 0.035, color=C(1.0, 0.7, 0.35), emissive=1.8)
            self.pendant(-6.0, 3.0, z0 + tz, 0.9)
        for k, (tx, tz) in enumerate(((1.5, 6.0), (1.5, 12.0), (5.2, 9.0))):                       # floor tables
            self.table(tx, z0 + tz, 0.9, 0.9, 0.75, "wood", C(0.7, 0.55, 0.45))
            for yaw, dx, dz in ((90, -0.7, 0), (-90, 0.7, 0), (0, 0, -0.7), (180, 0, 0.7)):
                pulled = 0.5 if (k == 1 and yaw == 90) else 0.0
                self.chair(tx + dx, z0 + tz + dz, yaw, color=C(0.5, 0.2, 0.2), pulled=pulled)
            self.ball(tx, 0.84, z0 + tz, 0.035, color=C(1.0, 0.7, 0.35), emissive=1.8)
            self.pendant(tx, 3.1, z0 + tz, 0.8)
        self.block(8.0, 0.55, z0 + 16.0, 1.3, 1.1, 8.0, "panel")                                  # bar
        self.box(8.0, 1.12, z0 + 16.0, 1.5, 0.06, 8.2, "marble", color=C(0.25, 0.22, 0.22))
        for k in range(4):
            zz = z0 + 13.0 + k * 1.9
            self.box(6.8, 0.38, zz, 0.36, 0.06, 0.36, color=C(0.5, 0.2, 0.2))
            self.box(6.8, 0.18, zz, 0.06, 0.36, 0.06, "metal")
            self.solid(6.6, 7.0, zz - 0.2, zz + 0.2, 0.5)
        self.block(2.6, 0.55, z0 + 2.0, 1.2, 1.1, 0.7, "wood")                                     # host stand
        self.quad(2.6, 1.25, z0 + 1.62, 0.7, 0.45, "-z", BANK.sign("CLOSED?", (255, 200, 170), (30, 10, 10), 256, 96), emissive=0.9, double=False)
        self.ad(8.78, 2.5, z0 + 7.0, 1.8, 2.4, "-x", f"{ch.idx}:rest0")

    def _bay(self, kinds, wts, s, lz):
        pick = self.rng.choices(kinds, wts)[0]                 # always consume the rng: the world stays deterministic
        return FORCED_BAYS.get((self.ch.idx, s, lz), pick) if self.w.level == "mall" else pick

    # ---- litter, loot, karts, NPCs
    def _litter(self):
        ch, rng, hw, xo = self.ch, self.rng, self.ch.hw, self.ch.xoff
        n = rng.randint(6, 13) if ch.kind in ("corridor", "garage", "pool", "skywindow") else rng.randint(3, 8)
        kinds = ("receipt", "paper", "bag", "cup", "can", "flyer", "scrap", "cart")
        wts = (0.20, 0.13, 0.10, 0.12, 0.07, 0.17, 0.13, 0.08 if ch.kind in ("corridor", "anchor", "garage", "skywindow") else 0.015)
        for _ in range(n):
            kind = rng.choices(kinds, wts)[0]
            x = rng.choice((-1, 1)) * rng.uniform(hw - 1.7, hw - 0.5) if rng.random() < 0.55 else rng.uniform(-hw + 1, hw - 1)
            z = ch.z0 + rng.uniform(0.5, CH - 0.5)
            if not self._clear(x, z, 0.5):
                continue
            pivot = ur.Entity(parent=ch.detail, position=(x, Litter.PROFILE[kind][2], z), rotation=(0, rng.uniform(0, 360), 0))
            kw = dict(lm=ch.lm, lamp=ch.lamp, xc=xo)
            if kind == "receipt":
                s_ = rng.uniform(0.05, 0.08)
                mk(pivot, "sphere", (0, 0, 0), (s_ * 1.3, s_ * 0.8, s_), color=C(0.93, 0.92, 0.88), emissive=0.05, **kw)
            elif kind == "paper":
                mk(pivot, "quad", (0, 0, 0), (0.22, 0.3, 1), color=C(0.9, 0.9, 0.86), rot=(90, 0, 0), double=True, **kw)
            elif kind == "flyer":
                mk(pivot, "quad", (0, 0, 0), (0.21, 0.30, 1), BANK.sign(rng.choice(self.PROMOS), (70, 45, 10), (232, 210, 140), 160, 224),
                   color=C(1.0, 0.95, 0.72), rot=(90, 0, 0), double=True, **kw)
            elif kind == "scrap":
                for _k in range(3):
                    mk(pivot, "quad", (rng.uniform(-0.08, 0.08), 0.001 * _k, rng.uniform(-0.08, 0.08)), (rng.uniform(0.05, 0.1), rng.uniform(0.05, 0.1), 1),
                       color=C(0.9, 0.88, 0.8), rot=(90, rng.uniform(0, 360), 0), double=True, **kw)
            elif kind == "cart":
                mt = C(0.72, 0.74, 0.8)
                mk(pivot, "cube", (0, 0.32, 0), (0.55, 0.03, 0.9), "metal", color=mt, **kw)
                for sx_ in (-0.27, 0.27):
                    mk(pivot, "cube", (sx_, 0.5, 0), (0.02, 0.34, 0.9), "glass", color=mt, alpha=True, **kw)
                for sz_ in (-0.44, 0.44):
                    mk(pivot, "cube", (0, 0.5, sz_), (0.55, 0.34, 0.02), "glass", color=mt, alpha=True, **kw)
                mk(pivot, "cube", (0, 0.88, -0.5), (0.58, 0.03, 0.03), color=C(0.1, 0.1, 0.12), **kw)
                for wx, wz in ((-0.22, -0.38), (0.22, -0.38), (-0.22, 0.38), (0.22, 0.38)):
                    mk(pivot, "sphere", (wx, 0.06, wz), (0.1, 0.1, 0.1), color=C(0.05, 0.05, 0.06), **kw)
            elif kind == "bag":
                col = rng.choice([C(0.75, 0.3, 0.3), C(0.3, 0.45, 0.7), C(0.85, 0.8, 0.6), C(0.2, 0.2, 0.22)])
                mk(pivot, "cube", (0, 0, 0), (0.24, 0.3, 0.14), color=col, **kw)
                for dx in (-0.06, 0.06):
                    mk(pivot, "cube", (dx, 0.2, 0), (0.015, 0.12, 0.015), color=col * 0.8, **kw)
            elif kind == "cup":
                mk(pivot, "cube", (0, 0, 0), (0.07, 0.1, 0.07), color=C(0.92, 0.92, 0.9), **kw)
                mk(pivot, "cube", (0, 0.06, 0), (0.08, 0.02, 0.08), color=C(0.2, 0.2, 0.22), **kw)
            else:
                mk(pivot, "cube", (0, 0, 0), (0.06, 0.12, 0.06), "metal", color=rng.choice([C(0.8, 0.15, 0.15), C(0.15, 0.4, 0.8)]), **kw)
            ch.litter.append(Litter(pivot, kind, x + xo, z, hw, xo))

    def _features(self):
        ch, rng, hw, lvl = self.ch, self.rng, self.ch.hw, self.w.level
        if ch.kind not in ("corridor", "garage", "pool", "skywindow", "cinema", "atrium", "office"):
            return
        boost = 2.0 if ch.kind in ("garage", "pool") else 1.0

        def spot():
            for _ in range(10):
                x = rng.choice((-1, 1)) * rng.uniform(hw - 2.2, hw - 0.9) if rng.random() < 0.6 else rng.uniform(-hw + 1.5, hw - 1.5)
                z = ch.z0 + rng.uniform(1.5, CH - 1.5)
                if self._clear(x, z, 0.9):
                    return x, z
            return None
        if lvl == "mall":
            for (i, kind), (fx, fz) in FORCED_LOOT.items():
                if i == ch.idx:
                    self.loot(kind, fx, ch.z0 + fz, "forced")
            if ch.idx == 1:
                self.w.g.spawn_kart(ch, ch.xoff - 3.0, ch.z0 + 16.0, 0.0, "k_start")
        sp = spot()
        if sp and rng.random() < 0.13 * boost:
            self.loot("cassette", sp[0], sp[1], "c0")
        sp = spot()
        if sp and rng.random() < 0.20 * boost:
            self.loot("fuel", sp[0], sp[1], "f0")
        sp = spot()
        if sp and ch.kind in ("corridor", "garage", "pool") and abs(ch.idx) > 1 and rng.random() < 0.07 * boost:
            self.w.g.spawn_kart(ch, ch.xoff + sp[0], sp[1], rng.choice((0.0, 180.0, 90.0)), "k0")

    def _npcs(self):
        ch, rng, lvl, hw = self.ch, self.rng, self.w.level, self.ch.hw
        xo, k = ch.xoff, ch.kind
        seats = [i for i in ch.inters if i.kind == "sit"]
        sp = []
        if lvl != "mall":
            if k == "pool" and rng.random() < 0.3:
                sp.append(("wander", None, "talk"))
        elif abs(ch.idx) > 1:
            if k == "corridor":
                r = rng.random()
                if r < 0.20:
                    sp.append(("wander", None, "talk"))
                elif r < 0.27 and seats:
                    sp.append(("sit", rng.choice(seats), "talk"))
                elif r < 0.34:
                    sp.append(("vanish", None, "talk"))
                elif r < 0.42:
                    sp.append(("guard", (xo + rng.choice((-1, 1)) * (hw - 0.9), ch.z0 + CH - 1.4), "talk"))
            elif k == "cafe":
                sp.append(("stand", (xo + 7.9, ch.z0 + 14.0), "shop"))
                for it in rng.sample(seats, min(len(seats), rng.randint(1, 3))):
                    sp.append(("sit", it, "talk"))
            elif k == "gym":
                sp += [("wander", None, "talk") for _ in range(rng.randint(1, 2))]
            elif k == "restaurant":
                if rng.random() < 0.6 and seats:
                    sp.append(("sit", rng.choice(seats), "talk"))
            elif k == "atrium":
                sp += [("wander", None, "talk") for _ in range(rng.randint(1, 2))]
                if rng.random() < 0.4:
                    sp.append(("vanish", None, "talk"))
            elif k == "anchor":
                sp += [("mannequin", None, "talk") for _ in range(rng.randint(3, 5))]
                if ch.part == 2:
                    sp.append(("stand", (xo + 8.0, ch.z0 + 18.7), "shop"))
            elif k == "skywindow":
                if seats and rng.random() < 0.6:
                    sp.append(("sit", rng.choice(seats), "talk"))
                else:
                    sp.append(("wander", None, "talk"))
            elif k == "cinema":
                if ch.part == 0:
                    sp.append(("stand", (xo + 5.0, ch.z0 + 21.9), "shop"))
                elif seats:
                    for it in rng.sample(seats, min(len(seats), rng.randint(1, 3))):
                        sp.append(("sit", it, "talk"))
            elif k == "security":
                sp.append(("guard", (xo + hw - 1.5, ch.z0 + CH - 1.5), "talk"))
        for mode, arg, role in sp:
            self.w.spawn_npc(ch, mode, arg, rng, role)

    def build(self, idx: int) -> Chunk:
        kind, part, xo, hw, h = self.w.geom(idx)
        ch = Chunk(idx, kind, part, xo, hw, h)
        ch.root = ur.Entity(position=(xo, 0, 0))
        ch.detail = ur.Entity(parent=ch.root)
        self.ch, self.par = ch, ch.root
        self.rng = seeded(self.seed, "chunk", self.w.level, idx)
        self._jog()
        getattr(self, "_build_" + kind)()
        self.par = ch.detail
        self._features()
        self._litter()
        self.rng = seeded(self.seed, "npc", self.w.level, idx)
        self._npcs()
        ch.occ = [c for c in ch.colliders if c[4] >= 2.5 and (c[1] - c[0]) >= 0.5 and (c[3] - c[2]) >= 0.5]
        return ch


ZONE_FG = {"cafe": (255, 232, 196), "gym": (120, 245, 255), "restaurant": (255, 205, 170), "corridor": (220, 230, 255)}
ZONE_BG = {"cafe": (44, 26, 14), "gym": (18, 10, 40), "restaurant": (36, 10, 12), "corridor": (14, 16, 24)}


class World:
    """Streams chunks around the player (builds ahead, destroys behind, rebuilds on the way back)
    and culls them: view frustum, fog distance, occlusion by tall walls/pillars, detail LOD."""
    def __init__(self, game: "Game") -> None:
        self.g = game
        self.seed = game.args.seed
        self.ads = game.ads
        self.chunks: dict = {}
        self.level = "mall"
        self.builder = ChunkBuilder(self)
        self.ahead, self.behind = 3, 3                     # strict sliding window: 3 chunks ahead, 3 behind, the rest is purged
        self.npcs: list = []
        self._occ_key = None
        self._occ = np.zeros((0, 4))
        self._occ_idx = np.zeros(0, dtype=int)
        self.stats = dict(shown=0, total=0, detail=0)
        self.keep = 1                                      # hysteresis: chunks stay loaded this much beyond the window (no rebuild thrash)
        self.last_built = 0
        self._next_ok = 0.0

    def geom(self, idx: int) -> tuple:
        return chunk_geom(self.seed, idx, self.level)

    def set_level(self, level: str) -> None:
        for idx in list(self.chunks):
            self.unload(idx)
        REAPER.pump(force=True)                            # a level change is hidden behind the lift effect: free everything now
        self.level = level

    def spawn_npc(self, ch: Chunk, mode: str, arg, rng, role: str = "talk") -> None:
        self.g.make_npc(ch, mode, arg, rng, role)

    def chunk_index(self, z: float) -> int:
        return int(math.floor(z / CH))

    def stream(self, z: float, budget: int = 1) -> int:
        pidx = self.chunk_index(z)
        for idx in [i for i in self.chunks if i < pidx - self.behind - self.keep or i > pidx + self.ahead + self.keep]:
            self.unload(idx)
        built = 0
        now = time.perf_counter()
        for idx in sorted((i for i in range(pidx - self.behind, pidx + self.ahead + 1) if i not in self.chunks),
                          key=lambda i: abs(i - pidx)):
            urgent = abs(idx - pidx) <= 1 or budget >= 99                # nearby ones never wait
            if urgent or (built < budget and now >= self._next_ok):
                self.chunks[idx] = self.builder.build(idx)
                built += 1
                if not urgent:
                    self._next_ok = time.perf_counter() + 0.25           # spread far-ahead builds out: one hitch-sized job at a time
        self.last_built = built
        return pidx

    def unload(self, idx: int) -> None:
        ch = self.chunks.pop(idx, None)
        if ch is None:
            return
        self.g.on_unload(ch)
        self.ads.remove_chunk(idx)
        REAPER.push(ch.root)                               # hidden now, freed a few ms at a time by the reaper

    def at(self, z: float) -> Optional[Chunk]:
        return self.chunks.get(self.chunk_index(z))

    def hw_at(self, z: float) -> float:
        ch = self.at(z)
        return ch.hw if ch else self.geom(self.chunk_index(z))[3]

    def xo_at(self, z: float) -> float:
        ch = self.at(z)
        return ch.xoff if ch else self.geom(self.chunk_index(z))[2]

    def near_chunks(self, z: float, spread: int = 1) -> list:
        i = self.chunk_index(z)
        return [self.chunks[k] for k in range(i - spread, i + spread + 1) if k in self.chunks]

    def colliders_near(self, z: float) -> list:
        out = []
        for ch in self.near_chunks(z):
            out += ch.colliders
        return out

    def floor_at(self, x: float, z: float) -> float:
        ch = self.at(z)
        return ch.floor_fn(x, z) if (ch is not None and ch.floor_fn is not None) else 0.0

    def resolve(self, x: float, y: float, z: float, r: float, step: float = 0.32):
        """Circle-vs-box push-out; returns (x, z, ground height under the feet, bumped)."""
        cols, bumped = self.colliders_near(z), False
        for _ in range(2):
            for (x0, x1, z0, z1, top) in cols:
                if x < x0 - r or x > x1 + r or z < z0 - r or z > z1 + r or y >= top - step:
                    continue
                cx, cz = min(max(x, x0), x1), min(max(z, z0), z1)
                dx, dz = x - cx, z - cz
                d2 = dx * dx + dz * dz
                if d2 >= r * r:
                    continue
                bumped = True
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    x += dx / d * (r - d)
                    z += dz / d * (r - d)
                else:
                    pens = (x - x0, x1 - x, z - z0, z1 - z)
                    m = min(pens)
                    if m == pens[0]:
                        x = x0 - r
                    elif m == pens[1]:
                        x = x1 + r
                    elif m == pens[2]:
                        z = z0 - r
                    else:
                        z = z1 + r
        ground = self.floor_at(x, z)
        for (x0, x1, z0, z1, top) in cols:
            if x0 - 0.02 <= x <= x1 + 0.02 and z0 - 0.02 <= z <= z1 + 0.02 and top <= y + step and top < 50:
                ground = max(ground, top)
        return x, z, ground, bumped

    # ---- culling ---------------------------------------------------------------------------------
    def _occluders(self) -> tuple:
        key = tuple(sorted(self.chunks))
        if key != self._occ_key:
            rows, idx = [], []
            for i, ch in self.chunks.items():
                for c in ch.occ:
                    rows.append(c[:4])
                    idx.append(i)
            self._occ = np.array(rows, dtype=float).reshape(-1, 4)
            self._occ_idx = np.array(idx, dtype=int)
            self._occ_key = key
        return self._occ, self._occ_idx

    @staticmethod
    def _blocked(ax: float, az: float, bx: float, bz: float, arr: np.ndarray) -> bool:
        if arr.shape[0] == 0:
            return False
        dx = (bx - ax) if abs(bx - ax) > 1e-9 else 1e-9
        dz = (bz - az) if abs(bz - az) > 1e-9 else 1e-9
        inside = ((arr[:, 0] < ax) & (ax < arr[:, 1]) & (arr[:, 2] < az) & (az < arr[:, 3])) | \
                 ((arr[:, 0] < bx) & (bx < arr[:, 1]) & (arr[:, 2] < bz) & (bz < arr[:, 3]))
        a = arr[~inside]
        if a.shape[0] == 0:
            return False
        tx1, tx2 = (a[:, 0] - ax) / dx, (a[:, 1] - ax) / dx
        tz1, tz2 = (a[:, 2] - az) / dz, (a[:, 3] - az) / dz
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(tz1, tz2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(tz1, tz2))
        return bool(np.any((tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)))

    def cull(self, cam: tuple, fwd: tuple, right: tuple, up: tuple, tan_v: float, tan_h: float, far: float,
             detail_dist: float, show_all: bool = False, occlusion: bool = True) -> None:
        """Hide chunks outside the frustum / beyond fog / behind walls; hide small props of distant ones."""
        cx, cy, cz = cam
        pidx = self.chunk_index(cz)
        ha, va = math.atan(tan_h) + 0.22, math.atan(tan_v) + 0.22
        sh, ch_, sv, cv = math.sin(ha), math.cos(ha), math.sin(va), math.cos(va)
        occ, occ_idx = self._occluders() if occlusion else (np.zeros((0, 4)), np.zeros(0, dtype=int))
        shown = detail = 0
        for i, ch in self.chunks.items():
            near = abs(i - pidx) <= 1
            zc = ch.z0 + CH / 2
            ch.dist = max(0.0, abs(zc - cz) - CH / 2)
            vis = True
            if not (show_all or near):
                R = math.sqrt((ch.hw + 1.0) ** 2 + (ch.h / 2 + 1.0) ** 2 + (CH / 2) ** 2)
                vx, vy, vz = ch.xoff - cx, ch.h / 2 - cy, zc - cz
                depth = vx * fwd[0] + vy * fwd[1] + vz * fwd[2]
                lat = vx * right[0] + vy * right[1] + vz * right[2]
                ver = vx * up[0] + vy * up[1] + vz * up[2]
                if ch.dist > far or depth + R < 0 or (lat * ch_ - depth * sh) > R or (-lat * ch_ - depth * sh) > R \
                        or (ver * cv - depth * sv) > R or (-ver * cv - depth * sv) > R:
                    vis = False
                elif occlusion and abs(i - pidx) >= 2 and occ.shape[0]:
                    sel = (occ_idx >= min(i, pidx)) & (occ_idx <= max(i, pidx))
                    arr = occ[sel]
                    vis = False
                    for fx in (0.0, -0.55, 0.55):
                        for fz in (0.5, 0.15):
                            if not self._blocked(cx, cz, ch.xoff + fx * ch.hw, ch.z0 + CH * fz, arr):
                                vis = True
                                break
                        if vis:
                            break
            want_detail = vis and (show_all or ch.dist < detail_dist)
            if vis != ch.shown:
                ch.shown = vis
                (ch.root.show if vis else ch.root.hide)()
                for n in ch.npcs:
                    (n.fig.root.show if vis else n.fig.root.hide)()
            if want_detail != ch.detail_shown and ch.detail is not None:
                ch.detail_shown = want_detail
                (ch.detail.show if want_detail else ch.detail.hide)()
            shown += vis
            detail += want_detail
        self.stats = dict(shown=shown, total=len(self.chunks), detail=detail)

    def culled_idx(self) -> set:
        return {i for i, c in self.chunks.items() if not (c.shown and c.detail_shown)}



# ============================================================================
# PEOPLE: player body, NPCs, dialogue
# ============================================================================
DIALOGUE = {
    "corridor": [
        "The lights never end. I counted. Then the count counted me.",
        "I've passed {n} pillars today. They keep being pillars.",
        "Behind you is also ahead of you. Don't think about it too hard.",
        "Don't look at the far end. It looks back, politely.",
        "The floor repeats. I find it kind.",
        "I used to know where the exits were. Now I know where they aren't.",
        "Something warm is playing, somewhere. It's always somewhere.",
        "My shadow got tired and went ahead without me.",
        "It was a beautiful mall once. I think it still is, quietly.",
        "You walk like someone who remembers the entrance. Hold on to that.",
        "Every shop is closed for a reason nobody wrote down.",
        "I whisper because the corridor is listening. It likes being listened to.",
        "{m} minutes? I stopped believing in minutes. Ceilings, though. Ceilings are real.",
        "If you reach the end, tell the beginning I said hello.",
    ],
    "cafe": [
        "Oat, almond, soy... do you think milk remembers being anything?",
        "They changed the playlist again. Same song, I think. Slower.",
        "My meeting is at three. There's no clock in here. It's always almost three.",
        "I've been waiting on this flat white since before the ceiling went up.",
        "The pastries are lovely. Nobody ever seems to bake them.",
        "I keep meaning to leave. I keep meaning to find out what leaving means.",
        "Four fifty for a coffee. Time itself is free, which should worry us.",
        "I used to have a commute. One of us ended.",
        "The barista says three sentences. I find it restful.",
        "Funny how the rain never reaches these windows. There are no windows.",
        "Is it a croissant if nobody remembers the oven?",
        "Wifi's down. It has always been down. It is a very honest network.",
    ],
    "gym": [
        "Reps are just counting with extra steps.",
        "Twelve more and I'll have earned an exit.",
        "The mirror shows a hallway behind me that isn't there. Good form, though.",
        "I run to arrive somewhere. The belt disagrees, gently.",
        "My heart rate has been 60 for six hours. I'm fine. I'm fine.",
        "Membership includes the infinite. Sauna is extra.",
        "The weights are always exactly as heavy as I'm ready for. Unsettling.",
        "Nobody here ever finishes a set. We just agree to stop looking.",
    ],
    "restaurant": [
        "Reservation for one. It's always for one.",
        "The special today is whatever you remember.",
        "I asked for the menu. They gave me a quieter room.",
        "Candles on every table, and no one to blow them out.",
        "They'll seat us soon. They've been seating us for years.",
        "The chairs are pushed in. Someone leaves them that way each night. No one leaves.",
        "I'm not waiting. I'm just early for a very long time.",
    ],
}
WHO = {"corridor": "figure", "cafe": "patron", "gym": "athlete", "restaurant": "diner"}
PALETTES = {
    "corridor": dict(coat=(0.62, 0.66, 0.74), legs=(0.45, 0.48, 0.56), skin=(0.86, 0.87, 0.92), em=0.12),
    "cafe": dict(coat=(0.62, 0.38, 0.28), legs=(0.28, 0.24, 0.30), skin=(0.90, 0.76, 0.66), em=0.05),
    "gym": dict(coat=(0.18, 0.55, 0.75), legs=(0.12, 0.12, 0.18), skin=(0.80, 0.66, 0.60), em=0.10),
    "restaurant": dict(coat=(0.22, 0.12, 0.14), legs=(0.12, 0.10, 0.12), skin=(0.82, 0.72, 0.68), em=0.04),
}


class Figure:
    """Low-poly human: torso, pivoting head, jointed arms (shoulder + elbow + hand) and legs on hip pivots."""
    def __init__(self, pal: dict, lm: float, lamp, em: float = 0.0, xc: float = 0.0) -> None:
        self.root = ur.Entity(parent=ur.scene)
        self.ents: list = []
        self.anim = None
        k = dict(lm=lm, lamp=lamp, xc=xc)
        self.k = k

        def part(model, pos, sc, col, parent=None, e=em):
            ent = mk(parent or self.root, model, pos, sc, color=C(*col), emissive=e, **k)
            self.ents.append(ent)
            return ent
        part("cube", (0, 1.05, 0), (0.5, 0.7, 0.28), pal["coat"])
        self.head = ur.Entity(parent=self.root, position=(0, 1.5, 0))
        part("sphere", (0, 0.12, 0), (0.30, 0.34, 0.30), pal["skin"], self.head)
        self.hip = [ur.Entity(parent=self.root, position=(x, 0.72, 0)) for x in (-0.13, 0.13)]
        self.sho = [ur.Entity(parent=self.root, position=(x, 1.38, 0)) for x in (-0.33, 0.33)]
        self.elb: list = []
        for h in self.hip:
            part("cube", (0, -0.36, 0), (0.18, 0.72, 0.2), pal["legs"], h)
        for s in self.sho:
            part("cube", (0, -0.16, 0), (0.14, 0.34, 0.16), pal["coat"], s)
            el = ur.Entity(parent=s, position=(0, -0.32, 0))
            part("cube", (0, -0.14, 0), (0.125, 0.30, 0.145), pal["coat"], el)
            part("sphere", (0, -0.31, 0), (0.11, 0.11, 0.11), pal["skin"], el)
            self.elb.append(el)
        self.phase = 0.0

    def place(self, x, y, z, yaw) -> None:
        self.root.position = (x, y, z)
        self.root.rotation_y = yaw

    def pose(self, dt: float, speed: float, sitting: bool = False) -> None:
        if self.anim is not None:                      # a scripted animation (eating, ...) owns the limbs
            try:
                keep = self.anim(self, dt)
            except Exception:
                keep = False
            if keep is False:
                self.anim = None
            else:
                return
        if sitting:
            for h in self.hip:
                h.rotation_x = -88
            for s in self.sho:
                s.rotation_x = -35
            for e in self.elb:
                e.rotation_x += (-25.0 - e.rotation_x) * min(1.0, 8.0 * dt)
            return
        for e in self.elb:
            e.rotation_x += (-8.0 * min(1.0, speed) - e.rotation_x) * min(1.0, 8.0 * dt)
        self.phase += dt * (2.0 + speed * 1.6)
        amp = min(38.0, speed * 9.0)
        sw = math.sin(self.phase * 3.0) * amp
        self.hip[0].rotation_x, self.hip[1].rotation_x = sw, -sw
        self.sho[0].rotation_x, self.sho[1].rotation_x = -sw * 0.8, sw * 0.8


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


DIALOGUE.update({
    "atrium": ["The fountain remembers having wishes in it. It's too polite to say so.", "That tree knows what month it is. I never do.",
               "Light comes down through the ceiling and nobody asks where it was before.", "I water nothing. It still looks thirsty.",
               "Take a fruit when it's ripe. Leave the hour it's not."],
    "anchor": ["Sale on everything that used to hang here.", "Men's, Women's, Children's. Mostly Absence.",
               "The racks go on for departments that never opened.", "Don't try on anything. It tries back.",
               "My torch died in Menswear. I've been listening since."],
    "skywindow": ["It's lovely up here. The ground is rumoured.", "Watch the weather. It's the only thing that changes its mind.",
                  "Some nights the sky does green things. Don't tell the ceiling.", "We're very high. I checked. I didn't check what we're on."],
    "cinema": ["The posters are locked. The films are not finished.", "Theater 7 is showing something you almost remember.",
               "Popcorn's free if you leave before the credits. They never end.", "The stars in the carpet were there before the carpet."],
    "restroom": ["Something's always dripping. I've stopped looking for it.", "The mirror is very long. Don't count the people in it.",
                 "I'm just washing my hands. For a while now."],
    "security": ["I watch the cameras. The cameras watch the hall. Nobody watches me. Good.", "Camera 4 shows a place we don't have.",
                 "Don't touch the dials. Touch the dials gently."],
    "garage": ["Level P-infinity. Validate your ticket with your memory.", "I parked here once. I was driving something."],
    "pool": ["Keep your shoes on. The water's only ankle-deep. Everywhere. Forever.", "I've been waiting for a lifeguard since the pool filled."],
})
GUARD_LINES = ["Move along. Or don't. The corner is mine, the hall is yours.", "I stand where the walls meet. It's very restful.",
               "If it gets dark, stay near a light. If there's no light, stay near me.", "Nothing to report. Nothing has ever been to report."]
WANDERER_LINES = ["I was just here. I was just there. Isn't that the point?", "Don't look too long. I'm shy in a way that works.",
                  "I'll be gone by the time you turn around, so say what you need."]
MANNEQUIN_LINES = ["...", "(it doesn't seem to have moved. It has.)", "(a faint smell of fabric softener and rain)"]
PALETTES.update({
    "atrium": dict(coat=(0.40, 0.50, 0.36), legs=(0.25, 0.22, 0.2), skin=(0.86, 0.74, 0.65), em=0.05),
    "anchor": dict(coat=(0.30, 0.30, 0.36), legs=(0.14, 0.14, 0.17), skin=(0.80, 0.78, 0.80), em=0.08),
    "skywindow": dict(coat=(0.55, 0.60, 0.70), legs=(0.25, 0.28, 0.34), skin=(0.88, 0.78, 0.70), em=0.06),
    "cinema": dict(coat=(0.55, 0.15, 0.20), legs=(0.12, 0.10, 0.14), skin=(0.86, 0.72, 0.66), em=0.06),
    "restroom": dict(coat=(0.4, 0.55, 0.6), legs=(0.2, 0.22, 0.26), skin=(0.88, 0.76, 0.7), em=0.05),
    "security": dict(coat=(0.16, 0.22, 0.30), legs=(0.10, 0.12, 0.16), skin=(0.84, 0.72, 0.66), em=0.05),
    "garage": dict(coat=(0.35, 0.30, 0.22), legs=(0.15, 0.15, 0.17), skin=(0.82, 0.70, 0.62), em=0.05),
    "pool": dict(coat=(0.6, 0.85, 0.9), legs=(0.4, 0.6, 0.7), skin=(0.9, 0.78, 0.7), em=0.08),
    "mannequin": dict(coat=(0.85, 0.85, 0.88), legs=(0.8, 0.8, 0.84), skin=(0.9, 0.9, 0.93), em=0.10),
})
WHO.update(dict(atrium="loiterer", anchor="shopper", skywindow="visitor", cinema="patron", restroom="visitor",
                security="guard", garage="driver", pool="swimmer"))
ZONE_FG.update(atrium=(220, 255, 220), anchor=(200, 210, 255), skywindow=(210, 235, 255), cinema=(255, 215, 120),
               restroom=(200, 255, 255), security=(160, 255, 200), garage=(255, 210, 110), pool=(200, 255, 255))
ZONE_BG.update(atrium=(10, 28, 16), anchor=(6, 6, 14), skywindow=(8, 18, 40), cinema=(40, 8, 20),
               restroom=(8, 40, 44), security=(6, 28, 18), garage=(30, 22, 6), pool=(8, 50, 60))


def set_xc(entities, xc: float) -> None:
    if FANCY:
        for e in entities:
            e.set_shader_input("xc", float(xc))


class GoKart:
    """Physics go-kart: engine force, speed-dependent steering, lateral grip that collapses while
    drifting (hold Space), a mini-boost when a drift ends, and a fuel tank (canisters / stores)."""
    R, MAXV, ACC = 0.62, 17.0, 14.0

    def __init__(self, key: str, x: float, z: float, yaw: float, fuel: float = 30.0) -> None:
        self.key, self.x, self.z, self.yaw, self.y = key, x, z, yaw, 0.0
        self.vx = self.vz = 0.0
        self.fuel, self.charge, self.skid, self.speed = fuel, 0.0, 0.0, 0.0
        self.root = None
        self.ents: list = []
        self.wheels: list = []
        self.inter = Interactable("kart", x, 0.0, z, 2.4, "Drive the go-kart", yaw, {"kart": self})
        self.steer_vis = 0.0
        self.xc = 0.0

    def build(self) -> None:
        if self.root is not None:
            return
        self.root = ur.Entity(parent=ur.scene)
        self.paint_parts = []
        self.glow = []
        base = tuple(getattr(self, "paint", None) or (0.80, 0.12, 0.14))

        def part(model, pos, sc, col, em=0.0, parent=None, tex="metal", rot=(0, 0, 0), paint=0.0):
            e = mk(parent or self.root, model, pos, sc, tex, color=col, emissive=em, lm=0.0, rot=rot)
            self.ents.append(e)
            if paint:
                self.paint_parts.append((e, paint))
            return e
        pc = lambda k: C(*[min(1.0, c * k) for c in base])
        blk, steel, chrome, white = C(0.07, 0.07, 0.09), C(0.30, 0.31, 0.36), C(0.84, 0.86, 0.92), C(0.96, 0.96, 0.98)
        part("cube", (0, 0.24, 0), (0.95, 0.15, 1.7), pc(1.0), paint=1.0)                         # ents[0]: chassis (repainted)
        part("cube", (0, 0.31, 0.95), (0.7, 0.16, 0.55), pc(1.08), paint=1.08)                     # ents[1]: nose (repainted)
        part("cube", (0, 0.29, 1.30), (0.44, 0.11, 0.32), pc(1.14), rot=(-9, 0, 0), paint=1.14)   # tapered nose cone
        part("cube", (0, 0.20, 1.50), (0.84, 0.05, 0.10), chrome, 0.12)                           # front bumper
        part("cube", (0, 0.09, 0.0), (0.82, 0.05, 1.85), blk)                                      # floor pan
        for s in (-1, 1):
            part("cube", (s * 0.56, 0.31, -0.05), (0.20, 0.20, 0.95), pc(0.92), paint=0.92)        # side pods
            part("cube", (s * 0.56, 0.42, 0.28), (0.16, 0.05, 0.42), pc(1.18), paint=1.18)
            self.glow.append(part("cube", (s * 0.47, 0.115, 0.0), (0.04, 0.035, 1.55), C(0.10, 0.95, 1.0), 1.5))    # underglow
            part("cube", (s * 0.29, 0.345, 1.47), (0.14, 0.07, 0.05), C(1.0, 0.97, 0.8), 1.7)     # headlights
            part("cube", (s * 0.33, 0.42, -1.12), (0.12, 0.06, 0.04), C(1.0, 0.1, 0.08), 1.6)     # tail lights
            part("cube", (s * 0.14, 0.34, -1.22), (0.08, 0.08, 0.36), chrome, 0.1)                # exhaust pipes
            part("cube", (s * 0.14, 0.34, -1.41), (0.07, 0.07, 0.04), C(1.0, 0.55, 0.15), 1.3)
            part("cube", (s * 0.28, 0.76, -1.06), (0.04, 0.44, 0.04), steel)                       # wing posts
            part("cube", (s * 0.46, 0.96, -1.08), (0.03, 0.22, 0.36), blk)                          # wing end plates
        part("cube", (0, 0.325, -0.05), (0.07, 0.01, 1.2), white, 0.25)                           # racing stripes
        part("cube", (0, 0.395, 0.95), (0.07, 0.01, 0.5), white, 0.25)
        part("sphere", (0, 0.40, 0.88), (0.30, 0.02, 0.30), white, 0.25)                           # number roundel
        part("sphere", (0, 0.41, 0.88), (0.14, 0.02, 0.14), C(0.1, 0.1, 0.12))
        part("cube", (0, 0.96, -1.08), (0.9, 0.04, 0.32), pc(0.85), rot=(-10, 0, 0), paint=0.85)  # rear wing
        part("cube", (0, 0.50, -0.38), (0.58, 0.12, 0.5), blk)                                     # bucket seat
        part("cube", (0, 0.80, -0.64), (0.58, 0.55, 0.10), blk)
        for s in (-1, 1):
            part("cube", (s * 0.31, 0.72, -0.52), (0.07, 0.40, 0.24), C(0.16, 0.16, 0.19))
            part("cube", (s * 0.14, 0.86, -0.585), (0.06, 0.50, 0.015), C(0.70, 0.10, 0.10))      # harness
        part("cube", (0, 1.13, -0.66), (0.30, 0.16, 0.08), blk)                                    # head rest
        part("cube", (0, 0.45, -0.92), (0.50, 0.40, 0.42), steel)                                  # engine
        part("cube", (0, 0.40, -0.92), (0.54, 0.06, 0.2), chrome, 0.1)
        part("cube", (0, 0.66, -0.95), (0.40, 0.2, 0.3), pc(0.8), paint=0.8)                       # fuel tank
        part("cube", (0, 0.7, 0.35), (0.05, 0.5, 0.05), steel)                                     # steering column
        self.wheel = ur.Entity(parent=self.root, position=(0, 0.95, 0.3), rotation=(-30, 0, 0))
        for pos, sc, col in (((0, 0, 0), (0.40, 0.035, 0.035), blk), ((-0.2, 0, 0), (0.06, 0.07, 0.07), C(0.7, 0.1, 0.1)),
                             ((0.2, 0, 0), (0.06, 0.07, 0.07), C(0.7, 0.1, 0.1)), ((0, 0, 0), (0.11, 0.06, 0.06), chrome)):
            self.ents.append(mk(self.wheel, "cube", pos, sc, color=col, lm=0.0))
        for sx, sz in ((-0.60, 0.68), (0.60, 0.68), (-0.60, -0.72), (0.60, -0.72)):
            pv = ur.Entity(parent=self.root, position=(sx, 0.2, sz))
            for sc, rot in (((0.20, 0.40, 0.28), (0, 0, 0)), ((0.20, 0.28, 0.40), (0, 0, 0)), ((0.20, 0.34, 0.34), (45, 0, 0))):
                self.ents.append(mk(pv, "cube", (0, 0, 0), sc, "rubber", color=blk, lm=0.0, rot=rot))
            out = 0.012 if sx > 0 else -0.012
            self.ents.append(mk(pv, "cube", (out, 0, 0), (0.205, 0.22, 0.22), "metal", color=chrome, emissive=0.1, lm=0.0, rot=(22, 0, 0)))
            self.ents.append(mk(pv, "cube", (out * 1.8, 0, 0), (0.205, 0.09, 0.09), "metal", color=pc(1.0), lm=0.0))
            self.wheels.append(pv)
        self.sync()

    def recolor(self, col) -> None:
        self.paint = tuple(col)
        for e, k in getattr(self, "paint_parts", ()):
            e.color = C(*[min(1.0, c * k) for c in col])

    def sync(self) -> None:
        if self.root is not None:
            self.root.position = (self.x, self.y, self.z)
            self.root.rotation_y = self.yaw
            self.wheel.rotation_z = self.steer_vis * 25
        self.inter.x, self.inter.z = self.x, self.z

    def drive(self, dt: float, w: "World", hk, boost_mult: float = 1.0) -> dict:
        fwd = hk["w"] + hk["up arrow"] - hk["s"] - hk["down arrow"]
        steer = hk["d"] + hk["right arrow"] - hk["a"] - hk["left arrow"]
        steer = max(-1, min(1, steer))
        want_drift = bool(hk["space"])
        engine = self.fuel > 0.0
        yr = math.radians(self.yaw)
        vf0 = self.vx * math.sin(yr) + self.vz * math.cos(yr)
        sf = min(1.0, abs(vf0) / 4.0)
        drifting = want_drift and abs(vf0) > 6.0 and steer != 0
        rate = 88.0 * (1.0 - 0.35 * min(1.0, abs(vf0) / self.MAXV)) * (1.35 if drifting else 1.0)
        self.yaw += steer * rate * sf * (1.0 if vf0 >= 0 else -1.0) * dt
        self.steer_vis += (steer - self.steer_vis) * min(1.0, 10 * dt)
        yr = math.radians(self.yaw)
        hx, hz, rx, rz = math.sin(yr), math.cos(yr), math.cos(yr), -math.sin(yr)
        vf = self.vx * hx + self.vz * hz
        vl = self.vx * rx + self.vz * rz
        burn = 0.12
        if engine and fwd > 0:
            vf += self.ACC * dt * max(0.0, 1.0 - vf / (self.MAXV * boost_mult))
            burn = 1.7
        elif fwd < 0:
            vf = vf - 22.0 * dt if vf > 0.5 else max(-4.0, vf - 7.0 * dt) if engine else vf - 8.0 * dt
        vf -= vf * 0.12 * dt
        if fwd == 0:
            vf *= math.exp(-0.55 * dt)
        vl *= math.exp(-(1.8 if drifting else 9.0) * dt)
        if drifting:
            self.charge += dt
        else:
            if self.charge > 0.8 and vf > 5:
                vf += min(3.5, self.charge * 2.0)
            self.charge = 0.0
        self.fuel = max(0.0, self.fuel - burn * dt)
        self.skid = min(1.0, abs(vl) / 3.5) if abs(vf) > 3 else 0.0
        self.vx, self.vz = hx * vf + rx * vl, hz * vf + rz * vl
        nx, nz = self.x + self.vx * dt, self.z + self.vz * dt
        nx, nz, ground, bumped = w.resolve(nx, self.y, nz, self.R, step=0.25)
        if bumped:
            self.vx *= 0.35
            self.vz *= 0.35
        xo, hw = w.xo_at(nz), w.hw_at(nz)
        lim = hw - 0.7
        if abs(nx - xo) > lim:
            nx = xo + math.copysign(lim, nx - xo)
            self.vx *= 0.4
        self.x, self.z = nx, nz
        self.y += (ground - self.y) * min(1.0, 18 * dt)
        self.speed = math.hypot(self.vx, self.vz)
        for wh in self.wheels:
            wh.rotation_y = self.steer_vis * 20 if wh.z > 0 else 0
        self.sync()
        return dict(drifting=drifting, charge=self.charge, skid=self.skid, rpm=min(1.0, self.speed / self.MAXV + (0.25 if fwd > 0 else 0.0)))


class Player:
    R, MAX_WALK, MAX_RUN, GRAV, JUMP = 0.38, 4.4, 7.2, 24.0, 7.4

    def __init__(self, g: "Game") -> None:
        self.g = g
        self.x, self.y, self.z = 0.0, 0.0, 2.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw_body = 0.0
        self.on_ground = True
        self.seat: Optional[Interactable] = None
        self.kart: Optional[GoKart] = None
        self.running = False
        self.wade = False
        z = ZONES["corridor"]
        self.fig = Figure(dict(coat=(0.10, 0.12, 0.18), legs=(0.08, 0.08, 0.10), skin=(0.92, 0.82, 0.74)),
                          0.0, z["lamp"], em=0.18)
        self.zone_kind = "corridor"

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vz)

    def sit(self, it: Interactable) -> None:
        self.seat = it
        it.occupied = True
        self.x, self.z, self.y, self.yaw_body = it.x, it.z, it.y - 0.72, it.yaw
        self.vx = self.vz = self.vy = 0.0

    def stand(self) -> None:
        it = self.seat
        if it is None:
            return
        s, c = math.sin(math.radians(it.yaw)), math.cos(math.radians(it.yaw))
        self.x, self.z, self.y = it.x + s * 0.8, it.z + c * 0.8, self.g.world.floor_at(it.x + s * 0.8, it.z + c * 0.8)
        it.occupied = False
        self.seat = None

    def mount(self, kart: GoKart) -> None:
        self.kart = kart
        kart.build()
        self.vx = self.vz = self.vy = 0.0
        self.x, self.z = kart.x, kart.z

    def dismount(self) -> None:
        k = self.kart
        if k is None:
            return
        yr = math.radians(k.yaw)
        self.x, self.z = k.x + math.cos(yr) * 1.2, k.z - math.sin(yr) * 1.2
        nx, nz, ground, _ = self.g.world.resolve(self.x, 0.0, self.z, self.R)
        self.x, self.z, self.y = nx, nz, ground
        self.vx = self.vz = 0.0
        self.kart = None
        self.g.sfx.engine["vol"] = 0.0
        self.g.sfx.engine["skid"] = 0.0

    def update(self, dt: float, w: "World", yaw: float, jump_pressed: bool) -> None:
        g, hk = self.g, ur.held_keys
        if self.kart is not None:
            k = self.kart
            st = k.drive(dt, w, hk, 1.0)
            self.x, self.z, self.y = k.x, k.z, k.y
            self.vx, self.vz = k.vx, k.vz
            self.yaw_body = k.yaw
            self.fig.place(k.x, k.y + 0.52 - 0.72 + 0.1, k.z - 0.0, k.yaw)
            self.fig.pose(dt, 0, True)
            g.sfx.engine.update(vol=k.engine_vol(), rpm=st["rpm"], skid=0.8 * st["skid"] if st["drifting"] else 0.0)
            return
        locked = g.ui is not None or g.elev is not None
        fwd = 0 if locked else (hk["w"] or hk["up arrow"]) - (hk["s"] or hk["down arrow"])
        strafe = 0 if locked else hk["d"] - hk["a"]
        if self.seat is not None:
            if (fwd or strafe) and not locked:
                self.stand()
            else:
                self.fig.place(self.x, self.y, self.z, self.yaw_body)
                self.fig.pose(dt, 0, True)
                self.running = False
                return
        sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        dx, dz = sy * fwd + cy * strafe, cy * fwd - sy * strafe
        n = math.hypot(dx, dz)
        want_run = bool(hk["shift"] or hk["left shift"]) and not g.vitals.exhausted and n > 1e-6 and not locked
        self.running = want_run
        vmax = (self.MAX_RUN if want_run else self.MAX_WALK) * g.vitals.speed() * (0.7 if self.wade else 1.0)
        tx, tz = (dx / n * vmax, dz / n * vmax) if n > 1e-6 else (0.0, 0.0)
        blend = 1.0 - math.exp(-(13.0 if self.on_ground else 3.0) * dt)
        self.vx += (tx - self.vx) * blend
        self.vz += (tz - self.vz) * blend
        if jump_pressed and self.on_ground and not locked:
            self.vy, self.on_ground = self.JUMP * math.sqrt(g.vitals.mult(2)), False
        self.vy -= self.GRAV * dt
        nx, nz = self.x + self.vx * dt, self.z + self.vz * dt
        for npc in g.nearby_npcs():                                        # soft body collision with people
            if npc.gone:
                continue
            ddx, ddz = nx - npc.x, nz - npc.z
            d = math.hypot(ddx, ddz)
            if d < self.R + 0.36 and self.y < 1.5:
                nx, nz = npc.x + ddx / (d + 1e-6) * (self.R + 0.36), npc.z + ddz / (d + 1e-6) * (self.R + 0.36)
        nx, nz, ground, bumped = w.resolve(nx, self.y, nz, self.R)
        hw, xo = w.hw_at(nz), w.xo_at(nz)
        nx = max(xo - hw + 0.1, min(xo + hw - 0.1, nx))
        if bumped:                                                          # kill velocity into walls
            self.vx = (nx - self.x) / dt * 0.5 if dt > 0 else 0.0
            self.vz = (nz - self.z) / dt * 0.5 if dt > 0 else 0.0
        self.x, self.z = nx, nz
        self.y += self.vy * dt
        self.wade = ground < -0.3
        if self.y <= ground:
            self.y, self.vy, self.on_ground = ground, 0.0, True
        elif self.y > ground + 0.03:
            self.on_ground = False
        if n > 1e-6 and self.speed > 0.3:
            self.yaw_body += wrap180(math.degrees(math.atan2(self.vx, self.vz)) - self.yaw_body) * min(1.0, 12.0 * dt)
        self.fig.place(self.x, self.y, self.z, self.yaw_body)
        self.fig.pose(dt, self.speed if self.on_ground else 0.5)


class Npc:
    """Sparse, lonely people.  Nobody speaks until you face them and press E.  Modes: wander, sit,
    stand, ai (the talkable one), shop, guard (corner guard), vanish (vanishing wanderer), mannequin."""
    def __init__(self, g: "Game", ch: Chunk, mode: str, arg, rng: random.Random, role: str = "talk") -> None:
        self.g, self.ch, self.mode, self.rng, self.role = g, ch, mode, rng, role
        self.kind = ch.kind
        pal = PALETTES["mannequin"] if mode == "mannequin" else PALETTES.get(ch.kind, PALETTES["corridor"])
        self.seat: Optional[Interactable] = None
        hw = ch.hw
        self.yaw = rng.uniform(0, 360)
        self.gone, self.gone_t, self.obs_t, self.shift_t, self.last_talk = False, 0.0, 0.0, rng.uniform(3, 8), -1e9
        self.history: list = []
        if mode == "sit":
            self.seat = arg
            self.x, self.z, self.yaw = arg.x, arg.z, arg.yaw
            arg.occupied = True
            self.y = arg.y - 0.72
        else:
            self.y = 0.0
            if mode in ("stand", "ai", "shop", "guard"):
                self.x, self.z = arg
                if mode == "guard":
                    self.yaw = self.home_yaw = 180.0 if rng.random() < 0.5 else 0.0
            else:
                self.x, self.z = self._free_spot(hw)
        self.fig = Figure(pal, ch.lm, ch.lamp, pal["em"], xc=ch.xoff)
        self.state = "sit" if mode == "sit" else ("wander" if mode in ("wander", "vanish") else "idle")
        self.timer = rng.uniform(1, 6)
        self.target = None
        self.stuck = 0.0
        self.last_line = -1
        self.speed = 0.0
        names = dict(guard="guard", vanish="wanderer", mannequin="mannequin")
        if mode in names:
            self.name = names[mode]
        elif role == "shop":
            self.name = {"cafe": "barista", "cinema": "usher", "anchor": "clerk"}.get(ch.kind, "vendor")
        else:
            self.name = WHO.get(ch.kind, "figure")
        if mode == "mannequin":
            for s in self.fig.sho:
                s.rotation_x = rng.uniform(-50, 30)
                s.rotation_z = rng.uniform(-20, 20)
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.mouth = rng.random()

    @property
    def radius(self) -> float:
        return 2.6

    def _free_spot(self, hw: float):
        ch = self.ch
        for _ in range(14):
            x, z = ch.xoff + self.rng.uniform(-hw + 1.4, hw - 1.4), ch.z0 + self.rng.uniform(1.5, CH - 1.5)
            if all(not (c[0] - 0.5 < x < c[1] + 0.5 and c[2] - 0.5 < z < c[3] + 0.5 and c[4] > 0.3) for c in ch.colliders):
                return x, z
        return ch.xoff, ch.z0 + CH / 2

    def pick_target(self) -> None:
        ch, hw = self.ch, self.ch.hw
        for _ in range(8):
            tx = ch.xoff + self.rng.uniform(-hw + 1.3, hw - 1.3)
            tz = ch.z0 + self.rng.uniform(1.2, CH - 1.2)
            if all(not (c[0] - 0.6 < tx < c[1] + 0.6 and c[2] - 0.6 < tz < c[3] + 0.6 and c[4] > 0.3) for c in ch.colliders):
                self.target = (tx, tz)
                return
        self.target = None

    def next_line(self) -> str:
        pool = {"guard": GUARD_LINES, "vanish": WANDERER_LINES, "mannequin": MANNEQUIN_LINES}.get(self.mode) \
            or DIALOGUE.get(self.kind) or DIALOGUE["corridor"]
        i = self.rng.randrange(len(pool))
        if i == self.last_line:
            i = (i + 1) % len(pool)
        self.last_line = i
        return (pool[i].replace("{n}", str(abs(self.ch.idx) * 3 + self.rng.randint(0, 2)))
                .replace("{m}", str(int(self.g.clock // 60))))

    def face(self, tyaw: float, dt: float, rate: float = 5.0) -> None:
        self.yaw += wrap180(tyaw - self.yaw) * min(1.0, rate * dt)

    def update(self, dt: float, w: "World") -> None:
        g = self.g
        px, pz = g.player.x, g.player.z
        dx, dz = px - self.x, pz - self.z
        d = math.hypot(dx, dz)
        if d > 52:
            return
        toward = math.degrees(math.atan2(dx, dz))
        observed = d < 16 and g.view_cos(self.x, self.z) > 0.62
        if self.mode == "vanish":
            if self.gone:
                self.gone_t -= dt
                if self.gone_t <= 0 and not observed:
                    self.x, self.z = self._free_spot(self.ch.hw)
                    self.gone = False
                    self.fig.root.show()
                return
            self.obs_t = self.obs_t + dt if (observed and d > 2.8) else max(0.0, self.obs_t - dt)
            if self.obs_t > 1.7:
                self.gone, self.gone_t, self.obs_t = True, self.rng.uniform(15, 40), 0.0
                self.fig.root.hide()
                return
        if self.mode == "mannequin":
            self.shift_t -= dt
            if not observed and self.shift_t <= 0:                         # moves only when unseen
                self.shift_t = self.rng.uniform(3, 8)
                if d > 2.4:
                    step = min(0.9, d - 2.2)
                    nx, nz, _, bumped = w.resolve(self.x + dx / d * step, 0.0, self.z + dz / d * step, 0.34)
                    if not bumped:
                        self.x, self.z = nx, nz
                self.yaw = toward + self.rng.uniform(-35, 35)
                for s in self.fig.sho:
                    s.rotation_x, s.rotation_z = self.rng.uniform(-60, 30), self.rng.uniform(-25, 25)
            self.fig.place(self.x, self.y, self.z, self.yaw)
            return
        moving = False
        if self.state == "sit":
            self.face(toward if d < 4.5 else (self.seat.yaw if self.seat else self.yaw), dt, 2.0 if d < 4.5 else 1.0)
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(dt, 0, True)
            return
        if self.mode == "guard":
            self.face(toward if d < 6.0 else self.home_yaw, dt, 1.2)
        elif self.mode in ("ai", "shop"):
            self.face(toward if d < 9 else self.yaw, dt, 1.5)
        elif d < 4.0 and self.state != "attend":
            self.state = "attend"
        if self.state == "attend":
            self.face(toward, dt, 3.5)
            if d > 6.5:
                self.state, self.timer = "idle", self.rng.uniform(1.5, 4)
        elif self.state == "idle":
            self.timer -= dt
            if self.mode in ("stand", "ai", "shop", "guard"):
                pass
            elif self.timer <= 0:
                self.pick_target()
                self.state = "wander" if self.target else "idle"
                self.timer = self.rng.uniform(2, 6)
        elif self.state == "wander":
            if self.target is None:
                self.pick_target()
                if self.target is None:
                    self.state = "idle"
            else:
                tx, tz = self.target
                ex, ez = tx - self.x, tz - self.z
                dist = math.hypot(ex, ez)
                if dist < 0.4:
                    self.state, self.target, self.timer = "idle", None, self.rng.uniform(3, 9)
                else:
                    self.face(math.degrees(math.atan2(ex, ez)), dt, 4.0)
                    step = 0.95 * dt
                    nx, nz, ground, bumped = w.resolve(self.x + ex / dist * step, 0.0, self.z + ez / dist * step, 0.34)
                    self.stuck = self.stuck + dt if (bumped or abs(nx - self.x) + abs(nz - self.z) < step * 0.3) else 0.0
                    self.x, self.z, self.y = nx, nz, ground
                    moving = True
                    if self.stuck > 1.4:
                        self.target, self.stuck, self.state, self.timer = None, 0.0, "idle", 1.0
        self.speed = 0.95 if moving else 0.0
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.pose(dt, self.speed * 1.4)


class Hud:
    def __init__(self, g: "Game") -> None:
        self.g = g
        mk_text = lambda **kw: ur.Text(**kw)
        self.zone = mk_text(text="", position=(-0.86, 0.46), scale=1.15, color=C(0.95, 0.97, 1.0, 0.9))
        self.panel = ur.Entity(parent=ur.camera.ui, model="quad", position=(0, -0.335), scale=(1.5, 0.105), color=C(0, 0, 0, 0.0), z=0.2)
        self.sub = mk_text(text="", position=(0, -0.33), origin=(0, 0), scale=1.05, color=C(1, 1, 1, 0))
        self.prompt = mk_text(text="", position=(0, -0.24), origin=(0, 0), scale=1.0, color=C(1, 0.95, 0.8, 0.95))
        self.note_t = mk_text(text="", position=(0, -0.17), origin=(0, 0), scale=1.0, color=C(0.7, 1.0, 0.8, 0))
        self.status = mk_text(text="", position=(0.86, 0.46), origin=(0.5, 0.5), scale=0.9, color=C(0.9, 0.95, 1, 0.9))
        self.dbg = mk_text(text="", position=(-0.86, -0.20), scale=0.75, color=C(0.7, 0.9, 1.0, 0.85))
        self.help = mk_text(text="WASD move  mouse look  Shift run  Space jump  E interact  G engage/hug  T use held  U store  H hold (in bag)  P phone  I inventory  F torch  V view  Tab mouse  F3 debug  Esc quit",
                            position=(0, 0.46), origin=(0, 0), scale=0.72, color=C(1, 1, 1, 0.75))
        self.sub_t, self.speaker, self.help_t, self.show_dbg = 0.0, None, 16.0, False
        self.note_life = 0.0
        self.bars = {}
        for i, (key, col) in enumerate((("hunger", (0.95, 0.65, 0.25)), ("thirst", (0.35, 0.7, 1.0)), ("stamina", (0.5, 0.95, 0.5)),
                                        ("fuel", (1.0, 0.4, 0.3)), ("torch", (1.0, 0.95, 0.6)))):
            y = -0.46 + 0.026 * i
            ur.Entity(parent=ur.camera.ui, model="quad", color=C(0, 0, 0, 0.45), position=(-0.86, y), origin=(-0.5, 0), scale=(0.2, 0.014), z=0.1)
            fill = ur.Entity(parent=ur.camera.ui, model="quad", color=C(*col, 0.9), position=(-0.86, y), origin=(-0.5, 0), scale=(0.2, 0.014), z=0.09)
            lab = ur.Text(text=key.upper(), position=(-0.645, y + 0.008), scale=0.55, color=C(1, 1, 1, 0.55))
            self.bars[key] = (fill, lab)

    def speak(self, npc, text: str) -> None:
        self.sub.text = f'[ {npc.name} ]  "{text}"'
        self.sub_t, self.speaker = 7.0, npc

    def note(self, text: str, secs: float = 2.6, good: bool = True) -> None:
        self.note_t.text, self.note_life = text, secs
        self.note_t.color = C(0.7, 1.0, 0.8, 1) if good else C(1.0, 0.7, 0.6, 1)

    def update(self, dt: float, zone_title: str, prompt: str, dbg: str) -> None:
        g = self.g
        self.zone.text = f"{zone_title}    z {g.player.z:+.0f} m" + ("    [ BLACKOUT ]" if g.grid.out else "")
        self.prompt.text = prompt
        if self.sub_t > 0:
            self.sub_t -= dt
            near = self.speaker is not None and math.hypot(self.speaker.x - g.player.x, self.speaker.z - g.player.z) < 9
            a = min(1.0, self.sub_t / 1.2, (7.0 - self.sub_t) / 0.4) if near else 0.0
            self.sub.color = C(1, 0.97, 0.9, max(0.0, a) * 0.95)
            self.panel.color = C(0, 0, 0, max(0.0, a) * 0.55)
        else:
            self.sub.color = C(1, 1, 1, 0)
            self.panel.color = C(0, 0, 0, 0)
        if self.note_life > 0:
            self.note_life -= dt
            c = self.note_t.color
            self.note_t.color = C(c[0], c[1], c[2], min(1.0, self.note_life / 0.6))
        self.help_t -= dt
        self.help.color = C(1, 1, 1, max(0.0, min(0.75, self.help_t / 3.0)))
        v, sky = g.vitals, g.sky
        self.status.text = f"{g.inv.coins} c    {sky.clock_text}  {sky.weather}"
        vals = dict(hunger=v.hunger / 100, thirst=v.thirst / 100, stamina=v.stamina / 100,
                    fuel=(g.player.kart.fuel / 100 if g.player.kart else 0.0), torch=g.torch_batt / 100)
        for k, (fill, lab) in self.bars.items():
            on = k not in ("fuel",) or g.player.kart is not None
            fill.scale_x = 0.2 * max(0.001, min(1.0, vals[k]))
            fill.enabled = lab.enabled = on
        self.dbg.text = dbg if self.show_dbg else ""



# ============================================================================
# AUDIO <-> GAME BRIDGE
# ============================================================================
class AudioStack:
    """Wires the original engine (indexer, conductor, mixer, DSP, master, ring, output) together
    for the game and starts it on a background thread so the window opens immediately."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.status = args, "silent"
        self.indexer = self.live = self.mixer = self.conductor = self.masterer = None
        self.renderer = self.output = self.cfg = None
        self.active = False
        self.dsp = None
        self._irs: dict = {}
        self._profile = "mall"
        self._prof_lock = threading.Lock()

    def prepare(self) -> None:
        a = self.args
        if a.silent:
            self.status = "audio disabled (--silent)"
            return
        if not a.library:
            self.status = "silent - pass --library DIR to start the orchestra"
            log.warning(self.status)
            return
        if _MISSING:
            self.status = "silent - missing " + ", ".join(_MISSING)
            log.warning("Audio disabled: %s", self.status)
            return
        if not Path(a.library).expanduser().is_dir():
            self.status = f"silent - library folder not found: {a.library}"
            log.error(self.status)
            return
        if a.harmonize_with:
            log.warning("--harmonize-with is ignored in game mode (use --audio-only for it)")
            a.harmonize_with = None
        try:
            device = int(a.device) if a.device and str(a.device).isdigit() else a.device
            sr = a.samplerate
            if sr is None:
                try:
                    sr = int(sd.query_devices(device, "output")["default_samplerate"])
                except Exception:
                    sr = 44_100
            a.library = Path(a.library).expanduser()
            cfg = self.cfg = build_config(a, sr)
            self.indexer = LibraryIndexer(cfg.library_dir, cfg.analysis,
                                          DiskCache(cfg.cache_dir, "analysis", cfg.use_cache), cfg.max_workers)
            self.mixer = Mixer(sr)
            dsp = EtherealDSPProcessor(sr, cfg.dsp, cfg.engine_block)
            self.live = LiveParams(cfg)
            dsp.live = self.live
            self.dsp = dsp
            self.conductor = OrchestralConductor(cfg, self.indexer, SmartSlicer(cfg.slicing, sr), dsp, self.mixer,
                                                 sequence_spec=a.sequence, pool=a.tracks)
            self.conductor.live = self.live
            self.masterer = StreamingMasterer(cfg.master, sr)
            ring = RingBuffer(int((cfg.buffer_s + 2.0) * sr))
            self.renderer = RenderThread(cfg, self.mixer, dsp.new_convolver(), self.masterer, ring, None, self.live)
            self.output = AudioOutput(cfg, ring)
        except Exception as exc:
            self.status = f"audio error: {exc}"
            log.error("Audio setup failed: %s", exc)
            return
        self.status = "scanning library..."
        threading.Thread(target=self._boot, name="audio-boot", daemon=True).start()

    def _boot(self) -> None:
        try:
            self.conductor.start()                    # scans, analyses first tracks, starts the lanes
            self.status = "analysing first tracks..."
            self.renderer.start()
            self.output.start()
            self.active = True
            self.status = "playing"
        except Exception as exc:
            self.status = f"audio error: {exc}"
            log.error("Audio start failed: %s", exc)

    def set_reverb_profile(self, name: str) -> None:
        """Swap the shared convolution IR (bathroom tile, concrete garage, atrium hall...) on a worker thread."""
        if not self.active or self.renderer is None or self.dsp is None or name == self._profile:
            return
        self._profile = name

        def work() -> None:
            try:
                with self._prof_lock:
                    if name not in self._irs:
                        if name == "mall":
                            self._irs[name] = self.dsp.ir
                        else:
                            rt, damp, pre, seed = REVERB_PROFILES[name]
                            c = dataclasses.replace(self.cfg.dsp, reverb_rt60_s=rt, damping_hz=damp, predelay_ms=pre, ir_seed=seed)
                            self._irs[name] = EtherealDSPProcessor(self.cfg.sample_rate, c, self.cfg.engine_block).ir
                    if self._profile == name:
                        self.renderer.conv = PartitionedConvolver(self._irs[name], self.cfg.engine_block)
            except Exception as exc:
                log.warning("Reverb profile %s failed: %s", name, exc)
        threading.Thread(target=work, name="reverb-profile", daemon=True).start()

    def level(self) -> float:
        return 0.0 if self.masterer is None else float(np.clip((self.masterer.peak_db + 36.0) / 30.0, 0.0, 1.0))

    def playing(self) -> dict:
        return self.mixer.playing_tracks() if (self.active and self.mixer is not None) else {}

    def stop(self) -> None:
        try:
            if self.conductor is not None:
                self.conductor.stop()
            if self.renderer is not None and self.active:
                self.renderer.start_fade(1.5)
                time.sleep(1.8)
            if self.output is not None:
                self.output.stop()
            if self.renderer is not None:
                self.renderer.stop_flag.set()
        finally:
            if self.indexer is not None:
                self.indexer.close()


class AudioBridge:
    """Turns where you are and what you do into DSP: a cold infinite corridor is dark, wet and
    smeared; a warm café opens the sound up; the gym is dense and dry; the empty restaurant
    thins to a single, drowned voice.  Speed, lingering at an ad, and sitting nudge it further."""
    TAU = 1.6

    def __init__(self, stack: AudioStack) -> None:
        self.stack = stack
        self.cur = dict(AUDIO_PRESETS["corridor"])
        self.still = 0.0
        self.linger = 0.0
        self.ad_track: Optional[int] = None
        self.zone = "corridor"
        self.duck, self.duck_target = 1.0, 1.0
        self.extra: dict = {}
        self.profile = "mall"

    def update(self, dt: float, zone: str, speed01: float, seated: bool, ad, ad_dist: float, machines: int) -> None:
        self.zone = zone
        t = dict(AUDIO_PRESETS[zone])
        ex = self.extra
        t["wet"] += ex.get("wet", 0.0)
        t["tone_hz"] += ex.get("tone", 0.0)
        prof = REVERB_OF.get(zone, "mall")
        if prof != self.profile:
            self.profile = prof
            self.stack.set_reverb_profile(prof)
        tau = 0.9 if self.duck_target < self.duck else 3.2          # fast duck, slow fade back in
        self.duck += (self.duck_target - self.duck) * (1.0 - math.exp(-dt / tau))
        self.still = self.still + dt if speed01 < 0.05 else 0.0
        t["wet"] -= 0.14 * speed01
        t["tone_hz"] += 2600.0 * speed01
        t["lanes"] += 1.2 * speed01 + 0.4 * min(machines, 3)
        if self.still > 6.0:                                       # standing still: dissolve into the smear
            k = min(1.0, (self.still - 6.0) / 10.0)
            t["stretch"] += 0.35 * k
            t["wet"] += 0.08 * k
        if ad is not None and ad_dist < 5.0:                       # an ad draws the sound closer
            k = 1.0 - ad_dist / 5.0
            t["wet"] -= 0.20 * k
            t["tone_hz"] += 3500.0 * k
            t["stretch"] = max(1.0, t["stretch"] - 0.4 * k)
        if seated:
            t["wet"] += 0.10
            t["gain"] *= 0.92
        a = 1.0 - math.exp(-dt / self.TAU)
        for k_, v in t.items():
            self.cur[k_] += (v - self.cur[k_]) * a
        live = self.stack.live
        if live is not None:
            live.wet = float(min(max(self.cur["wet"], 0.05), 0.95))
            live.tone_hz = float(self.cur["tone_hz"])
            live.gain = float(self.cur["gain"] * self.duck)
            live.stretch = float(max(1.0, self.cur["stretch"]))
            live.smear_prob = float(min(max(self.cur["smear_prob"], 0.0), 1.0))
            live.warmth_hz = float(self.cur["warmth_hz"])
            live.lanes = int(round(self.cur["lanes"]))
        cond = self.stack.conductor
        if cond is not None:                                       # linger at an ad -> its song is next
            if ad is not None and ad_dist < 4.0 and ad.track is not None and ad.track >= 0:
                if ad.track != self.ad_track:
                    self.ad_track, self.linger = ad.track, 0.0
                self.linger += dt
            else:
                self.linger = max(0.0, self.linger - dt * 2)
            if self.linger > 2.0 and self.ad_track is not None:
                cond.boost.clear()
                cond.boost[self.ad_track] = 16.0
            elif self.linger <= 0.0 and cond.boost:
                cond.boost.clear()

    def describe(self) -> str:
        c = self.cur
        return (f"audio[{self.zone}] wet {c['wet']:.2f}  tone {c['tone_hz']:.0f} Hz  stretch {c['stretch']:.2f}  "
                f"smear {c['smear_prob']:.2f}  lanes {c['lanes']:.1f}  gain {c['gain']:.2f}")


# ============================================================================
# FULL-SCREEN OVERLAYS: listening post, inventory album, shop, CCTV
# ============================================================================
class UiBase:
    mouse_free = False

    def __init__(self, g: "Game") -> None:
        self.g, self.ents = g, []

    def quad(self, pos=(0, 0), scale=(1, 1), col=None, tex=None, z=0.2):
        e = ur.Entity(parent=ur.camera.ui, model="quad", position=pos, scale=scale, color=col or C(0, 0, 0, 0.85), texture=tex, z=z)
        self.ents.append(e)
        return e

    def text(self, txt="", pos=(0, 0), scale=1.0, col=None, origin=(-0.5, 0)):
        t = ur.Text(text=txt, position=pos, scale=scale, color=col or C(0.85, 1.0, 0.92, 0.95), origin=origin)
        self.ents.append(t)
        return t

    def close(self) -> None:
        for e in self.ents:
            ur.destroy(e)
        self.ents = []

    def key(self, k: str) -> None: ...
    def update(self, dt: float) -> None: ...


def tex_from_pil(img):
    return BANK.to_texture(img, mip=False)


class KioskUI(UiBase):
    """The in-mall Listening Post: browse the library, preview tracks, insert cassettes to unlock more."""
    def __init__(self, g, seat: Interactable) -> None:
        super().__init__(g)
        self.seat, self.sel, self.msg, self.loading, self.playing = seat, 0, "", False, None
        self.tids = sorted(g.stack.indexer.tracks) if g.stack.indexer is not None else []
        g.player.sit(seat)
        g.bridge.duck_target = 0.0
        self.quad((0, 0), (1.2, 0.82), C(0.1, 0.9, 0.6, 0.35), z=0.21)
        self.quad((0, 0), (1.18, 0.80), C(0.015, 0.04, 0.035, 0.93))
        self.text("LISTENING POST   -   ETHEREAL FM", (-0.56, 0.35), 1.1)
        self.status = self.text("", (-0.56, 0.30), 0.8, C(1.0, 0.8, 0.4, 0.95))
        self.rows = [self.text("", (-0.56, 0.24 - 0.044 * i), 0.9) for i in range(11)]
        self.text("W/S select   Enter preview/stop   I insert cassette   E / Esc stand up", (-0.56, -0.36), 0.7, C(0.6, 0.8, 0.7, 0.9))
        self.refresh()

    def is_open(self, tid: int) -> bool:
        return tid in self.g.inv.unlocked or stable_hash(self.g.args.seed, "free", tid) % 4 == 0

    def refresh(self) -> None:
        g = self.g
        n = len(self.tids)
        self.status.text = (f"tracks {n}   cassettes in bag: {len(g.inv.cassettes)}   {self.msg}" if n else
                            "no library found - run with --library DIR")
        top = max(0, min(self.sel - 5, n - 11)) if n else 0
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= n:
                row.text = ""
                continue
            tid = self.tids[j]
            nm = g.covers.title(tid).replace("_", " ")[:48]
            open_ = self.is_open(tid)
            mark = ">" if j == self.sel else " "
            tag = "  [playing]" if self.playing == tid else ""
            row.text = f"{mark} {j + 1:3d}  {nm if open_ else '######## LOCKED ########'}{tag}"
            row.color = C(1.0, 0.95, 0.6, 1) if j == self.sel else (C(0.8, 1.0, 0.9, 0.95) if open_ else C(0.45, 0.5, 0.5, 0.8))

    def key(self, k: str) -> None:
        g, n = self.g, len(self.tids)
        if k in ("e", "escape"):
            g.close_ui()
        elif k in ("w", "up arrow") and n:
            self.sel, _ = (self.sel - 1) % n, g.sfx.play("tick", 0.6)
            self.refresh()
        elif k in ("s", "down arrow") and n:
            self.sel, _ = (self.sel + 1) % n, g.sfx.play("tick", 0.6)
            self.refresh()
        elif k == "enter" and n:
            tid = self.tids[self.sel]
            if not self.is_open(tid):
                self.msg = "locked - insert a cassette (I)"
            elif self.playing == tid:
                g.sfx.stop_music()
                self.playing, self.msg = None, "stopped"
            else:
                self.preview(tid)
            self.refresh()
        elif k == "i":
            if not g.inv.cassettes:
                self.msg = "no cassettes in your bag"
            else:
                tid = next((c for c in g.inv.cassettes if c in self.tids and not self.is_open(c)), None)
                if tid is None:
                    self.msg = "those tapes are already unlocked"
                else:
                    g.inv.cassettes.remove(tid)
                    g.inv.unlocked.add(tid)
                    g.inv.dirty = True
                    g.sfx.play("click")
                    g.sfx.play("pickup", 0.8)
                    self.sel, self.msg = self.tids.index(tid), "tape accepted - track unlocked forever"
            self.refresh()

    def preview(self, tid: int) -> None:
        g = self.g
        if not g.sfx.enabled or librosa is None:
            self.msg = "no audio output available"
            return
        info = g.stack.indexer.tracks.get(tid)
        if info is None:
            return
        self.loading, self.msg, self.playing = True, "loading...", tid
        sr = g.sfx.sr
        off = 0.25 * info.analysis.duration_s if info.analysis is not None else 0.0

        def work():
            try:
                y, _ = librosa.load(str(info.path), sr=sr, mono=False, offset=off, duration=75.0)
                y = np.stack([y, y]) if y.ndim == 1 else y[:2]
                if self.playing == tid and g.ui is self:
                    g.sfx.play_music(y, 0.85)
                    self.msg = "previewing"
            except Exception as exc:
                self.msg = f"cannot play ({type(exc).__name__})"
                self.playing = None
            self.loading = False
        threading.Thread(target=work, name="kiosk-preview", daemon=True).start()

    def update(self, dt: float) -> None:
        if self.playing is not None and not self.loading and not self.g.sfx.music_active:
            self.playing, self.msg = None, "ended"
            self.refresh()

    def close(self) -> None:
        g = self.g
        g.sfx.stop_music()
        g.bridge.duck_target = 1.0                     # the orchestra fades back in
        if g.player.seat is self.seat:
            g.player.stand()
        super().close()


class InventoryUI(UiBase):
    """Backpack + Album: consumables, fuel, fruit trophies, cassettes and booth photos."""
    def __init__(self, g) -> None:
        super().__init__(g)
        self.sel, self.big = 0, False
        self.cache: dict = {}
        self.quad((0, 0), (1.45, 0.86), C(0.3, 0.5, 0.8, 0.3), z=0.21)
        self.quad((0, 0), (1.43, 0.84), C(0.02, 0.025, 0.04, 0.92))
        self.text("BACKPACK  &  ALBUM", (-0.69, 0.38), 1.1)
        self.head = self.text("", (-0.69, 0.335), 0.85, C(1.0, 0.85, 0.5, 0.95))
        self.rows = [self.text("", (-0.69, 0.28 - 0.036 * i), 0.8) for i in range(15)]
        self.card = self.quad((0.43, 0.05), (0.36, 0.45), C(1, 1, 1, 1), None, 0.15)
        self.desc = self.text("", (0.25, -0.24), 0.75, C(0.8, 0.85, 1.0, 0.95))
        self.text("W/S select   E preview close-up   Enter/U use or eat   I / Esc close", (-0.69, -0.39), 0.7, C(0.6, 0.7, 0.9, 0.9))
        self.refresh()

    def entries(self) -> list:
        g, inv = self.g, self.g.inv
        out = []
        for key, it in ITEMS.items():
            if inv.n(key) > 0:
                out.append((key, key, f"{it['name']}  x{inv.n(key)}", it["desc"]))
        out += [("fruit", f, f"Fruit trophy: {f}", "Picked from the atrium tree.") for f in inv.fruits]
        out += [("cassette", t, f"Cassette: {g.covers.title(t).replace('_', ' ')[:30]}", "Insert at a Listening Post.") for t in inv.cassettes]
        out += [("photo", p, f"Polaroid #{p % 1000:03d}", "Someone else was in the booth.") for p in inv.photos]
        return out

    def refresh(self) -> None:
        inv, g = self.g.inv, self.g
        es = self.entries()
        self.es = es
        self.sel = max(0, min(self.sel, len(es) - 1))
        self.head.text = f"coins {inv.coins}    food {inv.n('food') + inv.n('snack')}    drinks {sum(inv.n(k) for k in ('water', 'coffee', 'tea', 'fizz', 'energy'))}    fuel {inv.n('fuel')}    fruit {len(inv.fruits)}    tapes {len(inv.cassettes)}    photos {len(inv.photos)}"
        top = max(0, min(self.sel - 7, len(es) - 15))
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= len(es):
                row.text = ""
                continue
            row.text = f"{'>' if j == self.sel else ' '} {es[j][2]}"
            row.color = C(1.0, 0.95, 0.6, 1) if j == self.sel else C(0.8, 0.9, 1.0, 0.95)
        if not es:
            self.rows[0].text = "(empty - the mall provides, eventually)"
            self.card.enabled = False
            self.desc.text = ""
            return
        kind, data, label, desc = es[self.sel]
        self.desc.text = desc
        key = (kind, data)
        if key not in self.cache:
            title = label.split(":", 1)[-1].strip()
            d = FRUIT_COL.get(data, (220, 120, 60)) if kind == "fruit" else data
            if kind in ITEMS:
                d, title = dict(desc=ITEMS[kind]["desc"]), ITEMS[kind]["name"]
            self.cache[key] = tex_from_pil(item_card(kind, d, title))
        self.card.enabled = True
        self.card.texture = self.cache[key]
        self.card.scale = (0.72, 0.9) if self.big else (0.36, 0.45)
        self.card.position = (0, 0.0) if self.big else (0.43, 0.05)
        self.card.z = 0.02 if self.big else 0.15

    def key(self, k: str) -> None:
        g = self.g
        if k in ("i", "escape"):
            if self.big:
                self.big = False
                self.refresh()
            else:
                g.close_ui()
        elif k in ("w", "up arrow"):
            self.sel -= 1
            self.big = False
            g.sfx.play("tick", 0.6)
            self.refresh()
        elif k in ("s", "down arrow"):
            self.sel += 1
            self.big = False
            g.sfx.play("tick", 0.6)
            self.refresh()
        elif k == "e":
            self.big = not self.big
            self.refresh()
        elif k in ("enter", "u") and self.es:
            kind, data = self.es[self.sel][0], self.es[self.sel][1]
            if kind in ITEMS and ITEMS[kind]["kind"] in ("drink", "food"):
                g.inv.add(kind, -1)
                g.vitals.eat(kind)
                g.sfx.play("gulp")
                g.hud.note(f"{ITEMS[kind]['name']}  -  {ITEMS[kind]['desc']}")
            elif kind == "fruit":
                g.inv.fruits.remove(data)
                g.inv.dirty = True
                g.vitals.hunger = min(100, g.vitals.hunger + 14)
                g.vitals.thirst = min(100, g.vitals.thirst + 12)
                g.sfx.play("gulp")
                g.hud.note("You eat the trophy. It was worth it.")
            elif kind == "fuel":
                g.hud.note("Fuel is for go-karts. Stand near one and press R.", good=False)
            self.refresh()


FRUIT_COL = {v[0]: v[1] for v in FRUITS.values()}
FRUITS_BY_NAME = {}


class ShopUI(UiBase):
    def __init__(self, g, npc) -> None:
        super().__init__(g)
        self.npc, self.sel, self.msg = npc, 0, ""
        self.quad((0, 0), (0.9, 0.62), C(0.9, 0.7, 0.3, 0.35), z=0.21)
        self.quad((0, 0), (0.88, 0.60), C(0.04, 0.03, 0.02, 0.93))
        self.text(f"{npc.name.upper()}  -  SUPPLIES", (-0.41, 0.26), 1.1, C(1.0, 0.9, 0.6, 1))
        self.wallet = self.text("", (-0.41, 0.21), 0.9, C(0.8, 1.0, 0.8, 1))
        self.rows = [self.text("", (-0.41, 0.14 - 0.05 * i), 0.95) for i in range(len(SHOP_STOCK))]
        self.text("W/S choose   Enter buy   E / Esc leave", (-0.41, -0.26), 0.7, C(0.8, 0.75, 0.6, 0.9))
        self.refresh()

    def refresh(self) -> None:
        g = self.g
        self.wallet.text = f"you have {g.inv.coins} c     {self.msg}"
        for i, (k, price) in enumerate(SHOP_STOCK):
            it = ITEMS[k]
            self.rows[i].text = f"{'>' if i == self.sel else ' '} {it['name']:<20} {price:>3} c    (have {g.inv.n(k)})"
            self.rows[i].color = C(1.0, 0.95, 0.6, 1) if i == self.sel else C(0.9, 0.85, 0.75, 0.95)

    def key(self, k: str) -> None:
        g = self.g
        if k in ("e", "escape"):
            g.close_ui()
        elif k in ("w", "up arrow"):
            self.sel = (self.sel - 1) % len(SHOP_STOCK)
            self.refresh()
        elif k in ("s", "down arrow"):
            self.sel = (self.sel + 1) % len(SHOP_STOCK)
            self.refresh()
        elif k == "enter":
            key, price = SHOP_STOCK[self.sel]
            if g.inv.coins < price:
                self.msg = "not enough coins"
            else:
                g.inv.coins -= price
                g.inv.add(key)
                g.sfx.play("coin")
                self.msg = f"bought {ITEMS[key]['name']}"
            self.refresh()


class CctvUI(UiBase):
    def __init__(self, g) -> None:
        super().__init__(g)
        self.idx = 0
        self.quad((0, 0), (1.9, 1.2), C(0, 0, 0, 1), z=0.3)
        self.screen = self.quad((0, 0.02), (1.28, 0.96), C(0.8, 1.0, 0.9, 1), None, 0.25)
        self.over = self.quad((0, 0.02), (1.28, 0.96), C(1, 1, 1, 0.22), BANK.get("static"), 0.24)
        self.lab = self.text("", (-0.62, 0.43), 1.0, C(0.7, 1.0, 0.8, 1))
        self.rec = self.text("REC", (0.5, 0.43), 1.0, C(1, 0.2, 0.2, 1))
        self.text("A/D switch camera   E / Esc leave", (-0.62, -0.47), 0.75, C(0.6, 0.8, 0.7, 0.9))
        g.cctv_ui = self
        g.feed_force = 0
        if g.feeds and g.feeds[0].ok:
            set_raw_texture(self.screen, g.feeds[0].tex)
        self.refresh()

    def refresh(self) -> None:
        pts = self.g.cctv_points()
        if pts:
            self.idx %= len(pts)
            self.g.feeds[0].point(*pts[self.idx])
            self.lab.text = pts[self.idx][5]
        else:
            self.lab.text = "NO SIGNAL"

    def key(self, k: str) -> None:
        if k in ("e", "escape"):
            self.g.close_ui()
        elif k in ("d", "right arrow", "a", "left arrow"):
            self.idx += 1 if k in ("d", "right arrow") else -1
            self.g.glitch = 1.0
            self.g.sfx.play("tick")
            self.refresh()

    def update(self, dt: float) -> None:
        g = self.g
        self.over.color = C(1, 1, 1, 0.16 + 0.8 * g.glitch)
        self.over.texture_offset = (random.random(), random.random())
        self.rec.color = C(1, 0.2, 0.2, 1.0 if int(g.clock * 1.5) % 2 == 0 else 0.1)
        t = int(g.clock)
        self.rec.text = f"REC  {t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"

    def close(self) -> None:
        self.g.cctv_ui = None
        super().close()


# ============================================================================
# THE GAME (window, camera, input, per-frame glue)
# ============================================================================
SEASON_LEN = 75.0
SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_SCALE = (0.8, 1.0, 0.9, 0.38)
PHONE_TEXT = ("...the number you have dialed is the number you are standing at...", "CALL BACK: 000-0000. YOU ARE EARLY.",
              "[static]  ...third floor...  [static]  ...don't take the lift...", "THANK YOU FOR HOLDING. YOU HAVE BEEN HELD SINCE 1994.",
              "(a child counting backwards, very calmly, from forty)", "LEAVE A MESSAGE AFTER THE TONE.  ...   (there is no tone)")
WISHES = ("The coin sinks. Something below the water says thank you.", "You wish for a door. The fountain files it under 'later'.",
          "A ripple answers, a beat too late.", "You hear the coin land somewhere much farther down than the pool.")


class Game:
    def __init__(self, args: argparse.Namespace, stack: AudioStack) -> None:
        global FANCY, SHADER, BANK
        _load_ursina()
        self.args, self.stack = args, stack
        FANCY = not args.no_shader
        try:
            self.app = ur.Ursina(title="Ethereal Mall", borderless=False, fullscreen=bool(args.fullscreen),
                                 vsync=True, development_mode=False, editor_ui_enabled=False)
        except TypeError:                                   # older/newer Ursina without those keywords
            self.app = ur.Ursina()
        for attr, val in (("exit_button.visible", False), ("fps_counter.enabled", True)):
            try:
                obj, name = attr.split(".")
                setattr(getattr(ur.window, obj), name, val)
            except Exception:
                pass
        self.clock, self.last = 0.0, time.perf_counter()
        self.look = dict(fog=ZONES["corridor"]["fog"], amb=ZONES["corridor"]["amb"], dens=ZONES["corridor"]["dens"])
        self.pulse = 1.0
        if FANCY:
            SHADER = ur.Shader(name="mall_shader", language=ur.Shader.GLSL, vertex=GLSL_VERT, fragment=GLSL_FRAG)
        # --- v2 state
        self.inv, self.vitals = Inventory(), Vitals()
        self.save = SaveFile(args.seed, not args.no_save)
        self.save.load(self.inv)
        self._v3_init()
        self.grid = PowerGrid(args.seed, not args.no_blackouts)
        self.sky = SkySystem(args.seed)
        self.sfx = SfxBus(stack, args.silent)
        self.ui: Optional[UiBase] = None
        self.elev: Optional[dict] = None
        self.cctv_ui = None
        self.feeds: list = []
        self.feed_t, self.glitch, self.feed_force = 0.0, 0.0, 0
        self.torch_on, self.torch_batt = False, 100.0
        self.karts: dict = {}
        self.kart_t = 0.0
        self.fruit_ready: dict = {}
        self.return_pos = None
        self.flick_k, self.last_buzz, self.last_drip = 1.0, 0.0, 0.0
        self.cull_t, self.rain_t, self.season_t, self.tw_t, self.save_t = 0.0, 0.0, 0.0, 0.0, 0.0
        self.cam = (0.0, 1.6, 0.0)
        self.cam_f = (0.0, 0.0, 1.0)
        self.fov = 78.0
        self.target: Optional[Interactable] = None
        self.errors: dict = {}
        self.cur_kind = "corridor"
        self.playing: dict = {}
        self.playing_t = 0.0
        self._push_look(0.0, 0.0, 0.0, 0.0)
        log.info("Generating procedural textures...")
        BANK = TextureBank(args.seed, args.flip_art)
        for name in ("white", "concrete", "marble", "glass", "metal", "shutter", "poster_blank"):
            BANK.get(name)
        self.covers = CoverArtService(stack.indexer, BANK, args.seed)
        self.ads = AdManager(self.covers, args.seed)
        self.ads.skip = set()
        self.world = World(self)
        self.player = Player(self)
        self.hud = Hud(self)
        self.dust = DustField()
        self.fx = ScreenFX()
        self.bridge = AudioBridge(stack)
        self.torch_ent = ur.Entity(parent=ur.camera, model="cube", position=(0.30, -0.26, 0.65), scale=(0.07, 0.07, 0.30), color=C(0.12, 0.12, 0.14))
        self.torch_lens = ur.Entity(parent=ur.camera, model="sphere", position=(0.30, -0.26, 0.82), scale=0.09, color=C(0.3, 0.3, 0.3))
        self.yaw, self.pitch, self.cam_dist, self.first_person = 0.0, 12.0, 4.2, False
        self.jump = False
        ur.camera.fov = 78
        ur.camera.clip_plane_near, ur.camera.clip_plane_far = 0.08, 260
        ur.mouse.locked = True
        log.info("Building the first chunks...")
        self.world.stream(self.player.z, budget=99)
        gc.collect()                                        # everything built so far lives for the whole session:
        gc.freeze()                                         # keep it out of future collections, and make those rare so
        gc.set_threshold(20000, 25, 100)                    # a full GC pass no longer lands as a multi-frame stall
        ctrl = ur.Entity()
        ctrl.update, ctrl.input = self.update, self.input
        self._ctrl = ctrl

    def run(self) -> None:
        self.app.run()

    def shutdown(self) -> None:
        self.save.save(self.inv, True)
        self.sfx.close()

    # ---- helpers used by chunk code
    def make_npc(self, ch: Chunk, mode: str, arg, rng, role: str = "talk") -> None:
        ch.npcs.append(Npc(self, ch, mode, arg, rng, role))

    def nearby_npcs(self) -> list:
        out: list = []
        for ch in self.world.near_chunks(self.player.z, 1):
            out += ch.npcs
        return out

    def spawn_kart(self, ch: Chunk, x: float, z: float, yaw: float, tag: str) -> None:
        key = f"{self.world.level}:{ch.idx}:{tag}"
        if key not in self.karts:
            k = GoKart(key, x, z, yaw, fuel=float(random.Random(key).choice((0, 12, 25, 40))))
            k.level = self.world.level
            self.karts[key] = k
        k = self.karts[key]
        if k.root is None:
            k.build()

    def on_unload(self, ch: Chunk) -> None:
        if self.player.seat is not None and self.player.seat in ch.inters:
            if isinstance(self.ui, KioskUI):
                self.close_ui()
            else:
                self.player.stand()
        for npc in ch.npcs:
            if self.hud.speaker is npc:
                self.hud.speaker = None
            if isinstance(self.ui, ShopUI) and self.ui.npc is npc:
                self.close_ui()
            REAPER.push(npc.fig.root)
        ch.npcs.clear()
        ch.litter.clear()
        w = self.world                                     # go-karts live in the scene, not in the chunk: free the ones left behind
        for k in self.karts.values():
            if k.root is not None and k is not self.player.kart and getattr(k, "level", "mall") == w.level \
                    and w.chunk_index(k.z) == ch.idx:
                self._drop_kart(k)

    def _drop_kart(self, k) -> None:
        root = k.root
        k.root, k.ents, k.wheels = None, [], []
        REAPER.push(root)                                  # GoKart.build() recreates it when its chunk is loaded again

    def _update_karts(self, dt: float) -> None:
        """Karts follow their chunk: built when it exists, hidden when it is culled, freed when it unloads."""
        self.kart_t -= dt
        if self.kart_t > 0:
            return
        self.kart_t = 0.2
        w, p = self.world, self.player
        for k in self.karts.values():
            if k is p.kart:
                if k.root is None:
                    k.build()
                continue
            ch = w.chunks.get(w.chunk_index(k.z)) if getattr(k, "level", "mall") == w.level else None
            if ch is None:
                if k.root is not None:
                    self._drop_kart(k)
                continue
            if k.root is None:
                k.build()
            vis = bool(ch.shown and ch.detail_shown)
            if k.root.enabled != vis:
                k.root.enabled = vis

    def view_cos(self, x: float, z: float) -> float:
        dx, dz = x - self.cam[0], z - self.cam[2]
        d = math.hypot(dx, dz) + 1e-6
        return (dx * self.cam_f[0] + dz * self.cam_f[2]) / (d * (math.hypot(self.cam_f[0], self.cam_f[2]) + 1e-6))

    # ---- overlays
    def open_ui(self, ui: UiBase) -> None:
        if self.ui is not None:
            self.close_ui()
        self.ui = ui

    def close_ui(self) -> None:
        ui, self.ui = self.ui, None
        if ui is not None:
            ui.close()
        ur.mouse.locked = True

    # ---- input
    def input(self, key: str) -> None:
        if self.elev is not None:
            return
        if self.ui is not None:
            if key.endswith(" up") or key.endswith(" hold") or key in ("left mouse down", "left mouse up", "scroll up", "scroll down"):
                return
            self.ui.key(key)
            return
        p = self.player
        if self._v3_input(key):
            return
        if key == "e":
            self.interact()
        elif key == "space":
            self.jump = True
        elif key == "v":
            self.first_person = not self.first_person
        elif key == "tab":
            ur.mouse.locked = not ur.mouse.locked
        elif key == "f3":
            self.hud.show_dbg = not self.hud.show_dbg
        elif key == "i":
            self.open_ui(InventoryUI(self))
        elif key == "f":
            self.torch_on = not self.torch_on and self.torch_batt > 1
            self.sfx.play("click", 0.7)
        elif key == "r":
            self.refuel()
        elif key == "scroll up":
            self.cam_dist = max(1.6, self.cam_dist - 0.4)
        elif key == "scroll down":
            self.cam_dist = min(8.0, self.cam_dist + 0.4)
        elif key == "escape":
            ur.application.quit()

    def refuel(self) -> None:
        p = self.player
        kart = p.kart or min((k for k in self.karts.values() if getattr(k, "level", "mall") == self.world.level),
                             key=lambda k: math.hypot(k.x - p.x, k.z - p.z), default=None)
        if kart.rec["eco"]:
            self.hud.note("Eco-Karts run on a battery. No fuel needed.", good=False)
            return
        if kart is None or math.hypot(kart.x - p.x, kart.z - p.z) > 4.0:
            return
        if self.inv.n("fuel") <= 0:
            self.hud.note("No fuel canisters. Look along the walls, or ask a vendor.", good=False)
        elif kart.fuel > 75:
            self.hud.note("The tank is nearly full.", good=False)
        else:
            self.inv.add("fuel", -1)
            kart.fuel = min(100.0, kart.fuel + 35.0)
            self.sfx.play("thud", 0.5)
            self.hud.note(f"Refuelled: {kart.fuel:.0f}%")

    # ---- interactions
    def _remove_inter(self, it: Interactable) -> None:
        it.alive, it.occupied = False, True
        ch = self.world.at(it.z)
        if ch is not None and it in ch.inters:
            ch.inters.remove(it)
        ent = it.data.get("ent")
        if ent is not None:
            ur.destroy(ent)

    def interact(self) -> None:
        p = self.player
        if p.kart is not None:
            if p.kart.speed < 9.0:
                p.dismount()
            return
        if p.seat is not None:
            p.stand()
            return
        it = self.target
        if it is None:
            return
        h = getattr(self, "_use_" + it.kind, None)
        if h is not None:
            h(it)

    def _use_sit(self, it) -> None:
        self.player.sit(it)

    def _use_treadmill(self, it) -> None:
        it.data["on"] = not it.data["on"]

    def _use_kiosk(self, it) -> None:
        self.open_ui(KioskUI(self, it))

    def _use_cctv(self, it) -> None:
        if not self.feeds:
            self.feeds = [FeedCam("cctvA"), FeedCam("cctvB")]
        self.open_ui(CctvUI(self))

    def _use_vend(self, it) -> None:
        if it.data["cool"] > 0.0:
            return
        key, price = it.data["stock"]
        if self.inv.coins < price:
            self.hud.note(f"{ITEMS[key]['name']} costs {price} c.  You have {self.inv.coins}.", good=False)
            self.sfx.play("tick", 0.7)
            return
        self.inv.coins -= price
        self.inv.add(key)
        it.data["cool"] = 1.4
        self.sfx.play("coin")
        self.hud.note(f"Bought {ITEMS[key]['name']}  (-{price} c)  -  it's in your bag (I)")
        tx, ty, tz = it.data["tray"]
        ch = self.world.at(tz)
        if ch is not None:
            pivot = ur.Entity(parent=ch.detail, position=(tx - ch.xoff, ty, tz))
            mk(pivot, "cube", (0, 0, 0), (0.06, 0.12, 0.06), "metal", color=C(*[c / 255 for c in ITEMS[key]["col"]]),
               lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            lit = Litter(pivot, "can", tx, tz, ch.hw, ch.xoff)
            lit.y, lit.vx, lit.vz, lit.vy, lit.awake = ty, it.data["dir"] * 1.4, random.uniform(-0.6, 0.6), 1.2, True
            ch.litter.append(lit)

    def _use_payphone(self, it) -> None:
        it.data["kick"] = 1.0
        it.data["n"] += 1
        if it.data["n"] % 2:
            self.sfx.play("phone", 0.9)
            self.hud.note("A muffled loop of static leaks out of the handset...")
        else:
            self.hud.speak(SimpleNamespace(name="payphone", x=it.x, z=it.z), random.choice(PHONE_TEXT))

    def _use_case(self, it) -> None:
        self.sfx.play("tick", 0.8)
        self.hud.note(random.choice(("The case is locked. The padlock is warm.", "LOCKED. The poster inside is slightly different each time.",
                                     "You rattle the case. Something inside rattles back.")), good=False)

    def _use_fountain(self, it) -> None:
        if self.inv.coins < 1:
            self.hud.note("You have no coin to give the fountain.", good=False)
            return
        self.inv.coins -= 1
        self.inv.dirty = True
        self.sfx.play("coin", 0.8)
        self.hud.note(random.choice(WISHES))

    def _use_booth(self, it) -> None:
        d = it.data
        if d["cool"] > 0:
            return
        d["cool"] = 6.0
        self.fx.flash_a = 1.0
        self.sfx.play("flash")
        seed = stable_hash(self.args.seed, "photo", int(self.clock * 10), len(self.inv.photos)) % 1_000_000
        sx, sz = d["slot"]
        ch = self.world.at(sz)
        if ch is None:
            return
        piv = ur.Entity(parent=ch.detail, position=(sx - ch.xoff, 0.02, sz), rotation=(0, random.uniform(0, 360), 0))
        k = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        mk(piv, "cube", (0, 0, 0), (0.26, 0.012, 0.31), color=C(0.95, 0.93, 0.88), **k)
        face = mk(piv, "quad", (0, 0.008, -0.02), (0.2, 0.2, 1), tex_from_pil(anomaly_photo(seed, 128)), rot=(90, 0, 0), double=True, emissive=0.2, **k)
        pk = Interactable("photo", sx, 0.0, sz, 1.4, "Pick up the photograph", 0.0, dict(seed=seed, ent=piv))
        ch.inters.append(pk)
        self.hud.note("*click*  A photograph slides out. It's still warm.")

    def _use_photo(self, it) -> None:
        self.inv.photos.append(it.data["seed"])
        self.inv.dirty = True
        self._remove_inter(it)
        self.sfx.play("pickup")
        self.hud.note("Photograph added to your album (I).  Don't look too long.")

    def _use_cassette(self, it) -> None:
        ids = self.covers.ids()
        tid = ids[stable_hash(self.args.seed, it.data["key"]) % len(ids)]
        self.inv.cassettes.append(tid)
        self.inv.collected.add(it.data["key"])
        self.inv.dirty = True
        self._remove_inter(it)
        self.sfx.play("pickup")
        self.hud.note(f"Cassette: {self.covers.title(tid).replace('_', ' ')[:34]}  -  bring it to a Listening Post")

    def _use_fuel(self, it) -> None:
        self.inv.add("fuel")
        self.inv.collected.add(it.data["key"])
        self._remove_inter(it)
        self.sfx.play("pickup")
        self.hud.note("Fuel canister (+1).  Near a go-kart, press R to refuel.")

    def _use_fruit(self, it) -> None:
        ch = self.world.at(it.z)
        key = (self.world.level, ch.idx if ch else 0, it.data["slot"])
        season = SEASONS[int(self._season_pos()) % 4]
        name = FRUITS.get(season, ("Orchard Fruit", (200, 150, 60)))[0]
        self.inv.fruits.append(name)
        self.inv.dirty = True
        self.fruit_ready[key] = self.clock + 110.0
        self.sfx.play("pickup", 0.8)
        self.hud.note(f"Picked: {name}.  Eat it, or keep it as a trophy (I).")

    def _use_kart(self, it) -> None:
        self.player.mount(it.data["kart"])
        self.hud.note("Go-kart: W/S gas & brake, A/D steer, hold Space to drift, R refuel, E dismount.")

    def _use_elevator(self, it) -> None:
        if self.elev is not None:
            return
        level = self.world.level
        target = "mall" if level != "mall" else ("garage" if stable_hash(self.args.seed, it.data["tag"]) % 2 == 0 else "pool")
        self.elev = dict(t=0.0, target=target, src=(it.x, it.z), switched=False, ding=False, rumble=False)
        if level == "mall":
            self.return_pos = (it.x, it.z)
        self.player.vx = self.player.vz = 0.0
        self.player.x, self.player.z = it.x, it.z

    def _use_npc(self, it) -> None:
        npc = it.data["npc"]
        if npc.role == "shop":
            self.hud.speak(npc, npc.next_line())
            self.open_ui(ShopUI(self, npc))
            return
        self.hud.speak(npc, npc.next_line())
        if self.clock - npc.last_talk > 240.0:
            n = random.randint(1, 3)
            self.inv.coins += n
            self.inv.dirty = True
            npc.last_talk = self.clock
            self.sfx.play("coin", 0.7)
            self.hud.note(f"+{n} c")

    # ---- look / lighting
    def _zone_target(self, kind: str) -> dict:
        z = dict(ZONES[kind])
        if kind in ("skywindow", "atrium"):
            k, wgt = self.sky.amb_k(), (1.0 if kind == "skywindow" else 0.55)
            z["amb"] = tuple(c * (0.45 + 2.6 * k * wgt) for c in z["amb"])
            sc = tuple(min(1.0, 0.18 + 0.9 * c * (0.35 + 0.65 * self.sky.light)) for c in self.sky.col)
            z["fog"] = tuple(f + (s - f) * 0.75 * wgt for f, s in zip(z["fog"], sc))
            z["dens"] = z["dens"] + 0.012 * self.sky.cloud * wgt
        return z

    def _push_look(self, px: float, py: float, pz: float, level: float) -> None:
        fog, amb, dens = self.look["fog"], self.look["amb"], self.look["dens"]
        p = self.grid.level if hasattr(self, "grid") else 1.0
        fl = getattr(self, "flick_k", 1.0)
        fog = tuple(c * (0.05 + 0.95 * p) for c in fog)
        amb = tuple(c * (0.03 + 0.97 * p) for c in amb)
        if FANCY:
            kind = getattr(self, "cur_kind", "corridor")
            pl_k = (0.1 + 0.9 * p) * (0.35 if kind == "anchor" else 1.0)
            ur.scene.set_shader_input("fog_col", ur.Vec3(*fog))
            ur.scene.set_shader_input("fog_dens", float(dens))
            ur.scene.set_shader_input("amb", ur.Vec3(*amb))
            ur.scene.set_shader_input("pl_pos", ur.Vec3(px, py + 1.3, pz))
            ur.scene.set_shader_input("pulse", float(1.0 + 0.10 * level))
            ur.scene.set_shader_input("lights_k", float(p * fl))
            ur.scene.set_shader_input("emis_scale", float(p))
            ur.scene.set_shader_input("pl_k", float(pl_k))
            on = getattr(self, "torch_on", False) and getattr(self, "torch_batt", 0) > 0
            cf = getattr(self, "cam_f", (0, 0, 1))
            cp = getattr(self, "cam", (px, py, pz))
            ur.scene.set_shader_input("fl_k", 1.0 if on else 0.0)
            ur.scene.set_shader_input("fl_pos", ur.Vec3(cp[0], cp[1] - 0.15, cp[2]))
            ur.scene.set_shader_input("fl_dir", ur.Vec3(*cf))
        else:
            ur.scene.fog_color = C(*fog)
            ur.scene.fog_density = float(dens)
        ur.window.color = C(*fog)

    def _ease_look(self, dt: float, kind: str) -> None:
        z = self._zone_target(kind)
        a = 1.0 - math.exp(-dt / 1.2)
        self.look["fog"] = tuple(c + (t - c) * a for c, t in zip(self.look["fog"], z["fog"]))
        self.look["amb"] = tuple(c + (t - c) * a for c, t in zip(self.look["amb"], z["amb"]))
        self.look["dens"] += (z["dens"] - self.look["dens"]) * a
        if kind != self.cur_kind:
            self._on_zone_change(self.cur_kind, kind)
            self.cur_kind = kind
            set_light_mode(self.player.fig.ents, 0.0 if kind in LM0_KINDS else 1.0, z["lamp"])
        set_xc(self.player.fig.ents, self.world.xo_at(self.player.z))

    def _on_zone_change(self, old: str, new: str) -> None:
        if new == "anchor" and self.grid.state == "on" and self.grid.cool <= 0 and random.random() < 0.7:
            self._grid_events(self.grid.trigger(False))
            self.grid.next = max(self.grid.next, 60)

    def _grid_events(self, evs: list) -> None:
        for e in evs:
            self.sfx.play(e, 1.0)

    # ---- camera
    def _solid_at(self, x: float, y: float, z: float, cols: list) -> bool:
        for (x0, x1, z0, z1, top) in cols:
            if x0 < x < x1 and z0 < z < z1 and y < top and top > 1.2:
                return True
        return False

    def _camera(self, dt: float) -> None:
        p = self.player
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        dx, dy, dz = math.sin(yr) * math.cos(pr), -math.sin(pr), math.cos(yr) * math.cos(pr)
        head = 1.62 if p.seat is None else 1.15
        if p.kart is not None:
            head = 1.25
        px, py, pz = p.x, p.y + head, p.z
        p.fig.root.enabled = not self.first_person
        if self.first_person:
            cx, cy, cz = px, py + 0.04, pz
        else:
            d = self.cam_dist + (0.8 if p.kart else 0.0)
            cols = self.world.colliders_near(pz)
            t_ok = 1.0
            for i in range(1, 17):                      # pull the camera in front of walls / pillars
                t = i / 16.0
                if self._solid_at(px - dx * d * t, py - dy * d * t, pz - dz * d * t, cols):
                    t_ok = max(0.12, (i - 1) / 16.0 - 0.05)
                    break
            d *= t_ok
            hw, xo = self.world.hw_at(pz), self.world.xo_at(pz)
            cx = max(xo - hw + 0.25, min(xo + hw - 0.25, px - dx * d))
            cy = max(0.25, py - dy * d)
            cz = pz - dz * d
        if p.kart is not None and p.kart.speed > 8:                     # speed shake
            amp = 0.012 * (p.kart.speed - 8)
            cx, cy = cx + random.uniform(-amp, amp), cy + random.uniform(-amp, amp)
        self.cam, self.cam_f = (cx, cy, cz), (dx, dy, dz)
        ur.camera.position = (cx, cy, cz)
        ur.camera.rotation = (self.pitch, self.yaw, 0)
        want = 78.0 + (min(16.0, p.kart.speed * 0.95) if p.kart else 0.0)
        self.fov += (want - self.fov) * min(1.0, 4 * dt)
        ur.camera.fov = self.fov

    def _cull(self, dt: float) -> None:
        self.cull_t -= dt
        if self.cull_t > 0:
            return
        self.cull_t = 0.1
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        f = self.cam_f
        r = (math.cos(yr), 0.0, -math.sin(yr))
        u = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
        dens = max(0.012, self.look["dens"])
        far = min(260.0, 4.2 / dens + 30.0)
        tan_v = math.tan(math.radians(self.fov / 2))
        show_all = bool(self.cctv_ui or (self.feeds and any(fc.ok and getattr(fc, "on", False) for fc in self.feeds)))
        self.world.cull(self.cam, f, r, u, tan_v, tan_v * ur.window.aspect_ratio, far, min(95.0, 3.4 / dens), show_all)
        ur.camera.clip_plane_far = 260.0 if self.sky.shown else min(260.0, far + 12.0)
        self.ads.skip = self.world.culled_idx()

    # ---- per frame
    def _guard(self, name: str, fn, *a) -> None:
        try:
            fn(*a)
        except Exception as exc:
            n = self.errors.get(name, 0) + 1
            self.errors[name] = n
            if n <= 3:
                log.exception("%s failed (%s: %s)", name, type(exc).__name__, exc)

    def update(self) -> None:
        now = time.perf_counter()
        dt = max(1e-4, min(0.05, now - self.last))
        self.last = now
        self.clock += dt
        hk = ur.held_keys
        free_look = self.ui is None and self.elev is None
        if ur.mouse.locked and free_look:
            self.yaw += ur.mouse.velocity[0] * self.args.sens
            self.pitch -= ur.mouse.velocity[1] * self.args.sens
        if free_look:
            self.yaw += (hk["right arrow"] - hk["left arrow"]) * 110 * dt * (0 if self.player.kart else 1)
        elif self.elev is not None:
            pass
        if self.player.kart is not None and free_look:                    # camera swings in behind the kart
            self.yaw += wrap180(self.player.kart.yaw - self.yaw) * min(1.0, 1.6 * dt)
        self.pitch = max(-55.0, min(78.0, self.pitch))
        p, w = self.player, self.world
        self._guard("sky", self._update_sky, dt)
        self._guard("power", self._update_power, dt)
        self._guard("vitals", self._update_vitals, dt)
        self._guard("world", w.stream, p.z, 1)
        if not w.last_built or len(REAPER.q) > 6:          # free old chunks/figures in frames that did not also build one
            self._guard("reap", REAPER.pump, 0.004)
        self._guard("karts", self._update_karts, dt)
        self._guard("player", p.update, dt, w, self.yaw, self.jump)
        self.jump = False
        ch = w.at(p.z)
        kind = ch.kind if ch else "corridor"
        self._guard("npcs", self._update_npcs, dt)
        self._guard("litter", self._update_litter, dt)
        self._guard("dust", self.dust.update, dt, p.x, p.y, p.z, p.vx, p.vz, w.hw_at(p.z), w.xo_at(p.z))
        self._guard("machines", self._update_machines, dt)
        self._guard("props", self._update_props, dt)
        self._guard("flicker", self._update_flicker, dt)
        self._guard("ambience", self._update_ambience, dt, kind)
        self._guard("trees", self._update_trees, dt)
        self._guard("elevator", self._update_elev, dt)
        if self.clock - self.playing_t > 0.2:
            self.playing, self.playing_t = self.stack.playing(), self.clock
        self._guard("ads", self.ads.update, dt, self.clock, self.playing, p.x, p.z)
        ad, ad_d = self.ads.nearest(p.x, p.y + 1.5, p.z)
        machines = sum(1 for c in w.near_chunks(p.z, 1) for it, _ in c.belts if it.data["on"])
        ex = dict(wet=0.0, tone=0.0)
        if self.sky.shown and kind == "skywindow":
            ex = dict(wet=0.08 * self.sky.rain, tone=-1800.0 * self.sky.rain)
        if self.grid.state in ("off", "pre"):
            ex = dict(wet=ex["wet"] + 0.08, tone=ex["tone"] - 1500.0)
        self.bridge.extra = ex
        self._guard("audio-bridge", self.bridge.update, dt, kind, min(1.0, p.speed / Player.MAX_RUN) if p.kart is None else min(1.0, p.speed / 18.0),
                    p.seat is not None, ad, ad_d, machines)
        self._guard("v3", self._update_v3, dt, kind)
        self._ease_look(dt, kind)
        level = self.stack.level()
        self.pulse += (level - self.pulse) * min(1.0, 6 * dt)
        self._camera(dt)
        self._guard("cull", self._cull, dt)
        self._guard("cctv", self._update_cctv, dt)
        self._push_look(p.x, p.y, p.z, self.pulse)
        self._guard("torch", self._update_torch, dt)
        self._guard("fx", self._update_fx, dt)
        self._guard("hud", self._update_hud, dt, kind, ad, ad_d)
        if self.ui is not None:
            self._guard("ui", self.ui.update, dt)
        self.save_t += dt
        if self.save_t > 20.0:
            self.save_t = 0.0
            self.save.save(self.inv)

    # -- subsystems
    def _update_sky(self, dt: float) -> None:
        sky, p, w = self.sky, self.player, self.world
        sky.tick(dt)
        near = [c for c in w.near_chunks(p.z, 3) if c.kind in ("skywindow", "atrium")]
        sky.show(bool(near))
        if near:
            sw = min(near, key=lambda c: abs(c.idx - w.chunk_index(p.z)))
            sky.rain_on = sw.kind in sky.RAIN_ZONES          # droplets only outside the sky-window glass; atrium: no weather particles
            if sw.kind == "skywindow":
                sky.rain_box = (sw.xoff + sw.hw + 0.8, p.y + 2.0, sw.z0 + CH / 2)
        else:
            sky.rain_on = False
        sky.apply(self.cam[0], self.cam[1], self.cam[2], dt, self.clock)
        while sky.events:
            if sky.events.pop() == "thunder" and sky.shown:
                self.sfx.play("thunder", 1.0, wet=True)
        self.rain_t -= dt
        if sky.shown and sky.rain > 0.2 and self.rain_t <= 0:
            self.rain_t = 2.6
            self.sfx.play("rain", sky.rain, wet=True)

    def _update_power(self, dt: float) -> None:
        self._grid_events(self.grid.update(dt))

    def _update_vitals(self, dt: float) -> None:
        v, p = self.vitals, self.player
        v.update(dt, p.running and p.speed > 1.0)

    def _update_npcs(self, dt: float) -> None:
        for ch in self.world.near_chunks(self.player.z, 2):
            if ch.shown or ch.dist < 20:
                for npc in ch.npcs:
                    npc.update(dt, self.world)

    def _update_litter(self, dt: float) -> None:
        p = self.player
        for ch in self.world.near_chunks(p.z, 1):
            for lit in ch.litter:
                lit.update(dt, p.x, p.z, p.speed, p.vx, p.vz)

    def _update_machines(self, dt: float) -> None:
        for ch in self.world.near_chunks(self.player.z, 1):
            for it, belt in ch.belts:
                if it.data["on"]:
                    it.data["ph"] = (it.data["ph"] + dt * 0.9) % 1.0
                    if FANCY:
                        belt.set_shader_input("uv_off", ur.Vec2(it.data["ph"] * 4, 0.0))
            for it in ch.inters:
                if it.kind == "vend":
                    it.data["cool"] = max(0.0, it.data["cool"] - dt)
                    if FANCY:
                        it.data["face"].set_shader_input("emissive", 1.1 + 2.5 * min(1.0, it.data["cool"]))
                elif it.kind == "booth":
                    it.data["cool"] = max(0.0, it.data["cool"] - dt)
            if ch.orb is not None and FANCY:
                ch.orb.set_shader_input("emissive", 0.5 + 2.2 * self.pulse)
                ch.orb.rotation_y += 30 * dt

    def _update_props(self, dt: float) -> None:
        """Stall doors, lift doors, payphone handsets, drips, cinema stars, static screens."""
        p = self.player
        self.glitch = max(0.0, self.glitch - dt * 2.0)
        near = self.world.near_chunks(p.z, 1)
        for ch in near:
            for d in ch.doors:
                dist = math.hypot(d["x"] - p.x, d["z"] - p.z)
                tgt = 95.0 if dist < 1.9 else float(d["ajar"])
                old = d["ang"]
                d["ang"] += max(-150 * dt, min(150 * dt, tgt - old))
                d["pivot"].rotation_y = d["ang"] + (math.sin(self.clock * 0.7 + d["z"]) * 1.5 if d["ajar"] else 0)
                if abs(d["ang"] - old) > 0.5 and d["cr"] <= 0 and dist < 6:
                    d["cr"] = 2.0
                    self.sfx.play("creak", 0.6, wet=True)
                d["cr"] = max(0.0, d["cr"] - dt)
            for sl in ch.sliders:
                closing = self.elev is not None and math.hypot(sl["x"] - self.elev["src"][0], sl["z"] - self.elev["src"][1]) < 4.0 and self.elev["t"] < 3.0
                want = 0.0 if closing else (1.0 if math.hypot(sl["x"] - p.x, sl["z"] - p.z) < 3.6 else 0.0)
                sl["open"] += max(-dt * 1.3, min(dt * 1.3, want - sl["open"]))
                for ent, zc, zo in sl["panels"]:
                    ent.z = sl["z"] + zc + (zo - zc) * sl["open"]
            for it in ch.swing:
                dd = it.data
                dd["kick"] = max(0.0, dd["kick"] * math.exp(-1.4 * dt) - 0.0)
                dd["pivot"].rotation_z = math.sin(self.clock * 5.5) * dd["kick"] * 22.0 + math.sin(self.clock * 0.8 + it.z) * 1.2
            for dr in ch.drips:
                dr["t"] -= dt
                if dr["t"] <= 0 and dr["fall"] < 0:
                    dr["t"] = random.uniform(1.2, 5.5)
                    dr["fall"] = 0.0
                if dr["fall"] >= 0:
                    dr["fall"] += dt
                    dr["ent"].enabled = True
                    dr["ent"].y = dr["y"] - (dr["y"] - 0.03) * min(1.0, (dr["fall"] / 0.5) ** 2)
                    if dr["fall"] >= 0.5:
                        dr["fall"] = -1.0
                        dr["ent"].enabled = False
                        if math.hypot(dr["x"] - p.x, dr["z"] - p.z) < 25:
                            self.sfx.play("drip", 0.9, pan=max(-1, min(1, (dr["x"] - p.x) / 12.0)), wet=True)
        self.tw_t -= dt
        if self.tw_t <= 0 and FANCY:
            self.tw_t = 0.09
            for ch in near:
                for e, ph in ch.twinkle:
                    e.set_shader_input("emissive", 1.0 + 0.9 * math.sin(self.clock * 3.0 + ph))
                for e in ch.statics:
                    e.set_shader_input("uv_off", ur.Vec2(random.random(), random.random()))

    def _update_flicker(self, dt: float) -> None:
        p = self.player
        mine = self.world.at(p.z)
        total = n = 0
        for ch in self.world.near_chunks(p.z, 2):
            for st in ch.strips:
                ent, base, cur, nxt, until, fk = st
                st[3] -= dt
                if st[3] <= 0:
                    st[4] = self.clock + random.uniform(0.25, 1.3)
                    st[3] = random.uniform(5.0, 24.0) / fk
                    if ch is mine and self.clock - self.last_buzz > 1.2 and math.hypot(ent.x + ch.xoff - p.x, ent.z - p.z) < 16:
                        self.last_buzz = self.clock
                        self.sfx.play("buzz", 0.8, pan=max(-1, min(1, (ent.x + ch.xoff - p.x) / 6.0)), wet=True)
                k = (0.1 if random.random() < 0.5 else 1.0) if self.clock < st[4] else 1.0
                if abs(k - cur) > 0.04:
                    st[2] = k
                    if FANCY:
                        ent.set_shader_input("emissive", base * k)
                if ch is mine:
                    total, n = total + st[2], n + 1
        self.flick_k = 0.6 + 0.4 * (total / n) if n else 1.0

    def _update_ambience(self, dt: float, kind: str) -> None:
        rate = DRIP_RATE.get(kind, 0.0)
        if rate > 0 and random.random() < rate * dt:
            self.sfx.play("drip", random.uniform(0.4, 0.9), pan=random.uniform(-0.8, 0.8), wet=True)

    def _season_pos(self) -> float:
        return (self.clock / SEASON_LEN) % 4.0

    def _update_trees(self, dt: float) -> None:
        self.season_t -= dt
        if self.season_t > 0:
            return
        self.season_t = 0.25
        sp = self._season_pos()
        i, f = int(sp), sp - int(sp)
        b = max(0.0, (f - 0.6) / 0.4)
        b = b * b * (3 - 2 * b)
        c0, c1 = ZoneBuilders.SEASON_COL[i], ZoneBuilders.SEASON_COL[(i + 1) % 4]
        col = tuple(a + (c - a) * b for a, c in zip(c0, c1))
        sc = SEASON_SCALE[i] + (SEASON_SCALE[(i + 1) % 4] - SEASON_SCALE[i]) * b
        season = SEASONS[i]
        fcol = FRUITS.get(season, (None, (200, 150, 60)))[1]
        for ch in self.world.near_chunks(self.player.z, 3):
            t = ch.tree
            if not t or not ch.shown:
                continue
            for e, rad, jit in t["leaves"]:
                e.color = C(min(1, col[0] * jit), min(1, col[1] * jit), min(1, col[2] * jit))
                e.scale = rad * 2 * sc
            for it, fe in t["fruit"]:
                key = (self.world.level, ch.idx, it.data["slot"])
                avail = season != "winter" and self.clock >= self.fruit_ready.get(key, 0.0) and f > 0.12
                it.occupied = not avail
                fe.enabled = avail
                if avail:
                    fe.color = C(*[c / 255 for c in fcol])

    def cctv_points(self) -> list:
        pts, n = [], 0
        pidx = self.world.chunk_index(self.player.z)
        for i in sorted(self.world.chunks):
            c = self.world.chunks[i]
            if abs(i - pidx) <= 4:
                n += 1
                title = ZONES[c.kind]["title"].split("  ")[0][:22]
                pts.append((c.xoff, min(c.h, 6.0) - 0.8, c.z0 + 2.0, 0.0, 14.0, f"CAM {n:02d}  {title}  (+Z)"))
                n += 1
                pts.append((c.xoff, min(c.h, 6.0) - 0.8, c.z0 + CH - 2.0, 180.0, 14.0, f"CAM {n:02d}  {title}  (-Z)"))
        return pts

    def _update_cctv(self, dt: float) -> None:
        near_sec = any(c.kind == "security" for c in self.world.near_chunks(self.player.z, 1))
        active = near_sec or self.cctv_ui is not None
        if not self.feeds and near_sec:
            self.feeds = [FeedCam("cctvA"), FeedCam("cctvB")]
        for k, fc in enumerate(self.feeds):
            was = getattr(fc, "on", False)
            fc.on = active
            if was != active:
                fc.active(active)
        if not active or not self.feeds:
            return
        self.feed_t -= dt
        pts = self.cctv_points()
        if self.feed_t <= 0 and pts:
            self.feed_t = 5.0
            self.glitch = 0.8
            n = int(self.clock / 5.0)
            if self.cctv_ui is None:
                self.feeds[0].point(*pts[n % len(pts)])
            self.feeds[1].point(*pts[(n * 3 + 1) % len(pts)])
        for ch in self.world.near_chunks(self.player.z, 1):
            for m in ch.monitors:
                if not m.get("bound") and self.feeds[m["feed"]].ok:
                    set_raw_texture(m["scr"], self.feeds[m["feed"]].tex)
                    m["bound"] = True
                if FANCY:
                    m["ov"].set_shader_input("emissive", 0.8 + 1.4 * self.glitch)

    def _update_torch(self, dt: float) -> None:
        on = self.torch_on and self.torch_batt > 0
        self.torch_batt = max(0.0, self.torch_batt - 0.25 * dt) if on else min(100.0, self.torch_batt + 0.6 * dt)
        if self.torch_on and self.torch_batt <= 0:
            self.torch_on = False
            self.hud.note("The torch dies. It will recharge, slowly, in your pocket.", good=False)
        self.torch_lens.color = C(1.0, 0.95, 0.7) if on else C(0.25, 0.25, 0.28)
        self.torch_ent.enabled = self.torch_lens.enabled = self.first_person or self.player.kart is None

    def _update_fx(self, dt: float) -> None:
        v, fx = self.vitals, self.fx
        tint = v.tint()
        vig = 0.55 * v.distress() + (0.35 if self.grid.out else 0.0)
        fx.static_a = 0.0
        e = self.elev
        if e is not None:
            fx.static_a = max(0.0, min(1.0, (e["t"] - 1.0) / 1.0)) if e["t"] < 3.4 else max(0.0, 1.0 - (e["t"] - 3.4) / 1.2)
            fx.fade_a = 0.0
        fx.update(dt, tint, vig)

    def _update_elev(self, dt: float) -> None:
        e = self.elev
        if e is None:
            return
        e["t"] += dt
        t, p = e["t"], self.player
        if t > 0.15 and not e["ding"]:
            e["ding"] = True
            self.sfx.play("ding")
        if t > 1.0 and not e["rumble"]:
            e["rumble"] = True
            self.sfx.play("rumble")
            self.sfx.play("static_long", 0.6)
        if t > 2.6 and not e["switched"]:
            e["switched"] = True
            tgt = e["target"]
            self._enter_level(tgt)
            p.y = p.vy = 0.0
            p.vx = p.vz = 0.0
            self.world.stream(p.z, budget=99)
            for k in self.karts.values():
                if k.root is not None:
                    k.root.enabled = getattr(k, "level", "mall") == tgt
            self.sfx.play("thud")
        if t > 4.6:
            self.elev = None

    def _update_hud(self, dt: float, kind: str, ad, ad_d: float) -> None:
        p = self.player
        prompt, best, bd = "", None, 1e9
        if self.ui is not None or self.elev is not None:
            self.target = None
            prompt = ""
        elif p.kart is not None:
            prompt = self._kart_prompt(p.kart)
            self.target = None
        elif p.seat is not None:
            prompt = "[E] Stand up"
            self.target = None
        else:
            for ch in self.world.near_chunks(p.z, 1):
                for it in ch.inters:
                    d = math.hypot(it.x - p.x, it.z - p.z)
                    if not it.occupied and d < it.radius and abs(it.y - p.y) < 1.6 and d < bd:
                        best, bd = it, d
            for k in self.karts.values():
                if getattr(k, "level", "mall") == self.world.level:
                    d = math.hypot(k.x - p.x, k.z - p.z)
                    if d < k.inter.radius and d < bd:
                        best, bd = k.inter, d
            for npc in self.nearby_npcs():                                  # people: only when you face them
                if npc.gone:
                    continue
                d = math.hypot(npc.x - p.x, npc.z - p.z)
                if d < npc.radius and self.view_cos(npc.x, npc.z) > 0.55 and d < bd + 0.6:
                    best, bd = Interactable("npc", npc.x, p.y, npc.z, npc.radius, f"Talk to the {npc.name.lower()}", 0.0, {"npc": npc}), d
            self.target = best
            if best is not None:
                label = best.label
                if best.kind == "treadmill":
                    label = "Stop treadmill" if best.data["on"] else "Start treadmill"
                elif best.kind == "vend":
                    k_, pr = best.data["stock"]
                    label = "Use the vending machine"
                elif best.kind == "elevator":
                    label = "Press the button  -  " + ("ride up to the mall" if self.world.level != "mall" else "ride down")
                elif best.kind == "npc" and best.data["npc"].role == "shop":
                    label = f"Browse the {best.data['npc'].name}'s supplies"
                prompt = f"[E] {label}"
        now = "  |  ".join(self.covers.title(t) for t, _ in sorted(self.playing.items(), key=lambda kv: -kv[1])[:2])
        st = self.world.stats
        dbg = (f"{self.stack.status}   now sounding: {now or '-'}\n{self.bridge.describe()}\n"
               f"chunks {st['shown']}/{st['total']} visible (detail {st['detail']})  ads {len(self.ads.ads)}  npcs {sum(len(c.npcs) for c in self.world.chunks.values())}  "
               f"level {self.world.level}  power {self.grid.state}  sky {self.sky.weather}  "
               f"reap-queue {len(REAPER.q)}  karts {sum(1 for k in self.karts.values() if k.root is not None)}/{len(self.karts)}")
        self.hud.update(dt, ZONES[kind]["title"], prompt, dbg)




# ============================================================================
# v3 EXPANSION  -  biomes, rare rooms, books, vending GUI, badges, backrooms, easter eggs
# (everything below is bolted onto the classes above; see _v3_install at the end of the block)
# ============================================================================
import json as _json

BIOMES = ("garage", "pool", "office")                  # infinite elevator biomes (branching junction chunks)
FIXED_LEVELS = {"backrooms": 8, "drain": 1}            # hand-made levels: number of chunks (idx 0..n-1)
JUNCTION_HW = 17.0                                      # width of a branching junction chunk
DEEP_IDX = 4                                            # the swimmable pool chunk in the poolrooms
BACKROOMS_ODDS = 3                                  # 1/100 per elevator ride
LEVEL_SPAWN = {"backrooms": (0.0, 4.0), "drain": (0.0, 3.0)}
V3_DIR = Path.home() / ".ethereal_mall"

ZONES.update({
    "office":    dict(title="LEVEL OFFICE  -  THE LONG QUARTER", fog=(0.78, 0.74, 0.55), dens=0.036, amb=(0.12, 0.11, 0.07),
                      lamp=(1.00, 0.95, 0.70, 1.25)),
    "server":    dict(title="DOCUMENTS & SERVER ROOM", fog=(0.62, 0.78, 0.92), dens=0.052, amb=(0.07, 0.09, 0.13),
                      lamp=(0.70, 0.88, 1.00, 1.15)),
    "lookout":   dict(title="THE LOOKOUT  -  WINDOW GALLERY", fog=(0.66, 0.70, 0.82), dens=0.030, amb=(0.09, 0.10, 0.14),
                      lamp=(0.90, 0.92, 1.00, 1.05)),
    "backrooms": dict(title="THE BACKROOMS", fog=(0.72, 0.66, 0.30), dens=0.060, amb=(0.20, 0.18, 0.07),
                      lamp=(1.00, 0.92, 0.45, 1.30)),
    "drain":     dict(title="BELOW THE DRAIN", fog=(0.05, 0.12, 0.12), dens=0.110, amb=(0.010, 0.020, 0.022),
                      lamp=(0.35, 0.80, 0.75, 0.55)),
})
AUDIO_PRESETS.update({
    "office":    dict(wet=0.62, tone_hz=5200.0, gain=0.92, stretch=1.40, smear_prob=0.40, warmth_hz=6000.0, lanes=2),
    "server":    dict(wet=0.30, tone_hz=9000.0, gain=0.90, stretch=1.00, smear_prob=0.10, warmth_hz=9000.0, lanes=3),
    "lookout":   dict(wet=0.50, tone_hz=8000.0, gain=0.95, stretch=1.30, smear_prob=0.30, warmth_hz=8000.0, lanes=3),
    "backrooms": dict(wet=0.86, tone_hz=2800.0, gain=0.90, stretch=2.00, smear_prob=0.80, warmth_hz=3200.0, lanes=1),
    "drain":     dict(wet=0.97, tone_hz=1800.0, gain=0.85, stretch=2.40, smear_prob=0.92, warmth_hz=2200.0, lanes=1),
})
REVERB_OF.update({"office": "hall", "backrooms": "garage", "drain": "garage", "server": "tile"})
DRIP_RATE.update({"drain": 0.9, "backrooms": 0.04})
KIND_HW.update(office=12.0, server=9.0, lookout=7.0, backrooms=12.0, drain=7.0)
KIND_H.update(office=3.3, server=3.6, lookout=6.0, backrooms=3.4, drain=3.2)
LM0_KINDS = LM0_KINDS + ("drain",)
MINOR_ZONES = (("restroom", 1.0), ("security", 0.8), ("lookout", 0.55), ("server", 0.35), (None, 1.6))
PALETTES.update({
    "office": dict(coat=(0.55, 0.52, 0.40), legs=(0.30, 0.28, 0.26), skin=(0.88, 0.76, 0.66), em=0.05),
    "backrooms": dict(coat=(0.50, 0.48, 0.30), legs=(0.26, 0.25, 0.20), skin=(0.80, 0.74, 0.60), em=0.05),
    "lookout": dict(coat=(0.40, 0.46, 0.62), legs=(0.20, 0.22, 0.30), skin=(0.86, 0.78, 0.70), em=0.08),
    "server": dict(coat=(0.80, 0.82, 0.86), legs=(0.25, 0.26, 0.30), skin=(0.88, 0.80, 0.74), em=0.06),
    "gold": dict(coat=(0.95, 0.78, 0.25), legs=(0.30, 0.22, 0.10), skin=(0.92, 0.82, 0.70), em=0.35),
    "nomad": dict(coat=(0.34, 0.30, 0.22), legs=(0.18, 0.17, 0.15), skin=(0.78, 0.64, 0.55), em=0.0),
    "craft": dict(coat=(0.55, 0.30, 0.12), legs=(0.20, 0.18, 0.16), skin=(0.84, 0.70, 0.60), em=0.05),
    "servant": dict(coat=(0.94, 0.94, 0.92), legs=(0.08, 0.08, 0.10), skin=(0.88, 0.76, 0.68), em=0.10),
    "trapped": dict(coat=(0.62, 0.60, 0.46), legs=(0.30, 0.28, 0.24), skin=(0.82, 0.74, 0.62), em=0.02),
})

# ---- new items --------------------------------------------------------------------------------
ITEMS.update({
    "cola":     dict(name="Retro Cola", kind="drink", thirst=24, hunger=2, fx="fizz", desc="Fizz with a flavour from a decade that never was.", col=(200, 40, 50)),
    "donut":    dict(name="Glazed Donut", kind="food", thirst=-3, hunger=30, fx=None, desc="Warm. Impossibly warm. Nobody is baking here.", col=(235, 175, 120), hold=True),
    "plush":    dict(name="Collectible Plushie", kind="keepsake", thirst=0, hunger=0, fx=None, desc="Soft. Huggable. Hold it (H), then hug it (G).", col=(190, 130, 90), hold=True),
    "trophy":   dict(name="The Secret Trophy", kind="keepsake", thirst=0, hunger=0, fx=None, desc="Engraved: FOR SWIMMING TOWARDS THE WRONG THING.", col=(240, 200, 60), hold=True),
    "phone":    dict(name="Mini-Map Phone", kind="tool", thirst=0, hunger=0, fx=None, desc="Press P: map, order app, pocket garage (store karts) and music. M plays downloads, N skips.", col=(70, 90, 120)),
    "popcorn":  dict(name="Cinema Popcorn", kind="food", thirst=-5, hunger=26, fx=None, desc="Buttered for a film that hasn't finished.", col=(250, 230, 150), hold=True),
    "soda":     dict(name="Cinema Soda", kind="drink", thirst=30, hunger=0, fx=None, desc="Mostly ice. Ice from a lake of ads.", col=(80, 150, 230), hold=True),
    "candy":    dict(name="Cinema Candy", kind="food", thirst=-4, hunger=14, fx="fizz", desc="Chewy, neon, vaguely apologetic.", col=(240, 90, 170), hold=True),
    "ticket":   dict(name="Theater 7 Ticket", kind="misc", thirst=0, hunger=0, fx=None, desc="ADMIT ONE. SEAT: WHEREVER YOU ARE.", col=(240, 210, 90)),
    "glasses":  dict(name="3D Glasses", kind="misc", thirst=0, hunger=0, fx=None, desc="Everything is still flat. But now on purpose.", col=(60, 200, 220)),
    "pasta":    dict(name="Ristorante Pasta", kind="food", thirst=-2, hunger=48, fx=None, desc="Al dente. Al absent.", col=(235, 195, 110), hold=True),
    "soup":     dict(name="Soup of the Hour", kind="food", thirst=6, hunger=34, fx="calm", desc="Tomato, and a little bit of closing time.", col=(215, 80, 50), hold=True),
    "steak":    dict(name="House Steak", kind="food", thirst=-5, hunger=62, fx="caffeine", desc="Cooked by someone who isn't here.", col=(140, 70, 50), hold=True),
    "tiramisu": dict(name="Tiramisu", kind="food", thirst=-3, hunger=28, fx="fizz", desc="Pick me up. It means it literally.", col=(190, 150, 110), hold=True),
    "lemonade": dict(name="Cold Lemonade", kind="drink", thirst=36, hunger=1, fx=None, desc="Sour like a Sunday evening.", col=(245, 230, 90), hold=True),
    "espresso": dict(name="Double Espresso", kind="drink", thirst=4, hunger=0, fx="caffeine", desc="One sip, three o'clock forever.", col=(90, 55, 35), hold=True),
    "wine":     dict(name="House Red (Unlabelled)", kind="drink", thirst=12, hunger=2, fx="calm", desc="Vintage: the year the lights came on.", col=(130, 30, 60), hold=True),
})
for _k, _v in ITEMS.items():
    if _v["kind"] in ("drink", "food"):
        _v.setdefault("hold", True)
VEND_ALL = (("water", 2), ("fizz", 3), ("cola", 3), ("tea", 3), ("coffee", 4), ("energy", 5))
CINEMA_STOCK = (("popcorn", 4), ("soda", 3), ("candy", 2), ("ticket", 5), ("glasses", 3))
CAFE_STOCK = (("water", 2), ("donut", 3), ("plush", 30), ("phone", 35))
MENU_STOCK = (("pasta", 6), ("soup", 4), ("steak", 11), ("tiramisu", 5), ("lemonade", 3), ("espresso", 3), ("wine", 7))
DRINK_KEYS = ("water", "fizz", "energy", "coffee", "tea", "cola", "soda", "lemonade", "espresso", "wine")
KART_COLORS = {"Crimson": (0.80, 0.12, 0.14), "Cobalt": (0.12, 0.28, 0.80), "Mint": (0.15, 0.75, 0.50),
               "Sunburst": (0.95, 0.72, 0.12), "Violet": (0.50, 0.20, 0.80), "Bone": (0.86, 0.84, 0.78),
               "Midnight": (0.06, 0.07, 0.12), "Hot Pink": (0.98, 0.30, 0.65)}

BADGES = {
    "descent": ("First Descent", "Rode an elevator out of the mall."),
    "wrongdoor": ("Wrong Door", "Stepped into an elevator that never came back."),
    "escaped": ("Back Out", "Found the exit of the Backrooms."),
    "nomad": ("Free Will", "Heard the nomad's last word."),
    "swim": ("Deep End", "Swam in the Poolrooms' deep pool."),
    "drain": ("Down the Drain", "Followed the water to its secret."),
    "rat": ("Rat Whisperer", "A rat told you something about the walls."),
    "reader": ("Bookworm", "Read three books in the archive."),
    "author": ("Marginalia", "Wrote something into a book."),
    "why": ("Archivist of Why", "Read every lore book there is."),
    "cold": ("Achoo!", "Stayed too long in the freezing server room."),
    "wish": ("Wishful", "Tossed a coin in the fountain."),
    "witness": ("Wishing Witness", "Watched a stranger toss coins at the fountain."),
    "archive": ("Lost & Found", "Examined a Lost & Found log."),
    "thirst": ("Thirsty Work", "Bought five drinks from vending machines."),
    "hug": ("Hugger", "Hugged something soft."),
    "plush": ("Soft Things", "Owned a Collectible Plushie."),
    "matinee": ("Matinee", "Bought everything from the cinema concession."),
    "paint": ("Fresh Paint", "Had your kart repainted by the craftsman."),
    "iron": ("Iron Arms", "Lifted 30 reps in the gym."),
    "lungs": ("Iron Lungs", "Ran on the treadmill for a full minute."),
    "dine": ("Table For One", "Was served a meal by the Ristorante waiter."),
    "window": ("Window Seat", "Listened to someone describe the scenery."),
    "rare": ("Rare Encounter", "Met someone who is almost never there."),
    "listener": ("Good Listener", "Heard every trapped soul in the Backrooms."),
    "phone": ("Reception", "Owned a Mini-Map Phone."),
    "trophy": ("Secret Trophy", "Brought the trophy up from below."),
}
BOOK_LORE = [
    ("PROSPECTUS, DRAFT 0", "This mall was never opened. It was imagined by a committee as the last place a person would need to go: "
     "everything in one building, so nobody would have to leave. Then the committee left. The building stayed, and kept imagining."),
    ("WHY THE CORRIDOR", "The Path is long because waiting is long. Every place people were told to wait - lobbies, platforms, "
     "ward corridors, car parks - sends a little of that waiting here. The mall is the sum of those minutes."),
    ("ON THE ELEVATORS", "The freight lifts are not built to take you anywhere. They only take you to where you were already waiting "
     "somewhere else: the garage, the pool, the long quarter. Once in a hundred rides, the lift takes you to where waiting hides."),
    ("THE ORCHESTRA (STAFF MEMO)", "The music is not recorded. Something in the walls remembers songs you once loved and plays them back "
     "from far away, slowed down so it can understand them. When you stand still, it dissolves. When you walk, it clears."),
    ("THE FIRST VISITORS", "Some came by accident. Some came on purpose, to be nowhere for a while. Those ones cannot leave: "
     "the place keeps what chooses it. Be careful what you choose. Be kind to the ones who chose already."),
    ("ON THE BACKROOMS", "There is a room behind the rooms. It has one door, which closes behind you. It is wallpaper, humming, "
     "and a small cafe that should not be open. If you must go, buy water and follow the cold air to the lift."),
    ("DRAIN INSPECTION LOG", "Pool 4 drains downward. The inspector wrote: 'It keeps draining. The level doesn't change.' "
     "Below is the only room in the building that was built on purpose. Please do not bring anything back up. - Management"),
]
RAT_FIRST = "psst. you. the walls lead to another room. don't tell the others."
RAT_LINES = ["Squeak. Rule one of rats: cheese is a metaphor.", "I've read every book here. They are mostly about me.",
             "I chewed a cable once. The lights turned the colour of Tuesday.", "Don't pet me. Do not look at me like that. Okay, once.",
             "I know a hole behind every hole. Ask nicely.", "The AC is a god. A very cold god. I pay tribute in crumbs.",
             "Paper tastes like unsent letters. Delicious.", "Squeak? That was a question. The answer is also squeak."]
NOMAD_LINES = ["Easy. Sit. I'm not asking for money. I came here by my own will. Nobody pushed me. Free will, you hear?",
               "I can't go back. Every door out opens onto more of this. I checked. I checked a lot.",
               "Do you hear it? The lights are humming a name. It's... mine."]
RARE_LINES = ["You found me. Hardly anyone does. Here - take this, it's yours anyway.",
              "I'm only on this corridor on the third Thursday of a month that doesn't exist. Lucky you.",
              "A secret: the fountain listens, the vending machines forgive, and the rat knows the way out."]
LOOKOUT_LINES = {
    "rain": ["It's raining out there. Only out there. Isn't that thoughtful?", "Rain that never reaches the glass. I've watched it for hours."],
    "fog": ["The fog comes in like it's reading the building. Slow, careful, polite.", "Every time the fog rolls past I think I see a skyline. Then it moves."],
    "dust": ["That's cosmic dust. I decided. It's prettier than the alternative.", "It looks like somebody shaking out a rug the size of a galaxy."],
}
TRAPPED = [
    ("TRAPPED - Marlowe", ["I only ever wanted a quiet place to read. Careful what you ask the walls for.", "The wallpaper is the book now. I'm on page nine hundred.", "Here. Take this. I found it under the carpet."]),
    ("TRAPPED - Ines", ["I took the lift on a dare. A hundred to one, they said. I'm the one.", "The cafe keeps me alive. The cafe also keeps me here.", "If you find the exit, tell the lift I said hello."]),
    ("TRAPPED - the Janitor", ["I clean the same hallway. It's never dirty. It's never clean.", "I stay because somebody has to turn the lights on in the next room.", "Take a coin. I can't spend it anywhere but here."]),
    ("TRAPPED - Odile", ["I'm not stuck. I'm staying. There's a difference and I'm very firm about it.", "Out there everyone is waiting for something. In here, nothing is waiting for me.", "Go. Be brave. Come back and tell me if the weather's real."]),
]


def _v3_synth(name: str, sr: int, seed: int = 0):
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    t = lambda d: np.arange(int(d * sr)) / sr
    if name == "sneeze":
        x = t(0.9)
        env = np.where(x < 0.45, (x / 0.45) ** 2 * 0.15, np.exp(-(x - 0.45) * 9.0))
        n = _lp(rng.standard_normal(x.size), 5200, sr, 2, "band" if False else "low")
        m = n * env * 1.4 + 0.25 * np.sin(2 * np.pi * (520 - 160 * np.clip(x - 0.45, 0, 1)) * x) * np.exp(-np.clip(x - 0.45, 0, 9) * 12) * (x > 0.45)
        return _st(m * 0.7)
    if name == "whisper":
        x = t(1.6)
        n = _lp(rng.standard_normal(x.size), 7000, sr, 2, "high") * _lp(rng.standard_normal(x.size), 6, sr, 1)
        env = np.minimum(1, np.minimum(x / 0.3, (1.6 - x) / 0.5)) * (0.55 + 0.45 * np.sin(2 * np.pi * 3.3 * x) ** 2)
        return _st(n * env * 3.0)
    if name == "splash":
        x = t(0.55)
        n = _lp(rng.standard_normal(x.size), 3800, sr, 2) * np.exp(-x * 9) * (1 - np.exp(-x * 400))
        n += 0.35 * np.sin(2 * np.pi * (500 + 900 * np.exp(-x * 20)) * x) * np.exp(-x * 24)
        return _st(n * 0.8)
    if name == "scream":
        x = t(1.7)
        f = 620 + 480 * np.sin(2 * np.pi * 0.9 * x) + 260 * np.sin(2 * np.pi * 6.0 * x)
        m = sum(np.sin(np.cumsum(2 * np.pi * f * k / sr)) / k for k in range(1, 6))
        m = m * 0.25 + 0.4 * _lp(rng.standard_normal(x.size), 3200, sr, 2)
        return _st(m * np.minimum(1, np.minimum(x / 0.05, (1.7 - x) / 0.4)) * 0.8)
    if name == "hug":
        x = t(0.7)
        m = _lp(rng.standard_normal(x.size), 600, sr, 2) * np.sin(np.pi * np.clip(x / 0.7, 0, 1)) ** 2
        m += 0.12 * np.sin(2 * np.pi * 220 * x) * np.exp(-x * 6)
        return _st(m * 0.9)
    if name == "squeak":
        x = t(0.28)
        f = 2300 + 1500 * np.sin(2 * np.pi * 9 * x)
        return _st(np.sin(np.cumsum(2 * np.pi * f / sr)) * np.exp(-x * 11) * (1 - np.exp(-x * 600)) * 0.45)
    if name == "clink":
        x = t(0.5)
        m = (np.sin(2 * np.pi * 3100 * x) + 0.6 * np.sin(2 * np.pi * 4700 * x)) * np.exp(-x * 14)
        return _st(m * 0.35)
    if name == "weights":
        x = t(0.6)
        m = _lp(rng.standard_normal(x.size), 900, sr, 2) * np.exp(-x * 14) + 0.5 * np.sin(2 * np.pi * 70 * x) * np.exp(-x * 9)
        return _st(m * 0.8)
    if name == "chime":
        x = t(1.4)
        m = sum(np.sin(2 * np.pi * f * x) * np.exp(-x * (2 + i)) for i, f in enumerate((880, 1318, 1760)))
        return _st(m * 0.22)
    return None


_synth_sfx_v2 = synth_sfx


def synth_sfx(name: str, sr: int, seed: int = 0) -> np.ndarray:      # noqa: F811  (extends the v2 synth)
    a = _v3_synth(name, sr, seed)
    return a if a is not None else _synth_sfx_v2(name, sr, seed)


# ---- tiny text helper -----------------------------------------------------------------------------
def wrap_text(text: str, width: int) -> list:
    out, line = [], ""
    for word in text.split(" "):
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    out.append(line)
    return out


# ---- the Backrooms: a fixed, hand-authored (non-random) maze -------------------------------------------
BR_COLS, BR_ROWS_PER_CHUNK = 12, 12                      # 2 m cells -> 24 x 24 m per chunk
BR_ROWS = BR_ROWS_PER_CHUNK * FIXED_LEVELS["backrooms"]
BR_ROOMS = {"start": (0, 3, 3, 8), "cafe": (46, 53, 2, 9), "exit": (88, 95, 6, 11)}      # r0, r1, c0, c1 inclusive


def build_backrooms_maze() -> tuple:
    """Deterministic maze (fixed constants, never reseeded by the world seed): returns (east, north, dead_ends).
    east[r][c]  : wall between (r,c) and (r,c+1);  north[r][c] : wall between (r,c) and (r+1,c)."""
    rng = random.Random(90210)
    east = [[True] * BR_COLS for _ in range(BR_ROWS)]
    north = [[True] * BR_COLS for _ in range(BR_ROWS)]
    seen = [[False] * BR_COLS for _ in range(BR_ROWS)]
    stack = [(0, BR_COLS // 2)]
    seen[0][BR_COLS // 2] = True
    while stack:
        r, c = stack[-1]
        opts = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < BR_ROWS and 0 <= c + dc < BR_COLS and not seen[r + dr][c + dc]]
        if not opts:
            stack.pop()
            continue
        nr, nc = rng.choice(opts)
        if nr == r + 1:
            north[r][c] = False
        elif nr == r - 1:
            north[nr][nc] = False
        elif nc == c + 1:
            east[r][c] = False
        else:
            east[r][nc] = False
        seen[nr][nc] = True
        stack.append((nr, nc))
    for (r0, r1, c0, c1) in BR_ROOMS.values():          # carve the rooms open (only ever removes walls)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if c < c1:
                    east[r][c] = False
                if r < r1:
                    north[r][c] = False

    def open_n(r, c):
        return (r < BR_ROWS - 1 and not north[r][c]) + (r > 0 and not north[r - 1][c]) + \
               (c < BR_COLS - 1 and not east[r][c]) + (c > 0 and not east[r][c - 1])
    dead = [(r, c) for r in range(8, BR_ROWS - 8) for c in range(BR_COLS) if open_n(r, c) == 1
            and not any(a <= r <= b and x <= c <= y for (a, b, x, y) in BR_ROOMS.values())]
    return east, north, dead


BR_EAST, BR_NORTH, BR_DEAD = build_backrooms_maze()


def pick_spread(cells: list, n: int, min_gap: int = 12) -> list:
    out = []
    for c in cells:
        if all(abs(c[0] - o[0]) >= min_gap for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


BR_TRAPPED_CELLS = pick_spread(BR_DEAD, 4, 14)
BR_COIN_CELLS = pick_spread(BR_DEAD[::-1], 6, 9)


def br_cell_xz(r: int, c: int) -> tuple:
    return -12.0 + 2.0 * c + 1.0, 2.0 * r + 1.0             # local x (chunk root = 0) , world z (chunk idx*24 folded in by r)


def br_solve(start=(1, BR_COLS // 2), goal=(BR_ROWS - 3, 8)) -> Optional[list]:
    """BFS through the fixed maze (used by the phone radar and by the self-check)."""
    from collections import deque as _dq
    prev, q = {start: None}, _dq([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        r, c = cur
        nb = []
        if r < BR_ROWS - 1 and not BR_NORTH[r][c]:
            nb.append((r + 1, c))
        if r > 0 and not BR_NORTH[r - 1][c]:
            nb.append((r - 1, c))
        if c < BR_COLS - 1 and not BR_EAST[r][c]:
            nb.append((r, c + 1))
        if c > 0 and not BR_EAST[r][c - 1]:
            nb.append((r, c - 1))
        for n in nb:
            if n not in prev:
                prev[n] = cur
                q.append(n)
    return None


# ---- persistent state: badges / flags ride along inside the existing save file ----------------------------
_inv_init_v2 = Inventory.__init__
_inv_to_v2, _inv_from_v2 = Inventory.to_json, Inventory.from_json


def _inv_init_v3(self) -> None:
    _inv_init_v2(self)
    self.badges: set = set()
    self.flags: dict = {}
    self.downloads: list = []                       # track ids downloaded to the phone


def _inv_to_v3(self) -> dict:
    d = _inv_to_v2(self)
    d["badges"], d["flags"] = sorted(self.badges), self.flags
    d["downloads"] = [int(t) for t in self.downloads]
    return d


def _inv_from_v3(self, d: dict) -> None:
    _inv_from_v2(self, d)
    self.badges = {str(b) for b in d.get("badges", [])}
    fl = d.get("flags", {})
    self.flags = fl if isinstance(fl, dict) else {}
    self.downloads = [int(t) for t in d.get("downloads", []) if isinstance(t, (int, float))]


class BookStore:
    """Notes written into books, kept in a plain JSON cache that you can delete any time:  ~/.ethereal_mall/book_cache.json"""
    def __init__(self) -> None:
        self.path = V3_DIR / "book_cache.json"
        self.data: dict = {}
        try:
            if self.path.is_file():
                d = _json.loads(self.path.read_text("utf-8"))
                self.data = {str(k): str(v) for k, v in d.items()}
        except Exception as exc:
            log.warning("book cache unreadable (%s)", exc)

    def get(self, bid: str) -> str:
        return self.data.get(bid, "")

    def put(self, bid: str, text: str) -> None:
        self.data[bid] = text
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(self.data, indent=1), "utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            log.warning("could not write book cache (%s)", exc)


def _artist_of(path: Path) -> str:
    try:
        from mutagen import File as _MF
        f = _MF(str(path), easy=True)
        if f is not None and f.get("artist"):
            return str(f["artist"][0])
    except Exception:
        pass
    stem = path.stem.replace("_", " ")
    return stem.split(" - ")[0].strip() if " - " in stem else "unknown artist"


def find_videos(args) -> list:
    exts = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")
    roots = []
    if getattr(args, "video_dir", ""):
        roots.append(Path(args.video_dir).expanduser())
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    roots.append(Path.cwd())
    out, seen = [], set()
    for r in roots:
        try:
            for p in sorted(r.iterdir()):
                if p.is_file() and p.suffix.lower() in exts and p not in seen:
                    seen.add(p)
                    out.append(p)
        except Exception:
            continue
    return out


class VideoScreen:
    """Smooth playback onto a wall: a worker thread decodes (OpenCV) ahead of time into a small queue and the
    render thread only uploads the frame whose time has come, so the game loop never waits on the decoder."""
    def __init__(self, path: Path, max_w: int = 960) -> None:
        self.path, self.max_w = path, max_w
        self.ok, self.tex, self.frames = False, None, deque(maxlen=10)
        self.stop, self.t0, self.fps, self.size = False, None, 24.0, (2, 2)
        self.last_up = -1.0
        try:
            import cv2
            self.cv2 = cv2
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError("cannot open")
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            if not (1.0 < self.fps < 120.0):
                self.fps = 24.0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            sc = min(1.0, self.max_w / max(1, w))
            self.size = (max(2, int(w * sc)), max(2, int(h * sc)))
            self.cap = cap
            from panda3d.core import Texture
            self.tex = Texture("cinema")
            self.tex.setup2dTexture(self.size[0], self.size[1], Texture.T_unsigned_byte, Texture.F_rgb)
            self.tex.setRamImage(bytes(self.size[0] * self.size[1] * 3))
            self.ok = True
            threading.Thread(target=self._decode, name="video-decode", daemon=True).start()
        except Exception as exc:
            log.warning("Cinema video unavailable (%s): install opencv-python for playback", exc)

    def _decode(self) -> None:
        cv2, n, spf = self.cv2, 0, 1.0 / self.fps
        while not self.stop:
            if len(self.frames) >= self.frames.maxlen:
                time.sleep(0.01)
                continue
            ok, fr = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                n = 0
                self.t0 = None
                continue
            if fr.shape[1] != self.size[0]:
                fr = cv2.resize(fr, self.size, interpolation=cv2.INTER_AREA)
            self.frames.append((n * spf, np.ascontiguousarray(fr[::-1]).tobytes()))
            n += 1

    def update(self, now: float) -> None:
        if not self.ok or not self.frames:
            return
        if self.t0 is None:
            self.t0 = now - self.frames[0][0]
        vt, pick = now - self.t0, None
        while self.frames and self.frames[0][0] <= vt:
            pick = self.frames.popleft()
        if pick is None and self.frames and vt < self.frames[0][0] - 2.0:       # loop restarted
            self.t0 = now - self.frames[0][0]
        if pick is not None:
            try:
                self.tex.setRamImageAs(pick[1], "BGR")
            except Exception:
                pass

    def close(self) -> None:
        self.stop = True
        try:
            self.cap.release()
        except Exception:
            pass


# ---- people -----------------------------------------------------------------------------------------
class Actor(Npc):
    """A person with hand-written behaviour.  Placed through Npc's 'stand' mode, then re-skinned."""
    PAL = "corridor"

    def __init__(self, g, ch, pos, rng, name: str, role: str = "talk", pal: Optional[str] = None) -> None:
        Npc.__init__(self, g, ch, "stand", pos, rng, role)
        self.mode, self.name = "actor", name
        p = PALETTES[pal or self.PAL]
        ur.destroy(self.fig.root)
        self.fig = Figure(p, ch.lm, ch.lamp, p["em"], xc=ch.xoff)
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.t, self.given, self.talks = 0.0, False, 0

    def dist(self) -> float:
        return math.hypot(self.g.player.x - self.x, self.g.player.z - self.z)

    def look_at_player(self, dt: float, rate: float = 3.0, within: float = 6.0, home: Optional[float] = None) -> None:
        g = self.g
        if self.dist() < within:
            self.face(math.degrees(math.atan2(g.player.x - self.x, g.player.z - self.z)), dt, rate)
        elif home is not None:
            self.face(home, dt, 1.0)

    def update(self, dt: float, w) -> None:
        if self.gone:
            return
        self.look_at_player(dt)
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.pose(dt, 0.0)

    def on_talk(self) -> None:
        self.talks += 1
        self.g.hud.speak(self, self.next_line())

    def walk_to(self, dt: float, tx: float, tz: float, speed: float = 1.7) -> bool:
        ex, ez = tx - self.x, tz - self.z
        d = math.hypot(ex, ez)
        if d < 0.08:
            return True
        self.face(math.degrees(math.atan2(ex, ez)), dt, 7.0)
        st = min(d, speed * dt)
        self.x, self.z = self.x + ex / d * st, self.z + ez / d * st
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.pose(dt, speed * 1.4)
        return False


class RareNpc(Actor):
    PAL = "gold"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "rare visitor", "talk")

    def next_line(self) -> str:
        return RARE_LINES[min(self.talks - 1, len(RARE_LINES) - 1)] if self.talks <= len(RARE_LINES) else self.rng.choice(RARE_LINES)

    def on_talk(self) -> None:
        super().on_talk()
        g = self.g
        if not self.given:
            self.given = True
            n = self.rng.randint(12, 28)
            g.inv.coins += n
            g.inv.dirty = True
            g.sfx.play("coin")
            g.hud.note(f"+{n} c  -  a rare gift")
            g.award("rare")


class NomadNpc(Actor):
    PAL = "nomad"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "nomad", "talk")
        self.state, self.trig, self.rise_t, self.run_t = "sit", None, 0.0, 0.0
        self.y = -0.34
        self.corner = (ch.xoff + rng.choice((-1, 1)) * (ch.hw - 1.2), ch.z0 + CH - 2.5)
        self.stage = 0
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def on_talk(self) -> None:
        if self.state != "sit":
            return
        self.talks += 1
        self.g.hud.speak(self, NOMAD_LINES[min(self.talks, len(NOMAD_LINES)) - 1])
        if self.talks == 1:
            self.trig = self.g.clock + 16.0
        if self.talks >= 3:
            self.trig = self.g.clock + 2.0

    def update(self, dt: float, w) -> None:
        g = self.g
        if self.gone:
            return
        if self.state == "sit":
            self.look_at_player(dt, 2.0, 5.0)
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(dt, 0, True)
            if self.trig is not None and g.clock >= self.trig:
                self.state = "rise"
                g.sfx.play("static", 0.6)
        elif self.state == "rise":
            self.rise_t += dt
            self.y = -0.34 * (1.0 - min(1.0, self.rise_t / 0.7))
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(dt, 0.0, self.rise_t < 0.4)
            if self.rise_t >= 0.9:
                self.state = "run"
                g.sfx.play("scream", 1.0, wet=True)
                g.noise_pulse = 0.9
                g.fx.flash_a = 0.5
                g.hud.speak(self, "THE LIGHTS ARE SAYING IT! I CHOSE! I CHOSE!!")
        elif self.state == "run":
            self.run_t += dt
            g.noise_pulse = max(g.noise_pulse, 0.35 + 0.25 * math.sin(self.run_t * 23.0))
            if int(self.run_t * 3.0) != int((self.run_t - dt) * 3.0):
                g.sfx.play("static", 0.5, pan=random.uniform(-0.6, 0.6))
            cx, cz = self.ch.xoff + 0.0, self.corner[1]
            target = (self.x, cz) if self.stage == 0 else self.corner
            if self.walk_to(dt, target[0], target[1], 7.6):
                if self.stage == 0:
                    self.stage = 1
                else:                                               # round the corner and gone, before the chunk unloads
                    self.gone = True
                    self.fig.root.hide()
                    g.noise_pulse = 0.0
                    g.award("nomad")
                    g.sfx.play("static_long", 0.3)


class CoinerNpc(Actor):
    """Looks around for a coin, leans down, throws it in the fountain, repeats until the chunk unloads."""
    PAL = "corridor"
    LINES = ["Shh. I'm in the middle of a wish.", "I keep losing the coin. The fountain finds it for me.",
             "Is it still a wish if you don't know what for?", "Penny, nickel, whatever it is down there. It accepts."]

    def __init__(self, g, ch, pos, rng, fountain) -> None:
        super().__init__(g, ch, pos, rng, "stranger", "talk")
        self.fount, self.phase, self.pt, self.lean, self.throws = fountain, "look", 0.0, 0.0, 0
        self.home_yaw = math.degrees(math.atan2(fountain[0] - self.x, fountain[2] - self.z))
        self.yaw = self.home_yaw

    def next_line(self) -> str:
        return self.rng.choice(self.LINES)

    def update(self, dt: float, w) -> None:
        g = self.g
        self.pt += dt
        if self.phase == "look":                         # scanning the floor for a coin
            self.yaw = self.home_yaw + 55.0 * math.sin(self.pt * 2.1)
            self.lean = max(0.0, self.lean - 60 * dt)
            if self.pt > 3.6:
                self.phase, self.pt = "down", 0.0
        elif self.phase == "down":
            self.yaw = self.home_yaw
            self.lean = min(38.0, self.lean + 70 * dt)
            if self.pt > 0.7:
                self.phase, self.pt = "throw", 0.0
        elif self.phase == "throw":
            if self.pt > 0.45:
                self.throws += 1
                fx, fy, fz = self.fount
                spawn_coin(self.ch, g, (self.x, 1.0, self.z), (fx, fy, fz), 0.9)
                self.phase, self.pt = "up", 0.0
                if self.throws >= 3 and self.dist() < 14.0:
                    g.award("witness")
        elif self.phase == "up":
            self.lean = max(0.0, self.lean - 45 * dt)
            if self.pt > 0.9:
                self.phase, self.pt = "look", 0.0
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.root.rotation_x = self.lean
        self.fig.pose(dt, 0.0)


class LookoutNpc(Actor):
    PAL = "lookout"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "window watcher", "talk")
        self.yaw = self.home = 90.0

    def next_line(self) -> str:
        wins = self.ch.extra.get("windows", [])
        mode = min(wins, key=lambda wd: abs(wd["z"] - self.z))["mode"] if wins else "fog"
        sky = ""
        if self.g.sky.weather == "storm" and mode == "rain":
            sky = " (and the thunder, the thunder, the thunder.)"
        return self.rng.choice(LOOKOUT_LINES[mode]) + sky

    def on_talk(self) -> None:
        super().on_talk()
        if self.talks >= 2:
            self.g.award("window")

    def update(self, dt: float, w) -> None:
        self.look_at_player(dt, 2.5, 3.2, self.home)
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.pose(dt, 0.0)


class TrappedNpc(Actor):
    PAL = "trapped"

    def __init__(self, g, ch, pos, rng, k: int) -> None:
        super().__init__(g, ch, pos, rng, TRAPPED[k][0], "talk")
        self.k = k

    def on_talk(self) -> None:
        g, lines = self.g, TRAPPED[self.k][1]
        self.talks += 1
        g.hud.speak(self, lines[min(self.talks, len(lines)) - 1])
        if self.talks >= len(lines) and not self.given:
            self.given = True
            g.inv.coins += 6
            g.inv.dirty = True
            g.sfx.play("coin")
            g.hud.note("+6 c  -  they press it into your hand")
            g.br_heard.add(self.k)
            if len(g.br_heard) >= len(TRAPPED):
                g.award("listener")


class CraftsmanNpc(Actor):
    PAL = "craft"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "craftsman", "talk")
        self.paint_price, self.fuel_price = rng.randint(6, 14), rng.randint(8, 20)      # differs for every craftsman

    def on_talk(self) -> None:
        self.g.hud.speak(self, self.rng.choice(["Got stuck here same as you. Might as well keep the karts pretty.",
                                                "Paint, fuel, a little love. Show me your cart.",
                                                "Prices are whatever I feel like today. Today I feel fair."]))
        self.g.open_ui(CraftUI(self.g, self))


class CafeVendor(Actor):
    PAL = "trapped"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "cafe keeper", "talk")

    def on_talk(self) -> None:
        self.g.hud.speak(self, self.rng.choice(["We're open. We've always been open. Water, donuts, and the occasional miracle.",
                                                "Don't ask where the donuts come from. Ask where they go.",
                                                "The phone shows you the way out. For a price. Everything is for a price down here."]))
        self.g.open_ui(StockShopUI(self.g, self, "BACKROOMS CAFE", CAFE_STOCK, finite=("phone", "plush")))


class CinemaUsher(Actor):
    PAL = "corridor"

    def on_talk(self) -> None:
        g = self.g
        g.hud.speak(self, self.rng.choice(["Welcome to Theater 7. Concessions first, always.", "Everything on the board. Buy it all and the screen remembers you."]))
        g.open_ui(StockShopUI(g, self, "THEATER 7  -  CONCESSIONS", CINEMA_STOCK, track=f"cinema:{self.ch.idx}"))


class ServantNpc(Actor):
    """Takes your order through a menu, walks to the kitchen, comes back and sets the dish (or glass) on your table."""
    PAL = "servant"

    def __init__(self, g, ch, pos, rng) -> None:
        super().__init__(g, ch, pos, rng, "waiter", "server")
        self.home = pos
        self.job, self.wp, self.carry, self.prep = None, [], None, 0.0
        self.aisle = ch.xoff - 2.8

    def on_talk(self) -> None:
        g = self.g
        if self.job is not None:
            g.hud.speak(self, "One moment, please. It's on its way.")
            return
        g.hud.speak(self, self.rng.choice(["Good evening. Sit anywhere. Order whenever the hour allows.", "Table for one? Table for you? Same table, different you."]))
        g.open_ui(StockShopUI(g, self, "RISTORANTE  -  MENU", MENU_STOCK, on_order=self.order))

    def order(self, key: str) -> None:
        ch, g = self.ch, self.g
        px, pz = g.player.x, g.player.z
        if not ch.tables:
            return
        tb = min(ch.tables, key=lambda t: math.hypot(t[0] + ch.xoff - px, t[1] - pz))
        tx, tz = tb[0] + ch.xoff, tb[1]
        sx = max(tx - tb[3] / 2 + 0.2, min(tx + tb[3] / 2 - 0.2, px))
        sz = max(tz - tb[4] / 2 + 0.2, min(tz + tb[4] / 2 - 0.2, pz))
        self.job = dict(key=key, slot=(sx, sz), table=(tx, tz))
        kz = ch.z0 + 21.8
        side = -1.0 if tx < self.aisle else 1.0
        self.wp = [(ch.xoff + 4.5, kz)]
        self.phase = "kitchen"
        self.prep = 3.2
        g.sfx.play("clink", 0.6, wet=True)

    def _item_ent(self, key: str, parent, pos):
        it = ITEMS[key]
        col = C(*[c / 255 for c in it["col"]])
        ch = self.ch
        kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        root = ur.Entity(parent=parent, position=pos)
        if key in DRINK_KEYS:
            mk(root, "cube", (0, 0.06, 0), (0.08, 0.12, 0.08), "glass", color=C(0.85, 0.95, 1.0, 0.45), alpha=True, **kw)
            mk(root, "cube", (0, 0.045, 0), (0.066, 0.085, 0.066), color=col, emissive=0.35, **kw)
        else:
            mk(root, "cube", (0, 0.01, 0), (0.30, 0.02, 0.30), "marble", color=C(0.95, 0.95, 0.92), **kw)
            mk(root, "sphere", (0, 0.06, 0), (0.17, 0.09, 0.17), color=col, emissive=0.2, **kw)
            mk(root, "sphere", (0.04, 0.10, 0.02), (0.07, 0.05, 0.07), color=col * 1.15, emissive=0.2, **kw)
        return root

    def update(self, dt: float, w) -> None:
        g, ch = self.g, self.ch
        if self.job is None:
            if self.wp and not self.walk_to(dt, *self.wp[0], 1.5):
                return
            if self.wp:
                self.wp.pop(0)
            if math.hypot(self.x - self.home[0], self.z - self.home[1]) > 0.1 and not self.wp:
                self.wp = [(self.aisle, ch.z0 + 21.8), self.home]
            self.look_at_player(dt, 2.0, 5.0, 180.0)
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(dt, 0.0)
            return
        ph = self.phase
        if ph == "kitchen":
            if self.wp:
                if self.walk_to(dt, *self.wp[0], 2.0):
                    self.wp.pop(0)
                return
            self.prep -= dt
            self.face(180.0, dt, 3.0)
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(dt, 0.0)
            if self.prep <= 0:
                self.carry = self._item_ent(self.job["key"], self.fig.root, (0.26, 0.95, 0.38))
                sx, sz = self.job["slot"]
                tx, tz = self.job["table"]
                ax = self.aisle
                ex, ez = sx + (0.9 if sx < ax else -0.9), sz
                self.wp = [(ax, ch.z0 + 21.8), (ax, sz), (ex, ez)]
                self.phase = "deliver"
        elif ph == "deliver":
            if self.walk_to(dt, *self.wp[0], 1.9):
                self.wp.pop(0)
            if not self.wp:
                sx, sz = self.job["slot"]
                key = self.job["key"]
                self.face(math.degrees(math.atan2(sx - self.x, sz - self.z)), dt, 12.0)
                ur.destroy(self.carry)
                self.carry = None
                table_h = 0.75
                for t in ch.tables:
                    if abs(t[0] + ch.xoff - self.job["table"][0]) < 0.01 and abs(t[1] - self.job["table"][1]) < 0.01:
                        table_h = t[2]
                ent = self._item_ent(key, ch.detail, (sx - ch.xoff, table_h + 0.03, sz))
                ch.served.append(dict(key=key, ent=ent, x=sx, z=sz))
                g.sfx.play("clink", 0.9, wet=True)
                g.hud.speak(self, f"Your {ITEMS[key]['name'].lower()}. Enjoy it while it's warm.")
                g.hud.note("Sit and press T to eat/drink it, or U to put it in your bag.")
                g.award("dine")
                self.job = None
                self.wp = [(self.aisle, sz), (self.aisle, ch.z0 + 21.8), self.home]
        self.fig.place(self.x, self.y, self.z, self.yaw)


class RatActor:
    """Not a Figure: a small rat.  Same duck-typed surface as Npc (x/z/gone/name/role/radius/fig.root)."""
    role, radius, mode, kind = "talk", 1.9, "rat", "server"

    def __init__(self, g, ch, x: float, z: float) -> None:
        self.g, self.ch, self.x, self.z, self.y = g, ch, x, z, 0.0
        self.name, self.gone, self.told, self.talks, self.last_talk = "rat", False, False, 0, -1e9
        self.yaw, self.target, self.timer, self.seat, self.history = random.uniform(0, 360), None, random.uniform(0.5, 2), None, []
        root = ur.Entity(parent=ur.scene)
        kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        fur = C(0.36, 0.33, 0.32)
        mk(root, "sphere", (0, 0.10, 0), (0.17, 0.15, 0.36), color=fur, **kw)
        mk(root, "sphere", (0, 0.11, 0.2), (0.11, 0.10, 0.16), color=fur, **kw)
        mk(root, "sphere", (0.05, 0.17, 0.23), (0.05, 0.05, 0.02), color=C(0.8, 0.6, 0.6), **kw)
        mk(root, "sphere", (-0.05, 0.17, 0.23), (0.05, 0.05, 0.02), color=C(0.8, 0.6, 0.6), **kw)
        mk(root, "sphere", (0.035, 0.13, 0.27), (0.02, 0.02, 0.02), color=C(0.02, 0.02, 0.02), emissive=0.5, **kw)
        mk(root, "sphere", (-0.035, 0.13, 0.27), (0.02, 0.02, 0.02), color=C(0.02, 0.02, 0.02), emissive=0.5, **kw)
        self.tail = ur.Entity(parent=root, position=(0, 0.07, -0.18))
        mk(self.tail, "cube", (0, 0, -0.14), (0.02, 0.02, 0.32), color=C(0.75, 0.55, 0.55), **kw)
        self.fig = SimpleNamespace(root=root)
        root.position = (x, 0, z)
        self.t = 0.0

    def next_line(self) -> str:
        return RAT_FIRST if not self.told else random.choice(RAT_LINES)

    def on_talk(self) -> None:
        g = self.g
        self.talks += 1
        g.hud.speak(self, self.next_line())
        g.sfx.play("squeak", 0.8, wet=True)
        if not self.told:
            self.told = True
            g.award("rat")
            g.ratwhisper = True

    def update(self, dt: float, w) -> None:
        g, ch = self.g, self.ch
        self.t += dt
        px, pz = g.player.x, g.player.z
        d = math.hypot(px - self.x, pz - self.z)
        speed = 0.0
        if d < 1.1 and g.player.speed > 2.5:                           # bolts if you charge it
            self.target = (self.x + (self.x - px) * 3.0, self.z + (self.z - pz) * 3.0)
            speed = 2.6
        elif self.target is not None:
            speed = 1.2
        self.timer -= dt
        if self.target is None and self.timer <= 0:
            for _ in range(8):
                tx = ch.xoff + random.uniform(-ch.hw + 1.3, ch.hw - 1.3)
                tz = ch.z0 + random.uniform(1.5, CH - 1.5)
                if all(not (c[0] - 0.4 < tx < c[1] + 0.4 and c[2] - 0.4 < tz < c[3] + 0.4 and c[4] > 0.2) for c in ch.colliders):
                    self.target = (tx, tz)
                    break
            self.timer = random.uniform(1.0, 3.5)
        if self.target is not None:
            ex, ez = self.target[0] - self.x, self.target[1] - self.z
            dist = math.hypot(ex, ez)
            if dist < 0.15:
                self.target = None
            else:
                self.yaw += wrap180(math.degrees(math.atan2(ex, ez)) - self.yaw) * min(1.0, 9.0 * dt)
                nx, nz, _g, bumped = w.resolve(self.x + ex / dist * speed * dt, 0.0, self.z + ez / dist * speed * dt, 0.15)
                if bumped:
                    self.target = None
                self.x, self.z = nx, nz
        self.fig.root.position = (self.x, 0.0, self.z)
        self.fig.root.rotation_y = self.yaw
        self.tail.rotation_y = math.sin(self.t * 6.0) * 25.0


def spawn_coin(ch, g, src: tuple, dst: tuple, dur: float = 0.9) -> None:
    """A small gold coin on a parabola into the fountain; splash + ripple on landing."""
    e = mk(ur.scene, "sphere", src, (0.09, 0.02, 0.09), color=C(1.0, 0.82, 0.2), emissive=0.8)
    st = dict(t=0.0)

    def tick(dt, game):
        st["t"] += dt
        k = min(1.0, st["t"] / dur)
        e.x = src[0] + (dst[0] - src[0]) * k
        e.z = src[2] + (dst[2] - src[2]) * k
        e.y = src[1] + (dst[1] - src[1]) * k + 1.6 * math.sin(math.pi * k)
        e.rotation_x += 600 * dt
        if k >= 1.0:
            ur.destroy(e)
            game.sfx.play("splash", 0.5, pan=max(-1, min(1, (dst[0] - game.player.x) / 10.0)), wet=True)
            fw = ch.extra.get("water")
            if fw:
                fw["ripple"](dst[0], dst[2])
            return False
        return True
    ch.tickers.append(tick)


# ---- UI screens ------------------------------------------------------------------------------------------
_SHIFT = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+",
          "[": "{", "]": "}", ";": ":", "'": '"', ",": "<", ".": ">", "/": "?", "\\": "|", "`": "~"}
_NAMED = {"comma": ",", "period": ".", "slash": "/", "minus": "-", "equals": "=", "semicolon": ";", "apostrophe": "'",
          "backslash": "\\", "left bracket": "[", "right bracket": "]", "space": " "}


def key_to_char(k: str) -> Optional[str]:
    shift = bool(ur.held_keys["shift"] or ur.held_keys["left shift"] or ur.held_keys["right shift"])
    k = _NAMED.get(k, k)
    if len(k) != 1:
        return None
    if k.isalpha():
        return k.upper() if shift else k.lower()
    return _SHIFT.get(k, k) if shift else k


class VendingUI(UiBase):
    """Pick your drink on the machine's own panel: arrows / W A S D move, number keys jump to a slot, Enter buys."""
    def __init__(self, g, it: Interactable) -> None:
        super().__init__(g)
        self.it, self.sel, self.msg = it, 0, ""
        self.menu = it.data.get("menu") or VEND_ALL
        self.quad((0, 0.0), (0.92, 0.86), C(0.8, 0.9, 1.0, 0.25), z=0.22)
        self.quad((0, 0.0), (0.90, 0.84), C(0.10, 0.11, 0.13, 0.96), z=0.21)
        self.quad((0, 0.30), (0.84, 0.14), C(0.02, 0.05, 0.06, 1.0), z=0.2)
        self.text("DRINKS  -  SELECT", (-0.40, 0.34), 1.0, C(0.5, 1.0, 0.8, 1))
        self.wallet = self.text("", (0.02, 0.34), 0.85, C(1.0, 0.9, 0.5, 1))
        self.tiles = []
        for i, (key, price) in enumerate(self.menu):
            cx, cy = -0.27 + 0.27 * (i % 3), 0.10 - 0.25 * (i // 3)
            col = C(*[c / 255 for c in ITEMS[key]["col"]])
            frame = self.quad((cx, cy), (0.245, 0.225), C(0.3, 0.3, 0.34, 1), z=0.19)
            self.quad((cx, cy + 0.02), (0.09, 0.14), col, z=0.18)
            self.text(f"{i + 1}", (cx - 0.11, cy + 0.09), 0.8, C(1, 1, 1, 0.9))
            self.text(ITEMS[key]["name"][:16], (cx, cy - 0.065), 0.58, C(1, 1, 1, 0.95), origin=(0, 0))
            self.text(f"{price} c", (cx, cy - 0.095), 0.62, C(1.0, 0.9, 0.5, 1), origin=(0, 0))
            self.tiles.append(frame)
        self.text("arrows/WASD move   1-6 jump   Enter buy   E / Esc leave", (-0.40, -0.39), 0.62, C(0.7, 0.8, 0.85, 0.9))
        self.refresh()

    def refresh(self) -> None:
        self.wallet.text = f"{self.g.inv.coins} c   {self.msg}"
        for i, f in enumerate(self.tiles):
            f.color = C(0.95, 0.85, 0.35, 1) if i == self.sel else C(0.3, 0.3, 0.34, 1)

    def key(self, k: str) -> None:
        g, n = self.g, len(self.menu)
        if k in ("e", "escape", "g"):
            g.close_ui()
            return
        if k in ("left arrow", "a"):
            self.sel = (self.sel - 1) % n
        elif k in ("right arrow", "d"):
            self.sel = (self.sel + 1) % n
        elif k in ("up arrow", "w"):
            self.sel = (self.sel - 3) % n
        elif k in ("down arrow", "s"):
            self.sel = (self.sel + 3) % n
        elif k.isdigit() and 1 <= int(k) <= n:
            self.sel = int(k) - 1
            self.buy()
        elif k in ("enter", "return"):
            self.buy()
        g.sfx.play("tick", 0.5)
        self.refresh()

    def buy(self) -> None:
        g = self.g
        if self.it.data["cool"] > 0:
            return
        key, price = self.menu[self.sel]
        if g.inv.coins < price:
            self.msg = "not enough coins"
            g.sfx.play("tick", 0.7)
            return
        g.inv.coins -= price
        g.inv.add(key)
        self.it.data["cool"] = 1.4
        g.sfx.play("coin")
        self.msg = f"{ITEMS[key]['name']} dispensed"
        g.inv.flags["vend"] = int(g.inv.flags.get("vend", 0)) + 1
        if g.inv.flags["vend"] >= 5:
            g.award("thirst")
        g._vend_drop(self.it, key)


class StockShopUI(UiBase):
    """Generic list shop (cinema concession, backrooms cafe, restaurant menu)."""
    def __init__(self, g, npc, title: str, stock: tuple, track: str = "", finite: tuple = (), on_order=None) -> None:
        super().__init__(g)
        self.npc, self.stock, self.sel, self.msg, self.track, self.finite, self.on_order = npc, stock, 0, "", track, finite, on_order
        n = len(stock)
        self.quad((0, 0), (0.96, 0.22 + 0.055 * n), C(0.9, 0.7, 0.3, 0.35), z=0.21)
        self.quad((0, 0), (0.94, 0.20 + 0.055 * n), C(0.04, 0.03, 0.02, 0.94))
        top = 0.08 + 0.0275 * n
        self.text(title, (-0.44, top), 1.05, C(1.0, 0.9, 0.6, 1))
        self.wallet = self.text("", (-0.44, top - 0.05), 0.85, C(0.8, 1.0, 0.8, 1))
        self.rows = [self.text("", (-0.44, top - 0.115 - 0.055 * i), 0.9) for i in range(n)]
        self.text("W/S choose   Enter buy   E / Esc leave", (-0.44, -top + 0.01), 0.65, C(0.8, 0.75, 0.6, 0.9))
        self.refresh()

    def bought(self) -> list:
        return list(self.g.inv.flags.get(self.track, [])) if self.track else []

    def refresh(self) -> None:
        g = self.g
        self.wallet.text = f"you have {g.inv.coins} c     {self.msg}"
        own = self.bought()
        for i, (k, price) in enumerate(self.stock):
            tag = "  [bought]" if k in own else (f"  (have {g.inv.n(k)})" if not self.on_order else "")
            self.rows[i].text = f"{'>' if i == self.sel else ' '} {ITEMS[k]['name']:<24} {price:>3} c{tag}"
            self.rows[i].color = C(1.0, 0.95, 0.6, 1) if i == self.sel else C(0.9, 0.85, 0.75, 0.95)

    def key(self, k: str) -> None:
        g = self.g
        if k in ("e", "escape", "g"):
            g.close_ui()
        elif k in ("w", "up arrow"):
            self.sel = (self.sel - 1) % len(self.stock)
            self.refresh()
        elif k in ("s", "down arrow"):
            self.sel = (self.sel + 1) % len(self.stock)
            self.refresh()
        elif k in ("enter", "return"):
            key, price = self.stock[self.sel]
            if key in self.finite and g.inv.n(key) > 0:
                self.msg = "you already own one"
            elif g.inv.coins < price:
                self.msg = "not enough coins"
            else:
                g.inv.coins -= price
                g.sfx.play("coin")
                if self.on_order:
                    self.on_order(key)
                    g.close_ui()
                    return
                g.inv.add(key)
                self.msg = f"bought {ITEMS[key]['name']}"
                if key == "plush":
                    g.award("plush")
                if key == "phone":
                    g.award("phone")
                if self.track:
                    lst = self.bought()
                    if key not in lst:
                        lst.append(key)
                    g.inv.flags[self.track] = lst
                    if all(s[0] in lst for s in self.stock):
                        g.cinema_complete()
            self.refresh()


class BookUI(UiBase):
    """Read a book; Tab switches to write mode.  Notes are saved to the local cache file as you type."""
    def __init__(self, g, bid: str, title: str, lore: str) -> None:
        super().__init__(g)
        self.bid, self.title, self.lore = bid, title, lore
        self.lore_lines = wrap_text(lore, 62) if lore else []
        self.text_buf = g.books.get(bid)
        self.write, self.page = False, 0
        self.quad((0, 0), (1.12, 0.86), C(0.85, 0.75, 0.5, 0.35), z=0.21)
        self.quad((0, 0), (1.10, 0.84), C(0.93, 0.89, 0.78, 0.97), z=0.2)
        self.head = self.text(title, (-0.50, 0.37), 1.1, C(0.25, 0.15, 0.08, 1))
        self.rows = [self.text("", (-0.50, 0.30 - 0.038 * i), 0.8, C(0.12, 0.1, 0.08, 1)) for i in range(15)]
        self.foot = self.text("", (-0.50, -0.39), 0.65, C(0.4, 0.25, 0.15, 1))
        if lore:
            self.mark_read()
        self.refresh()

    def mark_read(self) -> None:
        g = self.g
        lst = list(g.inv.flags.get("lore_read", []))
        if self.bid not in lst:
            lst.append(self.bid)
            g.inv.flags["lore_read"] = lst
            g.inv.dirty = True
        if len(lst) >= 3:
            g.award("reader")
        if all(f"lore{i}" in lst for i in range(len(BOOK_LORE))):
            g.award("why")

    def refresh(self) -> None:
        lines = list(self.lore_lines)
        if lines:
            lines.append("-" * 40)
            lines.append("notes:")
        note_lines = []
        for para in (self.text_buf + ("_" if self.write else "")).split("\n"):
            note_lines += wrap_text(para, 62) if para else [""]
        room = len(self.rows) - len(lines)
        shown = lines + (note_lines[-room:] if room > 0 else [])
        for i, row in enumerate(self.rows):
            row.text = shown[i] if i < len(shown) else ""
        self.foot.text = ("WRITING  -  type freely, Enter new line, Esc stop writing" if self.write
                          else "Tab: write in this book    E / Esc: close the book    (saved to ~/.ethereal_mall/book_cache.json)")

    def key(self, k: str) -> None:
        g = self.g
        if self.write:
            if k == "escape" or k == "tab":
                self.write = False
                g.books.put(self.bid, self.text_buf)
                if self.text_buf.strip():
                    g.award("author")
            elif k == "backspace":
                self.text_buf = self.text_buf[:-1]
            elif k in ("enter", "return"):
                self.text_buf += "\n"
            else:
                ch = key_to_char(k)
                if ch and len(self.text_buf) < 700:
                    self.text_buf += ch
            self.refresh()
            return
        if k in ("e", "escape", "g"):
            g.close_ui()
        elif k == "tab":
            self.write = True
            self.refresh()

    def close(self) -> None:
        self.g.books.put(self.bid, self.text_buf)
        super().close()


class ArchiveUI(UiBase):
    """Lost & Found desk log: metadata of a track the audio engine knows about."""
    def __init__(self, g, it: Interactable) -> None:
        super().__init__(g)
        ids = g.covers.ids() if g.covers is not None else []
        if ids:
            tid = ids[stable_hash(g.args.seed, "laf", it.data.get("tag", 0)) % len(ids)]
            info = g.stack.indexer.tracks.get(tid) if g.stack.indexer is not None else None
            title = g.covers.title(tid).replace("_", " ")
            artist = _artist_of(info.path) if info is not None else "unknown artist"
            dur = info.analysis.duration_s if (info is not None and info.analysis is not None) else None
            dtxt = f"{int(dur // 60)}:{int(dur % 60):02d}" if dur else "not yet measured"
            fname = info.path.name if info is not None else "?"
            lines = [f"TRACK    {title[:44]}", f"ARTIST   {artist[:44]}", f"LENGTH   {dtxt}", f"FILE     {fname[:44]}"]
        else:
            lines = ["TRACK    (shelf empty)", "ARTIST   -", "LENGTH   -", "NOTE     run with --library to fill the shelves"]
        self.quad((0, 0), (0.95, 0.5), C(0.6, 0.9, 0.7, 0.3), z=0.21)
        self.quad((0, 0), (0.93, 0.48), C(0.03, 0.05, 0.04, 0.95))
        self.text(f"LOST & FOUND  -  LOG #{abs(stable_hash(it.data.get('tag', 0))) % 9000 + 1000}", (-0.43, 0.19), 1.0, C(0.6, 1.0, 0.8, 1))
        for i, ln in enumerate(lines):
            self.text(ln, (-0.43, 0.10 - 0.055 * i), 0.85)
        self.text("LEFT BY    nobody  (an intercom whispers the title once more)", (-0.43, -0.14), 0.62, C(0.6, 0.75, 0.7, 0.9))
        self.text("E / Esc / G  close", (-0.43, -0.21), 0.62, C(0.6, 0.75, 0.7, 0.9))
        g.sfx.play("whisper", 0.8, wet=True)
        g.award("archive")

    def key(self, k: str) -> None:
        if k in ("e", "escape", "g", "enter"):
            self.g.close_ui()


class CraftUI(UiBase):
    def __init__(self, g, npc: CraftsmanNpc) -> None:
        super().__init__(g)
        self.npc, self.sel, self.msg = npc, 0, ""
        self.names = list(KART_COLORS)
        n = len(self.names) + 1
        self.quad((0, 0), (0.9, 0.22 + 0.05 * n), C(0.9, 0.55, 0.2, 0.35), z=0.21)
        self.quad((0, 0), (0.88, 0.20 + 0.05 * n), C(0.05, 0.04, 0.03, 0.95))
        top = 0.08 + 0.025 * n
        self.text(f"{npc.name.upper()}  -  PAINT & FUEL", (-0.41, top), 1.0, C(1.0, 0.85, 0.5, 1))
        self.wallet = self.text("", (-0.41, top - 0.05), 0.8, C(0.8, 1.0, 0.8, 1))
        self.rows = [self.text("", (-0.41, top - 0.11 - 0.05 * i), 0.85) for i in range(n)]
        self.text("W/S choose   Enter pay   E / Esc leave", (-0.41, -top + 0.01), 0.62, C(0.8, 0.75, 0.6, 0.9))
        self.refresh()

    def kart(self):
        g = self.g
        k = g.last_kart
        if k is None or getattr(k, "level", "mall") != g.world.level:
            k = min((k_ for k_ in g.karts.values() if getattr(k_, "level", "mall") == g.world.level),
                    key=lambda k_: math.hypot(k_.x - g.player.x, k_.z - g.player.z), default=None)
        return k

    def refresh(self) -> None:
        g, npc = self.g, self.npc
        self.wallet.text = f"you have {g.inv.coins} c   {self.msg}"
        for i, nm in enumerate(self.names):
            self.rows[i].text = f"{'>' if i == self.sel else ' '} repaint:  {nm:<10}  {npc.paint_price} c"
        self.rows[-1].text = f"{'>' if self.sel == len(self.names) else ' '} refuel to full           {npc.fuel_price} c"
        for i, r in enumerate(self.rows):
            r.color = C(1.0, 0.95, 0.6, 1) if i == self.sel else C(0.9, 0.85, 0.75, 0.95)

    def key(self, k: str) -> None:
        g, npc = self.g, self.npc
        n = len(self.names) + 1
        if k in ("e", "escape", "g"):
            g.close_ui()
        elif k in ("w", "up arrow"):
            self.sel = (self.sel - 1) % n
            self.refresh()
        elif k in ("s", "down arrow"):
            self.sel = (self.sel + 1) % n
            self.refresh()
        elif k in ("enter", "return"):
            kart = self.kart()
            price = npc.fuel_price if self.sel == len(self.names) else npc.paint_price
            if kart is None or math.hypot(kart.x - g.player.x, kart.z - g.player.z) > 16.0:
                self.msg = "bring your cart closer"
            elif g.inv.coins < price:
                self.msg = "not enough coins"
            else:
                g.inv.coins -= price
                g.inv.dirty = True
                g.sfx.play("coin")
                kart.build()
                if self.sel == len(self.names):
                    kart.fuel = 100.0
                    self.msg = "tank filled"
                else:
                    col = KART_COLORS[self.names[self.sel]]
                    kart.recolor(col)
                    self.msg = f"{self.names[self.sel]} - looks good"
                    g.award("paint")
            self.refresh()


class PhoneUI(UiBase):
    """Mini-map phone: in the Backrooms a live map + exit radar; elsewhere the next chunks ahead.  Also a delivery app."""
    mouse_free = False

    def __init__(self, g) -> None:
        super().__init__(g)
        self.t, self.msg = 0.0, ""
        self.quad((0, 0), (0.62, 0.92), C(0.4, 0.5, 0.7, 0.4), z=0.22)
        self.quad((0, 0), (0.60, 0.90), C(0.02, 0.025, 0.04, 0.98), z=0.21)
        self.screen = self.quad((0, 0.14), (0.52, 0.52), C(1, 1, 1, 1), None, 0.15)
        self.info = self.text("", (-0.26, -0.18), 0.72, C(0.6, 1.0, 0.8, 1))
        self.info2 = self.text("", (-0.26, -0.25), 0.72, C(1.0, 0.9, 0.6, 1))
        self.text("[1] order water  2 c    [2] order donut  3 c", (-0.26, -0.33), 0.6, C(0.8, 0.9, 1.0, 0.95))
        self.text("P / E / Esc close", (-0.26, -0.40), 0.6, C(0.6, 0.7, 0.85, 0.9))
        self.redraw()

    def redraw(self) -> None:
        g, p = self.g, self.g.player
        S = 256
        im = Image.new("RGBA", (S, S), (8, 14, 12, 255))
        d = ImageDraw.Draw(im)
        if g.world.level == "backrooms":
            pr, pc = int(p.z // 2), int((p.x + 12) // 2)
            R = 11
            cs = S / (2 * R + 1)
            path = br_solve((max(0, min(BR_ROWS - 1, pr)), max(0, min(BR_COLS - 1, pc))))
            for rr in range(pr - R, pr + R + 1):
                for cc in range(pc - R, pc + R + 1):
                    if not (0 <= rr < BR_ROWS and 0 <= cc < BR_COLS):
                        continue
                    x0, y0 = (cc - pc + R) * cs, (R - (rr - pr)) * cs
                    d.rectangle([x0 + 1, y0 + 1, x0 + cs - 1, y0 + cs - 1], fill=(26, 34, 26, 255))
                    if cc < BR_COLS - 1 and BR_EAST[rr][cc]:
                        d.line([x0 + cs, y0, x0 + cs, y0 + cs], fill=(170, 220, 160, 255), width=2)
                    if rr < BR_ROWS - 1 and BR_NORTH[rr][cc]:
                        d.line([x0, y0, x0 + cs, y0], fill=(170, 220, 160, 255), width=2)
            if path:
                for (rr, cc) in path[:40]:
                    if abs(rr - pr) <= R and abs(cc - pc) <= R:
                        x0, y0 = (cc - pc + R + 0.5) * cs, (R - (rr - pr) + 0.5) * cs
                        d.ellipse([x0 - 2, y0 - 2, x0 + 2, y0 + 2], fill=(255, 200, 80, 255))
            d.ellipse([S / 2 - 5, S / 2 - 5, S / 2 + 5, S / 2 + 5], fill=(80, 200, 255, 255))
            ex, ez = 6.0, BR_ROWS * 2.0 - 8.0
            dist = math.hypot(ex - p.x, ez - p.z)
            ang = math.degrees(math.atan2(ex - p.x, ez - p.z))
            arrow = "^>v<"[int(((ang + 45) % 360) // 90)] if False else ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((ang + 22.5) % 360) // 45)]
            self.info.text = f"EXIT ELEVATOR  {dist:4.0f} m  bearing {arrow}"
            self.info2.text = "follow the yellow trail" if path else "no route"
            r = (self.t * 90) % 120
            d.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], outline=(80, 200, 255, int(255 * (1 - r / 120))))
        else:
            idx = g.world.chunk_index(p.z)
            d.text((10, 8), "UPCOMING CHUNKS", fill=(120, 255, 200, 255), font=_font(16))
            for i in range(1, 9):
                kind = g.world.geom(idx + i)[0]
                nm = ZONES.get(kind, {}).get("title", kind)[:26]
                d.rectangle([10, 14 + 26 * i, 14 + (9 - i) * 14, 34 + 26 * i], fill=(30, 70, 60, 255))
                d.text((20, 17 + 26 * i), f"+{i}  {nm}", fill=(235, 250, 240, 255), font=_font(13))
            self.info.text = f"chunk {idx}   level {g.world.level}"
            self.info2.text = "no exit radar out here - the phone only knows the Backrooms"
        self.screen.texture = tex_from_pil(im)

    def update(self, dt: float) -> None:
        self.t += dt
        if self.t > 0.45:
            self.t = 0.0
            self.redraw()

    def key(self, k: str) -> None:
        g = self.g
        if k in ("p", "e", "escape", "g"):
            g.close_ui()
        elif k in ("1", "2"):
            key, price = (("water", 2), ("donut", 3))[int(k) - 1]
            if g.inv.coins < price:
                self.info2.text = "not enough coins"
                return
            g.inv.coins -= price
            g.inv.add(key)
            g.sfx.play("chime", 0.8)
            g.hud.note(f"A {ITEMS[key]['name']} appears in your bag. You did not hear it arrive.")


# ---- world builders --------------------------------------------------------------------------------------
class V3Builders:
    """New zones and props.  Installed onto ChunkBuilder (see _v3_install)."""

    # ---- helpers
    def _solid_void(self) -> None:
        ch = self.ch
        self.block(0, 3.0, ch.z0 + CH / 2, 2 * ch.hw + 2.0, 6.0, CH, "concrete", color=C(0.03, 0.03, 0.03))

    def _book(self, x, y, z, yaw, bid, title, lore, col=None) -> None:
        rng = self.rng
        col = col or C(*[rng.uniform(0.15, 0.7) for _ in range(3)])
        self.box(x, y + 0.03, z, 0.26, 0.06, 0.20, color=col, rot=(0, yaw, 0))
        self.box(x, y + 0.063, z, 0.24, 0.01, 0.18, color=C(0.92, 0.9, 0.82), rot=(0, yaw, 0))
        self.inter("book", x, 0.0, z, 1.4, "Read the book" if lore else "Open the blank book", yaw, dict(bid=bid, title=title, lore=lore))

    def _paper(self, x, z, kind="paper") -> None:
        ch, rng = self.ch, self.rng
        pivot = ur.Entity(parent=ch.detail, position=(x, Litter.PROFILE[kind][2], z), rotation=(0, rng.uniform(0, 360), 0))
        kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        if kind == "flyer":                                                     # manila folder
            mk(pivot, "cube", (0, 0, 0), (0.24, 0.012, 0.32), color=C(0.86, 0.74, 0.42), **kw)
        elif kind == "scrap":
            mk(pivot, "quad", (0, 0, 0), (0.1, 0.1, 1), color=C(0.9, 0.88, 0.8), rot=(90, 0, 0), double=True, **kw)
        else:
            mk(pivot, "quad", (0, 0, 0), (0.22, 0.3, 1), color=C(0.93, 0.93, 0.88), rot=(90, 0, 0), double=True, **kw)
        ch.litter.append(Litter(pivot, kind, x + ch.xoff, z, ch.hw, ch.xoff))

    # ---- branching junction chunks (biomes): partitions with doorways, wings, signs, an elevator
    def _junction(self) -> None:
        ch, rng, z0, H, lvl = self.ch, self.rng, self.ch.z0, self.ch.h, self.w.level
        base = KIND_HW[lvl]
        wallc = C(0.58, 0.58, 0.56)
        names = {"garage": ("P-WEST  BAYS", "P-EAST  BAYS"), "pool": ("CHANGING ROOMS", "SAUNA WING"), "office": ("ARCHIVES", "BREAK ROOM")}[lvl]
        for k, s in enumerate((-1, 1)):
            x = s * (base + 0.4)
            for (za, zb) in ((z0, z0 + 8.0), (z0 + 11.0, z0 + CH)):
                self.block(x, H / 2, (za + zb) / 2, 0.4, H, zb - za, "concrete", tile=((zb - za) / 4, H / 4), color=wallc)
            self.box(x, H - 0.45, z0 + 9.5, 0.4, 0.9, 3.0, "concrete", color=wallc)
            face = "+x" if s > 0 else "-x"
            self.quad(s * (base + 0.18), H - 1.2, z0 + 9.5, 3.0, 0.55, "-x" if s > 0 else "+x",
                      BANK.sign(("-> " if s > 0 else "<- ") + names[k], (255, 230, 140), (24, 20, 6), 512, 96), emissive=1.2, double=False)
        self.quad(0, H - 0.9, z0 + CH - 0.2, 8.0, 0.8, "-z", BANK.sign("<- LEFT     STRAIGHT ^     RIGHT ->", (255, 230, 140), (24, 20, 6), 768, 96),
                  emissive=1.1, double=False)
        es = rng.choice((-1, 1))
        self.elevator(es, z0 + 18.0)                                            # an exit lift hidden in one wing
        for s in (-1, 1):
            wx = s * (base + hw_mid(base))
            if lvl == "garage":
                for cz in (z0 + 3.5, z0 + 6.5):
                    if rng.random() < 0.8:
                        self.merged([(wx, 0.55, cz, 1.9, 0.7, 4.0)], "metal", color=rng.choice((C(0.5, 0.1, 0.1), C(0.1, 0.25, 0.45), C(0.6, 0.6, 0.62))))
                        self.merged([(wx, 1.15, cz, 1.7, 0.55, 2.2)], "glass", color=C(0.2, 0.25, 0.3))
                        self.solid(wx - 0.95, wx + 0.95, cz - 2.0, cz + 2.0, 1.3)
                if s != es:
                    for cz in (z0 + 15.0, z0 + 19.5):
                        self.merged([(wx, 0.55, cz, 1.9, 0.7, 4.0)], "metal", color=C(0.15, 0.4, 0.3))
                        self.solid(wx - 0.95, wx + 0.95, cz - 2.0, cz + 2.0, 1.3)
            elif lvl == "pool":
                self.lounge(wx, z0 + 4.0, 90 * -s, C(0.35, 0.8, 0.95))
                self.lounge(wx, z0 + 6.5, 90 * -s, C(1.0, 0.6, 0.75))
                self.plant(wx + s * 1.0, z0 + 21.5)
                self.ball(wx, 0.2, z0 + 14.0, 0.5, color=C(1.0, 0.85, 0.3), emissive=0.25)
            else:
                for cz in (z0 + 4.0, z0 + 7.0):
                    self.table(wx, cz, 2.0, 0.9, 0.74, "wood")
                    self.box(wx, 1.0, cz - 0.1, 0.5, 0.36, 0.05, color=C(0.05, 0.06, 0.08))
                    self.box(wx, 1.0, cz - 0.07, 0.44, 0.3, 0.02, color=C(0.3, 0.9, 0.7), emissive=0.9)
                    self.chair(wx, cz + 0.9, 180)
        for lz in (4, 12, 20):
            for s in (-1, 1):
                st = self.box(s * (base + 3.0), H - 0.05, z0 + lz, 1.2, 0.06, 0.5, color=C(1.0, 0.95, 0.75), emissive=1.5)
                self.strip(st, 1.5, 1.5)

    # ---- biome: the office
    def _build_office(self) -> None:
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        self._shell("carpet", "plaster", C(0.84, 0.80, 0.56), ceil_tex="plaster")
        for lz in (3, 9, 15, 21):
            for sx in (-8.0, -2.7, 2.7, 8.0):
                self.strip(self.box(sx, H - 0.04, z0 + lz, 1.6, 0.06, 0.6, color=C(1.0, 0.97, 0.8), emissive=1.5), 1.5, 1.3)
        self.par = ch.detail
        for zr in (6.0, 13.0, 20.0):
            for sx in (-8.5, -4.5, 4.5, 8.5):
                self.block(sx, 0.7, z0 + zr - 1.7, 3.0, 1.4, 0.1, "panel", color=C(0.55, 0.58, 0.52))
                self.block(sx - 1.5, 0.7, z0 + zr, 0.1, 1.4, 3.3, "panel", color=C(0.55, 0.58, 0.52))
                self.table(sx + 0.2, z0 + zr - 0.7, 1.6, 0.7, 0.74, "wood", C(0.7, 0.62, 0.5))
                self.box(sx + 0.2, 1.05, z0 + zr - 0.95, 0.5, 0.36, 0.05, color=C(0.05, 0.06, 0.08))
                self.box(sx + 0.2, 1.05, z0 + zr - 0.92, 0.44, 0.30, 0.02, color=C(0.3, 0.9, 0.7), emissive=0.8)
                if rng.random() < 0.6:
                    self.chair(sx + 0.2, z0 + zr + 0.3, 180, color=C(0.25, 0.28, 0.3))
        self.block(-hw + 0.8, 0.5, z0 + 22.5, 0.5, 1.0, 0.5, "metal", color=C(0.8, 0.85, 0.9))         # water cooler
        self.ball(-hw + 0.8, 1.2, z0 + 22.5, 0.22, color=C(0.3, 0.6, 1.0), emissive=0.6)
        for _ in range(5):
            self._paper(rng.uniform(-hw + 1, hw - 1), z0 + rng.uniform(1, 23), rng.choice(("paper", "flyer", "scrap")))
        self.ad(-hw + 0.3, 1.9, z0 + 12.0, 1.7, 2.2, "+x", f"{ch.idx}:off0")
        if ch.idx % 12 == 0:
            self.elevator(1, z0 + 12.0)

    # ---- rare mall room: documents archive + server room
    def _build_server(self) -> None:
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("rubber", "panel", C(0.52, 0.58, 0.66))
        for lz in (4, 12, 20):
            for sx in (-4.0, 0.0, 4.0):
                self.strip(self.box(sx, H - 0.04, z0 + lz, 0.45, 0.06, 3.0, color=C(0.8, 0.92, 1.0), emissive=1.5), 1.5, 2.2)
        self.par = ch.detail
        greens, reds = [], []
        for k in range(8):                                                       # server racks along the +x wall
            z = z0 + 2.0 + k * 2.7
            self.block(hw - 0.8, 1.1, z, 1.0, 2.2, 0.95, "metal", color=C(0.10, 0.11, 0.14))
            for j in range(9):
                (greens if rng.random() < 0.7 else reds).append((hw - 1.31, 0.3 + 0.2 * j, z + rng.uniform(-0.3, 0.3), 0.03, 0.03, 0.03))
        if greens:
            self.strip(self.merged(greens, "white", color=C(0.2, 1.0, 0.4), emissive=1.8), 1.8, 6.0)
        if reds:
            self.strip(self.merged(reds, "white", color=C(1.0, 0.2, 0.15), emissive=1.8), 1.8, 4.0)
        drawers = []
        for k in range(6):                                                       # archive cabinets along the -x wall
            z = z0 + 2.2 + k * 3.6
            self.block(-hw + 0.7, 0.95, z, 0.8, 1.9, 1.6, "metal", color=C(0.45, 0.48, 0.44))
            for j in range(4):
                drawers.append((-hw + 1.12, 0.35 + 0.45 * j, z, 0.04, 0.34, 1.3))
            if rng.random() < 0.5:
                self.box(-hw + 1.5, 0.8, z, 0.6, 0.3, 1.2, "metal", color=C(0.5, 0.52, 0.48))          # a drawer left hanging open
        self.merged(drawers, "metal", color=C(0.30, 0.33, 0.30))
        for tz in (7.0, 16.0):                                                   # reading tables
            self.table(-1.5, z0 + tz, 2.6, 1.2, 0.76, "wood", C(0.55, 0.5, 0.42))
        lore_ids = rng.sample(range(len(BOOK_LORE)), 4)
        slots = [(-hw + 0.7, 1.93, z0 + 2.2), (-hw + 0.7, 1.93, z0 + 9.4), (-hw + 0.7, 1.93, z0 + 16.6), (-1.5, 0.8, z0 + 7.0),
                 (-1.2, 0.8, z0 + 16.0), (-2.0, 0.8, z0 + 7.3), (-hw + 0.7, 1.93, z0 + 20.2), (-1.9, 0.8, z0 + 15.6), (hw - 0.8, 2.25, z0 + 5.0)]
        for i, (bx, by, bz) in enumerate(slots):
            if i < 4:
                bid, (title, lore) = f"lore{lore_ids[i]}", BOOK_LORE[lore_ids[i]]
            else:
                bid, title, lore = f"blank:{ch.idx}:{i}", "BLANK BOOK", ""
            self._book(bx, by, bz, rng.uniform(0, 180), bid, title, lore)
        wires = []                                                               # tangled wires across the floor
        for k in range(26):
            wx, wz = rng.uniform(-hw + 1.2, hw - 1.6), z0 + rng.uniform(0.8, CH - 0.8)
            L = rng.uniform(1.4, 3.6)
            e = self.box(wx, 0.03 + 0.012 * (k % 3), wz, 0.04, 0.035, L, color=rng.choice((C(0.05, 0.05, 0.06), C(0.5, 0.1, 0.1), C(0.1, 0.2, 0.5), C(0.7, 0.55, 0.1))),
                         rot=(0, rng.uniform(0, 180), 0))
            wires.append((e, e.rotation_y, rng.uniform(0, 6.28)))
        for _ in range(16):                                                      # paper & folders drifting on the floor
            self._paper(rng.uniform(-hw + 1.5, hw - 2.2), z0 + rng.uniform(1, CH - 1), rng.choice(("paper", "paper", "flyer", "scrap", "flyer")))
        # the freezing AC unit: bolted high on the -x wall (above the archive cabinets), so the whole walkway stays clear.
        # It is a plain box with no collider - nothing at floor level to bump into - and the cold air drifts off the wall.
        ax, az = -hw + 0.45, z0 + CH - 2.6
        self.box(ax, 2.9, az, 0.8, 0.9, 3.2, "metal", color=C(0.82, 0.86, 0.9))
        for k in range(6):
            self.box(ax + 0.42, 2.55, az - 1.2 + 0.48 * k, 0.04, 0.06, 0.34, color=C(0.1, 0.15, 0.2))
        self.box(ax + 0.42, 3.0, az + 1.3, 0.04, 0.12, 0.12, color=C(0.3, 0.9, 1.0), emissive=1.6)
        self.quad(ax + 0.43, 2.95, az - 0.3, 1.6, 0.3, "+x", BANK.sign("-18 C", (170, 240, 255), (4, 20, 30), 256, 64), emissive=1.0, double=False)
        frost = []
        for k in range(9):
            zb = az + rng.uniform(-1.4, 1.4)
            e = self.quad(ax + 0.6 + rng.uniform(0, 0.5), 2.0, zb, 1.4, 1.4, "+x", "white", alpha=True, emissive=0.5, color=C(0.85, 0.95, 1.0, 0.16))
            frost.append((e, rng.uniform(0, 1.0), zb))
        ch.extra["server"] = True
        ch.extra["cold"], ch.extra["sneeze_t"], ch.extra["rat_t"], ch.extra["rat"] = 0.0, 0.0, 5.0, None
        tbook = [0.0]

        def tick(dt, g):
            t = g.clock
            for e, base, ph in wires:
                e.rotation_y = base + math.sin(t * 0.9 + ph) * 2.6
                e.y = 0.03 + 0.006 * math.sin(t * 1.7 + ph)
            for e, ph, z_ in frost:
                k = (t * 0.22 + ph) % 1.0
                e.y = 2.4 - 2.2 * k
                e.z = z_ + math.sin(t * 0.5 + ph * 6) * 0.25
                e.color = C(0.85, 0.95, 1.0, 0.20 * math.sin(math.pi * k))
            tbook[0] -= dt
            if tbook[0] <= 0:                                                    # AC wind nudges the litter
                tbook[0] = random.uniform(1.8, 3.4)
                if ch.litter:
                    lit = random.choice(ch.litter)
                    lit.vx += random.uniform(0.3, 1.0)               # the unit blows out from the -x wall
                    lit.vz += random.uniform(-0.5, 0.5)
                    lit.vy = 0.5
                    lit.awake = True
            p = g.player
            here = g.world.at(p.z) is ch
            ex = ch.extra
            ex["cold"] = ex["cold"] + dt if here else max(0.0, ex["cold"] - dt * 2.0)
            if here and ex["cold"] > 32.0 and g.clock >= ex["sneeze_t"]:
                ex["sneeze_t"] = g.clock + random.uniform(18.0, 28.0)
                who = SimpleNamespace(name="you", x=p.x, z=p.z)
                if random.random() < 0.5:
                    g.sfx.play("sneeze", 0.9)
                    g.hud.speak(who, "...ah... AH-CHOO!")
                    g.award("cold")
                else:
                    g.hud.speak(who, random.choice(["It's freezing in here. Who thought this was a good idea for paper?", "My fingers have stopped replying.",
                                                    "Okay. This AC is personal. It's targeting me."]))
            if here:                                                             # rare rat: 1 in 200 per 5 s
                ex["rat_t"] -= dt
                if ex["rat_t"] <= 0:
                    ex["rat_t"] = 5.0
                    if (ex["rat"] is None or ex["rat"] not in ch.npcs) and random.randrange(200) == 0:
                        for _ in range(12):
                            rx, rz = ch.xoff + random.uniform(-hw + 2, hw - 2.5), ch.z0 + random.uniform(2, CH - 2)
                            if all(not (c[0] - 0.4 < rx < c[1] + 0.4 and c[2] - 0.4 < rz < c[3] + 0.4 and c[4] > 0.2) for c in ch.colliders):
                                ex["rat"] = RatActor(g, ch, rx, rz)
                                ch.npcs.append(ex["rat"])
                                break
            return True
        ch.tickers.append(tick)

    # ---- weather windows: weather exists only out there, behind the glass
    def _weather_window(self, cx: float, cy: float, w: float, h: float, mode: str, z: float, hw: float) -> None:
        ch, rng = self.ch, self.rng
        bx = hw + 3.4
        back = {"rain": C(0.10, 0.14, 0.22), "fog": C(0.55, 0.60, 0.66), "dust": C(0.015, 0.01, 0.06)}[mode]
        self.quad(bx, cy, z, w + 3.0, h + 1.6, "-x", "white", color=back, emissive=0.35 if mode != "fog" else 0.7, double=False)
        parts = []
        if mode == "rain":
            for _ in range(20):
                e = self.box(hw + 1.0, cy, z, 0.012, 0.55, 0.012, color=C(0.65, 0.82, 1.0, 0.8), emissive=0.9)
                parts.append([e, rng.uniform(0.2, 3.0), rng.uniform(-w / 2, w / 2), rng.uniform(cy - h / 2, cy + h / 2), rng.uniform(7, 11)])
        elif mode == "fog":
            for _ in range(5):
                e = self.quad(hw + 1.0, cy, z, w * 0.9, h * 1.1, "-x", "white", alpha=True, emissive=0.9, color=C(0.88, 0.92, 0.96, 0.25))
                parts.append([e, rng.uniform(0.3, 3.0), rng.uniform(-1.2, 1.2), 0.0, rng.uniform(0.15, 0.3)])
        else:
            hues = ((0.7, 0.4, 1.0), (0.3, 0.9, 1.0), (1.0, 0.8, 0.4), (1.0, 0.4, 0.8))
            for _ in range(18):
                e = self.ball(hw + 1.5, cy, z, 0.04, color=C(*rng.choice(hues)), emissive=1.5)
                parts.append([e, rng.uniform(0, 6.28), rng.uniform(-w / 2, w / 2), rng.uniform(cy - h / 2, cy + h / 2), rng.uniform(0.3, 0.8)])
            for _ in range(7):
                self.ball(bx - 0.05, rng.uniform(cy - h / 2, cy + h / 2), z + rng.uniform(-w / 2, w / 2), 0.06, color=C(1, 1, 1), emissive=2.0)

        def tick(dt, g):
            t = g.clock
            if mode == "rain":
                for p_ in parts:
                    p_[3] -= p_[4] * dt
                    if p_[3] < cy - h / 2:
                        p_[3], p_[2], p_[1] = cy + h / 2, rng.uniform(-w / 2, w / 2), rng.uniform(0.2, 3.0)
                    p_[0].position = (hw + p_[1], p_[3], z + p_[2])
            elif mode == "fog":
                for p_ in parts:
                    p_[2] += p_[4] * dt
                    if p_[2] > w * 0.6:
                        p_[2] = -w * 0.6
                    e = p_[0]
                    e.position = (hw + p_[1] + math.sin(t * 0.2 + p_[1]) * 0.3, cy, z + p_[2])
                    e.color = C(0.88, 0.92, 0.96, 0.28 * (1 - abs(p_[2]) / (w * 0.62)))
            else:
                for p_ in parts:
                    p_[0].position = (hw + 0.8 + 1.4 * math.sin(t * p_[4] + p_[1]) + 1.0, p_[3] + 0.3 * math.sin(t * 0.6 + p_[1]),
                                      z + p_[2] + 0.5 * math.cos(t * p_[4] * 0.7 + p_[1]))
            return True
        ch.tickers.append(tick)

    def _build_lookout(self) -> None:
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("terrazzo", "plaster", C(0.74, 0.76, 0.86), walls=(-1,))
        self.block(hw + 0.25, 0.45, zc, 0.5, 0.9, CH, "concrete", color=C(0.55, 0.57, 0.62))
        self.box(hw + 0.25, H - 0.55, zc, 0.5, 1.1, CH, "concrete", color=C(0.55, 0.57, 0.62))
        self.solid(hw, hw + 1.0, z0 - 0.1, z0 + CH + 0.1)
        for pz, pw in ((0.6, 1.2), (8.0, 2.4), (16.0, 2.4), (23.4, 1.2)):
            self.box(hw + 0.25, H / 2, z0 + pz, 0.5, H, pw, "concrete", color=C(0.55, 0.57, 0.62))
        self.strip(self.box(0, H - 0.05, zc, 0.3, 0.05, CH - 2.0, color=C(0.9, 0.93, 1.0), emissive=1.2), 1.2, 0.6)
        self.par = ch.detail
        wins = []
        for z in (z0 + 4.0, z0 + 12.0, z0 + 20.0):
            mode = rng.choice(("rain", "fog", "dust"))
            wins.append(dict(mode=mode, z=z))
            self.quad(hw - 0.05, 2.7, z, 5.6, 3.6, "-x", "glass", alpha=True, color=C(0.75, 0.88, 1.0, 0.16))
            self._weather_window(hw, 2.7, 5.6, 3.6, mode, z, hw)
            self.box(hw - 0.6, 0.45, z, 0.5, 0.08, 1.8, "wood")
            self.box(hw - 0.6, 0.22, z, 0.4, 0.44, 1.6, "metal", color=C(0.2, 0.2, 0.22))
            self.solid(hw - 0.9, hw - 0.3, z - 0.9, z + 0.9, 0.5)
            self.inter("sit", hw - 0.6, 0.5, z, 1.7, "Sit and watch the window", 90.0)
        ch.extra["windows"] = wins
        self.ad(-(hw - 0.22), 2.4, z0 + 12.0, 1.9, 2.5, "+x", f"{ch.idx}:lk0")
        self.plant(-hw + 1.0, z0 + 3.0)
        self.plant(-hw + 1.0, z0 + 21.0)

    # ---- the swimmable pool (poolrooms) and its drain
    def _build_pool_deep(self) -> None:
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc, xo = z0 + CH / 2, self.ch.xoff
        W2 = hw + 0.5
        px0, px1, pz0, pz1 = -6.0, 6.0, z0 + 4.0, z0 + 20.0
        self._shell("tile", "tile", C(0.95, 1.0, 1.0), floor=False)
        self.box(0, -0.2, z0 + 2.0, 2 * W2, 0.4, 4.0, "tile", tile=(8, 1))
        self.box(0, -0.2, z0 + 22.0, 2 * W2, 0.4, 4.0, "tile", tile=(8, 1))
        for s in (-1, 1):
            self.box(s * (W2 + 6.0) / 2, -0.2, zc, W2 - 6.0, 0.4, 16.0, "tile", tile=(3, 4))
        self.box(0, -3.8, zc, 12.4, 0.4, 16.4, "tile", tile=(4, 5), color=C(0.20, 0.45, 0.50))
        for s in (-1, 1):                                                        # pool walls + the shallow ledge
            self.box(s * 6.1, -1.9, zc, 0.2, 3.8, 16.4, "tile", tile=(5, 2), color=C(0.55, 0.85, 0.9))
            self.box(s * 5.4, -0.5, zc, 1.4, 0.1, 16.0, "tile", color=C(0.7, 0.95, 1.0))
            self.box(0, -1.9, z0 + 4.0 + s * 8.0 + 8.0 * (1 if s > 0 else 0) * 0, 12.4, 3.8, 0.2, "tile", tile=(5, 2), color=C(0.55, 0.85, 0.9)) if False else None
        self.box(0, -1.9, pz0, 12.4, 3.8, 0.2, "tile", tile=(5, 2), color=C(0.55, 0.85, 0.9))
        self.box(0, -1.9, pz1, 12.4, 3.8, 0.2, "tile", tile=(5, 2), color=C(0.55, 0.85, 0.9))
        self.box(0, -0.5, pz0 + 0.7, 10.0, 0.1, 1.4, "tile", color=C(0.7, 0.95, 1.0))
        self.box(0, -0.5, pz1 - 0.7, 10.0, 0.1, 1.4, "tile", color=C(0.7, 0.95, 1.0))

        def floor_fn(x, z, xo=xo):
            lx = x - xo
            if px0 <= lx <= px1 and pz0 <= z <= pz1:
                return -0.45 if min(lx - px0, px1 - lx, z - pz0, pz1 - z) < 1.2 else -3.6
            return 0.0
        ch.floor_fn = floor_fn
        water = mk(self.par, "quad", (0, -0.28, zc), (12, 16, 1), "glass", (4, 5), color=C(0.15, 0.62, 0.85, 0.55), rot=(90, 0, 0), double=True, alpha=True,
                   lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        mk(self.par, "quad", (0, -3.59, zc), (1.5, 1.5, 1), color=C(0.01, 0.02, 0.02), rot=(90, 0, 0), double=True, lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        for a in (0, 45, 90, 135):
            self.box(0, -3.57, zc, 1.5, 0.02, 0.05, "metal", color=C(0.25, 0.28, 0.3), rot=(0, a, 0))
        glow = self.ball(0, -3.45, zc, 0.07, color=C(0.3, 1.0, 0.9), emissive=1.6)
        ch.extra["swim"] = dict(x0=px0 + xo, x1=px1 + xo, z0=pz0, z1=pz1, surface=-0.28, floor=-3.6, drain=(xo, zc), water=water)
        for lz in (4, 12, 20):
            for sx in (-6, 0, 6):
                self.strip(self.box(sx, H - 0.05, z0 + lz, 1.2, 0.06, 1.2, color=C(1.0, 1.0, 0.95), emissive=1.3), 1.3, 0.7)
        self.par = ch.detail
        for sx in (-8.5, 8.5):
            for pz in (6.0, 18.0):
                self.block(sx, H / 2, z0 + pz, 0.9, H, 0.9, "tile", tile=(1, 2), color=C(0.95, 1.0, 1.0))
        self.block(7.6, 0.6, z0 + 3.0, 0.12, 1.2, 1.3, "metal", color=C(0.7, 0.72, 0.75))
        self.quad(7.52, 0.75, z0 + 3.0, 1.2, 0.5, "-x", BANK.sign("DEEP END", (255, 255, 255), (10, 60, 90), 256, 96, "DRAIN 4 - DO NOT DIVE"), emissive=0.8, double=False)
        t0 = [0.0]

        def tick(dt, g):
            t0[0] += dt
            glow.scale = 0.14 + 0.05 * math.sin(g.clock * 2.0)
            water.color = C(0.15, 0.62, 0.85, 0.50 + 0.06 * math.sin(g.clock * 0.9))
            return True
        ch.tickers.append(tick)
        if ch.idx % 12 == 0:
            self.elevator(1, z0 + 12.0)

    def _build_drain(self) -> None:
        ch, z0, rng, hw, H = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("tile", "tile", C(0.35, 0.5, 0.5))
        self.par = ch.detail
        for lz in (5, 14):
            st = self.box(0, H - 0.05, z0 + lz, 0.4, 0.05, 2.4, color=C(0.5, 1.0, 0.9), emissive=0.9)
            self.strip(st, 0.9, 6.0)
        for s in (-1, 1):                                                        # pipes
            self.box(s * (hw - 0.4), H - 0.5, zc, 0.35, 0.35, CH - 1.0, "metal", color=C(0.22, 0.28, 0.28))
            for lz in (3, 9, 15, 21):
                self.box(s * (hw - 0.4), H / 2, z0 + lz, 0.2, H, 0.2, "metal", color=C(0.2, 0.26, 0.26))
        for _ in range(7):
            px, pz = rng.uniform(-hw + 1.5, hw - 1.5), z0 + rng.uniform(2, 22)
            mk(self.par, "circle", (px, 0.015, pz), (rng.uniform(0.6, 1.8), rng.uniform(0.6, 1.8), 1), "glass", color=C(0.15, 0.4, 0.4, 0.55), rot=(90, 0, 0),
               double=True, alpha=True, lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        self.quad(0, 2.2, z0 + CH - 0.3, 4.6, 0.9, "-z", BANK.sign("YOU WERE NOT MEANT TO COME UP THIS WAY", (140, 255, 230), (4, 14, 14), 768, 96), emissive=0.8, double=False)
        self.quad(-hw + 0.2, 1.4, z0 + 9.0, 1.3, 0.9, "+x", BANK.sign("it keeps draining", (110, 200, 190), (4, 12, 12), 384, 96, "the level does not change"), emissive=0.5, double=False)
        self.block(0, 0.4, z0 + 19.0, 1.4, 0.8, 1.4, "tile", color=C(0.4, 0.55, 0.55))              # pedestal
        if not self.w.g.inv.flags.get("trophy_taken"):
            pv = ur.Entity(parent=ch.detail, position=(0, 0.8, z0 + 19.0))
            kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            gold = C(1.0, 0.8, 0.25)
            mk(pv, "cube", (0, 0.03, 0), (0.3, 0.06, 0.3), color=C(0.25, 0.18, 0.08), **kw)
            mk(pv, "cube", (0, 0.2, 0), (0.07, 0.3, 0.07), color=gold, emissive=0.7, **kw)
            mk(pv, "sphere", (0, 0.42, 0), (0.30, 0.26, 0.30), color=gold, emissive=0.9, **kw)
            for sx in (-0.17, 0.17):
                mk(pv, "sphere", (sx, 0.42, 0), (0.1, 0.14, 0.04), color=gold, emissive=0.7, **kw)
            self.inter("trophy", 0, 0.0, z0 + 19.0, 1.7, "Take the secret trophy", 0.0, dict(ent=pv))
        self.box(hw - 0.35, 1.3, z0 + 3.0, 0.08, 2.6, 0.1, "metal", color=C(0.5, 0.52, 0.52))      # ladder
        self.box(hw - 0.35, 1.3, z0 + 3.6, 0.08, 2.6, 0.1, "metal", color=C(0.5, 0.52, 0.52))
        for k in range(8):
            self.box(hw - 0.35, 0.3 + 0.3 * k, z0 + 3.3, 0.08, 0.05, 0.6, "metal", color=C(0.5, 0.52, 0.52))
        self.inter("ladder", hw - 1.3, 0.0, z0 + 3.3, 1.6, "Climb the ladder back up", 90.0, dict())
        nx = [random.uniform(8, 14)]

        def tick(dt, g):
            nx[0] -= dt
            if nx[0] <= 0:
                nx[0] = random.uniform(8, 16)
                g.sfx.play(random.choice(("whisper", "thud", "creak")), 0.5, pan=random.uniform(-0.9, 0.9), wet=True)
            return True
        ch.tickers.append(tick)

    # ---- the Backrooms (fixed layout)
    def _build_backrooms(self) -> None:
        ch, z0, rng, hw, H, g = self.ch, self.ch.z0, self.rng, self.ch.hw, self.ch.h, self.w.g
        idx = ch.idx
        r0 = idx * BR_ROWS_PER_CHUNK
        self._shell("carpet", "plaster", C(0.92, 0.84, 0.44), ceil_tex="plaster")
        boxes, WT = [], 0.3
        for c in range(1, BR_COLS):
            r = 0
            while r < BR_ROWS_PER_CHUNK:
                if BR_EAST[r0 + r][c - 1]:
                    rs = r
                    while r + 1 < BR_ROWS_PER_CHUNK and BR_EAST[r0 + r + 1][c - 1]:
                        r += 1
                    za, zb, x = z0 + 2 * rs, z0 + 2 * (r + 1), -12.0 + 2 * c
                    boxes.append((x, H / 2, (za + zb) / 2, WT, H, zb - za + WT))
                    self.solid(x - WT / 2, x + WT / 2, za - WT / 2, zb + WT / 2)
                r += 1
        for b in range(0 if r0 > 0 else 1, BR_ROWS_PER_CHUNK):
            rr, c = r0 + b - 1, 0
            while c < BR_COLS:
                if BR_NORTH[rr][c]:
                    cs = c
                    while c + 1 < BR_COLS and BR_NORTH[rr][c + 1]:
                        c += 1
                    xa, xb, z = -12.0 + 2 * cs, -12.0 + 2 * (c + 1), z0 + 2.0 * b
                    boxes.append(((xa + xb) / 2, H / 2, z, xb - xa + WT, H, WT))
                    self.solid(xa - WT / 2, xb + WT / 2, z - WT / 2, z + WT / 2)
                c += 1
        if idx == 0:
            self.block(0, H / 2, z0 - 0.15, 2 * hw + 1, H, 0.3, "plaster", color=C(0.9, 0.82, 0.42))
        if idx == FIXED_LEVELS["backrooms"] - 1:
            self.block(0, H / 2, z0 + CH + 0.15, 2 * hw + 1, H, 0.3, "plaster", color=C(0.9, 0.82, 0.42))
        if boxes:
            self.merged(boxes, "plaster", tile=(2, 1), color=C(0.93, 0.85, 0.44))
        for rl in (1, 4, 7, 10):
            for cl in (2, 6, 10):
                x, z = br_cell_xz(rl, cl)[0], z0 + 2.0 * rl + 1.0
                self.strip(self.box(x, H - 0.05, z, 1.6, 0.06, 0.5, color=C(1.0, 0.97, 0.7), emissive=1.5), 1.5, 2.0)
        self.par = ch.detail
        if idx == 0:
            self.quad(0, 2.0, z0 + 7.6, 4.0, 0.7, "-z", BANK.sign("THE LIFT IS GONE", (60, 40, 10), (230, 210, 120), 512, 96), emissive=0.5, double=False)
        if idx == 4:                                                             # the cafe that should not be open
            self.block(-2.0, 0.55, z0 + 7.2, 8.0, 1.1, 1.0, "wood", color=C(0.7, 0.5, 0.35))
            self.box(-2.0, 1.13, z0 + 7.2, 8.2, 0.06, 1.2, "marble", color=C(0.95, 0.9, 0.8))
            self.quad(-2.0, 2.7, z0 + 11.7, 4.4, 0.8, "-z", BANK.sign("CAFE  -  OPEN", (255, 235, 190), (60, 25, 10), 640, 112, "water | donuts | miracles"), emissive=1.2, double=False)
            self.ball(-5.0, 1.3, z0 + 7.2, 0.14, color=C(1.0, 0.8, 0.5), emissive=1.5)
            for tx, tz in ((-6.0, 3.0), (2.0, 3.0)):
                self.table(tx, z0 + tz, 0.9, 0.9, 0.75, "wood")
                self.chair(tx - 0.85, z0 + tz, FACE_YAW["+x"])
                self.chair(tx + 0.85, z0 + tz, FACE_YAW["-x"])
            ch.npcs.append(CafeVendor(g, ch, (ch.xoff - 2.0, z0 + 8.5), rng))
            ch.npcs[-1].yaw = 180.0
        if idx == FIXED_LEVELS["backrooms"] - 1:
            self.elevator(1, z0 + 16.0)
            self.strip(self.box(8.0, H - 0.06, z0 + 16.0, 3.0, 0.06, 0.4, color=C(0.3, 1.0, 0.5), emissive=1.8), 1.8, 3.0)
        for k, (r, c) in enumerate(BR_TRAPPED_CELLS):
            if r // BR_ROWS_PER_CHUNK == idx:
                x, z = br_cell_xz(r, c)[0], z0 + 2.0 * (r % BR_ROWS_PER_CHUNK) + 1.0
                self.ball(x, 0.06, z, 0.09, color=C(1.0, 0.85, 0.5), emissive=1.8)
                npc = TrappedNpc(g, ch, (ch.xoff + x, z), rng, k)
                ch.npcs.append(npc)
        for k, (r, c) in enumerate(BR_COIN_CELLS):
            if r // BR_ROWS_PER_CHUNK == idx and f"br:{k}" not in g.br_coins:
                x, z = br_cell_xz(r, c)[0], z0 + 2.0 * (r % BR_ROWS_PER_CHUNK) + 1.0
                pv = ur.Entity(parent=ch.detail, position=(x, 0, z))
                kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
                for j in range(5):
                    mk(pv, "cube", (rng.uniform(-0.12, 0.12), 0.02 + 0.02 * j, rng.uniform(-0.12, 0.12)), (0.09, 0.015, 0.09), color=C(1.0, 0.82, 0.2), emissive=0.9, **kw)
                self.inter("coinpile", x, 0.0, z, 1.4, "Pick up the coins", 0.0, dict(key=f"br:{k}", ent=pv))

    # ---- fountain with a working water system (hooked after the original fountain())
    def _fountain_water(self, cx: float, cz: float, rim_y0: float) -> None:
        ch = self.ch
        rng = random.Random(stable_hash(self.seed, ch.idx, "fw"))
        N, y0, wl, G = 26, rim_y0 + 2.2, rim_y0 + 0.25, 9.8
        drops = []
        for i in range(N):
            vy, hv = rng.uniform(2.6, 3.6), rng.uniform(0.9, 1.9)
            T = (vy + math.sqrt(vy * vy + 2 * G * (y0 - wl))) / G
            e = mk(self.par, "sphere", (cx, y0, cz), (0.07, 0.07, 0.07), color=C(0.7, 0.9, 1.0, 0.85), emissive=0.8, alpha=True, lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            drops.append([e, 2 * math.pi * i / N + rng.uniform(-0.1, 0.1), vy, hv, T, rng.uniform(0, T), False])
        rings = []
        for _ in range(8):
            e = mk(self.par, "circle", (cx, wl + 0.02, cz), (0.1, 0.1, 1), color=C(0.8, 0.95, 1.0, 0.0), rot=(90, 0, 0), double=True, alpha=True,
                   lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
            rings.append([e, 9.0, 0.0, 0.0])                                   # ent, age (inactive when >= 1.6), x, z
        xo, nxt = ch.xoff, [0]

        def ripple(wx, wz):
            r = rings[nxt[0] % len(rings)]
            nxt[0] += 1
            r[1], r[2], r[3] = 0.0, wx - xo, wz

        splash_t = [0.0]

        def tick(dt, g):
            for d in drops:
                d[5] = (d[5] + dt) % d[4]
                t = d[5]
                r = d[3] * t
                d[0].position = (cx + math.cos(d[1]) * r, y0 + d[2] * t - 0.5 * G * t * t, cz + math.sin(d[1]) * r)
                if t < dt * 1.5 and d[6] and random.random() < 0.12:
                    ripple(cx + xo + math.cos(d[1]) * d[3] * d[4], cz + math.sin(d[1]) * d[3] * d[4])
                d[6] = True
            for rg in rings:
                if rg[1] < 1.6:
                    rg[1] += dt
                    rg[0].position = (rg[2], wl + 0.02, rg[3])
                    s = 0.2 + rg[1] * 1.3
                    rg[0].scale = (s, s, 1)
                    rg[0].color = C(0.8, 0.95, 1.0, 0.35 * (1 - rg[1] / 1.6))
                else:
                    rg[0].color = C(0.8, 0.95, 1.0, 0.0)
            splash_t[0] -= dt
            if splash_t[0] <= 0:
                splash_t[0] = random.uniform(0.5, 1.1)
                if math.hypot(cx + xo - g.player.x, cz - g.player.z) < 16:
                    g.sfx.play("splash", 0.14, pan=max(-1, min(1, (cx + xo - g.player.x) / 10.0)), wet=True)
            return True
        ch.tickers.append(tick)
        ch.extra["fountain"] = (cx + xo, wl, cz)
        ch.extra["water"] = dict(ripple=ripple)

    # ---- per-chunk extras bolted onto existing zones
    def _v3_features(self) -> None:
        ch, g, lvl, hw, z0 = self.ch, self.w.g, self.w.level, self.ch.hw, self.ch.z0
        rng = seeded(self.seed, "v3feat", lvl, ch.idx)
        if lvl == "mall" and ch.kind == "corridor" and abs(ch.idx) > 1 and rng.random() < 0.07:          # Lost & Found archive desk
            s, z = rng.choice((-1, 1)), z0 + rng.choice((2.0, 6.0, 10.0, 14.0, 18.0, 22.0))
            if self._clear(s * 3.4, z, 1.3):
                face = "-x" if s > 0 else "+x"
                self.block(s * 3.4, 0.55, z, 0.9, 1.1, 2.2, "wood", color=C(0.6, 0.45, 0.3))
                self.box(s * 3.4, 1.13, z, 1.0, 0.05, 2.3, "marble", color=C(0.85, 0.82, 0.78))
                self.ball(s * 3.4, 1.3, z - 0.7, 0.07, color=C(0.4, 1.0, 0.6), emissive=1.6)
                self.quad(s * 2.93, 0.8, z, 1.6, 0.3, face, BANK.sign("LOST & FOUND", (200, 255, 220), (10, 40, 24), 384, 96), emissive=0.9, double=False)
                for k in range(4):
                    self.box(s * 3.35, 1.18 + 0.02 * k, z + rng.uniform(-0.7, 0.7), 0.25, 0.02, 0.18, color=C(0.9, 0.9, 0.84))
                self.inter("archive", s * 2.6, 0.0, z, 1.8, "Examine the Lost & Found log", FACE_YAW[face], dict(tag=f"{ch.idx}:{z:.0f}"))
        if ch.kind == "restaurant" and lvl == "mall":                                                  # kitchen pass for the waiter
            self.block(4.5, 0.55, z0 + 23.2, 3.2, 1.1, 0.9, "panel", color=C(0.55, 0.3, 0.25))
            self.box(4.5, 1.13, z0 + 23.2, 3.3, 0.05, 1.0, "marble", color=C(0.9, 0.85, 0.8))
            self.quad(4.5, 2.2, z0 + 23.7, 2.2, 0.5, "-z", BANK.sign("KITCHEN", (255, 220, 170), (40, 14, 10), 256, 72), emissive=1.1, double=False)
            self.ball(4.5, 1.5, z0 + 23.2, 0.12, color=C(1.0, 0.7, 0.35), emissive=1.8)
        if ch.kind == "gym" and lvl == "mall":                                                          # lifting stations
            for k, (wx, wz, label) in enumerate(((-3.0, z0 + 17.2, "Lift dumbbells"), (1.6, z0 + 7.0, "Bench press reps"))):
                if k == 0:
                    self.block(-3.0, 0.45, z0 + 18.4, 1.6, 0.9, 0.5, "metal", color=C(0.2, 0.2, 0.26))
                    for j in range(6):
                        self.box(-3.6 + 0.24 * j, 1.0, z0 + 18.4, 0.1, 0.16, 0.16, color=C(0.07, 0.07, 0.09), emissive=0.15)
                self.inter("weights", wx, 0.0, wz, 1.5, label, 180.0, dict(station=k))
        if ch.kind == "cinema" and ch.part == 1 and lvl == "mall":
            ch.extra["cinema_screen"] = True

    def _v3_npcs(self) -> None:
        ch, g, lvl, hw, z0, xo, k = self.ch, self.w.g, self.w.level, self.ch.hw, self.ch.z0, self.ch.xoff, self.ch.kind
        rng = seeded(self.seed, "v3npc", lvl, ch.idx)
        if lvl == "mall" and abs(ch.idx) > 1:
            if k == "corridor":
                r = rng.random()
                if r < 0.012 or r > 0.992:
                    spot = None
                    for _ in range(8):
                        x, z = rng.uniform(-3.0, 3.0), z0 + rng.uniform(3, 21)
                        if self._clear(x, z, 0.7):
                            spot = (xo + x, z)
                            break
                    if spot:
                        ch.npcs.append(RareNpc(g, ch, spot, rng) if r < 0.012 else CraftsmanNpc(g, ch, spot, rng))
            elif k == "atrium" and ch.extra.get("fountain") and rng.random() < 0.12:
                fx, fy, fz = ch.extra["fountain"]
                ch.npcs.append(CoinerNpc(g, ch, (fx, fz + 4.05), rng, (fx, fy, fz)))
            elif k == "restaurant":
                ch.npcs.append(ServantNpc(g, ch, (xo + 4.5, z0 + 21.8), rng))
            elif k == "lookout":
                ch.npcs.append(LookoutNpc(g, ch, (xo + hw - 2.4, z0 + 8.0), rng))
        elif lvl == "garage" and abs(ch.idx) > 2 and seeded(self.seed, "nomad", ch.idx).random() < 1.0 / 20.0:
            sx, sz = rng.choice((-1, 1)) * 2.6, z0 + rng.uniform(4, 10)
            self.box(sx, 0.17, sz, 0.8, 0.34, 0.8, color=C(0.62, 0.48, 0.30))
            self.box(sx + 0.3, 0.02, sz + 0.5, 0.5, 0.02, 0.4, color=C(0.58, 0.44, 0.27), rot=(0, 20, 0))
            ch.npcs.append(NomadNpc(g, ch, (xo + sx, sz), rng))


def hw_mid(base: float) -> float:
    return (JUNCTION_HW - base) / 2.0


def _b_build(self, idx: int):
    """ChunkBuilder.build with the v3 hooks: fixed levels, deep pool, junctions, extra features and people."""
    kind, part, xo, hw, h = self.w.geom(idx)
    lvl = self.w.level
    ch = Chunk(idx, kind, part, xo, hw, h)
    ch.root = ur.Entity(position=(xo, 0, 0))
    ch.detail = ur.Entity(parent=ch.root)
    self.ch, self.par = ch, ch.root
    self.rng = seeded(self.seed, "chunk", lvl, idx)
    if lvl in FIXED_LEVELS and not (0 <= idx < FIXED_LEVELS[lvl]):
        self._solid_void()
        return ch
    self._jog()
    name = "_build_" + kind
    if lvl == "pool" and idx == DEEP_IDX:
        name = "_build_pool_deep"
    getattr(self, name)()
    if lvl in BIOMES and hw > KIND_HW[lvl] + 0.1:
        self._junction()
    self.par = ch.detail
    if lvl not in FIXED_LEVELS:
        self._features()
        try:
            self._v3_features()
        except Exception:
            log.exception("v3 features failed for chunk %s", idx)
        self._litter()
        self.rng = seeded(self.seed, "npc", lvl, idx)
        self._npcs()
        try:
            self._v3_npcs()
        except Exception:
            log.exception("v3 npcs failed for chunk %s", idx)
    ch.occ = [c for c in ch.colliders if c[4] >= 2.5 and (c[1] - c[0]) >= 0.5 and (c[3] - c[2]) >= 0.5]
    return ch


class GameV3:
    """Methods bolted onto Game: badges, held items, activities, swimming, elevators, books, cinema video."""

    def _v3_init(self) -> None:
        self.activity, self.held, self.hold_ent, self.hug_t = None, None, None, 0.0
        self.last_kart, self.noise_pulse, self.underwater = None, 0.0, False
        self.books = BookStore()
        self.br_coins, self.br_heard = set(), set()
        self.pool_ret, self.swim, self.ratwhisper = None, None, False
        self.video, self.vid_paths, self.vid_checked = None, [], False
        self._tick_err: set = set()

    # ---- badges
    def award(self, bid: str) -> None:
        if bid in self.inv.badges or bid not in BADGES:
            return
        self.inv.badges.add(bid)
        self.inv.dirty = True
        self.sfx.play("chime", 0.8)
        self.hud.note(f"BADGE UNLOCKED:  {BADGES[bid][0]}   ({len(self.inv.badges)}/{len(BADGES)})", 4.0)

    def cinema_complete(self) -> None:
        self.inv.flags["cinema_all"] = True
        self.inv.dirty = True
        self.award("matinee")
        self.hud.note("Everything bought. Somewhere down the corridor, a screen flickers awake.", 5.0)

    # ---- held items
    def hold(self, key: str) -> None:
        if self.inv.n(key) <= 0:
            return
        self.drop_held(quiet=True)
        self.held = key
        it = ITEMS[key]
        col = C(*[c / 255 for c in it["col"]])
        root = ur.Entity(parent=ur.camera, position=(0.30, -0.26, 0.70))
        if key == "plush":
            body = (190, 130, 90)
            for pos, sc, c in (((0, 0, 0), (0.16, 0.18, 0.12), col), ((0, 0.14, 0), (0.12, 0.12, 0.10), col),
                               ((-0.05, 0.21, 0), (0.05, 0.05, 0.03), col), ((0.05, 0.21, 0), (0.05, 0.05, 0.03), col),
                               ((-0.09, 0.02, 0), (0.06, 0.12, 0.06), col), ((0.09, 0.02, 0), (0.06, 0.12, 0.06), col)):
                ur.Entity(parent=root, model="sphere", position=pos, scale=sc, color=c)
        elif key == "trophy":
            gold = C(1.0, 0.8, 0.25)
            ur.Entity(parent=root, model="cube", position=(0, -0.08, 0), scale=(0.1, 0.03, 0.1), color=C(0.25, 0.18, 0.08))
            ur.Entity(parent=root, model="cube", position=(0, 0.0, 0), scale=(0.025, 0.12, 0.025), color=gold)
            ur.Entity(parent=root, model="sphere", position=(0, 0.1, 0), scale=(0.11, 0.1, 0.11), color=gold)
        elif it["kind"] == "drink":
            ur.Entity(parent=root, model="cube", scale=(0.06, 0.14, 0.06), color=col)
            ur.Entity(parent=root, model="cube", position=(0, 0.075, 0), scale=(0.05, 0.015, 0.05), color=C(0.8, 0.8, 0.85))
        else:
            ur.Entity(parent=root, model="sphere", scale=(0.16, 0.07, 0.16), color=col)
            ur.Entity(parent=root, model="sphere", position=(0.02, 0.05, 0), scale=(0.08, 0.05, 0.08), color=col * 1.1)
        self.hold_ent = root
        self.hud.note(f"Holding {it['name']}.  T use   U put away" + ("   G hug" if key == "plush" else ""), 3.0)

    def drop_held(self, quiet: bool = False) -> None:
        if self.hold_ent is not None:
            ur.destroy(self.hold_ent)
        self.hold_ent, self.held = None, None
        if not quiet:
            self.hud.note("You put it away.")

    def use_held_or_served(self) -> None:
        p = self.player
        s = self._served_near() if p.seat is not None else None
        if s is not None:
            ch = self.world.at(p.z)
            ch.served.remove(s)
            ur.destroy(s["ent"])
            self.vitals.eat(s["key"])
            self.sfx.play("gulp")
            self.hud.note(f"{ITEMS[s['key']]['name']}  -  {ITEMS[s['key']]['desc']}")
            return
        key = self.held
        if key is None:
            self.hud.note("Nothing in your hands.  Hold something from your bag (I, then H).", good=False)
            return
        it = ITEMS[key]
        if it["kind"] in ("drink", "food"):
            self.inv.add(key, -1)
            self.vitals.eat(key)
            self.sfx.play("gulp")
            self.hud.note(f"{it['name']}  -  {it['desc']}")
            if self.inv.n(key) <= 0:
                self.drop_held(True)
        elif key == "plush":
            self.hug()
        else:
            self.hud.note(it["desc"], 3.5)

    def store_held_or_served(self) -> None:
        p = self.player
        s = self._served_near() if p.seat is not None else None
        if s is not None:
            self.world.at(p.z).served.remove(s)
            ur.destroy(s["ent"])
            self.inv.add(s["key"])
            self.sfx.play("pickup")
            self.hud.note(f"{ITEMS[s['key']]['name']} stored in your bag (I).")
        elif self.held is not None:
            self.drop_held()

    def _served_near(self):
        p = self.player
        ch = self.world.at(p.z)
        if ch is None:
            return None
        best, bd = None, 2.3
        for s in ch.served:
            d = math.hypot(s["x"] - p.x, s["z"] - p.z)
            if d < bd:
                best, bd = s, d
        return best

    def hug(self) -> None:
        self.hug_t = 0.6
        self.sfx.play("hug", 0.9)
        self.hud.note(random.choice(("You hug the plushie. It forgives you for everything.", "Squish. Something in the walls sighs.", "Soft. Unreasonably soft.")))
        self.award("hug")

    def engage(self) -> None:
        p = self.player
        if self.activity is not None or p.kart is not None or self.target is not None:
            self.interact()
            return
        if self.held == "plush":
            self.hug()
            return
        for npc in self.nearby_npcs():
            if not npc.gone and math.hypot(npc.x - p.x, npc.z - p.z) < 2.2 and self.view_cos(npc.x, npc.z) > 0.5:
                self.sfx.play("hug", 0.7)
                self.hud.speak(npc, random.choice(("...oh. Thank you.", "Nobody's done that in a long time.", "(a stiff, surprised, grateful hug)")))
                self.award("hug")
                return

    # ---- input / hints
    def _v3_input(self, key: str) -> bool:
        if self.activity is not None and key in ("w", "a", "s", "d", "up arrow", "down arrow", "left arrow", "right arrow", "escape"):
            self.activity = None
            self.hud.note("You step away.")
            return key == "escape"
        if key == "g":
            self.engage()
        elif key == "t":
            self.use_held_or_served()
        elif key == "u":
            self.store_held_or_served()
        elif key == "p":
            if self.inv.n("phone") > 0:
                self.open_ui(PhoneUI(self))
            else:
                self.hud.note("You don't own a phone.  (The Backrooms cafe sells one.)", good=False)
        else:
            return False
        return True

    def _v3_hint(self) -> str:
        p, a = self.player, self.activity
        if self.ui is not None:
            return ""
        out = []
        if a is not None:
            out.append("[E] / move to step off  -  running %.0fs" % a["t"] if a["kind"] == "treadmill" else "[E] lift a rep (%d)   move to stop" % a["reps"])
        elif self.swim is not None and self.swim_active():
            out.append("[Space] up   [C] dive   swim to the edge to climb out")
        if p.seat is not None and self._served_near() is not None:
            out.append("[T] eat/drink   [U] store it")
        if self.held is not None:
            out.append(f"holding {ITEMS[self.held]['name']}:  T use  U put away" + ("  G hug" if self.held == "plush" else ""))
        return "   |   ".join(out)

    # ---- elevators & levels
    def _enter_level(self, tgt: str) -> None:
        p, w = self.player, self.world
        prev = w.level
        w.set_level(tgt)
        self.activity = None
        if tgt == "mall":
            p.x, p.z = self.return_pos if self.return_pos else (3.6, 60.0)
            if prev == "backrooms":
                self.award("escaped")
        elif tgt in LEVEL_SPAWN:
            p.x, p.z = LEVEL_SPAWN[tgt]
            if tgt == "backrooms":
                self.br_coins, self.br_heard = set(), set()
                self.award("wrongdoor")
                self.hud.note("The lift doors close behind you. There is no lift.", 5.0, good=False)
        else:
            p.x, p.z = KIND_HW[tgt] - 1.4, 12.0
            self.award("descent")

    def _g_use_elevator(self, it) -> None:
        if self.elev is not None:
            return
        level = self.world.level
        if level != "mall":
            target = "mall"
        else:
            target = BIOMES[stable_hash(self.args.seed, it.data["tag"]) % len(BIOMES)]
        if level in ("mall",) + BIOMES and random.randrange(BACKROOMS_ODDS) == 0:
            target = "backrooms"
        self.elev = dict(t=0.0, target=target, src=(it.x, it.z), switched=False, ding=False, rumble=False)
        if level == "mall":
            self.return_pos = (it.x, it.z)
        self.player.vx = self.player.vz = 0.0
        self.player.x, self.player.z = it.x, it.z

    def _enter_drain(self) -> None:
        p, sw = self.player, self.swim
        self.pool_ret = (sw["x0"] - 0.2, sw["z0"] - 2.0)
        self.world.set_level("drain")
        p.x, p.z = LEVEL_SPAWN["drain"]
        p.y = p.vy = p.vx = p.vz = 0.0
        self.world.stream(p.z, budget=99)
        self.fx.flash_a = 1.0
        self.sfx.play("rumble", 0.8)
        self.hud.note("The drain pulls you down. Down. Further than the pool was deep.", 5.0, good=False)
        self.award("drain")

    def _use_ladder(self, it) -> None:
        p = self.player
        self.world.set_level("pool")
        p.x, p.z = self.pool_ret if self.pool_ret else (-8.0, DEEP_IDX * CH + 2.5)
        p.y = p.vy = p.vx = p.vz = 0.0
        self.world.stream(p.z, budget=99)
        self.fx.flash_a = 0.8
        self.sfx.play("thud")

    def _use_trophy(self, it) -> None:
        self.inv.add("trophy")
        self.inv.flags["trophy_taken"] = True
        self._remove_inter(it)
        self.sfx.play("pickup")
        self.award("trophy")
        self.hud.note("The Secret Trophy.  Hold it from your bag (I, H).", 4.0)

    def _use_coinpile(self, it) -> None:
        if it.data["key"] in self.br_coins:
            return
        self.br_coins.add(it.data["key"])
        self.inv.coins += 8
        self.inv.dirty = True
        self._remove_inter(it)
        self.sfx.play("coin")
        self.hud.note("+8 c   (someone dropped these a long time ago)")

    def _use_book(self, it) -> None:
        d = it.data
        self.sfx.play("tick", 0.6)
        self.open_ui(BookUI(self, d["bid"], d["title"], d["lore"]))

    def _use_archive(self, it) -> None:
        self.open_ui(ArchiveUI(self, it))

    def _use_weights(self, it) -> None:
        self.activity = dict(kind="lift", it=it, t=0.0, reps=0, lift=0.0, total=0)
        self.hud.note("Press E to lift. Move to stop.")

    def _g_use_treadmill(self, it) -> None:
        it.data["on"] = not it.data["on"]
        if it.data["on"]:
            self.activity = dict(kind="treadmill", it=it, t=0.0, reps=0, lift=0.0, total=0)
            self.hud.note("Running. Press E (or move) to step off.")
        else:
            self.activity = None

    def _activity_press(self) -> None:
        a = self.activity
        if a["kind"] == "treadmill":
            a["it"].data["on"] = False
            self.activity = None
        else:
            a["reps"] += 1
            a["lift"] = 0.6
            self.inv.flags["reps"] = int(self.inv.flags.get("reps", 0)) + 1
            self.vitals.stamina = max(0.0, self.vitals.stamina - 4.0)
            self.sfx.play("weights", 0.8)
            if a["reps"] % 10 == 0:
                self.inv.coins += 2
                self.hud.note("+2 c   (the gym pays in encouragement and coins)")
            if self.inv.flags["reps"] >= 30:
                self.award("iron")
            if self.vitals.stamina <= 0:
                self.activity = None
                self.hud.note("You're spent.", good=False)

    def _g_interact(self) -> None:
        if self.activity is not None:
            self._activity_press()
            return
        _orig_interact(self)

    def _activity_player(self, dt: float) -> None:
        a, p, it = self.activity, self.player, self.activity["it"]
        a["t"] += dt
        if a["kind"] == "treadmill":
            p.x, p.y, p.z, p.yaw_body = it.x - 0.2, 0.36, it.z - 1.0, -90.0
            p.fig.place(p.x, p.y, p.z, p.yaw_body)
            p.fig.pose(dt, 5.5)
            self.vitals.thirst = max(0.0, self.vitals.thirst - 0.35 * dt)
            if int(a["t"] / 30) > a["total"]:
                a["total"] += 1
                self.inv.coins += 1
                self.inv.dirty = True
                self.hud.note("+1 c  -  30 seconds of honest running")
            if a["t"] >= 60:
                self.award("lungs")
        else:
            p.x, p.y, p.z, p.yaw_body = it.x, 0.0, it.z, it.yaw
            a["lift"] = max(0.0, a["lift"] - dt)
            p.fig.place(p.x, p.y, p.z, p.yaw_body)
            p.fig.pose(dt, 0.0)
            ang = -150.0 * math.sin(math.pi * min(1.0, a["lift"] / 0.6)) if a["lift"] > 0 else -15.0
            for s in p.fig.sho:
                s.rotation_x = ang
        p.vx = p.vz = 0.0

    # ---- swimming
    def swim_active(self) -> bool:
        p = self.player
        if p.kart is not None or p.seat is not None or self.elev is not None:
            return False
        ch = self.world.at(p.z)
        sw = ch.extra.get("swim") if ch is not None else None
        if not sw:
            return False
        self.swim = sw
        inside = sw["x0"] <= p.x <= sw["x1"] and sw["z0"] <= p.z <= sw["z1"]
        return inside and self.world.floor_at(p.x, p.z) < -1.0 and p.y < -0.75

    def _swim_update(self, dt: float, w, yaw: float) -> None:
        p, hk, sw = self.player, ur.held_keys, self.swim
        locked = self.ui is not None
        fwd = 0 if locked else (hk["w"] or hk["up arrow"]) - (hk["s"] or hk["down arrow"])
        strafe = 0 if locked else hk["d"] - hk["a"]
        sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        dx, dz = sy * fwd + cy * strafe, cy * fwd - sy * strafe
        n = math.hypot(dx, dz)
        sp = 3.4 if (hk["shift"] or hk["left shift"]) else 2.3
        tx, tz = (dx / n * sp, dz / n * sp) if n > 1e-6 else (0.0, 0.0)
        b = 1.0 - math.exp(-3.0 * dt)
        p.vx += (tx - p.vx) * b
        p.vz += (tz - p.vz) * b
        up, down = (hk["space"] and not locked), ((hk["c"] or hk["left control"]) and not locked)
        if up or down:
            tvy = (1.7 if up else 0.0) - (1.9 if down else 0.0)
        else:
            tvy = (-1.15 - p.y) * 1.3 if p.y > -1.9 else -0.12
        p.vy += (tvy - p.vy) * (1.0 - math.exp(-4.0 * dt))
        nx = max(sw["x0"] + 0.3, min(sw["x1"] - 0.3, p.x + p.vx * dt))
        nz = max(sw["z0"] + 0.3, min(sw["z1"] - 0.3, p.z + p.vz * dt))
        p.x, p.z = nx, nz
        p.y = max(sw["floor"], min(-1.15, p.y + p.vy * dt))
        p.on_ground, p.wade, p.running = False, False, False
        if n > 1e-6 and p.speed > 0.3:
            p.yaw_body += wrap180(math.degrees(math.atan2(p.vx, p.vz)) - p.yaw_body) * min(1.0, 8.0 * dt)
        p.fig.place(p.x, p.y, p.z, p.yaw_body)
        p.fig.pose(dt, p.speed * 0.8)
        if random.random() < dt * 0.35:
            self.sfx.play("drip", 0.35, wet=True)
        self.award("swim")
        if p.y <= sw["floor"] + 0.3 and math.hypot(p.x - sw["drain"][0], p.z - sw["drain"][1]) < 1.0:
            self._enter_drain()

    # ---- cinema screens
    def _cinema_setup(self, ch) -> None:
        ent = ch.statics[0]
        ch.statics.remove(ent)
        if not self.vid_checked:
            self.vid_checked, self.vid_paths = True, find_videos(self.args)
            if self.vid_paths:
                self.video = VideoScreen(self.vid_paths[0])
        ok = self.video is not None and self.video.ok
        try:
            if FANCY:
                ent.set_shader_input("uv_scale", ur.Vec2(1.0, 1.0))
                ent.set_shader_input("uv_off", ur.Vec2(0.0, 0.0))
                ent.set_shader_input("emissive", 1.25 if ok else 0.0)
            else:
                ent.texture_scale = (1, 1)
            if ok:
                ent.color = C(1, 1, 1)
                set_raw_texture(ent, self.video.tex)
            else:
                ent.color = C(0, 0, 0)
        except Exception:
            log.exception("cinema screen setup failed")
        ch.extra["screen_on"] = True

    def _update_v3(self, dt: float, kind: str) -> None:
        p, w = self.player, self.world
        for ch in w.near_chunks(p.z, 1):
            for fn in list(ch.tickers):
                try:
                    if fn(dt, self) is False:
                        ch.tickers.remove(fn)
                except Exception as exc:
                    if fn in ch.tickers:
                        ch.tickers.remove(fn)
                    if len(self._tick_err) < 4:
                        self._tick_err.add(str(exc))
                        log.exception("chunk ticker failed (%s)", exc)
            if self.inv.flags.get("cinema_all") and ch.extra.get("cinema_screen") and not ch.extra.get("screen_on") and ch.statics:
                self._cinema_setup(ch)
        if self.video is not None and self.video.ok:
            near_screen = any(c.extra.get("screen_on") for c in w.near_chunks(p.z, 2))
            if near_screen:
                self.video.update(self.clock)
        self.noise_pulse = max(0.0, self.noise_pulse - dt * 0.7)
        a = self.activity
        if a is not None:
            ch = w.at(a["it"].z)
            if self.ui is not None or ch is None or a["it"] not in ch.inters:
                self.activity = None
        self.underwater = bool(self.swim_active() and self.cam[1] < self.swim["surface"])
        if self.hug_t > 0 and self.hold_ent is not None:
            self.hug_t -= dt
            s = 1.0 - 0.25 * math.sin(math.pi * max(0.0, self.hug_t) / 0.6)
            self.hold_ent.scale = (s, s, s)
            self.hold_ent.position = (0.30 - 0.18 * (1 - s) * 4, -0.26 + 0.1 * (1 - s) * 4, 0.70 - 0.2 * (1 - s) * 4)
        elif self.hold_ent is not None:
            self.hold_ent.scale = (1, 1, 1)
            self.hold_ent.position = (0.30, -0.26 + 0.008 * math.sin(self.clock * 1.6), 0.70)

    def _g_update_fx(self, dt: float) -> None:
        _orig_update_fx(self, dt)
        fx = self.fx
        if self.noise_pulse > 0.01:
            fx.static_a = max(fx.static_a, min(0.9, self.noise_pulse))
        if self.underwater:
            fx.tint.color = C(0.08, 0.45, 0.62, 0.38)
            fx.vig.color = C(0.0, 0.05, 0.08, 0.5)

    # ---- overrides of existing handlers
    def _vend_drop(self, it, key: str) -> None:
        tx, ty, tz = it.data["tray"]
        ch = self.world.at(tz)
        if ch is None:
            return
        pivot = ur.Entity(parent=ch.detail, position=(tx - ch.xoff, ty, tz))
        mk(pivot, "cube", (0, 0, 0), (0.06, 0.12, 0.06), "metal", color=C(*[c / 255 for c in ITEMS[key]["col"]]), lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        lit = Litter(pivot, "can", tx, tz, ch.hw, ch.xoff)
        lit.y, lit.vx, lit.vz, lit.vy, lit.awake = ty, it.data["dir"] * 1.4, random.uniform(-0.6, 0.6), 1.2, True
        ch.litter.append(lit)

    def _g_use_vend(self, it) -> None:
        if it.data["cool"] > 0.0:
            return
        self.sfx.play("click", 0.7)
        self.open_ui(VendingUI(self, it))

    def _g_use_fountain(self, it) -> None:
        if self.inv.coins < 1:
            self.hud.note("You have no coin to give the fountain.", good=False)
            return
        self.inv.coins -= 1
        self.inv.dirty = True
        ch = self.world.at(it.z)
        if ch is not None and ch.extra.get("fountain"):
            p = self.player
            spawn_coin(ch, self, (p.x, 1.2, p.z), ch.extra["fountain"], 0.8)
        else:
            self.sfx.play("coin", 0.8)
        self.hud.note(random.choice(WISHES))
        self.award("wish")

    def _g_use_kart(self, it) -> None:
        self.last_kart = it.data["kart"]
        _orig_use_kart(self, it)

    def _g_use_npc(self, it) -> None:
        npc = it.data["npc"]
        h = getattr(npc, "on_talk", None)
        if h is not None:
            h()
            return
        if npc.role == "shop" and npc.kind == "cinema":
            self.hud.speak(npc, "Concessions are on the board. Buy everything and the screen remembers you.")
            self.open_ui(StockShopUI(self, npc, "THEATER 7  -  CONCESSIONS", CINEMA_STOCK, track=f"cinema:{npc.ch.idx}"))
            return
        _orig_use_npc(self, it)


def _player_update_v3(self, dt, w, yaw, jump_pressed):
    g = self.g
    if self.kart is None and g.activity is not None:
        g._activity_player(dt)
        return
    if g.swim_active():
        g._swim_update(dt, w, yaw)
        return
    _orig_player_update(self, dt, w, yaw, jump_pressed)


def _hud_update_v3(self, dt, zone_title, prompt, dbg):
    _orig_hud_update(self, dt, zone_title, prompt, dbg)
    h = self.g._v3_hint()
    if h:
        self.prompt.text = (self.prompt.text + "   |   " if self.prompt.text else "") + h


_orig_item_card = item_card


def item_card(kind: str, data, title: str = "", W: int = 320, H: int = 400):      # noqa: F811
    if kind in ("badge", "hdr", "keepsake"):
        im = Image.new("RGBA", (W, H), (22, 24, 30, 255))
        d = ImageDraw.Draw(im)
        if kind == "badge":
            d.ellipse([W / 2 - 90, 70, W / 2 + 90, 250], fill=(240, 200, 70, 255), outline=(255, 250, 220, 255), width=6)
            d.ellipse([W / 2 - 70, 90, W / 2 + 70, 230], outline=(120, 80, 10, 255), width=4)
            d.text((W / 2 - 40, 140), "*  *  *", fill=(120, 80, 10, 255), font=_font(30))
            d.text((18, H - 98), title[:26], fill=(235, 238, 250, 255), font=_font(22))
            d.text((18, H - 62), BADGES.get(str(data), ("", ""))[1][:40], fill=(150, 160, 190, 255), font=_font(13))
        else:
            d.text((18, H // 2 - 14), title[:24], fill=(235, 238, 250, 255), font=_font(24))
        return im
    return _orig_item_card(kind, data, title, W, H)


_orig_inv_entries, _orig_inv_key = InventoryUI.entries, InventoryUI.key


def _inv_entries_v3(self) -> list:
    out = _orig_inv_entries(self)
    inv = self.g.inv
    out.append(("hdr", "badges", f"---- BADGES  {len(inv.badges)}/{len(BADGES)} ----", "Earned by exploring, helping, swimming and being curious."))
    for bid in BADGES:
        if bid in inv.badges:
            out.append(("badge", bid, f"Badge: {BADGES[bid][0]}", BADGES[bid][1]))
    return out


def _inv_key_v3(self, k: str) -> None:
    if k == "h" and getattr(self, "es", None):
        kind = self.es[self.sel][0]
        if kind in ITEMS and ITEMS[kind].get("hold"):
            self.g.hold(kind)
            self.g.close_ui()
            return
        self.g.hud.note("That can't be held.", good=False)
        return
    _orig_inv_key(self, k)


def _v3_install() -> None:
    global _orig_interact, _orig_update_fx, _orig_use_kart, _orig_use_npc, _orig_player_update, _orig_hud_update
    Inventory.__init__, Inventory.to_json, Inventory.from_json = _inv_init_v3, _inv_to_v3, _inv_from_v3
    ChunkBuilder.__bases__ = (V3Builders,) + ChunkBuilder.__bases__
    ChunkBuilder.build = _b_build
    _of = ZoneBuilders.fountain

    def _fountain_v3(self, cx, cz, rim_y0):
        _of(self, cx, cz, rim_y0)
        try:
            self._fountain_water(cx, cz, rim_y0)
        except Exception:
            log.exception("fountain water failed")
    ZoneBuilders.fountain = _fountain_v3
    _orig_interact, _orig_update_fx = Game.interact, Game._update_fx
    _orig_use_kart, _orig_use_npc = Game._use_kart, Game._use_npc
    _orig_player_update, _orig_hud_update = Player.update, Hud.update
    for name in ("_v3_init", "award", "cinema_complete", "hold", "drop_held", "use_held_or_served", "store_held_or_served", "_served_near",
                 "hug", "engage", "_v3_input", "_v3_hint", "_enter_level", "_enter_drain", "_use_ladder", "_use_trophy", "_use_coinpile",
                 "_use_book", "_use_archive", "_use_weights", "_activity_press", "_activity_player", "swim_active", "_swim_update",
                 "_cinema_setup", "_update_v3", "_vend_drop"):
        setattr(Game, name, getattr(GameV3, name))
    Game._use_elevator = GameV3._g_use_elevator
    Game._use_treadmill = GameV3._g_use_treadmill
    Game.interact = GameV3._g_interact
    Game._update_fx = GameV3._g_update_fx
    Game._use_vend = GameV3._g_use_vend
    Game._use_fountain = GameV3._g_use_fountain
    Game._use_kart = GameV3._g_use_kart
    Game._use_npc = GameV3._g_use_npc
    Player.update = _player_update_v3
    Hud.update = _hud_update_v3
    InventoryUI.entries, InventoryUI.key = _inv_entries_v3, _inv_key_v3


_v3_install()


# ============================================================================
# v4 EXPANSION  -  SAKURA-YA (rare Japanese izakaya: kaiten sushi + ramen bar), phone garage & music player
# (everything below is bolted onto the classes above; see _v4_install at the end of the block)
# ============================================================================
ZONES["japan"] = dict(title="SAKURA-YA  -  SUSHI & RAMEN IZAKAYA", fog=(0.34, 0.17, 0.13), dens=0.036,
                      amb=(0.20, 0.125, 0.095), lamp=(1.00, 0.66, 0.38, 1.30))
AUDIO_PRESETS["japan"] = dict(wet=0.34, tone_hz=9500.0, gain=0.92, stretch=1.05, smear_prob=0.10, warmth_hz=9500.0, lanes=3)
REVERB_OF["japan"] = "tile"
KIND_HW["japan"] = 17.0
KIND_H["japan"] = 5.2
WHO["japan"] = "diner"
ZONE_FG["japan"], ZONE_BG["japan"] = (255, 214, 190), (52, 10, 12)
for _d in (ZONES, AUDIO_PRESETS, REVERB_OF, KIND_HW, KIND_H, WHO, ZONE_FG, ZONE_BG):
    _d["jpcourt"] = _d["japan"]                 # the forecourts share the restaurant's look, sound and size
PALETTES.update({
    "jp_chef_r": dict(coat=(0.10, 0.11, 0.20), legs=(0.08, 0.08, 0.12), skin=(0.90, 0.76, 0.64), em=0.05),     # ramen chef: indigo jinbei
    "jp_chef_s": dict(coat=(0.96, 0.96, 0.94), legs=(0.12, 0.12, 0.16), skin=(0.90, 0.76, 0.64), em=0.10),     # sushi chef: whites
    "jp_server": dict(coat=(0.16, 0.20, 0.44), legs=(0.10, 0.10, 0.16), skin=(0.92, 0.80, 0.70), em=0.06),      # indigo happi
    "jp_c0": dict(coat=(0.55, 0.20, 0.22), legs=(0.14, 0.14, 0.20), skin=(0.90, 0.76, 0.66), em=0.04),
    "jp_c1": dict(coat=(0.22, 0.42, 0.40), legs=(0.18, 0.18, 0.22), skin=(0.88, 0.74, 0.64), em=0.04),
    "jp_c2": dict(coat=(0.80, 0.74, 0.62), legs=(0.22, 0.20, 0.20), skin=(0.92, 0.80, 0.70), em=0.04),
    "jp_c3": dict(coat=(0.30, 0.30, 0.36), legs=(0.12, 0.12, 0.16), skin=(0.86, 0.72, 0.62), em=0.04),
    "jp_c4": dict(coat=(0.62, 0.40, 0.55), legs=(0.20, 0.16, 0.22), skin=(0.92, 0.78, 0.68), em=0.04),
})
DIALOGUE["japan"] = ["The broth is eleven hours old and still learning.", "The belt never stops. I have made peace with that.",
                     "Salmon, then tuna, then the egg. In that order, or the plates get offended.",
                     "Nobody here is in a hurry. It is rude to be in a hurry.",
                     "This place has no back door. It has a front door at both ends.",
                     "Ring the little bell if you need us. It is a very polite bell."]

# ---- menu ---------------------------------------------------------------------------------------
ITEMS.update({
    "ramen_shoyu": dict(name="Shoyu Ramen", kind="food", thirst=-4, hunger=54, fx=None, desc="Clear soy broth, curly noodles, one patient egg.", col=(205, 140, 70), hold=True),
    "ramen_miso": dict(name="Miso Ramen", kind="food", thirst=-5, hunger=58, fx="calm", desc="Cloudy, buttery, and quietly forgiving.", col=(215, 160, 80), hold=True),
    "ramen_tonkotsu": dict(name="Tonkotsu Ramen", kind="food", thirst=-6, hunger=66, fx="caffeine", desc="Pork broth so rich it has opinions.", col=(235, 215, 180), hold=True),
    "gyoza": dict(name="Pan-fried Gyoza", kind="food", thirst=-3, hunger=28, fx=None, desc="Crisp underneath, soft on top, gone in a minute.", col=(225, 180, 110), hold=True),
    "karaage": dict(name="Karaage", kind="food", thirst=-5, hunger=30, fx=None, desc="Fried chicken with a lemon and no regrets.", col=(220, 160, 60), hold=True),
    "onigiri": dict(name="Warm Onigiri", kind="food", thirst=-2, hunger=24, fx=None, desc="A rice triangle, microwaved until it forgives you.", col=(240, 240, 235), hold=True),
    "miso_soup": dict(name="Miso Soup", kind="food", thirst=8, hunger=14, fx="calm", desc="Tofu, wakame, a small moon of steam.", col=(190, 130, 70), hold=True),
    "chawanmushi": dict(name="Chawanmushi", kind="food", thirst=2, hunger=26, fx="calm", desc="Steamed egg custard. Wobbles like a secret.", col=(245, 220, 120), hold=True),
    "curry": dict(name="Katsu Curry", kind="food", thirst=-4, hunger=64, fx=None, desc="Warmed in the microwave, respected by everyone.", col=(170, 100, 40), hold=True),
    "nigiri_set": dict(name="Chef's Nigiri Set", kind="food", thirst=-2, hunger=46, fx="caffeine", desc="Five pieces, each pressed for you personally.", col=(250, 120, 90), hold=True),
    "salmon_roll": dict(name="Salmon Maki", kind="food", thirst=-1, hunger=34, fx=None, desc="Six neat wheels of salmon and rice.", col=(250, 130, 100), hold=True),
    "tuna_nigiri": dict(name="Tuna Nigiri", kind="food", thirst=-1, hunger=22, fx=None, desc="Two pieces. Deep red. Very calm.", col=(200, 50, 60), hold=True),
    "sashimi": dict(name="Sashimi Fan", kind="food", thirst=-1, hunger=30, fx="calm", desc="Thin slices, no rice, all attitude.", col=(240, 110, 110), hold=True),
    "green_tea": dict(name="Hot Green Tea", kind="drink", thirst=26, hunger=0, fx="calm", desc="Grassy, hot, and endlessly refilled.", col=(120, 170, 80), hold=True),
    "ramune": dict(name="Ramune", kind="drink", thirst=30, hunger=1, fx="fizz", desc="Press the marble. Hear it rattle.", col=(110, 200, 240), hold=True),
    "sake": dict(name="Warm Sake", kind="drink", thirst=10, hunger=2, fx="calm", desc="A small cup that makes the lanterns softer.", col=(240, 235, 215), hold=True),
    "mochi": dict(name="Daifuku Mochi", kind="food", thirst=-2, hunger=18, fx=None, desc="Pillows of rice with sweet red bean inside.", col=(250, 200, 210), hold=True),
    "plate_tamago": dict(name="Tamago Plate", kind="food", thirst=0, hunger=14, fx=None, desc="Sweet egg on rice. Off the white plate.", col=(245, 215, 110), hold=True),
    "plate_salmon": dict(name="Salmon Plate", kind="food", thirst=0, hunger=17, fx=None, desc="Off the red plate. Fresh from the belt.", col=(250, 130, 100), hold=True),
    "plate_unagi": dict(name="Unagi Plate", kind="food", thirst=0, hunger=20, fx=None, desc="Glazed eel from the black plate.", col=(110, 70, 40), hold=True),
    "plate_otoro": dict(name="Otoro Plate", kind="food", thirst=0, hunger=24, fx="caffeine", desc="The gold plate. Fatty tuna. Hush.", col=(250, 150, 150), hold=True),
})
DRINK_KEYS = tuple(DRINK_KEYS) + ("green_tea", "ramune", "sake")
JP_MENU = (   # (section, item, price, how the chef makes it)
    ("RAMEN  (ramen bar)", "ramen_shoyu", 6, "ramen"), ("RAMEN  (ramen bar)", "ramen_miso", 7, "ramen"),
    ("RAMEN  (ramen bar)", "ramen_tonkotsu", 8, "ramen"),
    ("SIDES  (pan & fryer)", "gyoza", 4, "fry"), ("SIDES  (pan & fryer)", "karaage", 5, "fry"),
    ("WARMED  (microwave)", "onigiri", 3, "micro"), ("WARMED  (microwave)", "miso_soup", 3, "micro"),
    ("WARMED  (microwave)", "chawanmushi", 5, "micro"), ("WARMED  (microwave)", "curry", 7, "micro"),
    ("SUSHI  (private chef, made for you - dearer than the belt)", "nigiri_set", 9, "sushi"),
    ("SUSHI  (private chef, made for you - dearer than the belt)", "salmon_roll", 7, "sushi"),
    ("SUSHI  (private chef, made for you - dearer than the belt)", "tuna_nigiri", 5, "sushi"),
    ("SUSHI  (private chef, made for you - dearer than the belt)", "sashimi", 8, "sushi"),
    ("DRINKS & SWEETS", "green_tea", 2, "drink"), ("DRINKS & SWEETS", "ramune", 3, "drink"),
    ("DRINKS & SWEETS", "sake", 5, "drink"), ("DRINKS & SWEETS", "mochi", 3, "drink"),
)
JP_HOW = {k: how for _s, k, _p, how in JP_MENU}
JP_PRICE = {k: p for _s, k, p, _h in JP_MENU}
BELT_PLATES = (("plate_tamago", 2, (0.97, 0.95, 0.88)), ("plate_salmon", 3, (0.82, 0.14, 0.14)),
               ("plate_unagi", 4, (0.07, 0.07, 0.09)), ("plate_otoro", 6, (0.95, 0.74, 0.20)))
JP_PRICE.update({k: p for k, p, _c in BELT_PLATES})
JP_KEYS = set(JP_PRICE)
BADGES.update({
    "sakura": ("Sakura-ya", "Found the rare Japanese izakaya in the mall."),
    "conveyor": ("Kaiten", "Took a plate off the sushi conveyor belt."),
    "omakase": ("Table Service", "Ordered from a Sakura-ya server and ate it with chopsticks."),
    "pocketkart": ("Pocket Garage", "Stored a go-kart inside your phone."),
    "playlist": ("Shuffle", "Played music from your phone while walking the mall."),
})

# ---- sound effects -------------------------------------------------------------------------------
_VOWELS = {"a": (780, 1250, 2600), "i": (300, 2300, 3100), "u": (330, 1300, 2500), "e": (480, 1950, 2650), "o": (520, 950, 2500)}


def _bp(x, lo, hi, sr, order=2):
    lo, hi = max(40.0, lo), min(hi, sr * 0.45)
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=sr, output="sos"), x)


def _voice(sr, syll, f0=135.0, sd=0.14, rise=0.10, seed=1):
    """Tiny formant voice: [(vowel, lead)] -> shouted syllables.  lead: ''|'sh'|'r'|'m'."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (v, lead) in enumerate(syll):
        n = int(sd * sr)
        x = np.arange(n) / sr
        f = f0 * (1.0 + rise * math.sin(math.pi * i / max(1, len(syll) - 1)) - 0.05 * i / len(syll))
        ph = 2 * np.pi * np.cumsum(np.full(n, f)) / sr
        src = sum(np.sin(k * ph) / k ** 0.9 for k in range(1, 22))
        f1, f2, f3 = _VOWELS[v]
        sig = _bp(src, f1 * 0.85, f1 * 1.15, sr) * 1.2 + _bp(src, f2 * 0.88, f2 * 1.12, sr) * 0.7 + _bp(src, f3 * 0.9, f3 * 1.1, sr) * 0.3
        env = np.minimum(1.0, x / 0.02) * np.minimum(1.0, (sd - x) / 0.04) * (0.85 + 0.15 * np.sin(np.pi * x / sd))
        sig = sig * env
        if lead == "sh":
            nn = int(0.07 * sr)
            sh = _bp(rng.standard_normal(nn), 3000, 9000, sr) * np.hanning(nn) * 0.9
            sig = np.concatenate([sh, sig])
        elif lead == "r":
            sig[:int(0.02 * sr)] *= 0.25
        elif lead == "m":
            sig[:int(0.03 * sr)] *= 0.4
        out.append(sig)
    m = np.concatenate(out)
    return m / (np.max(np.abs(m)) + 1e-9)


def _v4_synth(name: str, sr: int, seed: int = 0):
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    t = lambda d: np.arange(int(d * sr)) / sr
    if name == "door_slide":
        x = t(0.95)
        stick = 0.55 + 0.45 * (np.sin(2 * np.pi * 23 * x) > -0.2)
        m = _lp(rng.standard_normal(x.size), 900, sr, 2) * 0.9 * stick + _lp(rng.standard_normal(x.size), 180, sr, 2) * 1.2
        m *= np.minimum(1, np.minimum(x / 0.08, (0.95 - x) / 0.25)) * np.where(x < 0.7, 1.0, 0.7)
        tok = np.exp(-np.clip(x - 0.78, 0, 9) * 55) * (x > 0.78) * np.sin(2 * np.pi * 140 * x)
        return _st((m * 0.55 + tok * 0.6) * 0.8)
    if name == "door_bell":
        x = t(2.2)
        m = np.zeros(x.size)
        for t0, g in ((0.0, 1.0), (0.17, 0.8), (0.46, 0.55)):
            xs = np.clip(x - t0, 0, None)
            for f, a in ((2093, 1.0), (2637, 0.6), (3136, 0.45), (4186, 0.25), (5274, 0.12)):
                m += g * a * np.sin(2 * np.pi * f * xs + 0.3 * f) * np.exp(-xs * (5.5 + f / 900)) * (x >= t0)
        wind = _lp(rng.standard_normal(x.size), 6500, sr, 2, "high") * 0.02 * np.exp(-x * 1.5)
        return _st((m * 0.16 + wind) * 0.9)
    if name == "irasshai":
        m = _voice(sr, [("i", ""), ("a", "r"), ("a", "sh"), ("i", ""), ("a", "m"), ("e", "")], 150.0, 0.15, 0.14, 3)
        return _st(m * 0.75)
    if name == "oishii":
        m = _voice(sr, [("o", ""), ("i", ""), ("i", "sh"), ("i", "")], 190.0, 0.15, 0.2, 5)
        return _st(m * 0.6)
    if name == "arigato":
        m = _voice(sr, [("a", ""), ("i", "r"), ("a", ""), ("o", ""), ("u", "")], 140.0, 0.14, 0.12, 7)
        return _st(m * 0.6)
    if name == "itadaki":
        m = _voice(sr, [("i", ""), ("a", ""), ("a", ""), ("i", ""), ("a", ""), ("u", "")], 175.0, 0.13, 0.12, 9)
        return _st(m * 0.5)
    if name == "chop":
        x = t(0.9)
        m = np.zeros(x.size)
        for k in range(6):
            t0 = 0.04 + k * 0.135
            xs = np.clip(x - t0, 0, None)
            m += (rng.standard_normal(x.size) * np.exp(-xs * 380) * 0.55 + np.sin(2 * np.pi * 210 * xs) * np.exp(-xs * 55) * 0.8) * (x >= t0)
        return _st(_lp(m, 5000, sr, 2) * 0.7)
    if name == "sizzle":
        x = t(2.2)
        n = _lp(rng.standard_normal(x.size), 2500, sr, 2, "high")
        crack = (rng.random(x.size) > 0.9985) * rng.standard_normal(x.size) * 3.0
        m = (n * (0.35 + 0.35 * np.sin(2 * np.pi * 7 * x) ** 2) + _lp(crack, 6000, sr, 2)) * np.minimum(1, np.minimum(x / 0.12, (2.2 - x) / 0.5))
        return _st(m * 0.22)
    if name == "boil":
        x = t(2.6)
        m = _lp(rng.standard_normal(x.size), 700, sr, 2) * 0.35
        for _ in range(26):
            t0 = rng.uniform(0, 2.4)
            f = rng.uniform(220, 520)
            xs = np.clip(x - t0, 0, None)
            m += np.sin(2 * np.pi * np.cumsum(f * (1 + 2.0 * np.exp(-xs * 40))) / sr) * np.exp(-xs * 30) * 0.28 * (x >= t0)
        return _st(m * np.minimum(1, np.minimum(x / 0.2, (2.6 - x) / 0.4)) * 0.55)
    if name == "mw_hum":
        x = t(2.4)
        m = (np.sin(2 * np.pi * 120 * x) + 0.5 * np.sin(2 * np.pi * 240 * x) + 0.2 * np.sin(2 * np.pi * 360 * x)) * 0.30
        m += _lp(rng.standard_normal(x.size), 1400, sr, 2) * 0.10
        return _st(m * np.minimum(1, np.minimum(x / 0.1, (2.4 - x) / 0.1)) * 0.6)
    if name == "mw_beep":
        x = t(0.9)
        m = np.sin(2 * np.pi * 2000 * x) * ((x % 0.3) < 0.13) * 0.35
        return _st(m)
    if name == "mw_ding":
        x = t(1.4)
        m = (np.sin(2 * np.pi * 1760 * x) + 0.4 * np.sin(2 * np.pi * 2637 * x)) * np.exp(-x * 3.2) * 0.35
        return _st(m)
    if name == "mw_door":
        x = t(0.35)
        m = _lp(rng.standard_normal(x.size), 1800, sr, 2) * np.exp(-x * 14) * 0.5 + np.sin(2 * np.pi * 95 * x) * np.exp(-x * 25) * 0.8
        return _st(m * 0.7)
    if name == "plate":
        x = t(0.45)
        m = sum(np.sin(2 * np.pi * f * x) * np.exp(-x * d) * a for f, d, a in ((3150, 40, 0.5), (4380, 55, 0.35), (1980, 28, 0.3))) + rng.standard_normal(x.size) * np.exp(-x * 600) * 0.3
        return _st(m * 0.6)
    if name == "slurp":
        x = t(1.0)
        fc = 700 + 1500 * x
        nz = rng.standard_normal(x.size)
        m = np.zeros(x.size)
        for a, b in zip(range(0, x.size - 800, 800), range(800, x.size, 800)):
            m[a:b] = _bp(nz[a:b], fc[a] * 0.8, fc[a] * 1.3, sr)[:b - a]
        m *= (0.6 + 0.4 * np.sin(2 * np.pi * 9 * x)) * np.minimum(1, np.minimum(x / 0.08, (1.0 - x) / 0.2))
        return _st(m * 0.9)
    if name == "bell_table":
        x = t(1.8)
        m = np.zeros(x.size)
        for t0 in (0.0, 0.28):
            xs = np.clip(x - t0, 0, None)
            for f, a in ((1568, 1.0), (4327, 0.5), (8470, 0.2)):
                m += a * np.sin(2 * np.pi * f * xs) * np.exp(-xs * (4 + f / 1500)) * (x >= t0)
        return _st(m * 0.28)
    if name == "whoosh":
        x = t(0.45)
        m = _bp(rng.standard_normal(x.size), 800, 3800, sr) * np.sin(np.pi * x / 0.45) ** 2
        return _st(m * 0.55)
    if name == "pour":
        x = t(1.6)
        m = _bp(rng.standard_normal(x.size), 900, 3200, sr) * (0.6 + 0.4 * np.sin(2 * np.pi * 16 * x)) * np.minimum(1, np.minimum(x / 0.1, (1.6 - x) / 0.3))
        return _st(m * 0.35)
    if name == "pat":
        x = t(0.18)
        m = np.sin(2 * np.pi * 120 * x) * np.exp(-x * 40) + _lp(rng.standard_normal(x.size), 900, sr, 2) * np.exp(-x * 60) * 0.5
        return _st(m * 0.7)
    if name == "murmur":
        x = t(4.0)
        m = np.zeros(x.size)
        for k in range(5):
            f0 = rng.uniform(110, 240)
            seg = _voice(sr, [(rng.choice(list("aiueo")), "") for _ in range(rng.integers(8, 14))], f0, 0.17, 0.3, int(rng.integers(0, 999)))
            o = int(rng.uniform(0, 3.0) * sr)
            seg = seg[:x.size - o]
            m[o:o + seg.size] += seg * rng.uniform(0.2, 0.5)
        m = _lp(m, 2200, sr, 2)
        return _st(m * 0.10)
    if name.startswith("koto"):
        k = int(name[4:] or 0)
        f = 220.0 * 2 ** (((0, 2, 3, 7, 8, 12, 14)[k % 7]) / 12.0) * (2 if k >= 7 else 1)
        n = int(2.4 * sr)
        N = max(2, int(sr / f))
        buf = rng.uniform(-1, 1, N)
        exc = np.zeros(n)
        exc[:N] = buf
        den = np.zeros(N + 2)
        den[0], den[N], den[N + 1] = 1.0, -0.4985, -0.4985
        out = signal.lfilter([1.0], den, exc)
        out *= np.exp(-np.arange(n) / sr * 1.6)
        return _st(_lp(out, 4200, sr, 2) * 0.6)
    if name == "belt_hum":
        x = t(3.0)
        m = (np.sin(2 * np.pi * 54 * x) + 0.3 * np.sin(2 * np.pi * 108 * x)) * 0.05 + _lp(rng.standard_normal(x.size), 500, sr, 2) * 0.03
        return _st(m * np.minimum(1, np.minimum(x / 0.4, (3.0 - x) / 0.4)))
    return None


_synth_sfx_v3 = synth_sfx


def synth_sfx(name: str, sr: int, seed: int = 0) -> np.ndarray:      # noqa: F811  (adds the v4 sounds)
    a = _v4_synth(name, sr, seed)
    return a if a is not None else _synth_sfx_v3(name, sr, seed)


def jp_say(g, x: float, z: float, name: str, gain: float = 1.0, far: float = 24.0) -> None:
    """Positional one-shot: gain falls off with distance and pans by x relative to the player."""
    p = g.player
    d = math.hypot(x - p.x, z - p.z)
    if d > far:
        return
    k = (1.0 - d / far) ** 1.4
    g.sfx.play(name, gain * k, pan=max(-1.0, min(1.0, (x - p.x) / 9.0)))


# ---- CJK-aware signs ---------------------------------------------------------------------------------
_CJK_FONT = [None, False]


def _cjk_font(size: int):
    if not _CJK_FONT[1]:
        _CJK_FONT[1] = True
        for nm in ("YuGothB.ttc", "YuGothM.ttc", "meiryob.ttc", "meiryo.ttc", "msgothic.ttc", "NotoSansCJK-Bold.ttc", "NotoSansCJK-Regular.ttc",
                   "NotoSansCJKjp-Bold.otf", "NotoSansJP-Bold.otf", "Hiragino Sans GB.ttc", "HiraginoSans-W6.ttc", "ipag.ttf", "ipam.ttf",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                   "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"):
            try:
                ImageFont.truetype(nm, 20)
                _CJK_FONT[0] = nm
                break
            except Exception:
                continue
    return ImageFont.truetype(_CJK_FONT[0], size) if _CJK_FONT[0] else None


def jp_sign(jp: str, roman: str, fg=(255, 226, 190), bg=(60, 10, 12), w: int = 512, h: int = 128, sub: str = "", vertical: bool = False):
    """Sign texture: Japanese script when the machine has a CJK font, romaji otherwise."""
    key = f"jpsign:{jp}:{roman}:{fg}:{bg}:{sub}:{w}x{h}"
    if key not in BANK._tex:
        im = Image.new("RGBA", (w, h), bg + (255,))
        d = ImageDraw.Draw(im)
        d.rectangle([3, 3, w - 4, h - 4], outline=fg + (255,), width=3)
        d.rectangle([9, 9, w - 10, h - 10], outline=fg + (120,), width=1)
        f = _cjk_font(int(h * (0.52 if sub else 0.6)))
        txt = jp if f is not None else roman
        if f is None:
            f = _font(int(h * (0.40 if sub else 0.5)))
        if vertical and txt is jp:
            fs = int(w * 0.62)
            f = _cjk_font(fs)
            y = 8
            for chx in jp:
                d.text(((w - d.textlength(chx, font=f)) / 2, y), chx, fill=fg + (255,), font=f)
                y += int(fs * 1.05)
        else:
            tw = d.textlength(txt, font=f)
            while tw > w - 24 and getattr(f, "size", 12) > 10:
                f = (_cjk_font if txt is jp else _font)(max(10, int(f.size * 0.88)))
                tw = d.textlength(txt, font=f)
            d.text(((w - tw) / 2, h * (0.08 if sub else 0.2)), txt, fill=fg + (255,), font=f)
            if sub:
                f2 = _font(int(h * 0.17))
                d.text(((w - d.textlength(sub, font=f2)) / 2, h * 0.72), sub, fill=fg + (210,), font=f2)
        BANK._tex[key] = BANK.to_texture(im)
    return BANK._tex[key]


# ---- people: styling, arms, speech bubbles -----------------------------------------------------------
def _fm(fig, parent, model, pos, sc, col, em=0.0):
    e = mk(parent, model, pos, sc, color=C(*col), emissive=em, **fig.k)
    fig.ents.append(e)
    return e


def jp_style(fig, style: str, rng: random.Random) -> None:
    """Hair, eyes and uniforms that make a low-poly figure read as a Japanese restaurant person."""
    hair = rng.choice(((0.04, 0.04, 0.05), (0.10, 0.07, 0.06), (0.06, 0.05, 0.08)))
    H = fig.head
    _fm(fig, H, "sphere", (0, 0.20, -0.02), (0.34, 0.21, 0.34), hair)
    _fm(fig, H, "cube", (0, 0.07, -0.13), (0.32, 0.26, 0.08), hair)
    for sx in (-0.07, 0.07):
        _fm(fig, H, "sphere", (sx, 0.12, 0.135), (0.035, 0.045, 0.02), (0.02, 0.02, 0.03), 0.4)
    if style == "ramen":
        _fm(fig, H, "cube", (0, 0.20, 0.0), (0.345, 0.07, 0.345), (0.95, 0.95, 0.92))                  # tenugui headband
        _fm(fig, fig.root, "cube", (0, 0.82, 0.14), (0.54, 0.62, 0.03), (0.92, 0.92, 0.9))               # apron
    elif style == "sushi":
        _fm(fig, H, "cube", (0, 0.27, 0.0), (0.34, 0.13, 0.34), (0.97, 0.97, 0.95))                      # chef cap
        _fm(fig, H, "cube", (0, 0.20, 0.0), (0.35, 0.05, 0.35), (0.10, 0.10, 0.3))                       # indigo band
        _fm(fig, fig.root, "cube", (0, 0.80, 0.14), (0.54, 0.7, 0.03), (0.95, 0.95, 0.95))
        _fm(fig, fig.root, "cube", (0, 1.28, 0.145), (0.5, 0.09, 0.03), (0.15, 0.2, 0.45))
    elif style == "server":
        _fm(fig, H, "sphere", (0, 0.36, -0.06), (0.12, 0.12, 0.12), hair)                                # hair bun
        _fm(fig, fig.root, "cube", (0, 0.87, 0.0), (0.53, 0.10, 0.31), (0.78, 0.12, 0.14))              # obi
        _fm(fig, fig.root, "cube", (0, 0.58, 0.145), (0.40, 0.42, 0.03), (0.95, 0.95, 0.92))             # waist apron
    elif style == "guest" and rng.random() < 0.35:
        _fm(fig, H, "cube", (0, 0.13, 0.15), (0.26, 0.05, 0.015), (0.08, 0.08, 0.1))                    # round glasses band


def _arms(self, dt: float, ls=0.0, le=0.0, rs=0.0, re=0.0, lz=0.0, rz=0.0, hx=0.0, hy=0.0, rate: float = 14.0, hips=None) -> None:
    a = min(1.0, rate * dt)
    tg = ((self.sho[0], ls, lz), (self.sho[1], rs, rz))
    for e, x, z in tg:
        e.rotation_x += (x - e.rotation_x) * a
        e.rotation_z += (z - e.rotation_z) * a
    for e, x in ((self.elb[0], le), (self.elb[1], re)):
        e.rotation_x += (x - e.rotation_x) * a
    self.head.rotation_x += (hx - self.head.rotation_x) * a
    self.head.rotation_y += (hy - self.head.rotation_y) * a
    hv = -88.0 if hips == "sit" else 0.0
    for h in self.hip:
        h.rotation_x += (hv - h.rotation_x) * a


class Bubble:
    """A speech bubble that pops up above a head (parented to the figure, billboarded to the camera)."""
    def __init__(self, fig, h: float = 2.05) -> None:
        self.root = ur.Entity(parent=fig.root, position=(0, h, 0))
        self.root.billboard = True
        self.bg = ur.Entity(parent=self.root, model="quad", color=C(0.05, 0.03, 0.03, 0.78), z=0.02)
        self.edge = ur.Entity(parent=self.root, model="quad", color=C(1.0, 0.82, 0.5, 0.9), z=0.03)
        self.txt = ur.Text(text="", parent=self.root, origin=(0, 0), scale=6.0, color=C(1.0, 0.96, 0.86, 1), z=0.0)
        self.t = 0.0
        self.root.enabled = False

    def say(self, text: str, dur: float = 3.4) -> None:
        lines = wrap_text(text, 24)
        self.txt.text = "\n".join(lines)
        try:
            wpx = max(ur.Text.get_width(l) for l in lines) * 6.0
        except Exception:
            wpx = max(len(l) for l in lines) * 0.085
        wpx = max(0.5, min(3.2, wpx)) + 0.18
        hh = 0.2 * len(lines) + 0.12
        self.bg.scale = (wpx, hh)
        self.edge.scale = (wpx + 0.045, hh + 0.045)
        self.edge.z = 0.03
        self.root.y = 2.0 + 0.1 * len(lines)
        self.t = dur
        self.root.enabled = True

    def tick(self, dt: float, near: bool = True) -> None:
        if self.t > 0:
            self.t -= dt
            if self.t <= 0 or not near:
                self.root.enabled = False
                self.t = 0.0


# ---- dishes -------------------------------------------------------------------------------------------
def jp_dish(key: str, parent, pos, kw: dict, empty: bool = False):
    """A little hand-built model of every dish. The first child is always the plate/bowl (kept when 'empty')."""
    root = ur.Entity(parent=parent, position=pos)

    def P(model, p, s, col, em=0.0, tex=None):
        return mk(root, model, p, s, tex, color=C(*col), emissive=em, **kw)
    if key.startswith("ramen"):
        P("sphere", (0, 0.07, 0), (0.30, 0.15, 0.30), (0.62, 0.10, 0.10))
        if not empty:
            broth = {"ramen_shoyu": (0.55, 0.30, 0.10), "ramen_miso": (0.78, 0.50, 0.22), "ramen_tonkotsu": (0.93, 0.88, 0.78)}[key]
            P("cube", (0, 0.12, 0), (0.25, 0.012, 0.25), broth, 0.25)
            P("cube", (0, 0.135, 0), (0.17, 0.02, 0.17), (0.96, 0.82, 0.36), 0.1)
            P("sphere", (-0.06, 0.15, 0.05), (0.07, 0.05, 0.07), (0.97, 0.96, 0.9))
            P("sphere", (-0.06, 0.155, 0.05), (0.035, 0.03, 0.035), (0.98, 0.7, 0.1), 0.2)
            P("cube", (0.07, 0.15, 0.03), (0.09, 0.015, 0.07), (0.95, 0.6, 0.6))
            P("cube", (0.0, 0.155, -0.08), (0.1, 0.02, 0.05), (0.04, 0.14, 0.06))
            P("cube", (0.05, 0.16, -0.04), (0.04, 0.012, 0.04), (0.2, 0.7, 0.2))
    elif key == "gyoza":
        P("cube", (0, 0.01, 0), (0.34, 0.02, 0.22), (0.95, 0.95, 0.92))
        if not empty:
            for i in range(5):
                P("sphere", (-0.12 + 0.06 * i, 0.045, 0.0), (0.075, 0.05, 0.1), (0.93, 0.78, 0.45))
    elif key == "karaage":
        P("cube", (0, 0.01, 0), (0.30, 0.02, 0.30), (0.2, 0.2, 0.24))
        if not empty:
            for i in range(5):
                P("sphere", (-0.07 + 0.06 * (i % 3), 0.055 + 0.04 * (i // 3), -0.04 + 0.07 * (i % 2)), (0.1, 0.085, 0.1), (0.85, 0.55, 0.2))
            P("sphere", (0.12, 0.04, 0.1), (0.07, 0.05, 0.07), (0.95, 0.9, 0.3))
    elif key == "onigiri":
        P("cube", (0, 0.01, 0), (0.28, 0.02, 0.2), (0.3, 0.2, 0.15))
        if not empty:
            for dx in (-0.07, 0.07):
                P("diamond" if False else "cube", (dx, 0.07, 0), (0.1, 0.1, 0.1), (0.97, 0.97, 0.95), 0.1)
                P("cube", (dx, 0.04, 0.052), (0.09, 0.06, 0.012), (0.03, 0.08, 0.05))
    elif key == "miso_soup":
        P("sphere", (0, 0.05, 0), (0.17, 0.10, 0.17), (0.5, 0.07, 0.05))
        if not empty:
            P("cube", (0, 0.095, 0), (0.13, 0.01, 0.13), (0.72, 0.5, 0.25), 0.2)
            P("cube", (0.02, 0.1, 0.01), (0.04, 0.02, 0.04), (0.97, 0.95, 0.85))
    elif key == "chawanmushi":
        P("sphere", (0, 0.05, 0), (0.15, 0.10, 0.15), (0.92, 0.92, 0.95))
        if not empty:
            P("cube", (0, 0.098, 0), (0.12, 0.015, 0.12), (0.98, 0.86, 0.35), 0.2)
            P("cube", (0.02, 0.11, 0), (0.03, 0.02, 0.03), (0.2, 0.7, 0.25))
    elif key == "curry":
        P("cube", (0, 0.01, 0), (0.38, 0.02, 0.30), (0.95, 0.95, 0.92))
        if not empty:
            P("sphere", (-0.07, 0.05, 0), (0.17, 0.08, 0.2), (0.98, 0.98, 0.96))
            P("sphere", (0.09, 0.04, 0), (0.15, 0.06, 0.2), (0.45, 0.22, 0.06), 0.1)
            P("cube", (0.0, 0.1, 0.0), (0.1, 0.03, 0.15), (0.8, 0.55, 0.2))
    elif key in ("nigiri_set", "tuna_nigiri", "salmon_roll", "sashimi") or key.startswith("plate_"):
        base = (0.55, 0.36, 0.2) if key == "nigiri_set" else (0.95, 0.95, 0.92)
        if key.startswith("plate_"):
            base = next(c for k, _p, c in BELT_PLATES if k == key)
        P("cube", (0, 0.012, 0), (0.46 if key == "nigiri_set" else 0.3, 0.024, 0.2 if key == "nigiri_set" else 0.3), base)
        if not empty:
            top = {"nigiri_set": [(0.95, 0.4, 0.3), (0.85, 0.15, 0.2), (0.98, 0.82, 0.3), (0.95, 0.55, 0.6), (0.95, 0.95, 0.9)],
                   "tuna_nigiri": [(0.8, 0.12, 0.18)] * 2, "salmon_roll": [(0.05, 0.1, 0.06)] * 3,
                   "sashimi": [(0.95, 0.35, 0.4), (0.95, 0.45, 0.5), (0.98, 0.7, 0.7)],
                   "plate_tamago": [(0.98, 0.82, 0.3)] * 2, "plate_salmon": [(0.96, 0.45, 0.28)] * 2,
                   "plate_unagi": [(0.28, 0.14, 0.08)] * 2, "plate_otoro": [(0.98, 0.62, 0.66)] * 2}[key]
            n = len(top)
            for i, c in enumerate(top):
                x = (i - (n - 1) / 2) * (0.075 if n > 3 else 0.11)
                if key == "salmon_roll":
                    P("cube", (x, 0.045, 0), (0.07, 0.06, 0.07), c)
                    P("cube", (x, 0.077, 0), (0.05, 0.01, 0.05), (0.98, 0.5, 0.35))
                elif key == "sashimi":
                    P("cube", (x, 0.03 + 0.01 * i, 0), (0.08, 0.025, 0.14), c, 0.08)
                else:
                    P("cube", (x, 0.04, 0), (0.07, 0.04, 0.12), (0.97, 0.97, 0.95))
                    P("cube", (x, 0.07, 0), (0.08, 0.025, 0.13), c, 0.1)
    elif key == "green_tea":
        P("cube", (0, 0.045, 0), (0.07, 0.09, 0.07), (0.9, 0.92, 0.88))
        if not empty:
            P("cube", (0, 0.088, 0), (0.058, 0.008, 0.058), (0.45, 0.62, 0.22), 0.3)
    elif key == "ramune":
        P("cube", (0, 0.1, 0), (0.06, 0.2, 0.06), (0.6, 0.85, 0.95))
        if not empty:
            P("cube", (0, 0.11, 0), (0.05, 0.14, 0.05), (0.4, 0.75, 0.95), 0.3)
            P("sphere", (0, 0.21, 0), (0.035, 0.035, 0.035), (0.9, 0.95, 1.0))
    elif key == "sake":
        P("sphere", (0, 0.07, 0), (0.09, 0.14, 0.09), (0.93, 0.93, 0.9))
        if not empty:
            P("cube", (0.14, 0.03, 0.0), (0.04, 0.05, 0.04), (0.97, 0.97, 0.95))
            P("cube", (0.14, 0.055, 0.0), (0.03, 0.006, 0.03), (0.96, 0.95, 0.8), 0.3)
    elif key == "mochi":
        P("cube", (0, 0.01, 0), (0.30, 0.02, 0.16), (0.9, 0.8, 0.7))
        if not empty:
            for i, c in enumerate(((0.99, 0.95, 0.96), (0.98, 0.74, 0.8), (0.82, 0.92, 0.7))):
                P("sphere", (-0.09 + 0.09 * i, 0.05, 0), (0.08, 0.07, 0.08), c, 0.1)
    else:
        P("sphere", (0, 0.05, 0), (0.2, 0.1, 0.2), (0.9, 0.9, 0.9))
    return root

# ---- SAKURA-YA layout (chunk-local x; z is relative to the chunk start) -----------------------------------------
JP_L = dict(
    stool_x=-7.0, counter_x=-8.0, dish_x=-7.7, chef_front=-9.0, chef_back=-10.2, pass_dish=(-7.7, 21.9), pass_srv=(-6.5, 21.9),
    ramen_z=tuple(5.6 + 1.7 * k for k in range(10)),
    belt_x=9.0, belt_z0=4.3, belt_z1=20.6, rstool_x=7.5, rdish_x=8.2, belt_seat_z=tuple(5.0 + 1.35 * k for k in range(12)),
    chefR=(10.35, 5.0),
    lanes_x=(-5.5, 0.0, 5.5), lanes_z=(3.0, 9.0, 15.0, 21.8),
    tables=((-2.7, 6.0), (2.7, 6.0), (-2.7, 12.0), (2.7, 12.0), (-2.7, 18.0), (2.7, 18.0)),
    # ramen chef stations: (x, z, yaw)   yaw -90 = facing the back counter, +90 = facing the customers
    st=dict(pot=(-10.2, 7.7, -90.0), pan=(-10.2, 10.0, -90.0), board=(-10.2, 12.3, -90.0), micro=(-10.2, 14.6, -90.0),
            sushi=(-10.2, 17.4, -90.0), tea=(-10.2, 20.2, -90.0), dress=(-9.2, 9.0, 90.0), idle=(-9.6, 13.0, 90.0)),
    doors_z=(0.3, 23.7),
)


class V4Builders:
    """The rare SAKURA-YA zone.  Installed onto ChunkBuilder (see _v4_install)."""

    def _jp_ell(self, x, y, z, sx, sy, sz, col, em=0.0, shape="sphere"):
        return mk(self.par, shape, (x, y, z), (sx, sy, sz), "white", lm=self.ch.lm, lamp=self.ch.lamp, xc=self.ch.xoff, color=col, emissive=em)

    def _jp_stool(self, x, z, yaw, h, data, col):
        self.box(x, h, z, 0.40, 0.08, 0.40, "wood", color=col)
        self.box(x, h / 2, z, 0.07, h, 0.07, "metal", color=C(0.22, 0.2, 0.2))
        self.box(x, 0.03, z, 0.34, 0.05, 0.34, "metal", color=C(0.2, 0.19, 0.19))
        self.solid(x - 0.2, x + 0.2, z - 0.2, z + 0.2, h + 0.02)
        it = self.inter("sit", x, h + 0.02, z, 1.55, "Sit", yaw, data)
        return it

    def _jp_shoji(self, x, y, z, w, h, face, lat=True):
        """Paper screen with a dark lattice (emissive paper, merged bars)."""
        self.quad(x, y, z, w, h, face, "white", color=C(1.0, 0.90, 0.72), emissive=0.75, double=False)
        bars = []
        s = 1 if face == "-x" else -1
        for i in range(-int(w / 0.6) // 2, int(w / 0.6) // 2 + 1):
            bars.append((x + s * -0.02, y, z + i * 0.6, 0.04, h, 0.04))
        for j in range(5):
            bars.append((x + s * -0.02, y - h / 2 + j * h / 4, z, 0.04, 0.04, w))
        self.merged(bars, "wood", color=C(0.2, 0.11, 0.07))

    def _jp_wave_poster(self):
        key = "jp_wave"
        if key not in BANK._tex:
            im = Image.new("RGBA", (384, 224), (238, 226, 196, 255))
            d = ImageDraw.Draw(im)
            d.ellipse([280, 28, 344, 92], fill=(200, 40, 36, 255))
            for i in range(8):
                y = 110 + i * 14
                d.rectangle([0, y, 384, y + 14], fill=(24 + i * 6, 52 + i * 8, 110 + i * 9, 255))
            for k in range(7):
                cx = 40 + k * 52
                d.pieslice([cx - 42, 78 + (k % 2) * 10, cx + 42, 170 + (k % 2) * 10], 180, 360, outline=(250, 248, 240, 255), width=5)
                d.arc([cx - 28, 92, cx + 28, 150], 190, 330, fill=(250, 248, 240, 255), width=3)
            d.rectangle([0, 0, 383, 223], outline=(70, 40, 24, 255), width=8)
            BANK._tex[key] = BANK.to_texture(im)
        return BANK._tex[key]

    def _jp_building_shell(self):
        ch, z0, hw, H = self.ch, self.ch.z0, self.ch.hw, self.ch.h
        zc = z0 + CH / 2
        self._shell("wood", "panel", C(0.30, 0.17, 0.11), walls=(), ceil_tex="panel")
        jp_street_walls(self)
        # floors: tatami in the dining hall, slate by the belt, steel-grey tile in both kitchens
        self.box(0, 0.012, zc, 10.8, 0.02, CH - 0.8, "plaster", tile=(6, 6), color=C(0.66, 0.66, 0.40))
        lines = [(0, 0.024, z0 + 0.4 + 1.8 * k, 10.8, 0.01, 0.05) for k in range(14)] + [(x, 0.024, zc, 0.05, 0.01, CH - 0.8) for x in (-5.4, -1.8, 1.8, 5.4)]
        self.merged(lines, "white", color=C(0.10, 0.07, 0.05))
        self.box(7.1, 0.012, zc, 3.4, 0.02, CH - 0.8, "marble", tile=(1.5, 8), color=C(0.28, 0.30, 0.34))
        self.box(-6.9, 0.012, zc, 3.0, 0.02, CH - 0.8, "wood", tile=(1.5, 8), color=C(0.45, 0.30, 0.20))
        self.box(-10.2, 0.014, z0 + 14.0, 3.6, 0.02, 18.2, "marble", tile=(1.4, 8), color=C(0.62, 0.62, 0.66))
        self.box(10.75, 0.014, z0 + 12.7, 2.5, 0.02, 18.4, "marble", tile=(1.2, 8), color=C(0.62, 0.62, 0.66))
        # ceiling: dark beams + lantern rows
        beams = [(0, H - 0.2, z0 + 1.5 + 3.0 * k, 23.6, 0.2, 0.3) for k in range(8)] + [(x, H - 0.3, zc, 0.3, 0.2, CH - 0.8) for x in (-9, -4.5, 0, 4.5, 9)]
        self.merged(beams, "wood", color=C(0.20, 0.11, 0.07))
        self.par = ch.detail
        caps, cords, minis = [], [], []
        for lz in (3.0, 9.0, 15.0, 21.0):
            for lx in (-6.8, 0.0, 6.8):
                y = H - 1.5
                e = self._jp_ell(lx, y, z0 + lz, 0.55, 0.72, 0.55, C(0.95, 0.20, 0.12), 1.55)
                self.strip(e, 1.55, 0.35)
                caps += [(lx, y + 0.38, z0 + lz, 0.30, 0.06, 0.30), (lx, y - 0.38, z0 + lz, 0.30, 0.06, 0.30)]
                cords.append((lx, (y + 0.4 + H - 0.3) / 2, z0 + lz, 0.02, H - 0.3 - y - 0.4, 0.02))
        for mz in range(1, 12):
            for mx in (-8.45, 8.55):
                minis.append((mx, 3.05, z0 + 2.0 * mz, 0.2, 0.28, 0.2))
        self.merged(caps, "white", color=C(0.04, 0.03, 0.03))
        self.merged(cords, "white", color=C(0.05, 0.04, 0.04))
        self.strip(self.merged(minis, "white", color=C(0.95, 0.25, 0.14), emissive=1.35), 1.35, 0.25)

    # ---- entrance doors (both ends: the Path runs straight through the restaurant) ------------------------------------
    def _jp_door(self, zrel, outward):
        ch, z0, hw, H, xo = self.ch, self.ch.z0, self.ch.hw, self.ch.h, self.ch.xoff
        z = z0 + zrel
        inner = "+z" if outward == "-z" else "-z"
        dark = C(0.30, 0.17, 0.11)
        gap = 1.9
        for s in (-1, 1):
            self.block(s * (gap + (12.0 - gap) / 2), H / 2, z, 12.0 - gap, H, 0.4, "panel", color=dark)
        self.box(0, 3.95, z, 2 * gap, 2.5, 0.4, "panel", color=dark)
        self.quad(0, 4.1, z + (-0.22 if outward == "-z" else 0.22), 3.6, 1.2, outward,
                  jp_sign("さくら屋", "SAKURA-YA", (255, 226, 190), (70, 12, 14), 640, 160, "sushi  /  ramen  /  izakaya"), emissive=1.3, double=False)
        self.quad(0, 4.1, z + (0.22 if outward == "-z" else -0.22), 3.6, 1.2, inner,
                  jp_sign("いらっしゃいませ", "IRASSHAIMASE", (255, 226, 190), (70, 12, 14), 640, 160, "welcome"), emissive=1.3, double=False)
        for s in (-1, 1):
            self.box(s * gap, 1.5, z, 0.22, 3.0, 0.5, "wood", color=C(0.16, 0.08, 0.05))
        self.box(0, 3.02, z, 2 * gap + 0.3, 0.16, 0.5, "wood", color=C(0.16, 0.08, 0.05))
        # sliding shoji panels
        zoff = 0.26 if outward == "-z" else -0.26
        panels = []
        for s in (-1, 1):
            pv = ur.Entity(parent=ch.detail, position=(s * gap / 2, 0, z + zoff))
            for (px, py, pz, sx, sy, sz, col, em) in (
                    (0, 1.45, 0, gap - 0.04, 2.9, 0.06, (1.0, 0.9, 0.72), 0.7),
                    (0, 0.05, 0, gap - 0.04, 0.1, 0.1, (0.18, 0.1, 0.06), 0.0), (0, 2.9, 0, gap - 0.04, 0.1, 0.1, (0.18, 0.1, 0.06), 0.0),
                    (-(gap - 0.04) / 2, 1.45, 0, 0.08, 2.9, 0.1, (0.18, 0.1, 0.06), 0.0), ((gap - 0.04) / 2, 1.45, 0, 0.08, 2.9, 0.1, (0.18, 0.1, 0.06), 0.0),
                    (0, 1.45, 0, 0.05, 2.9, 0.09, (0.18, 0.1, 0.06), 0.0), (0, 0.95, 0, gap - 0.04, 0.05, 0.09, (0.18, 0.1, 0.06), 0.0),
                    (0, 1.9, 0, gap - 0.04, 0.05, 0.09, (0.18, 0.1, 0.06), 0.0)):
                mk(pv, "cube", (px, py, pz), (sx, sy, sz), "white", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(*col), emissive=em)
            panels.append((pv, s))
        strips = []
        for k in range(4):
            px = -1.2 + 0.8 * k
            pv = ur.Entity(parent=ch.detail, position=(px, 2.95, z + (0.55 if outward == "-z" else -0.55)))
            mk(pv, "cube", (0, -0.55, 0), (0.74, 1.1, 0.03), "white", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(0.07, 0.09, 0.24))
            mk(pv, "cube", (0, -0.62, 0.018), (0.34, 0.34, 0.01), "white", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(0.95, 0.93, 0.88), emissive=0.2)
            strips.append((pv, k * 1.3))
        coll = (-gap + xo, gap + xo, z - 0.2, z + 0.2, 3.0)
        ch.colliders.append(coll)
        ch.extra.setdefault("jp_doors", []).append(dict(z=z, panels=panels, strips=strips, open=0.0, coll=coll, gap=gap, solid=True, was=False, side=outward))

    # ---- left: ramen bar + kitchen -------------------------------------------------------------------------------------
    def _jp_ramen(self):
        ch, z0, rng, H, xo = self.ch, self.ch.z0, self.rng, self.ch.h, self.ch.xoff
        L, V = JP_L, ch.extra["jp_vis"]
        lacq, dark = C(0.16, 0.07, 0.05), C(0.12, 0.06, 0.04)
        self.block(L["counter_x"], 0.52, z0 + 13.9, 1.0, 1.04, 17.8, "wood", tile=(1, 6), color=C(0.34, 0.15, 0.10))
        self.box(L["counter_x"], 1.06, z0 + 13.9, 1.2, 0.07, 18.0, "wood", tile=(1, 6), color=C(0.55, 0.32, 0.16))
        self.box(-7.45, 0.55, z0 + 13.9, 0.06, 1.0, 17.8, color=C(0.55, 0.08, 0.08))
        self.block(-11.4, 0.55, z0 + 4.95, 1.2, 1.1, 0.3, "wood", color=C(0.28, 0.13, 0.08))          # north end: a doorway into the washing nook stays open
        self.block(-10.25, 0.55, z0 + 22.8 + 0.2, 3.5, 1.1, 0.3, "wood", color=C(0.28, 0.13, 0.08))
        self.box(-8.0, 1.5, z0 + 5.0, 0.06, 0.8, 0.06, "wood", color=dark)
        # back counter & stations
        self.block(-11.35, 0.48, z0 + 14.0, 0.9, 0.96, 17.0, "metal", tile=(1, 6), color=C(0.72, 0.74, 0.78))
        self.box(-11.35, 0.98, z0 + 14.0, 1.0, 0.04, 17.1, "metal", color=C(0.85, 0.87, 0.9))
        st = L["st"]
        zp = z0 + st["pot"][1]
        self.box(-11.3, 1.08, zp, 0.8, 0.2, 0.9, color=C(0.1, 0.1, 0.12))
        self.box(-11.25, 1.55, zp, 0.62, 0.6, 0.62, "metal", color=C(0.84, 0.86, 0.9))
        self.box(-11.25, 1.88, zp, 0.68, 0.06, 0.68, "metal", color=C(0.7, 0.72, 0.78))
        self.ball(-10.98, 1.35, zp, 0.05, color=C(0.1, 0.1, 0.12))
        self.box(-10.9, 1.1, zp + 0.25, 0.04, 0.05, 0.04, color=C(1.0, 0.45, 0.1), emissive=1.8)
        V["steam"] = [(self.ball(-11.25, 2.0 + 0.12 * k, zp, 0.12, color=C(0.95, 0.95, 0.98, 0.35), emissive=0.5, alpha=True), k * 0.8) for k in range(6)]
        V["pot_z"] = zp
        zp2 = z0 + st["pan"][1]
        self.box(-11.3, 1.04, zp2, 0.8, 0.12, 0.8, color=C(0.1, 0.1, 0.12))
        self.box(-11.3, 1.12, zp2, 0.55, 0.05, 0.55, color=C(0.07, 0.07, 0.08))
        self.box(-11.0, 1.13, zp2 + 0.5, 0.05, 0.03, 0.5, color=C(0.1, 0.08, 0.06))
        zb = z0 + st["board"][1]
        self.box(-11.3, 1.01, zb, 0.62, 0.05, 1.0, "wood", color=C(0.78, 0.60, 0.38))
        self.merged([(-11.28, 1.045, zb - 0.28 + 0.1 * k, 0.05, 0.03, 0.05) for k in range(5)], "white", color=C(0.25, 0.65, 0.22))
        self.box(-11.55, 1.06, zb + 0.55, 0.12, 0.12, 0.35, "wood", color=C(0.2, 0.1, 0.06))
        zm = z0 + st["micro"][1]
        self.box(-11.5, 1.18, zm, 0.55, 0.40, 0.58, "metal", color=C(0.82, 0.84, 0.88))
        self.box(-11.25, 1.20, zm - 0.02, 0.01, 0.30, 0.40, "glass", alpha=True, color=C(0.05, 0.06, 0.08, 0.85))
        V["mw_light"] = self.box(-11.30, 1.20, zm - 0.02, 0.3, 0.28, 0.36, color=C(1.0, 0.85, 0.5), emissive=0.0)
        self.quad(-11.215, 1.20, zm + 0.26, 0.12, 0.30, "+x", "white", color=C(0.12, 0.12, 0.15), emissive=0.1)
        self.quad(-11.21, 1.28, zm + 0.26, 0.10, 0.05, "+x", BANK.sign("0:00", (120, 255, 150), (4, 20, 8), 128, 48), emissive=1.4, double=False)
        hv = ur.Entity(parent=ch.detail, position=(-11.19, 1.20, zm - 0.24))
        mk(hv, "cube", (0.0, 0, 0.21), (0.03, 0.30, 0.42), "white", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(0.1, 0.11, 0.14))
        mk(hv, "cube", (0.02, 0, 0.21), (0.012, 0.26, 0.38), "glass", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(0.55, 0.65, 0.7, 0.4), alpha=True)
        V["mw_door"], V["mw_z"] = hv, zm
        zs = z0 + st["sushi"][1]
        self.box(-11.3, 1.01, zs, 0.7, 0.05, 2.0, "wood", color=C(0.82, 0.66, 0.42))
        self.box(-11.28, 1.06, zs - 0.65, 0.5, 0.14, 0.5, "wood", color=C(0.62, 0.4, 0.22))
        self.box(-11.28, 1.14, zs - 0.65, 0.42, 0.03, 0.42, color=C(0.98, 0.98, 0.95))
        for k, col in enumerate(((0.92, 0.42, 0.30), (0.78, 0.10, 0.16), (0.98, 0.78, 0.30))):
            self.box(-11.35, 1.06, zs + 0.0 + 0.3 * k, 0.28, 0.07, 0.24, color=C(*col), emissive=0.12)
        self.ball(-11.2, 1.07, zs + 0.9, 0.05, color=C(0.35, 0.7, 0.2))
        self.box(-11.0, 1.04, zs + 0.5, 0.38, 0.015, 0.38, "wood", color=C(0.85, 0.8, 0.5))
        self.quad(-11.75, 2.45, zs, 1.5, 0.35, "+x", jp_sign("特別握り", "PRIVATE SUSHI", (255, 226, 190), (40, 10, 14), 384, 96), emissive=1.1, double=False)
        zt = z0 + st["tea"][1]
        self.box(-11.35, 1.35, zt - 0.3, 0.32, 0.7, 0.32, "metal", color=C(0.7, 0.72, 0.76))
        self.box(-11.1, 1.3, zt - 0.3, 0.08, 0.05, 0.08, color=C(0.2, 0.2, 0.22))
        self.merged([(-11.3, 1.1, zt + 0.25 + 0.14 * k, 0.07, 0.2, 0.07) for k in range(5)], "white", color=C(0.5, 0.82, 0.95), emissive=0.2)
        self.merged([(-11.3, 1.07, zt + 0.95 + 0.0, 0.08, 0.14, 0.08), (-11.3, 1.07, zt + 1.1, 0.08, 0.14, 0.08)], "white", color=C(0.93, 0.93, 0.88))
        # wall shelves with bowls & bottles
        self.merged([(-11.7, 2.05, z0 + 14.0, 0.36, 0.05, 17.0), (-11.7, 2.65, z0 + 14.0, 0.36, 0.05, 17.0)], "wood", color=C(0.30, 0.16, 0.09))
        items = []
        for k in range(36):
            zz = z0 + 5.8 + k * 0.45
            items.append((-11.65, 2.12 + 0.6 * (k % 2), zz, 0.16, 0.14 + 0.1 * (k % 3 == 0), 0.2))
        self.merged(items, "white", color=C(0.88, 0.9, 0.92))
        self.merged([(-11.65, 2.12 + 0.6 * (k % 2) + 0.2, z0 + 6.0 + k * 0.9, 0.08, 0.24, 0.08) for k in range(18)], "white", color=C(0.30, 0.52, 0.34), emissive=0.15)
        for zz in (8.5, 13.0, 17.5):
            self.quad(-11.88, 3.6, z0 + zz, 1.6, 1.0, "+x", jp_sign("らーめん", ("RAMEN", "GYOZA", "SAKE")[(int(zz) // 5) % 3], (255, 226, 190), (46, 10, 12), 384, 128), emissive=1.2, double=False)
        # outside the counter: stools, bells, banners, menu boards
        for k, z in enumerate(L["ramen_z"]):
            it = self._jp_stool(L["stool_x"], z0 + z, FACE_YAW["-x"], 0.64, {}, C(0.55, 0.1, 0.1))
            rec = dict(kind="ramen", sid=k, x=it.x, z=it.z, y=it.y, yaw=it.yaw, it=it, slot=(L["dish_x"] + xo, z0 + z),
                       stand=(L["chef_front"] + xo, z0 + z), occ=None,
                       bell=(-7.75 + xo, z0 + L["ramen_z"][min(9, (k // 2) * 2 + 1)] - 0.97))
            it.data["jp"] = rec
            ch.extra["jp_seats"].append(rec)
        for zz in (L["ramen_z"][1] - 0.97, L["ramen_z"][3] - 0.97, L["ramen_z"][5] - 0.97, L["ramen_z"][7] - 0.97, L["ramen_z"][9] - 0.97):        
            self.ball(-7.75, 1.12, z0 + zz, 0.06, color=C(0.95, 0.78, 0.25), emissive=0.3)
            self.box(-7.75, 1.10, z0 + zz, 0.14, 0.02, 0.14, color=C(0.2, 0.1, 0.05))
        self.merged([(-7.95, 1.12, z0 + z + dz, 0.04, 0.03, 0.2) for z in L["ramen_z"] for dz in (-0.12,)], "white", color=C(0.1, 0.06, 0.04))
        for zz in (7.0, 13.0, 19.0):
            self.box(-6.55, 3.55, z0 + zz, 0.03, 2.2, 0.02, color=C(0.1, 0.1, 0.1))
            self.quad(-6.55, 3.8, z0 + zz, 0.62, 1.7, "+x", jp_sign("ラーメン", "RAMEN", (255, 240, 220), (150, 14, 14), 128, 384, vertical=True), emissive=1.1, double=True)

    # ---- right: kaiten (conveyor) sushi ---------------------------------------------------------------------------------
    def _jp_belt(self):
        ch, z0, rng, H, xo = self.ch, self.ch.z0, self.rng, self.ch.h, self.ch.xoff
        L, V = JP_L, ch.extra["jp_vis"]
        self.block(8.75, 0.5, z0 + 12.7, 1.5, 1.0, 17.4, "wood", tile=(1, 6), color=C(0.30, 0.13, 0.09))
        self.box(8.4, 1.005, z0 + 12.7, 0.9, 0.04, 17.4, "wood", tile=(1, 6), color=C(0.60, 0.38, 0.20))
        V["belt"] = self.box(L["belt_x"], 1.03, z0 + 12.45, 0.42, 0.04, 16.5, "rubber", tile=(0.4, 14), color=C(0.55, 0.57, 0.62))
        for dx in (-0.25, 0.25):
            self.box(L["belt_x"] + dx, 1.05, z0 + 12.45, 0.04, 0.05, 16.5, "metal", color=C(0.85, 0.86, 0.9))
        self.box(9.6, 1.55, z0 + 12.7, 0.05, 1.1, 17.4, "glass", alpha=True, color=C(0.7, 0.85, 0.9, 0.35))
        for zz, nm in ((3.5, "a"), (21.0, "b")):                        # tunnels the plates vanish into
            self.block(9.0, 1.45, z0 + zz + (0.35 if nm == "a" else 0.0), 1.5, 1.0, 0.8, "wood", color=C(0.20, 0.09, 0.06))
        self.box(9.0, 2.15, z0 + 21.0, 1.5, 0.25, 0.9, "wood", color=C(0.45, 0.10, 0.08))
        self.block(9.0, 0.5, z0 + 3.7, 2.0, 1.0, 0.4, "wood", color=C(0.30, 0.13, 0.09))                  # north gate leaves an aisle for Hiro
        self.block(10.0, 0.5, z0 + 21.7, 4.0, 1.0, 0.4, "wood", color=C(0.30, 0.13, 0.09))
        # chef's side
        self.block(11.35, 0.48, z0 + 5.3, 0.9, 0.96, 3.6, "metal", color=C(0.78, 0.8, 0.84))
        self.box(11.35, 0.98, z0 + 5.3, 1.0, 0.04, 3.7, "wood", color=C(0.82, 0.66, 0.42))
        self.box(11.3, 1.02, z0 + 4.3, 0.55, 0.05, 0.9, "wood", color=C(0.88, 0.76, 0.52))
        self.box(11.22, 1.12, z0 + 5.8, 0.42, 0.16, 0.42, "wood", color=C(0.58, 0.37, 0.2))
        self.box(11.22, 1.21, z0 + 5.8, 0.36, 0.03, 0.36, color=C(0.98, 0.98, 0.95))
        for k, col in enumerate(((0.93, 0.42, 0.30), (0.78, 0.10, 0.16), (0.98, 0.78, 0.30), (0.2, 0.12, 0.08))):
            self.box(11.3, 1.06, z0 + 6.4 + 0.3 * k, 0.28, 0.08, 0.24, color=C(*col), emissive=0.12)
        self.block(11.4, 0.55, z0 + 9.8, 0.8, 1.1, 3.2, "metal", color=C(0.82, 0.86, 0.9))       # fridge case
        self.box(10.98, 0.62, z0 + 9.8, 0.04, 0.8, 3.0, "glass", alpha=True, color=C(0.7, 0.95, 1.0, 0.4))
        self.merged([(11.3, 0.3 + 0.28 * (k % 3), z0 + 8.4 + 0.5 * k, 0.4, 0.1, 0.34) for k in range(6)], "white", color=C(0.93, 0.45, 0.35), emissive=0.2)
        self.merged([(11.3, 0.3 + 0.28 * ((k + 1) % 3), z0 + 8.65 + 0.5 * k, 0.4, 0.1, 0.34) for k in range(5)], "white", color=C(0.85, 0.14, 0.2), emissive=0.2)
        self.merged([(11.7, 2.05, z0 + 12.7, 0.36, 0.05, 15.0), (11.7, 2.7, z0 + 12.7, 0.36, 0.05, 15.0)], "wood", color=C(0.30, 0.16, 0.09))
        self.merged([(11.65, 2.12 + 0.65 * (k % 2), z0 + 6.0 + k * 0.5, 0.16, 0.14, 0.2) for k in range(26)], "white", color=C(0.9, 0.9, 0.93))
        self.quad(11.88, 3.65, z0 + 9.5, 3.4, 1.0, "-x", jp_sign("回転寿司", "KAITEN SUSHI", (255, 226, 190), (30, 8, 38), 512, 144, "plates 2 - 6 c"), emissive=1.2, double=False)
        self.quad(11.88, 3.3, z0 + 16.8, 3.0, 1.7, "-x", self._jp_wave_poster(), emissive=0.7, double=False)
        self.quad(11.88, 2.3, z0 + 20.0, 1.3, 1.3, "-x", BANK.board("PLATES", ["white   2 c", "red     3 c", "black   4 c", "gold    6 c", "E: take plate"]), emissive=0.6, double=False)
        for k, z in enumerate(L["belt_seat_z"]):
            it = self._jp_stool(L["rstool_x"], z0 + z, FACE_YAW["+x"], 0.64, {}, C(0.10, 0.12, 0.30))
            rec = dict(kind="belt", sid=k, x=it.x, z=it.z, y=it.y, yaw=it.yaw, it=it, bell=None, slot=(L["rdish_x"] + xo, z0 + z), occ=None)
            it.data["jp"] = rec
            ch.extra["jp_seats"].append(rec)
        self.merged([(8.15, 1.04, z0 + z + dz, 0.05, 0.03, 0.2) for z in L["belt_seat_z"] for dz in (-0.18,)], "white", color=C(0.1, 0.06, 0.04))
        self.merged([(8.15, 1.07, z0 + z + 0.2, 0.07, 0.1, 0.07) for z in L["belt_seat_z"]], "white", color=C(0.30, 0.10, 0.06))
        # plates ride on this pool of entities (see JpRestaurant)
        ch.extra["belt_ents"] = []

    # ---- middle: dining hall ---------------------------------------------------------------------------------------------
    def _jp_tables(self):
        ch, z0, rng, xo = self.ch, self.ch.z0, self.rng, self.ch.xoff
        L = JP_L
        wood = C(0.30, 0.16, 0.10)
        for tid, (tx, tz) in enumerate(L["tables"]):
            z = z0 + tz
            self.table(tx, z, 1.1, 1.6, 0.76, "wood", wood)
            self.box(tx, 0.8, z, 0.12, 0.05, 0.12, color=C(0.95, 0.78, 0.25))                                  # bell base
            self.ball(tx, 0.85, z, 0.055, color=C(0.97, 0.80, 0.30), emissive=0.35)
            self._jp_ell(tx, 0.96, z + 0.55, 0.17, 0.24, 0.17, C(1.0, 0.82, 0.55), 1.3, "cube")                # andon lamp
            self.merged([(tx + 0.12, 0.79, z - 0.55, 0.06, 0.06, 0.06), (tx + 0.1, 0.8, z + 0.0, 0.04, 0.04, 0.22)], "white", color=C(0.4, 0.22, 0.12))
            self.merged([(tx, 0.775, z - 0.05, 0.9, 0.012, 1.4)], "white", color=C(0.62, 0.62, 0.40))
            for side, yaw in ((-1, 90), (1, -90)):
                for dz in (0.0,):
                    cx, cz = tx + side * 0.85, z + dz
                    self.chair(cx, cz, yaw, color=C(0.52, 0.14, 0.12))
                    it = ch.inters[-1]
                    if abs(it.x - (cx + xo)) > 1e-6:
                        raise RuntimeError("jp chair/interactable mismatch")
                    srv_x = -1.0 if tx < 0 else 1.0
                    rec = dict(kind="table", tid=tid, x=it.x, z=it.z, y=it.y, yaw=it.yaw, it=it, bell=(tx + xo, z), slot=(tx - side * 0.35 + xo, z + dz),
                               spot=(srv_x + xo, z), occ=None, tz=z)
                    it.data["jp"] = rec
                    ch.extra["jp_seats"].append(rec)
        ch.extra["jp_tables"] = [(tx + xo, z0 + tz) for tx, tz in L["tables"]]

    # ---- entrance details + decor --------------------------------------------------------------------------------------
    def _jp_decor(self):
        ch, z0, rng, H, xo = self.ch, self.ch.z0, self.rng, self.ch.h, self.ch.xoff
        V = ch.extra["jp_vis"]
        # lucky cat cashier
        self.block(4.0, 0.55, z0 + 1.7, 1.3, 1.1, 0.7, "wood", color=C(0.30, 0.15, 0.09))
        self.box(4.0, 1.12, z0 + 1.7, 1.4, 0.05, 0.8, "wood", color=C(0.62, 0.40, 0.22))
        self.box(4.5, 1.25, z0 + 1.6, 0.35, 0.22, 0.3, color=C(0.14, 0.14, 0.17))
        self.box(4.5, 1.37, z0 + 1.5, 0.3, 0.02, 0.2, color=C(0.3, 0.9, 0.5), emissive=0.8)
        wh = (0.97, 0.96, 0.92)
        self.ball(3.65, 1.42, z0 + 1.7, 0.2, color=C(*wh), emissive=0.15)
        self.ball(3.65, 1.7, z0 + 1.7, 0.16, color=C(*wh), emissive=0.15)
        for sx in (-0.07, 0.07):
            self.box(3.65 + sx, 1.86, z0 + 1.7, 0.05, 0.07, 0.04, color=C(*wh))
        self.box(3.65, 1.58, z0 + 1.52, 0.2, 0.05, 0.03, color=C(0.85, 0.1, 0.1))
        self.ball(3.65, 1.5, z0 + 1.5, 0.045, color=C(0.97, 0.8, 0.25), emissive=0.6)
        pv = ur.Entity(parent=ch.detail, position=(3.8, 1.55, z0 + 1.6))
        mk(pv, "cube", (0.0, 0.1, 0.0), (0.07, 0.24, 0.07), "white", lm=ch.lm, lamp=ch.lamp, xc=xo, color=C(*wh), emissive=0.15)
        V["neko"] = pv
        # shoe cabinet
        self.block(-4.0, 0.55, z0 + 1.4, 1.6, 1.1, 0.7, "wood", color=C(0.25, 0.12, 0.08))
        self.merged([(-4.0 + dx, 0.3 + dy, z0 + 1.03, 0.4, 0.2, 0.02) for dx in (-0.5, 0.0, 0.5) for dy in (0.0, 0.45)], "white", color=C(0.12, 0.06, 0.04))
        # sakura trees + bonsai
        petals = []
        for tx, tz in ((-4.9, 22.8), (4.9, 22.8), (-4.9, 1.0)):
            self.block(tx, 0.3, z0 + tz, 0.7, 0.6, 0.7, "wood", color=C(0.2, 0.1, 0.07))
            self.box(tx, 1.2, z0 + tz, 0.14, 1.4, 0.14, color=C(0.22, 0.12, 0.08), rot=(0, 0, 6))
            for k, (dx, dy, dz, r) in enumerate(((0, 2.2, 0, 0.6), (0.4, 2.0, 0.2, 0.45), (-0.4, 2.05, -0.2, 0.45), (0.1, 2.5, -0.2, 0.4), (-0.2, 2.35, 0.35, 0.38))):
                self.ball(tx + dx, dy, z0 + tz + dz, r, color=C(1.0, 0.72 + 0.04 * (k % 2), 0.80), emissive=0.3)
            for k in range(7):
                e = self.box(tx, 2.0, z0 + tz, 0.07, 0.01, 0.05, color=C(1.0, 0.75, 0.82), emissive=0.5)
                petals.append([e, tx, tz, rng.uniform(0, 6.28), rng.uniform(0.25, 0.5)])
        V["petals"] = petals
        # banners & signage on the side walls
        self.ad(-11.65, 2.5, z0 + 2.4, 1.6, 2.2, "+x", f"{ch.idx}:jp0")
        self.ad(11.65, 2.5, z0 + 2.4, 1.6, 2.2, "-x", f"{ch.idx}:jp1")
        self.quad(-11.9, 3.7, z0 + 2.4, 1.4, 0.7, "+x", jp_sign("ようこそ", "WELCOME", (255, 226, 190), (46, 10, 12), 256, 128), emissive=1.1, double=False)
        # maneki-neko style paper charms hung in the dining hall
        self.merged([(x, H - 0.9, z0 + z, 0.03, 0.9, 0.03) for x in (-1.4, 1.4) for z in (6.0, 12.0, 18.0)], "white", color=C(0.04, 0.03, 0.03))
        self.merged([(x, H - 1.45, z0 + z, 0.28, 0.28, 0.02) for x in (-1.4, 1.4) for z in (6.0, 12.0, 18.0)], "white", color=C(0.95, 0.4, 0.5), emissive=0.8)

    def _build_japan(self):
        ch = self.ch
        ch.extra["jp"] = True
        ch.extra["jp_vis"] = {}
        ch.extra["jp_seats"] = []
        self._jp_building_shell()
        self._jp_door(JP_L["doors_z"][0], "-z")
        self._jp_door(JP_L["doors_z"][1], "+z")
        self.par = ch.detail
        self._jp_ramen()
        self._jp_belt()
        self._jp_tables()
        self._jp_decor()

# ---- people of SAKURA-YA ---------------------------------------------------------------------------------------------
JP_COMPLIMENTS = ("Oishii! Taisho, sugoi!", "Umai! Best in the mall!", "Mmm! The chef is a genius!", "Oishii desu! Arigatou!", "So good. Gochisousama!")
JP_CHAT = ("Nee, try the gyoza.", "The belt never ends. I love it.", "Is it three o'clock already?", "Just one more plate...", "This broth is a hug.",
           "Kanpai!", "Hm. Hm-hm. Delicious.", "Another tea, please.")
JP_STAFF_GREET = ("Irasshaimase!!", "Irasshaimase! Welcome!", "Irasshai! Please, come in!")


def _an(a, name: str, e: float, d: float) -> None:
    s = math.sin
    if name == "boil":
        k = 0.5 + 0.5 * s(e * 2.2)
        a.pose(ls=-62 - 18 * k, le=-25, rs=-62 - 18 * k, re=-25, hx=18)
    elif name == "ladle":
        a.pose(ls=-30, le=-10, rs=-75, re=-32 + 26 * s(e * 2.0), hx=22)
    elif name == "chop":
        c = max(0.0, s(e * 14.0))
        a.pose(ls=-60, le=-38, rs=-56 + 16 * c, re=-30 - 28 * c, hx=26)
    elif name == "fry":
        a.pose(ls=-28, le=-5, rs=-70 + 7 * s(e * 11), re=-12, hx=18)
    elif name == "flip":
        k = min(1.0, e / max(0.1, d))
        a.pose(ls=-28, le=-5, rs=-70 - 34 * s(k * math.pi), re=-12 - 44 * s(k * math.pi), hx=15)
    elif name == "topping":
        a.pose(ls=-40, le=-30, rs=-60 + 8 * s(e * 5), re=-60 + 10 * s(e * 5 + 1), hx=20)
    elif name == "mw_load":
        a.pose(ls=-40, le=-25, rs=-82, re=-10 - 10 * s(e * 3), hx=6)
    elif name == "mw_cook":
        a.pose(ls=-10, le=-5, rs=-66 + 5 * (1 if s(e * 6) > 0.8 else 0), re=-30, hx=8)
    elif name == "mw_take":
        a.pose(ls=-60, le=-30, rs=-70, re=-30, hx=8)
    elif name == "press":
        a.pose(ls=-70, le=-55 - 12 * s(e * 5), rs=-70, re=-55 - 12 * s(e * 5 + 0.6), hx=24)
    elif name == "pour":
        a.pose(ls=-55, le=-22, rs=-72 - 12 * s(e * 2.4), re=-45, hx=12)
    elif name == "place":
        a.pose(ls=-82, le=-12, rs=-82, re=-12, hx=10)
    elif name == "wipe":
        a.pose(ls=-30, le=-10, rs=-72, re=-18 + 0, rz=0, hx=18)
        a.fig.sho[1].rotation_z = s(e * 5.0) * 22.0
    elif name == "cross":
        a.pose(ls=-42, le=-118, rs=-42, re=-118, lz=32, rz=-32, hx=0)
    elif name == "bow":
        k = s(min(1.0, e / max(0.1, d)) * math.pi)
        a.pose(ls=0, le=0, rs=0, re=0, hx=48 * k)
    elif name == "write":
        a.pose(ls=-72, le=-72, rs=-62, re=-82 + 8 * s(e * 9), hx=18)
    elif name == "taste":
        k = min(1.0, e / max(0.1, d))
        a.pose(ls=-20, le=-10, rs=-45, re=-90 - 40 * s(k * math.pi), hx=10)
    elif name == "wave":
        a.pose(ls=0, le=0, rs=-170, re=-12 + 8 * s(e * 9), rz=-10 + 14 * s(e * 9), hx=0)
    elif name == "wash":
        a.pose(ls=-50, le=-55, rs=-62, re=-60 + 12 * s(e * 9), hx=22)
        a.fig.sho[1].rotation_z = s(e * 9.0) * 16.0
    else:
        a.pose(hx=0)


class JpActor(Actor):
    """Restaurant person driven by a generator ("brain"): every yield is one frame."""
    STYLE = "guest"

    def __init__(self, g, ch, pos, rng, name, pal, style, rest, role="talk") -> None:
        Actor.__init__(self, g, ch, pos, rng, name, role, pal)
        self.rest, self.style, self.gen, self.dt = rest, style, None, 0.016
        self.bub: Optional[Bubble] = None
        self.props: dict = {}
        self.carry_ent = None
        self.hips = None
        self.alert = 0.0
        self.dead = False
        jp_style(self.fig, style, rng)
        self.fig.anim = None

    # -- helpers -----------------------------------------------------------------------------------------------------
    def say(self, text: str, dur: float = 3.4, sfx: Optional[str] = None, gain: float = 0.9) -> None:
        if self.bub is None:
            self.bub = Bubble(self.fig)
        self.bub.say(text, dur)
        if sfx:
            jp_say(self.g, self.x, self.z, sfx, gain)

    def pose(self, ls=0.0, le=0.0, rs=0.0, re=0.0, lz=0.0, rz=0.0, hx=0.0, hy=None, rate=12.0) -> None:
        if hy is None:
            hy = 0.0
            if self.dist() < 7.0 or self.alert > 0:
                hy = max(-55.0, min(55.0, wrap180(math.degrees(math.atan2(self.g.player.x - self.x, self.g.player.z - self.z)) - self.yaw)))
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.fig.arms(self.dt, ls, le, rs, re, lz, rz, hx, hy, rate, self.hips)

    def mkprop(self, parent, parts):
        root = ur.Entity(parent=parent)
        for m, p, s, c, e in parts:
            _fm(self.fig, root, m, p, s, c, e)
        root.enabled = False
        return root

    def show_prop(self, name: Optional[str]) -> None:
        for k, v in self.props.items():
            v.enabled = (k == name)

    def set_carry(self, key: Optional[str]) -> None:
        if self.carry_ent is not None:
            ur.destroy(self.carry_ent)
            self.carry_ent = None
        if key:
            self.carry_ent = jp_dish(key, self.fig.root, (0, 1.02, 0.42), self.fig.k)

    def has_work(self) -> bool:
        return False

    def update(self, dt: float, w) -> None:
        if self.gone or self.dead:
            return
        try:
            self.dt = dt
            if self.bub is not None:
                self.bub.tick(dt, self.dist() < 24)
            self.alert = max(0.0, self.alert - dt)
            if self.gen is None:
                self.gen = self.brain()
            try:
                next(self.gen)
            except StopIteration:
                self.gen = None
        except Exception:
            self.dead = True
            log.exception("restaurant person %s failed; freezing them", self.name)

    # -- generator building blocks ----------------------------------------------------------------------------------
    def walk(self, x, z, speed=1.7):
        self.hips = None
        f = self.fig
        while not self.walk_to(self.dt, x, z, speed):
            if self.carry_ent is not None:
                f.sho[0].rotation_x = f.sho[1].rotation_x = -62.0
                f.elb[0].rotation_x = f.elb[1].rotation_x = -35.0
            else:
                for e in f.elb:
                    e.rotation_x *= 0.8
                for e in f.sho:
                    e.rotation_z *= 0.8
            f.head.rotation_x *= 0.8
            f.head.rotation_y *= 0.8
            yield
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def turn(self, yaw, t=0.35):
        e = 0.0
        while e < t or abs(wrap180(yaw - self.yaw)) > 6.0:
            e += self.dt
            self.face(yaw, self.dt, 9.0)
            self.pose()
            if e > 1.5:
                break
            yield

    def hold(self, t, anim=None, d=None, brk=False):
        e = 0.0
        while e < t:
            e += self.dt
            if brk and self.has_work():
                return
            if anim:
                _an(self, anim, e, d or t)
            else:
                self.pose()
            yield

    def route(self, tx, tz):
        L, xo, z0 = JP_L, self.ch.xoff, self.ch.z0
        lx, lz = [xo + v for v in L["lanes_x"]], [z0 + v for v in L["lanes_z"]]
        cells = [(i, j) for i in range(3) for j in range(4)]
        near = lambda x, z: min(cells, key=lambda c: math.hypot(lx[c[0]] - x, lz[c[1]] - z))
        a, b = near(self.x, self.z), near(tx, tz)
        if math.hypot(self.x - tx, self.z - tz) < 2.2:
            return [(tx, tz)]
        return [(lx[a[0]], lz[a[1]]), (lx[a[0]], lz[b[1]]), (lx[b[0]], lz[b[1]]), (tx, tz)]

    def go(self, tx, tz, speed=1.7):
        for (x, z) in self.route(tx, tz):
            yield from self.walk(x, z, speed)

    def on_talk(self) -> None:
        self.g.hud.speak(self, self.rng.choice(DIALOGUE["japan"]))


# ---- SAKURA-YA v5: staffing, plates, trays, washing pipeline -------------------------------------------------------
JP_L.update(
    wash_sink=(-11.3, 2.6), cleaner=(-10.35, 2.9), cleaner_sink=(-10.45, 2.3), silver=(-9.3, 3.0), silver_drop=(-8.3, 3.0), wash_spot=(-8.6, 2.0),
    rack=(-11.3, 4.3), rack_stand=(-10.35, 4.3), cab_L=(-11.4, 15.7), cab_R=(11.45, 7.65), cabR_stand=(10.35, 7.65),
    door_L=(-9.7, 5.2), door_R=(10.45, 3.0), tray_x=(-8.2, -7.65), tray_z=(21.1, 21.7, 22.3), tray_chef_x=-9.0, tray_srv_x=-6.6,
)
JP_L["st"]["plates_L"] = (-10.2, 15.7, -90.0)
JP_EST = {"ramen": 10.0, "fry": 7.2, "micro": 9.3, "sushi": 6.5, "drink": 2.2}      # seconds of work per dish type
JP_FIRST = {"ramen": "pot", "fry": "pan", "micro": "micro", "sushi": "sushi", "drink": "tea"}
JP_LOCK = ("pot", "pan", "board", "micro", "sushi", "tea")                           # one chef at a time
JP_RACK_CAP, JP_SILVER_CAP = 33, 36
JP_COMPLIMENTS = ("Oishii! Taisho, sugoi!", "Umai! Best in the mall!", "Mmm! The chef is a genius!", "Oishii desu! Arigatou!",
                  "So good. Gochisousama!", "Ahh... that hit the spot.", "I could cry. It's the broth.", "Perfect. Exactly as I remembered.",
                  "Nobody does it like this place.", "Delicious! My compliments to the kitchen.")
JP_CHAT = ("Nee, try the gyoza.", "The belt never ends. I love it.", "Is it three o'clock already?", "Just one more plate...", "This broth is a hug.",
           "Kanpai!", "Hm. Hm-hm. Delicious.", "Another tea, please.", "The lanterns are so warm today.", "Do you hear the koto? It's always playing.",
           "I come here when the corridors get too long.", "Careful, the tea is hot.", "Eat slowly. The mall isn't going anywhere.",
           "That plate was gold. I am being very calm about it.", "The sakura never drops its petals on the food. Respect.", "Mm, the egg is perfect today.")
JP_WAIT = ("Sumimasen... is my order coming?", "I can smell it. It's close.", "Patience. Patience. Tea.", "Hungry. Very hungry.",
           "Is that my tray? No. Ah.", "They are very busy today.", "Any minute now...")
JP_LEAVE = ("Gochisousama deshita!", "That was lovely. Back to the mall I go.", "Thank you for the meal!", "I'll be back tomorrow. Or whenever tomorrow is.",
            "Time to walk it off. Mata ne!", "Okanjo, please... ah, it's already paid. Arigatou!")
JP_STAY = ("One more tea, I think.", "I'll sit a little longer.", "The stool is too comfortable to leave.", "Maybe a sweet, later...",
           "Let me digest. Slowly.", "Mm. Not yet. Five more minutes.")
JP_ARRIVE = ("Konnichiwa!", "Is this seat free? Ah, good.", "It smells wonderful in here.", "Finally, somewhere warm.", "Irasshai... I mean, thank you!")
JP_REORDER = ("Sumimasen! One more, please.", "Could I see the menu again?", "I'm hungry again. Is that rude?", "Same again, please.", "Hm... something different this time.")
JP_STACK = ("Oh, this one's still good.", "Waste not. Itadakimasu!", "Someone left this beauty. I'll take it.", "Nobody will mind, right?")
JP_STAFF_GREET = ("Irasshaimase!!", "Irasshaimase! Welcome!", "Irasshai! Please, come in!")
JP_STAFF_FAREWELL = ("Arigatou gozaimashita!", "Mata kite kudasai ne!", "Thank you! Please come again!", "Ki wo tsukete!")
JP_RUSH = ("Isogu zo! Orders are piling up!", "Faster, faster! Hai!", "Rush hour! Stay sharp!", "Three tickets, one pot. Hai!")
JP_HIRO = ("Plates coming through!", "Fresh plates! Still warm from the rack.", "Stack's low on the left - I'm on it!", "Hiro, delivery. Always Hiro.",
           "Mind the stack, please!", "Left wing, right wing, left wing... I live in the corridor.")
JP_CLEAN = ("Scrub, scrub, rinse. The sound is the best part.", "A clean plate is a small miracle.", "Hiro is waiting on these. No pressure.",
            "Listen to the water. It's very nice, isn't it?", "Twenty plates an hour, and I count every one.")
DIALOGUE["japan"] = list(DIALOGUE["japan"]) + ["Hiro runs the plates. If he stops, the kitchens stop.", "Wash, dry, deliver. That's the whole restaurant.",
                                               "Table numbers are on the trays. A tray goes out only when it's complete.", "The silver table is where all the dirty plates go to be forgiven."]


def jp_tray_slot(tid: int) -> tuple:
    return JP_L["tray_x"][tid % 2], JP_L["tray_z"][(tid // 2) % 3]


class JpChef(JpActor):
    """which: 'ramen' = the dedicated left-wing bar chef (serves only the left bar), 'mid' = a core chef (left side bar) who works
    the middle-area rush, 'sushi' = the belt chef of the right wing."""

    def __init__(self, g, ch, pos, rng, rest, which: str, idx: int = 0) -> None:
        nm = {"ramen": "bar chef", "mid": "chef", "sushi": "sushi chef"}[which]
        pal = {"ramen": "jp_chef_r", "mid": "jp_chef_m", "sushi": "jp_chef_s"}[which]
        super().__init__(g, ch, pos, rng, nm, pal, "sushi" if which == "sushi" else "ramen", rest, "talk")
        self.which, self.idx = which, idx
        e1 = self.fig.elb[1]
        self.props = dict(
            knife=self.mkprop(e1, [("cube", (0, -0.42, 0.12), (0.014, 0.06, 0.24), (0.86, 0.88, 0.92), 0.2), ("cube", (0, -0.42, -0.04), (0.024, 0.03, 0.1), (0.1, 0.06, 0.04), 0)]),
            ladle=self.mkprop(e1, [("cube", (0, -0.42, 0.16), (0.016, 0.016, 0.34), (0.8, 0.82, 0.86), 0.1), ("sphere", (0, -0.42, 0.34), (0.1, 0.06, 0.1), (0.8, 0.82, 0.86), 0.1)]),
            pan=self.mkprop(e1, [("cube", (0, -0.42, 0.14), (0.02, 0.02, 0.28), (0.1, 0.1, 0.12), 0), ("cube", (0, -0.42, 0.36), (0.26, 0.03, 0.26), (0.08, 0.08, 0.1), 0)]),
        )
        self.last_cmp = 0.0
        self.cur_st = None
        self.idle_z = {0: 13.0, 1: 8.6, 2: 18.2}.get(idx, 13.0)
        self.dress_dz = {0: 0.0, 1: -0.8, 2: 0.8}.get(idx, 0.0)
        self.last_cry = -99.0
        self.buf = 0

    # ---- rush-hour scaling (core chefs only) --------------------------------------------------------------------------
    @property
    def spd(self) -> float:
        return 1.7 + (1.05 * self.rest.rush if self.which == "mid" else 0.0)

    def tscale(self) -> float:
        return 1.0 - (0.36 * self.rest.rush if self.which == "mid" else 0.0)

    def has_work(self) -> bool:
        return self.which != "sushi" and (self.rest.pool_pending(self) or (self.which == "ramen" and self.rest.asks_pending()))

    def shout(self, text: Optional[str] = None, sfx="irasshai") -> None:
        self.say(text or self.rng.choice(JP_STAFF_GREET), 3.0, sfx, 0.9)

    def thanks(self) -> None:
        self.gen = self._thanks_gen(self.gen)

    def _thanks_gen(self, old):
        yield from self.hold(0.5)
        self.say("Arigatou gozaimasu!", 3.0, "arigato", 0.7)
        yield from self.hold(1.3, "bow", 1.3)
        self.gen = old
        yield

    def on_talk(self) -> None:
        g, rest = self.g, self.rest
        if self.which == "sushi":
            g.hud.speak(self, "Fresh off my board and onto the belt! Sit on a stool on this side and press E to take a plate. I need Hiro's plates to keep cooking.")
        elif self.which == "ramen":
            rec = rest.p_rec
            if rec is not None and rec["kind"] == "ramen":
                rest.player_call_chef(rec)
                g.hud.speak(self, "Hai! One moment, I'm coming around. I only cook for this bar.")
            else:
                g.hud.speak(self, "Irasshai! I cook for the bar stools only. Take a stool and I'll be with you the second my hands are free.")
        else:
            g.hud.speak(self, self.rng.choice(("Middle tables are ours! Sit at a table and a server will take your order.", "Busy, busy! Orders go out on numbered trays.",
                                               "If the plate stack runs dry, ask Hiro. Not me.")) if rest.rush < 0.5 else self.rng.choice(JP_RUSH))

    # ---- stations -------------------------------------------------------------------------------------------------------
    def stand_at(self, key):
        x, z, yaw = JP_L["st"][key]
        if key == "dress":
            z += self.dress_dz
        if key == "idle":
            z = self.idle_z
        yield from self.walk(x + self.ch.xoff, self.ch.z0 + z, self.spd)
        yield from self.turn(yaw)

    def to_station(self, key):
        rest = self.rest
        if key != self.cur_st:
            rest.release(self.cur_st, self)
            self.cur_st = None
        if key in JP_LOCK:                                    # occupied workstation: wait for it (the order picker already preferred a free one)
            w = 0.0
            while not rest.acquire(key, self):
                w += self.dt
                self.pose(rs=-24.0 if w < 6 else 0.0)
                yield
        self.cur_st = key
        yield from self.stand_at(key)

    def idle_routine(self):
        if self.which == "sushi":
            r = self.rng.random()
            if r < 0.4:
                yield from self.knife_trick()
            elif r < 0.7:
                yield from self.hold(self.rng.uniform(3, 5), "wipe")
            else:
                yield from self.hold(self.rng.uniform(3, 5), "cross")
            return
        r = self.rng.random()
        if r < 0.28:
            yield from self.stand_at("idle")
            yield from self.hold(self.rng.uniform(4, 7), "cross", brk=True)
        elif r < 0.55:
            yield from self.to_station("board")
            yield from self.knife_trick(brk=True)
            self.rest.release("board", self)
            self.cur_st = None
        elif r < 0.75:
            yield from self.walk(JP_L["st"]["dress"][0] + self.ch.xoff, self.ch.z0 + self.rng.uniform(8, 19), self.spd)
            yield from self.turn(90.0)
            yield from self.hold(self.rng.uniform(3, 5), "wipe", brk=True)
        elif r < 0.9:
            yield from self.to_station("pot")
            self.show_prop("ladle")
            yield from self.hold(3.4, "taste", 3.4, brk=True)
            self.show_prop(None)
            self.rest.release("pot", self)
            self.cur_st = None
        else:
            yield from self.stand_at("idle")
            yield from self.hold(2.5, "wave", brk=True)

    def knife_trick(self, brk=False):
        self.show_prop("knife")
        k = self.props["knife"]
        e = 0.0
        jp_say(self.g, self.x, self.z, "whoosh", 0.6)
        while e < 3.2:
            e += self.dt
            if brk and self.has_work():
                break
            _an(self, "write", e, 3.2)
            fl = (e % 1.6) / 1.6
            k.rotation_x = 0 if fl > 0.55 else fl / 0.55 * 720.0
            if abs(fl - 0.0) < self.dt / 1.6 * 1.5:
                jp_say(self.g, self.x, self.z, "whoosh", 0.5)
            yield
        k.rotation_x = 0
        self.show_prop(None)

    # ---- brain ----------------------------------------------------------------------------------------------------------
    def brain(self):
        rest = self.rest
        if self.which == "sushi":
            yield from self.sushi_loop()
            return
        while True:
            ask = rest.next_ask(self) if self.which == "ramen" else None
            if ask is not None and not rest.bar_in_progress():
                yield from self.ask_counter(ask)
                continue
            it = rest.claim_item(self)
            if it is not None:
                yield from self.cook_item(it)
            elif ask is not None:
                yield from self.ask_counter(ask)
            elif rest.pool_pending(self):
                yield from self.wait_blocked()
            else:
                yield from self.idle_routine()

    def wait_blocked(self):
        rest = self.rest
        if rest.plates["L"] - rest.plate_res["L"] < 1 and rest.clock - self.last_cry > 9.0:
            self.last_cry = rest.clock
            self.say(self.rng.choice(("No plates! Hiro-san, plates please!", "The cabinet is empty... where is Hiro?")), 3.2)
        elif rest.rush > 0.6 and rest.clock - self.last_cry > 14.0 and self.which == "mid":
            self.last_cry = rest.clock
            self.say(self.rng.choice(JP_RUSH), 3.0)
        e = 0.0
        while e < 0.6:
            e += self.dt
            self.pose(rs=-20.0)
            yield

    def ask_counter(self, req):
        rest, rec, cust = self.rest, req["rec"], req["cust"]
        yield from self.walk(rec["stand"][0], rec["stand"][1], self.spd)
        yield from self.turn(90.0)
        if cust is None:
            if rest.p_rec is not rec:
                req["state"] = "gone"
                return
            self.say("Irasshai! What will it be today?", 5.0, "irasshai", 0.5)
            pr = rest.open_prompt(self, rec, "counter")
            e = 0.0
            while pr["state"] in ("open", "menu") and e < 25.0:
                e += self.dt
                _an(self, "write", e, 25.0)
                yield
            if pr["state"] == "ordered":
                self.say("Hai! Coming right up!", 3.0)
                rest.new_ticket(rec, None, pr["items"], "bar", prio=True)
                req["state"] = "done"
            else:
                self.say("Take your time. Ring the bell (R) when you're ready.", 4.0)
                req["state"] = "declined"
                rest.close_prompt(pr)
        else:
            yield from self.hold(1.4, "write", 1.4)
            items = cust.choose_order()
            cust.ordered = items
            rest.new_ticket(rec, cust, items, "bar")
            req["state"] = "done"

    STEPS = {
        "ramen": (("pot", "boil", 3.2, "boil", 2.6), ("board", "chop", 2.4, "chop", 0.9), ("pot", "ladle", 2.2, "pour", 1.6), ("dress", "topping", 2.2, None, 0)),
        "fry": (("pan", "fry", 4.2, "sizzle", 2.2), ("pan", "flip", 1.4, "whoosh", 1.0), ("dress", "topping", 1.6, None, 0)),
        "micro": (("micro", "mw_load", 1.6, "mw_door", 9.0), ("micro", "mw_cook", 4.6, "mw_hum", 2.4), ("micro", "mw_take", 1.5, "mw_ding", 9.0), ("dress", "topping", 1.6, None, 0)),
        "sushi": (("sushi", "chop", 2.3, "chop", 0.9), ("sushi", "press", 2.6, "pat", 0.5), ("sushi", "topping", 1.6, None, 0)),
        "drink": (("tea", "pour", 2.2, "pour", 1.6),),
    }

    def cook_item(self, it):
        g, rest, V = self.g, self.rest, self.ch.extra["jp_vis"]
        key, tk = it["key"], it["tk"]
        how = JP_HOW[key]
        it["state"] = "cooking"
        PROP = {"chop": "knife", "ladle": "ladle", "fry": "pan", "flip": "pan", "boil": None}
        yield from self.to_station("plates_L")                  # every dish starts with a clean plate from the left-wing cabinet
        yield from self.hold(0.8 * self.tscale(), "place", 0.8)
        rest.take_plate("L")
        jp_say(g, self.x, self.z, "plate", 0.5, 16.0)
        for (st, anim, dur, sfx, period) in self.STEPS[how]:
            yield from self.to_station(st)
            dur *= self.tscale()
            self.show_prop(PROP.get(anim))
            e, nxt = 0.0, 0.0
            while e < dur:
                e += self.dt
                if sfx and e >= nxt:
                    jp_say(g, self.x, self.z, sfx, 0.8 if sfx != "mw_hum" else 0.55, 16.0)
                    nxt += period
                _an(self, anim, e, dur)
                if st == "micro":
                    k = min(1.0, e / dur)
                    ang = {"mw_load": 85.0 * min(1.0, k * 2.5) * (1.0 if k < 0.75 else 1.0 - (k - 0.75) * 4), "mw_cook": 0.0,
                           "mw_take": 85.0 * min(1.0, k * 3.0) * (1.0 if k < 0.7 else 1.0 - (k - 0.7) * 3.3)}[anim]
                    V["mw_door"].rotation_y = ang
                    if FANCY:
                        V["mw_light"].set_shader_input("emissive", 1.6 if (anim == "mw_cook" or (anim == "mw_load" and k > 0.8)) else 0.0)
                    if anim == "mw_cook" and abs(e - dur + 0.35) < self.dt:
                        jp_say(g, self.x, self.z, "mw_beep", 0.7, 16.0)
                yield
            if st == "micro" and FANCY:
                V["mw_light"].set_shader_input("emissive", 0.0)
            self.show_prop(None)
        rest.release(self.cur_st, self)
        self.cur_st = None
        if tk["dest"] == "mid":
            self.set_carry(key)
            sx, sz = jp_tray_slot(tk["tid"])
            yield from self.walk(JP_L["tray_chef_x"] + self.ch.xoff, self.ch.z0 + sz, self.spd)
            yield from self.turn(90.0)
            yield from self.hold(0.8 * self.tscale(), "place", 0.8)
            self.set_carry(None)
            rest.tray_put(tk, it)
            jp_say(g, self.x, self.z, "plate", 0.8)
        else:
            tk["plated"].append(it)
            it["state"] = "done"
            if len(tk["plated"]) < len(tk["items"]):            # the rest of this order is cooked back-to-back, then it all goes out together
                return
            self.set_tray([i["key"] for i in tk["items"]], "BAR")
            rec = tk["rec"]
            yield from self.walk(rec["stand"][0], rec["stand"][1], self.spd)
            yield from self.turn(90.0)
            yield from self.hold(0.9, "place", 0.9)
            self.set_carry(None)
            rest.put_counter_tray(tk)
            jp_say(g, self.x, self.z, "plate", 0.8)
            self.say("Dozo! Enjoy!", 2.6)

    # ---- sushi chef: forever feeding the belt (needs plates from the right-wing cabinet) -------------------------------
    def sushi_loop(self):
        rest, g = self.rest, self.g
        cx, cz = JP_L["chefR"]
        while True:
            if rest.belt_load() >= 7:
                yield from self.walk(cx + self.ch.xoff, self.ch.z0 + cz)
                yield from self.turn(90.0)
                yield from self.idle_routine()
                continue
            if self.buf <= 0:
                if rest.plates["R"] <= 0:                          # kitchen starved of plates: wait for Hiro
                    if rest.clock - self.last_cry > 10.0:
                        self.last_cry = rest.clock
                        self.say("No plates on the right! Hiro-san!", 3.0)
                    yield from self.walk(cx + self.ch.xoff, self.ch.z0 + cz)
                    yield from self.turn(90.0)
                    yield from self.hold(1.0, "cross")
                    continue
                yield from self.walk(JP_L["cabR_stand"][0] + self.ch.xoff, self.ch.z0 + JP_L["cabR_stand"][1])
                yield from self.turn(90.0)
                yield from self.hold(0.8, "place", 0.8)
                self.buf = min(3, rest.plates["R"])
                rest.plates["R"] -= self.buf
                jp_say(g, self.x, self.z, "plate", 0.5, 16.0)
            yield from self.walk(cx + self.ch.xoff, self.ch.z0 + cz)
            yield from self.turn(90.0)
            r = self.rng.random()
            kind = 0 if r < 0.38 else (1 if r < 0.68 else (2 if r < 0.88 else 3))
            for (anim, dur, sfx, period, prop) in (("chop", 2.2, "chop", 0.9, "knife"), ("press", 2.0, "pat", 0.5, None), ("topping", 1.1, None, 0, None)):
                self.show_prop(prop)
                e, nxt = 0.0, 0.0
                while e < dur:
                    e += self.dt
                    if sfx and e >= nxt:
                        jp_say(g, self.x, self.z, sfx, 0.7, 18.0)
                        nxt += period
                    _an(self, anim, e, dur)
                    yield
            self.show_prop(None)
            self.set_carry(BELT_PLATES[kind][0])
            yield from self.turn(-90.0)
            e = 0.0
            fired = False
            while e < 1.0:
                e += self.dt
                _an(self, "place", e, 1.0)
                if e > 0.55 and not fired:
                    fired = True
                    self.set_carry(None)
                    self.buf -= 1
                    rest.belt_add(kind, (self.x - self.ch.xoff - 0.5, 1.25, self.z))
                    jp_say(g, self.x, self.z, "plate", 0.6, 18.0)
                yield


class JpServer(JpActor):
    """Middle-area waiter.  Takes orders, carries ONE numbered tray with the complete order, clears the plates piled beside every seat
    and drops them on the silver table of the washing area."""

    def __init__(self, g, ch, pos, rng, rest, nm: str) -> None:
        super().__init__(g, ch, pos, rng, nm, "jp_server", "server", rest, "server")
        self.home = pos
        self.job = None

    def has_work(self) -> bool:
        return self.rest.server_has_work()

    def shout(self, text=None, sfx="irasshai") -> None:
        self.say(text or self.rng.choice(JP_STAFF_GREET), 3.0, sfx, 0.85)

    def on_talk(self) -> None:
        g, rest = self.g, self.rest
        rec = rest.p_rec
        if rec is not None and rec["kind"] == "table":
            rest.player_ring(rec, quick=True)
            g.hud.speak(self, "Hai, sumimasen! Right away.")
        else:
            g.hud.speak(self, self.rng.choice(("Please sit at any table in the middle. I'll be with you in a few seconds.",
                                               "Ramen bar on the left, kaiten sushi on the right. Tables are served by us - one tray, whole order, table number on it.",
                                               "Dirty plates go to the silver table, at the back by the entrance. You can wash a few for tips, you know.")))

    def brain(self):
        rest = self.rest
        while True:
            job = rest.pop_job(self)
            if job is None:
                self.job = None
                yield from self.patrol()
                continue
            self.job = job
            t = job["type"]
            if t == "ask":
                yield from self.do_ask(job)
            elif t == "tray":
                yield from self.do_tray(job)
            elif t == "clear":
                yield from self.do_clear(job)
            self.job = None

    def interruptible(self) -> bool:
        j = self.job
        if j is None or self.carry_ent is not None:
            return j is None
        if j["type"] == "ask":
            return j.get("state") == "claimed" and j.get("cust") is not None and j["cust"].ordered is None
        return j["type"] == "clear"

    def interrupt(self) -> None:
        j, rest = self.job, self.rest
        self.gen, self.job = None, None
        self.set_carry(None)
        if j is not None:
            if j["type"] == "ask":
                j["state"] = "new"
            elif j["type"] == "clear":
                j["claim"] = None

    def update(self, dt: float, w) -> None:
        if self.rest.want_server and self.interruptible() and self.job is not None:
            self.interrupt()
        super().update(dt, w)

    def set_tray(self, keys, label) -> None:
        """Carry a numbered tray holding every dish of the order at once."""
        if self.carry_ent is not None:
            ur.destroy(self.carry_ent)
        root = ur.Entity(parent=self.fig.root, position=(0, 1.02, 0.42))
        mk(root, "cube", (0, 0, 0), (0.46, 0.03, 0.34), color=C(0.10, 0.06, 0.05), **self.fig.k)
        mk(root, "cube", (0, 0.02, 0.18), (0.14, 0.04, 0.012), color=C(0.95, 0.78, 0.25), emissive=0.5, **self.fig.k)
        for i, k in enumerate(keys[:4]):
            jp_dish(k, root, ((i - (len(keys[:4]) - 1) / 2) * 0.16, 0.03, -0.02), self.fig.k).scale = 0.62
        self.carry_ent = root

    def patrol(self):
        rest, L = self.rest, JP_L
        if self.rng.random() < 0.55:
            nx = self.rng.choice(L["lanes_x"])
            nz = self.rng.choice(L["lanes_z"])
            yield from self.go(nx + self.ch.xoff, nz + self.ch.z0, 1.3)
        t = self.rng.uniform(2.0, 5.5)
        e = 0.0
        while e < t:
            e += self.dt
            if self.has_work():
                return
            self.pose(rs=-20 if (self.alert > 0) else 0.0)
            yield

    def do_ask(self, job):
        rest, rec, cust = self.rest, job["rec"], job["cust"]
        yield from self.go(rec["spot"][0], rec["spot"][1])
        yield from self.turn(-90.0 if rec["spot"][0] < self.ch.xoff else 90.0, 0.3)
        want_yaw = math.degrees(math.atan2(rec["x"] - self.x, rec["z"] - self.z))
        self.face(want_yaw, 1.0, 20.0)
        if cust is None:
            if rest.p_rec is not rec:
                job["state"] = "gone"
                self.say("Oh... they've gone.", 2.5)
                return
            self.shout("Irasshaimase! Are you ready to order?", "irasshai")
            pr = rest.open_prompt(self, rec, "table")
            e = 0.0
            while pr["state"] in ("open", "menu") and e < 25.0:
                e += self.dt
                self.face(math.degrees(math.atan2(self.g.player.x - self.x, self.g.player.z - self.z)), self.dt, 6.0)
                _an(self, "write", e, 25.0)
                yield
            if pr["state"] == "ordered":
                self.say("Hai! Kashikomarimashita!", 3.0)
                job["state"] = "done"
                yield from self.hold(0.7, "write")
                yield from self.go(*rest.pass_srv)
                yield from self.turn(-90.0)
                yield from self.hold(1.0, "write", 1.0)
                rest.new_ticket(rec, None, pr["items"], "mid", prio=True)
                jp_say(self.g, self.x, self.z, "bell_table", 0.4, 14.0)
            else:
                if pr["state"] != "gone":
                    self.say("Take your time. Ring the bell (R) when you're ready.", 4.0)
                job["state"] = "declined"
                rest.close_prompt(pr)
        else:
            self.say("Hai, nani ni shimasu ka?", 2.6)
            yield from self.hold(1.8, "write")
            items = cust.choose_order()
            cust.ordered = items
            job["state"] = "done"
            yield from self.go(*rest.pass_srv)
            yield from self.turn(-90.0)
            yield from self.hold(1.0, "write", 1.0)
            rest.new_ticket(rec, cust, items, "mid")

    def do_tray(self, job):
        rest, g = self.rest, self.g
        tk = job["tk"]
        sx, sz = jp_tray_slot(tk["tid"])
        yield from self.go(JP_L["tray_srv_x"] + self.ch.xoff, sz + self.ch.z0)
        yield from self.turn(-90.0)
        yield from self.hold(0.9, "place", 0.9)
        rest.tray_clear(tk)                                     # the whole tray leaves the pass at once
        self.set_tray([i["key"] for i in tk["items"]], tk["no"])
        jp_say(g, self.x, self.z, "tray_set", 0.7, 16.0)
        rec = tk["rec"]
        yield from self.go(rec["spot"][0], rec["spot"][1])
        yield from self.turn(-90.0 if rec["spot"][0] < self.ch.xoff else 90.0, 0.3)
        yield from self.hold(0.9, "place", 0.9)
        self.set_carry(None)
        rest.put_table_tray(tk)
        jp_say(g, self.x, self.z, "plate", 0.85)
        self.say(f"Table {tk['no']}!  Douzo! Please enjoy.", 3.0)

    def do_clear(self, pile):
        rest = self.rest
        rec = pile["rec"]
        yield from self.go(*rec["clear_spot"])
        yield from self.hold(0.9, "place", 0.9)
        n = rest.take_pile(pile)
        if n <= 0:
            return
        if self.carry_ent is not None:
            ur.destroy(self.carry_ent)
        root = ur.Entity(parent=self.fig.root, position=(0, 1.02, 0.42))
        for i in range(min(n, 8)):
            mk(root, "cube", (0, 0.03 * i, 0), (0.30, 0.025, 0.30), color=C(0.93, 0.93, 0.9), **self.fig.k)
        self.carry_ent = root
        jp_say(self.g, self.x, self.z, "dish_clink", 0.6)
        x, z = JP_L["silver_drop"]
        yield from self.go(x + self.ch.xoff, z + self.ch.z0)
        yield from self.turn(-90.0)
        w = 0.0
        while rest.silver >= JP_SILVER_CAP and w < 12.0:         # silver table full: wait for the dishwasher
            w += self.dt
            yield
        yield from self.hold(0.8, "place", 0.8)
        self.set_carry(None)
        rest.silver_add(n)
        jp_say(self.g, self.x, self.z, "dish_stack", 0.8, 18.0)


class JpHiro(JpActor):
    """Hiro: picks clean plates off the drying rack and delivers them to the cabinet of the left or the right wing."""

    def __init__(self, g, ch, pos, rng, rest) -> None:
        super().__init__(g, ch, pos, rng, "Hiro", "jp_hiro", "server", rest, "talk")

    def has_work(self) -> bool:
        return False

    def on_talk(self) -> None:
        r = self.rest
        self.g.hud.speak(self, self.rng.choice(JP_HIRO) + f"   (left {r.plates['L']}, right {r.plates['R']}, rack {r.rack})")

    def route_pts(self, wing):
        L = JP_L
        if wing == "L":
            return [L["door_L"], (L["cab_L"][0] + 1.2, L["cab_L"][1]), L["st"]["plates_L"][:2]]
        return [(-6.0, 3.0), (7.0, 3.0), L["door_R"], L["cabR_stand"]]

    def path(self, pts, speed=1.9):
        for (x, z) in pts:
            yield from self.walk(x + self.ch.xoff, z + self.ch.z0, speed)

    def brain(self):
        rest, L = self.rest, JP_L
        while True:
            wing = rest.hiro_target()
            if wing is None:
                yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0)
                yield from self.turn(-90.0)
                yield from self.hold(self.rng.uniform(2.0, 4.0), "wipe" if self.rng.random() < 0.5 else None, brk=False)
                continue
            yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0, 1.9)
            yield from self.turn(-90.0)
            n = rest.rack_take(8)
            if n <= 0:
                continue
            yield from self.hold(0.9, "place", 0.9)
            if self.carry_ent is not None:
                ur.destroy(self.carry_ent)
            root = ur.Entity(parent=self.fig.root, position=(0, 1.02, 0.42))
            for i in range(min(n, 8)):
                mk(root, "cube", (0, 0.03 * i, 0), (0.30, 0.025, 0.30), color=C(0.97, 0.97, 0.95), **self.fig.k)
            self.carry_ent = root
            if self.rng.random() < 0.4:
                self.say(self.rng.choice(JP_HIRO), 2.8)
            yield from self.path(self.route_pts(wing))
            yield from self.turn(-90.0 if wing == "L" else 90.0)
            yield from self.hold(1.0, "place", 1.0)
            self.set_carry(None)
            rest.plates[wing] += n
            rest.last_deliver = rest.clock
            jp_say(self.g, self.x, self.z, "dish_stack", 0.8, 18.0)
            if wing == "R":
                yield from self.path(list(reversed(self.route_pts("R")[:-1])))
            else:
                yield from self.path(list(reversed(self.route_pts("L")[:-1])))


class JpCleaner(JpActor):
    """Dishwasher: takes dirty plates from the silver table, washes them with running water and sponge sounds, stacks them on the rack."""

    def __init__(self, g, ch, pos, rng, rest) -> None:
        super().__init__(g, ch, pos, rng, "dishwasher", "jp_cleaner", "server", rest, "talk")

    def on_talk(self) -> None:
        self.g.hud.speak(self, self.rng.choice(JP_CLEAN))

    def has_work(self) -> bool:
        return False

    def brain(self):
        rest, L = self.rest, JP_L
        while True:
            if rest.silver > 0 and rest.rack < JP_RACK_CAP:
                n = rest.silver_take(3)
                yield from self.walk(L["cleaner"][0] + self.ch.xoff, L["cleaner"][1] + self.ch.z0, 1.5)
                yield from self.turn(90.0)
                yield from self.hold(0.6, "place", 0.6)
                yield from self.walk(L["cleaner_sink"][0] + self.ch.xoff, L["cleaner_sink"][1] + self.ch.z0, 1.2)
                yield from self.turn(-90.0)
                for _ in range(n):
                    e, nxt = 0.0, 0.0
                    while e < 2.8:
                        e += self.dt
                        if e >= nxt:
                            jp_say(self.g, self.x, self.z, "dish_water" if int(nxt / 2.0) % 2 == 0 else "dish_scrub", 0.8, 20.0)
                            nxt += 1.7
                        _an(self, "wash", e, 2.8)
                        yield
                    jp_say(self.g, self.x, self.z, "dish_clink", 0.6, 16.0)
                yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0, 1.3)
                yield from self.turn(-90.0)
                yield from self.hold(0.8, "place", 0.8)
                rest.rack_add(n)
                jp_say(self.g, self.x, self.z, "dish_stack", 0.7, 18.0)
            else:
                yield from self.walk(L["cleaner"][0] + self.ch.xoff, L["cleaner"][1] + self.ch.z0, 1.2)
                yield from self.turn(-90.0)
                yield from self.hold(self.rng.uniform(2.0, 4.0), "wipe")


JpActor.set_tray = JpServer.set_tray
if not hasattr(JpActor, "shout"):
    def _generic_shout(self, text=None, sfx="irasshai") -> None:
        self.say(text or self.rng.choice(JP_STAFF_GREET), 3.0, sfx, 0.85)
    JpActor.shout = _generic_shout


class JpCustomer(JpActor):
    def __init__(self, g, ch, rng, rest, rec, arrive: Optional[int] = None) -> None:
        pal = f"jp_c{rng.randrange(5)}"
        start = (rec["x"], rec["z"])
        if arrive is not None:
            zr = JP_L["doors_z"][arrive]
            start = (ch.xoff + rng.uniform(-0.6, 0.6), ch.z0 + zr + (-2.6 if arrive == 0 else 2.6))
        super().__init__(g, ch, start, rng, rng.choice(("diner", "diner", "regular", "guest")), pal, "guest", rest, "talk")
        self.srec, self.arrive = rec, arrive
        rec["occ"], rec["it"].occupied = self, True
        self.ordered = None
        self.got_list: list = []
        self.meals, self.reorder_at, self.state_tag = 0, 0.0, "idle"
        self.chop = self.mkprop(self.fig.elb[1], [("cube", (0.014, -0.42, 0.15), (0.012, 0.012, 0.26), (0.62, 0.45, 0.28), 0), ("cube", (-0.014, -0.42, 0.15), (0.012, 0.012, 0.26), (0.62, 0.45, 0.28), 0)])
        if arrive is None:
            self.y = rec["y"] - 0.72
            self.yaw = rec["yaw"]
            self.hips = "sit"
            self.fig.place(self.x, self.y, self.z, self.yaw)
            self.fig.pose(0.0, 0.0, True)
        else:
            self.y, self.hips = 0.0, None
            self.yaw = 0.0 if arrive == 0 else 180.0
            self.fig.place(self.x, self.y, self.z, self.yaw)

    # ---- ordering ---------------------------------------------------------------------------------------------------
    def choose_order(self) -> list:
        """A whole order: a main (or two), maybe a side, maybe a drink - the kitchen cooks them in parallel and the tray leaves complete."""
        rng = self.rng
        mains = [k for s, k, p, h in JP_MENU if h in ("ramen", "micro", "sushi") and not (h == "sushi" and rng.random() < 0.35)]
        sides = [k for s, k, p, h in JP_MENU if h == "fry"] + ["mochi"]
        drinks = [k for s, k, p, h in JP_MENU if h == "drink" and k != "mochi"]
        items = [rng.choice(mains)]
        if rng.random() < 0.40:
            items.append(rng.choice(sides))
        if rng.random() < 0.55:
            items.append(rng.choice(drinks))
        if self.srec["kind"] == "ramen":
            items = [k for k in items if JP_HOW[k] != "sushi"] or [rng.choice(("ramen_shoyu", "ramen_miso", "ramen_tonkotsu"))]
        return items

    def choose(self) -> str:
        return self.choose_order()[0]

    def received(self, key, ent) -> None:
        self.got_list.append((key, ent))

    def on_talk(self) -> None:
        tag, rng = self.state_tag, self.rng
        if tag == "waiting":
            line = rng.choice(JP_WAIT)
        elif tag == "eating":
            line = "Mmf. (chewing)  ...Oishii."
        else:
            line = rng.choice(DIALOGUE["japan"] + list(JP_CHAT))
        self.g.hud.speak(self, line)

    def idle(self, t, anim=None):
        e, nxt = 0.0, self.rng.uniform(3.0, 8.0)
        while e < t:
            e += self.dt
            nxt -= self.dt
            if nxt <= 0:
                nxt = self.rng.uniform(5.0, 12.0)
                if self.dist() < 16 and self.rng.random() < 0.18:
                    self.say(self.rng.choice(JP_CHAT), 3.0)
            sway = math.sin(self.rest.clock * 0.6) * 4.0
            if anim == "hand":
                self.pose(rs=-165, re=-10, rz=-8 + 12 * math.sin(e * 7), hx=0, hy=None)
            elif self.hips == "sit":
                self.pose(ls=-30, le=-40, rs=-30, re=-40, hx=sway * 0.3)
            else:
                self.pose()
            yield

    # ---- life cycle -------------------------------------------------------------------------------------------------
    def brain(self):
        rec, rest, rng = self.srec, self.rest, self.rng
        first = self.arrive is None
        if self.arrive is not None:
            yield from self.enter_and_sit()
            yield from self.idle(rng.uniform(4.0, 10.0))
        while True:
            if first:
                yield from self.idle(rng.uniform(10.0, 80.0) if rec["kind"] == "table" else rng.uniform(2.0, 30.0))
            else:                                         # never reorder sooner than 30 s after the last meal ended
                yield from self.idle(max(0.0, self.reorder_at - rest.clock) + rng.uniform(1.0, 18.0))
                self.say(rng.choice(JP_REORDER), 2.8)
            first = False
            gave_up = False
            if rec["kind"] == "table":
                gave_up = yield from self.order_table()
            elif rec["kind"] == "ramen":
                gave_up = yield from self.order_counter()
            else:
                yield from self.order_belt()
            self.state_tag = "idle"
            self.reorder_at = rest.clock + rng.uniform(32.0, 58.0)
            if gave_up and rng.random() < 0.8:
                yield from self.leave(annoyed=True)
                return
            if self.meals >= 1:
                if rng.random() < 0.5:                    # finished eating: leave or stay, at random
                    yield from self.leave()
                    return
                self.say(rng.choice(JP_STAY), 3.0)

    def exit_point(self):
        rec = self.srec
        s, c = math.sin(math.radians(rec["yaw"])), math.cos(math.radians(rec["yaw"]))
        return rec["x"] - s * 0.85, rec["z"] - c * 0.85           # behind the seat, away from the table / counter

    def enter_and_sit(self):
        rest, rec, rng = self.rest, self.srec, self.rng
        d = self.arrive
        inward = 1.0 if d == 0 else -1.0
        rest.npc_door(d, True, self)
        yield from self.walk(self.ch.xoff + rng.uniform(-0.5, 0.5), self.ch.z0 + JP_L["doors_z"][d] + inward * 1.8, 1.5)
        if rng.random() < 0.5:
            self.say(rng.choice(JP_ARRIVE), 2.8)
        ex, ez = self.exit_point()
        yield from self.go(ex, ez, 1.5)
        yield from self.turn(rec["yaw"], 0.3)
        self.x, self.z, self.y, self.yaw, self.hips = rec["x"], rec["z"], rec["y"] - 0.72, rec["yaw"], "sit"
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def leave(self, annoyed: bool = False):
        rest, rec, rng = self.rest, self.srec, self.rng
        self.state_tag = "leaving"
        self.say("Mou ii, I'll go elsewhere..." if annoyed else rng.choice(JP_LEAVE), 3.2, None if annoyed else "gochisou", 0.6)
        yield from self.idle(1.0)
        ex, ez = self.exit_point()
        self.chop.enabled = False
        self.hips = None
        self.y = 0.0
        yield from self.walk(ex, ez, 1.4)
        rec["occ"], rec["it"].occupied = None, False
        d = 0 if rec["z"] < self.ch.z0 + JP_L["doors_z"][1] / 2 else 1
        inward = 1.0 if d == 0 else -1.0
        yield from self.go(self.ch.xoff + rng.uniform(-0.5, 0.5), self.ch.z0 + JP_L["doors_z"][d] + inward * 1.6, 1.5)
        rest.npc_door(d, False, self)
        yield from self.walk(self.ch.xoff + rng.uniform(-0.4, 0.4), self.ch.z0 + JP_L["doors_z"][d] - inward * 2.8, 1.5)
        rest.retire(self)

    def order_table(self):
        rest = self.rest
        rest.call_server(self, self.srec)
        self.say("Sumimasen!", 2.5)
        jp_say(self.g, self.srec["bell"][0], self.srec["bell"][1], "bell_table", 0.7)
        t = 0.0
        self.state_tag = "waiting"
        while self.ordered is None:
            t += self.dt
            yield from self.idle(0.1, "hand" if t < 8.0 else None)
            if t > 40.0:
                jp_say(self.g, self.srec["bell"][0], self.srec["bell"][1], "bell_table", 0.6)
                t = 0.0
        return (yield from self.wait_and_eat())

    def order_counter(self):
        rest = self.rest
        rest.counter_wait(self, self.srec)
        t = 0.0
        self.state_tag = "waiting"
        while self.ordered is None:
            t += self.dt
            yield from self.idle(0.1)
            if t > 30.0:
                t = 0.0
                jp_say(self.g, self.srec["bell"][0], self.srec["bell"][1], "bell_table", 0.6)
                rest.counter_wait(self, self.srec)
        return (yield from self.wait_and_eat())

    def wait_and_eat(self):
        t = 0.0
        self.state_tag = "waiting"
        tk = self.ticket()
        while not self.got_list:
            t += self.dt
            yield from self.idle(0.1)
            if t > 150.0:
                if tk is not None:
                    tk["void"] = True
                self.ordered = None
                return True
        yield from self.idle(0.6)                              # the whole tray lands together
        items, self.got_list = self.got_list, []
        self.ordered = None
        self.state_tag = "eating"
        for key, ent in items:
            yield from self.eat(key, ent)
            self.rest.finish_dish(self.srec, key, ent)
        self.meals += 1
        yield from self.compliment()
        return False

    def ticket(self):
        for tk in self.rest.tickets:
            if tk["cust"] is self and tk["state"] != "closed":
                return tk
        return None

    def eat(self, key, ent):
        self.say("Itadakimasu!", 2.4, "itadaki", 0.55)
        yield from self.idle(1.2)
        self.chop.enabled = True
        n = 5 if key.startswith("ramen") else 4
        per = 2.7
        for i in range(n):
            e = 0.0
            while e < per:
                e += self.dt
                ph = e / per
                if ph < 0.30:
                    self.pose(ls=-30, le=-40, rs=-78, re=-22, rz=-6, hx=14)
                elif ph < 0.45:
                    self.pose(ls=-30, le=-40, rs=-62, re=-45, rz=-8, hx=12)
                elif ph < 0.65:
                    self.pose(ls=-30, le=-40, rs=-52, re=-132, rz=-18, hx=-4)
                    if key.startswith("ramen") and 0.5 < ph < 0.52 and i % 2 == 0:
                        jp_say(self.g, self.x, self.z, "slurp", 0.55, 14.0)
                else:
                    self.pose(ls=-30, le=-40, rs=-48, re=-128, rz=-18, hx=6 * math.sin(e * 16))
                yield
        self.chop.enabled = False

    def compliment(self):
        rest = self.rest
        self.say(self.rng.choice(JP_COMPLIMENTS), 3.6, "oishii", 0.6)
        k = self.srec["kind"]
        chef = rest.chefL if k == "ramen" else (rest.chefR if k == "belt" else self.rng.choice(rest.chefs_mid))
        if chef is not None and rest.clock - chef.last_cmp > 6.0:
            chef.last_cmp = rest.clock
            chef.thanks()
        yield from self.idle(2.5)

    def eat_belt_plate(self, pl_key, ent):
        yield from self.idle(0.6)
        yield from self.eat(pl_key, ent)
        self.rest.finish_dish(self.srec, pl_key, ent)
        self.meals += 1
        yield from self.compliment()

    def order_belt(self):
        rest, rec, rng = self.rest, self.srec, self.rng
        self.state_tag = "waiting"
        st = rest.find_stack(self)                              # plates left uneaten around here: eat one, or wait for a fresh one?
        if st is not None and rng.random() < 0.55:
            rest.claim_stack(st)
            self.say(rng.choice(JP_STACK), 2.8)
            e = 0.0
            while e < 0.9:
                e += self.dt
                self.pose(ls=-30, le=-40, rs=-82, re=-8, rz=-4, hx=18)
                yield
            ent = rest.take_stack(st, rec)
            self.state_tag = "eating"
            yield from self.eat_belt_plate(st["key"], ent)
            return
        t, pl = 0.0, None
        while pl is None and t < 60.0:
            t += self.dt
            pl = rest.belt_near(rec["slot"][1], 0.3)
            if pl is not None and rng.random() < 0.35:
                pl = None
                for _k in range(60):
                    yield from self.idle(0.02)
            yield from self.idle(0.05)
        if pl is None:
            return
        pl["taken"] = True
        e = 0.0
        while e < 0.9:
            e += self.dt
            self.pose(ls=-30, le=-40, rs=-82, re=-8, rz=-4, hx=18)
            yield
        ent = rest.belt_grab(pl, rec)
        if rng.random() < 0.28:                                  # grabbed one for later: it stays stacked beside them
            pl2 = None
            for _ in range(80):
                pl2 = rest.belt_near(rec["slot"][1], 0.3)
                if pl2 is not None:
                    break
                yield from self.idle(0.1)
            if pl2 is not None:
                pl2["taken"] = True
                rest.stash(rec, pl2["key"], rest.belt_grab(pl2, rec))
        self.state_tag = "eating"
        yield from self.eat_belt_plate(pl["key"], ent)


class JpRestaurant:
    """Per-chunk controller of SAKURA-YA: belt, doors, tickets & trays, kitchen queues with station locks, plate pools, washing pipeline."""
    BELT_SPEED = 0.46

    def __init__(self, g, ch) -> None:
        self.g, self.ch, self.xo, self.z0 = g, ch, ch.xoff, ch.z0
        self.V = ch.extra["jp_vis"]
        self.rng = random.Random(stable_hash(g.args.seed, "jp", ch.idx))
        self.clock = 0.0
        self.belt: list = []
        self.reqs: list = []
        self.counter_reqs: list = []
        self.timers: list = []
        self.servers: list = []
        self.chefL = self.chefR = None
        self.chefs_mid: list = []
        self.hiro = self.cleaner = None
        self.customers: list = []
        self.p_rec = None
        self.p_state, self.p_t, self.p_req = None, 0.0, None
        self.prompt = None
        self.pass_srv = (JP_L["pass_srv"][0] + self.xo, self.z0 + JP_L["pass_srv"][1])
        self.amb_t = [1.5, 2.0, 3.0]
        self.belt_ph = 0.0
        self.want_server = False
        self.kw = dict(lm=ch.lm, lamp=ch.lamp, xc=ch.xoff)
        # v5 pipeline
        self.tickets: list = []
        self.mid_items: list = []
        self.bar_items: list = []
        self.tk_seq = 0
        self.tray_owner: dict = {}
        self.pass_ready: list = []
        self.piles: dict = {}
        self.silver = 0
        self.rack = 0
        self.plates = {"L": 22, "R": 14}
        self.plate_res = {"L": 0, "R": 0}
        self.station_owner: dict = {}
        self.rush = 0.0
        self.last_deliver = 0.0
        self.door_t = [0.0, 0.0]
        self.door_cool = [0.0, 0.0]
        self.pside = [None, None]
        self.retired: list = []
        self.wash = None
        self.vis_t = 0.0
        self.arrive_cap = 10

    # ---- tickets, trays, kitchen queues -------------------------------------------------------------------------------
    def new_ticket(self, rec, cust, items, dest, prio=False):
        self.tk_seq += 1
        tid = rec.get("tid", -1)
        tk = dict(id=self.tk_seq, rec=rec, cust=cust, dest=dest, prio=prio, t0=self.clock, started=False, void=False, done=[], plated=[], items=[],
                  state="open", tid=tid, no=tid + 1)
        est = {k: JP_EST[JP_HOW[k]] for k in items}
        mx = max(est.values())
        for key in sorted(items, key=lambda k: -est[k]):          # longest dish first; short ones are held back so everything finishes together
            tk["items"].append(dict(key=key, tk=tk, est=est[key], not_before=self.clock + (mx - est[key]) * 0.8, state="queued", chef=None, ent=None))
        self.tickets.append(tk)
        (self.bar_items if dest == "bar" else self.mid_items).extend(tk["items"])
        return tk

    def pool_of(self, chef):
        return self.bar_items if chef.which == "ramen" else self.mid_items

    def pool_pending(self, chef) -> bool:
        return any(it["state"] == "queued" and not it["tk"]["void"] for it in self.pool_of(chef))

    def bar_in_progress(self) -> bool:
        return any(it["state"] in ("queued", "claimed", "cooking") and it["tk"]["started"] and not it["tk"]["void"] for it in self.bar_items)

    def claim_item(self, chef):
        now, best, bs = self.clock, None, -1e9
        for it in self.pool_of(chef):
            if it["state"] != "queued":
                continue
            tk = it["tk"]
            if tk["void"]:
                it["state"] = "void"
                continue
            if it["not_before"] > now:
                continue
            if tk["dest"] == "mid" and self.tray_owner.get(tk["tid"]) not in (None, tk):
                continue                                          # that table's tray is still busy with an earlier order
            free = self.station_owner.get(JP_FIRST[JP_HOW[it["key"]]]) in (None, chef)
            s = (now - tk["t0"]) * 0.8 + it["est"] * 0.3 + (5.0 if tk["started"] else 0.0) + (60.0 if tk["prio"] else 0.0) - (0.0 if free else 9.0)
            if s > bs:
                best, bs = it, s
        if best is None:
            return None
        if self.station_owner.get(JP_FIRST[JP_HOW[best["key"]]]) not in (None, chef):
            return None                                           # every candidate needs a busy station: wait a moment
        if self.plates["L"] - self.plate_res["L"] < 1:
            return None                                           # no clean plates in the left wing: the kitchen struggles
        self.plate_res["L"] += 1
        best["state"], best["chef"] = "claimed", chef
        tk = best["tk"]
        tk["started"] = True
        if tk["dest"] == "mid":
            self.tray_owner[tk["tid"]] = tk
        return best

    def take_plate(self, wing) -> None:
        self.plates[wing] = max(0, self.plates[wing] - 1)
        self.plate_res[wing] = max(0, self.plate_res[wing] - 1)

    def acquire(self, key, who) -> bool:
        o = self.station_owner.get(key)
        if o is None or o is who or getattr(o, "dead", False):
            self.station_owner[key] = who
            return True
        return False

    def release(self, key, who) -> None:
        if key is not None and self.station_owner.get(key) is who:
            del self.station_owner[key]

    def tray_put(self, tk, it) -> None:
        n = len(tk["done"])
        sx, sz = jp_tray_slot(tk["tid"])
        it["ent"] = jp_dish(it["key"], self.ch.detail, (sx, 1.13 + 0.02 * n, self.z0 + sz + (n - 1) * 0.1), self.kw)
        it["state"] = "done"
        tk["done"].append(it)
        if len(tk["done"]) == len(tk["items"]):
            tk["state"] = "ready"
            self.pass_ready.append(tk)
            if tk["cust"] is None:
                self.want_server = True                           # the player's tray: any waiter who is not carrying something comes now
            self._tray_lamp(tk["tid"], True)
            jp_say(self.g, sx + self.xo, sz + self.z0, "bell_table", 0.4, 14.0)

    def _tray_lamp(self, tid, on) -> None:
        tr = self.V.get("trays", {}).get(tid)
        if tr is not None:
            tr.color = C(0.25, 1.0, 0.35) if on else C(0.15, 0.10, 0.10)
            if FANCY:
                tr.set_shader_input("emissive", 1.6 if on else 0.0)

    def tray_clear(self, tk) -> None:
        for it in tk["items"]:
            if it.get("ent") is not None:
                ur.destroy(it["ent"])
                it["ent"] = None
        self._tray_lamp(tk["tid"], False)
        if self.tray_owner.get(tk["tid"]) is tk:
            del self.tray_owner[tk["tid"]]

    def _dish_at(self, key, x, y, z):
        return jp_dish(key, self.ch.detail, (x - self.xo, y, z), self.kw)

    def put_table_tray(self, tk) -> None:
        rec = tk["rec"]
        x, z = rec["slot"]
        for i, it in enumerate(tk["items"]):
            ent = self._dish_at(it["key"], x, 0.80, z + 0.13 * (i - (len(tk["items"]) - 1) / 2))
            self._give(rec, it["key"], ent, tk["cust"], x, z + 0.13 * (i - (len(tk["items"]) - 1) / 2))
        tk["state"] = "closed"

    def put_counter_tray(self, tk) -> None:
        rec = tk["rec"]
        x, z = rec["slot"]
        for i, it in enumerate(tk["items"]):
            zz = z + 0.17 * (i - (len(tk["items"]) - 1) / 2)
            ent = self._dish_at(it["key"], x, 1.10, zz)
            self._give(rec, it["key"], ent, tk["cust"], x, zz)
        tk["state"] = "closed"

    def _give(self, job_rec, key, ent, cust, x, z) -> None:
        g = self.g
        if cust is not None and cust.srec is job_rec and not cust.dead and not getattr(cust, "gone", False):
            cust.received(key, ent)
        else:
            self.ch.served.append(dict(key=key, ent=ent, x=x, z=z, rec=job_rec, t=self.clock))
            if self.p_rec is job_rec:
                g.hud.note(f"Your {ITEMS[key]['name']} has arrived.  Press T to eat with chopsticks, or U to put it in your bag.", 5.0)

    # ---- waiter jobs ----------------------------------------------------------------------------------------------------
    def server_has_work(self) -> bool:
        return bool(self.pass_ready) or any(r["state"] == "new" and self._req_ok(r) for r in self.reqs) or any(not p["claim"] and p["plates"] for p in self.piles.values())

    def _req_ok(self, r) -> bool:
        if r["cust"] is None:
            return self.p_rec is r["rec"]
        return r["cust"].srec is r["rec"] and not r["cust"].dead

    def pop_job(self, srv):
        for r in self.reqs:                                   # the player is always served first
            if r["state"] == "new" and r["cust"] is None and self._req_ok(r):
                r["state"], r["type"] = "claimed", "ask"
                self.want_server = False
                return r
        for tk in self.pass_ready:                            # a tray leaves only when the whole order is on it
            if not tk.get("claim") and tk["state"] == "ready":
                tk["claim"], tk["type"] = srv, "tray"
                self.pass_ready.remove(tk)
                if tk["cust"] is None:
                    self.want_server = False
                return dict(type="tray", tk=tk)
        big = [p for p in self.piles.values() if not p["claim"] and p["plates"] and (len(p["plates"]) >= 3 or self.clock - p["t"] > 30.0)]
        if big:
            p = big[0]
            p["claim"] = srv
            p["type"] = "clear"
            return p
        for r in self.reqs:
            if r["state"] == "new" and self._req_ok(r):
                r["state"], r["type"] = "claimed", "ask"
                return r
        self.reqs = [r for r in self.reqs if r["state"] in ("new", "claimed")]
        for p in self.piles.values():
            if not p["claim"] and p["plates"] and (self.clock - p["t"] > 6.0 or p["rec"]["occ"] is None):
                p["claim"] = srv
                p["type"] = "clear"
                return p
        return None

    def call_server(self, cust, rec):
        r = dict(state="new", rec=rec, cust=cust)
        self.reqs.append(r)
        return r

    def counter_wait(self, cust, rec):
        if not any(r["cust"] is cust and r["state"] in ("waiting", "claimed") for r in self.counter_reqs):
            self.counter_reqs.append(dict(state="waiting", rec=rec, cust=cust))

    def asks_pending(self) -> bool:
        return any(r["state"] == "waiting" and self._req_ok(r) for r in self.counter_reqs)

    def next_ask(self, chef):
        for r in self.counter_reqs:
            if r["state"] == "waiting" and self._req_ok(r):
                r["state"] = "claimed"
                return r
        self.counter_reqs = [r for r in self.counter_reqs if r["state"] in ("waiting", "claimed")]
        return None

    # ---- dirty plates: piles beside the seats -> silver table -> dishwasher -> rack -> Hiro -> cabinets -----------------
    def finish_dish(self, rec, key, ent) -> None:
        if ent is None:
            return
        for c in list(ent.children)[1:]:
            c.enabled = False
        pile = self.piles.get(id(rec))
        if pile is None:
            pile = self.piles[id(rec)] = dict(rec=rec, plates=[], t=self.clock, claim=None)
        if not pile["plates"]:
            pile["t"] = self.clock
        px, pz = rec["pile"]
        base = {"table": 0.80, "ramen": 1.10}.get(rec["kind"], 1.08)
        ent.position = (px - self.xo, base + 0.05 * min(8, len(pile["plates"])), pz)
        pile["plates"].append(ent)
        jp_say(self.g, px, pz, "dish_clink", 0.4, 14.0)

    def take_pile(self, pile) -> int:
        n = len(pile["plates"])
        for e in pile["plates"]:
            try:
                ur.destroy(e)
            except Exception:
                pass
        pile["plates"], pile["claim"] = [], None
        return n

    def silver_add(self, n) -> None:
        self.silver += n

    def silver_take(self, n) -> int:
        k = min(n, self.silver)
        self.silver -= k
        return k

    def rack_add(self, n) -> None:
        self.rack += n

    def rack_take(self, n) -> int:
        k = min(n, self.rack)
        self.rack -= k
        return k

    def hiro_target(self):
        if self.rack <= 0:
            return None
        lo = min(self.plates["L"], self.plates["R"])
        if self.rack >= 4 or lo <= 4 or self.clock - self.last_deliver > 25.0:
            return "L" if self.plates["L"] <= self.plates["R"] else "R"
        return None

    def player_wash(self) -> None:
        """The player's cleaning spot: grab a dirty dish off the silver table, scrub it, earn $1-4; the plate goes on the drying rack."""
        g = self.g
        if self.wash is not None:
            return
        if self.silver <= 0:
            g.hud.note("No dirty dishes on the silver table right now.", good=False)
            return
        self.silver -= 1
        self.wash = dict(t=0.0, nxt=0.0)
        g.sfx.play("dish_water", 0.8)
        fig = g.player.fig
        st = self.wash

        def anim(f, dt):
            e = st["t"]
            f.arms(dt, -45, -50, -62, -62, 0, 0, 20, 0, 12.0, None)
            f.sho[1].rotation_z = math.sin(e * 9) * 18.0
            return e < 3.2
        fig.anim = anim

    def _tick_wash(self, dt) -> None:
        w, g = self.wash, self.g
        if w is None:
            return
        p = g.player
        if math.hypot(p.x - (JP_L["wash_spot"][0] + self.xo), p.z - (JP_L["wash_spot"][1] + self.z0)) > 3.2 or p.kart is not None or p.seat is not None:
            self.silver += 1                                      # walked away: the dish goes back
            self.wash = None
            p.fig.anim = None
            return
        w["t"] += dt
        w["nxt"] -= dt
        if w["nxt"] <= 0:
            w["nxt"] = 1.7
            g.sfx.play("dish_scrub", 0.7)
        if w["t"] >= 3.2:
            pay = self.rng.randint(1, 4)
            g.inv.coins += pay
            g.inv.dirty = True
            self.rack += 1
            self.wash = None
            p.fig.anim = None
            g.sfx.play("coin", 0.7)
            g.sfx.play("dish_clink", 0.6)
            g.hud.note(f"Squeaky clean.  +${pay}   (clean plate on the drying rack)")

    # ---- uneaten plates & stacks -----------------------------------------------------------------------------------------
    def stash(self, rec, key, ent) -> None:
        x, z = rec["slot"]
        ent.position = (x - self.xo, 1.08, z + 0.34)
        self.ch.served.append(dict(key=key, ent=ent, x=x, z=z + 0.34, rec=rec, t=self.clock, stash=True))

    def find_stack(self, cust):
        rec, g = cust.srec, self.g
        sx, sz = rec["slot"]
        best, bd = None, 2.8
        for s in self.ch.served:
            if s["key"] not in ("plate_tamago", "plate_salmon", "plate_unagi", "plate_otoro") or s.get("claimed"):
                continue
            d = math.hypot(s["x"] - sx, s["z"] - sz)
            if g.player.seat is not None and s.get("rec") is not None and g.player.seat.data.get("jp") is s.get("rec") and d < 1.0:
                continue
            if d < bd:
                best, bd = s, d
        return best

    def claim_stack(self, s) -> None:
        s["claimed"] = True

    def take_stack(self, s, rec):
        if s in self.ch.served:
            self.ch.served.remove(s)
        s["ent"].position = (rec["slot"][0] - self.xo, 1.08, rec["slot"][1])
        return s["ent"]

    def _tidy(self) -> None:
        """Uneaten plates nobody wants any more get cleared eventually, so plates keep circulating."""
        p = self.g.player
        for s in list(self.ch.served):
            if s.get("rec") is not None and s.get("claimed") is None and self.clock - s.get("t", self.clock) > 240.0 \
                    and math.hypot(s["x"] - p.x, s["z"] - p.z) > 5.0:
                self.ch.served.remove(s)
                self.finish_dish(s["rec"], s["key"], s["ent"])

    # ---- player: seating, bells, prompts ----------------------------------------------------------------------------
    def player_call_chef(self, rec) -> None:
        if not any(r["cust"] is None and r["state"] in ("waiting", "claimed") for r in self.counter_reqs):
            self.counter_reqs.append(dict(state="waiting", rec=rec, cust=None))

    def player_ring(self, rec, quick: bool = False) -> None:
        g = self.g
        jp_say(g, rec["bell"][0], rec["bell"][1], "bell_table", 0.9)
        if rec["kind"] == "ramen":
            self.player_call_chef(rec)
            g.hud.note("*ding ding*  The bar chef will come as soon as their hands are free.")
        else:
            if not any(r["cust"] is None and r["state"] in ("new", "claimed") for r in self.reqs) and self.prompt is None:
                self.p_state, self.p_t = "ring", 1.0 if quick else 2.2
            g.hud.note("*ding ding*  A server heard the bell.")

    def open_prompt(self, who, rec, kind):
        pr = dict(state="open", who=who, rec=rec, kind=kind, items=None, t=0.0)
        self.prompt = pr
        self.g.jp_prompt = pr
        return pr

    def close_prompt(self, pr) -> None:
        if self.prompt is pr:
            self.prompt = None
        if getattr(self.g, "jp_prompt", None) is pr:
            self.g.jp_prompt = None

    def _player_changed(self, rec) -> None:
        old = self.p_rec
        self.p_rec = rec
        if self.prompt is not None and rec is not old:
            self.prompt["state"] = "gone"
            self.close_prompt(self.prompt)
        if rec is None:
            self.p_state = None
            return
        g = self.g
        if rec["kind"] == "table":
            self.p_state, self.p_t = "wait", self.rng.uniform(5.0, 10.0)
            g.hud.note("Take a seat. A server will come to you shortly.   R = ring the table bell", 4.0)
        elif rec["kind"] == "ramen":
            self.p_state, self.p_t = "wait", 2.5
            g.hud.note("Ramen bar: the bar chef will ask for your order when their hands are free.   R = ring the bell", 4.0)
        else:
            self.p_state = None
            g.hud.note("Kaiten sushi: press E to take a plate as it passes in front of you.", 4.0)

    def _player_logic(self, dt) -> None:
        if self.p_state in ("wait", "ring") and self.prompt is None:
            self.p_t -= dt
            if self.p_t <= 0:
                rec = self.p_rec
                if rec["kind"] == "table":
                    if not any(r["cust"] is None and r["state"] in ("new", "claimed") for r in self.reqs):
                        self.reqs.append(dict(state="new", rec=rec, cust=None))
                        self.want_server = True
                    self.p_state = "called"
                else:
                    self.player_call_chef(rec)
                    self.p_state = "called"

    def player_order(self, pr, items: list) -> None:
        pr["items"] = list(items)
        pr["state"] = "ordered"
        self.g.award("omakase" if pr["kind"] == "table" else "sakura")

    # ---- belt ------------------------------------------------------------------------------------------------------------
    def belt_load(self) -> int:
        return len(self.belt)

    def belt_add(self, kind: int, start) -> None:
        key = BELT_PLATES[kind][0]
        ent = jp_dish(key, self.ch.detail, (start[0], start[1], start[2]), self.kw)
        self.belt.append(dict(kind=kind, key=key, ent=ent, z=self.z0 + JP_L["belt_z0"] + 0.5, tw=0.0, a=start, taken=False))

    def belt_near(self, z: float, reach: float = 0.3):
        for pl in self.belt:
            if not pl["taken"] and pl["tw"] >= 0.5 and abs(pl["z"] - z) <= reach:
                return pl
        return None

    def belt_grab(self, pl, rec):
        if pl in self.belt:
            self.belt.remove(pl)
        ent = pl["ent"]
        x, z = rec["slot"]
        ent.position = (x - self.xo, 1.08, z)
        return ent

    def _tick_belt(self, dt) -> None:
        self.belt_ph = (self.belt_ph + dt * self.BELT_SPEED * 0.8) % 1.0
        if FANCY and self.V.get("belt") is not None:
            self.V["belt"].set_shader_input("uv_off", ur.Vec2(0.0, -self.belt_ph * 14.0 * 0.4))
        for pl in list(self.belt):
            if pl["tw"] < 0.5:
                pl["tw"] += dt
                k = min(1.0, pl["tw"] / 0.5)
                ax, ay, az = pl["a"]
                bx, by, bz = JP_L["belt_x"], 1.06, pl["z"]
                pl["ent"].position = (ax + (bx - ax) * k, ay + (by - ay) * k + 0.25 * math.sin(math.pi * k), az + (bz - az) * k)
            elif not pl["taken"]:
                pl["z"] += self.BELT_SPEED * dt
                pl["ent"].position = (JP_L["belt_x"], 1.06, pl["z"])
                if pl["z"] > self.z0 + JP_L["belt_z1"]:
                    ur.destroy(pl["ent"])
                    self.belt.remove(pl)
                    self.plates["R"] += 1                          # an unsold plate rides back into the right-wing kitchen

    def player_take_plate(self) -> bool:
        g, p = self.g, self.g.player
        rec = self.p_rec
        if rec is None or rec["kind"] != "belt":
            return False
        pl = self.belt_near(rec["slot"][1], 0.55)
        if pl is None:
            g.hud.note("No plate in reach yet - wait for one to glide past.", good=False)
            return True
        price = BELT_PLATES[pl["kind"]][1]
        if g.inv.coins < price:
            g.hud.note(f"This plate costs {price} c.", good=False)
            return True
        g.inv.coins -= price
        g.inv.dirty = True
        pl["taken"] = True
        ent = self.belt_grab(pl, rec)
        self.ch.served.append(dict(key=pl["key"], ent=ent, x=rec["slot"][0], z=rec["slot"][1], rec=rec, t=self.clock))
        g.sfx.play("plate", 0.7)
        g.hud.note(f"{ITEMS[pl['key']]['name']}  (-{price} c).  Press T to eat it with chopsticks, U to bag it.   Uneaten plates stay on the counter.", 4.5)
        g.award("conveyor")
        if self.chefR is not None and self.rng.random() < 0.25:
            g.hud.speak(self.chefR, "Hai, douzo!")
        return True

    # ---- doors: entering and leaving are different events ------------------------------------------------------------------
    def staff(self) -> list:
        w = [self.chefL, self.chefR] + self.chefs_mid + self.servers + [self.hiro, self.cleaner]
        return [s for s in w if s is not None and not s.dead]

    def _staff_react(self, d, entering: bool) -> None:
        """Only the nearest one or two staff react, and they say different things for a guest coming in and a guest going out."""
        i = 0 if d["side"] == "-z" else 1
        if self.door_cool[i] > 0:
            return
        self.door_cool[i] = 5.0
        who = sorted(self.staff(), key=lambda w: math.hypot(w.x - self.xo, w.z - d["z"]))
        who = [w for w in who if math.hypot(w.x - self.xo, w.z - d["z"]) < 16.0]
        if not who:
            return
        if entering:
            who[0].shout()
            if len(who) > 1 and self.rng.random() < 0.6:
                who[1].shout(None, "irasshai")
            for c in self.customers:                              # only diners near the door glance up
                if not c.dead and math.hypot(c.x - self.xo, c.z - d["z"]) < 6.0:
                    c.alert = 2.0
        else:
            who[0].say(self.rng.choice(JP_STAFF_FAREWELL), 3.0, "arigato", 0.8)
            if hasattr(who[0], "thanks") and self.rng.random() < 0.5:
                who[0].thanks()

    def npc_door(self, i: int, entering: bool, cust) -> None:
        d = self.ch.extra["jp_doors"][i]
        self.door_t[i] = 3.5
        jp_say(self.g, self.xo, d["z"], "door_slide", 0.8, 30.0)
        jp_say(self.g, self.xo, d["z"], "door_bell", 0.8, 34.0)
        self._staff_react(d, entering)

    def retire(self, cust) -> None:
        cust.gone = True
        self.retired.append(cust)

    def _tick_doors(self, dt) -> None:
        g, p, ch = self.g, self.g.player, self.ch
        t = self.clock
        for i in (0, 1):
            self.door_cool[i] = max(0.0, self.door_cool[i] - dt)
            self.door_t[i] = max(0.0, self.door_t[i] - dt)
        for i, d in enumerate(ch.extra.get("jp_doors", ())):
            near = abs(p.z - d["z"]) < 3.4 and abs(p.x - self.xo) < 3.4
            inward = 1.0 if d["side"] == "-z" else -1.0
            inside = (p.z - d["z"]) * inward > 0.0
            if abs(p.x - self.xo) < 2.4 and abs(p.z - d["z"]) < 4.0:       # which way did the player cross?
                if self.pside[i] is not None and inside != self.pside[i]:
                    self._staff_react(d, inside)
                    if inside:
                        g.award("sakura")
                self.pside[i] = inside
            else:
                self.pside[i] = None
            opening = near or self.door_t[i] > 0
            if opening and not d["was"]:
                jp_say(g, self.xo, d["z"], "door_slide", 0.9, 30.0)
                jp_say(g, self.xo, d["z"], "door_bell", 1.0, 34.0)
            elif d["was"] and not opening and d["open"] > 0.5:
                jp_say(g, self.xo, d["z"], "door_slide", 0.5, 22.0)
            d["was"] = opening
            d["open"] = max(0.0, min(1.0, d["open"] + (1.7 if opening else -1.4) * dt))
            for pv, s in d["panels"]:
                pv.x = s * (d["gap"] / 2 + d["open"] * d["gap"] * 0.98)
            if d["open"] > 0.6 and d["coll"] in ch.colliders:
                ch.colliders.remove(d["coll"])
            elif d["open"] < 0.35 and d["coll"] not in ch.colliders:
                ch.colliders.append(d["coll"])
            push = max(0.0, 1.0 - math.hypot(p.x - self.xo, p.z - d["z"]) / 1.6) * 24.0
            for pv, ph in d["strips"]:
                pv.rotation_x = math.sin(t * 1.3 + ph) * 3.0 + (push if (p.z - d["z"]) * (1 if d["side"] == "-z" else -1) > 0 else -push)

    def _tick_decor(self, dt) -> None:
        V, t = self.V, self.clock
        for e, ph in V.get("steam", ()):
            k = (t * 0.35 + ph) % 1.0
            e.position = (-11.25 + 0.12 * math.sin(t * 2 + ph), 2.0 + 0.9 * k, V["pot_z"])
            e.color = C(0.95, 0.95, 0.98, 0.35 * math.sin(math.pi * k))
            e.scale = 0.12 + 0.25 * k
        if V.get("neko") is not None:
            V["neko"].rotation_z = math.sin(t * 3.2) * 24.0
        for e, tx, tz, ph, sp in V.get("petals", ()):
            k = (t * sp * 0.5 + ph) % 1.0
            e.position = (tx + 0.7 * math.sin(t * 0.7 + ph) * k, 2.3 - 2.28 * k, self.z0 + tz + 0.6 * math.cos(t * 0.5 + ph) * k)
            e.rotation_y = t * 90 + ph * 40

    def _tick_ambience(self, dt) -> None:
        g, p = self.g, self.g.player
        if abs(p.z - (self.z0 + 12.0)) > 16.0 or getattr(g, "ambient_muted", False):
            return
        a = self.amb_t
        a[0] -= dt
        a[1] -= dt
        a[2] -= dt
        if a[0] <= 0:
            a[0] = 3.6
            g.sfx.play("murmur", 0.55)
        if a[1] <= 0:
            a[1] = self.rng.uniform(1.4, 4.2)
            g.sfx.play(f"koto{self.rng.randrange(14)}", 0.22, pan=self.rng.uniform(-0.4, 0.4))
        if a[2] <= 0:
            a[2] = 2.9
            if p.x - self.xo > 3.0:
                g.sfx.play("belt_hum", 0.35, pan=0.5)

    def _tick_pipeline(self, dt) -> None:
        load = sum(1 for it in self.mid_items if it["state"] == "queued") + 0.6 * len(self.pass_ready) + sum(1 for r in self.reqs if r["state"] == "new")
        self.rush += (max(0.0, min(1.0, (load - 2.0) / 5.0)) - self.rush) * min(1.0, dt * 0.8)
        self.mid_items = [i for i in self.mid_items if i["state"] in ("queued", "claimed", "cooking")]
        self.bar_items = [i for i in self.bar_items if i["state"] in ("queued", "claimed", "cooking") or not i["tk"]["state"] == "closed"]
        self.tickets = [t for t in self.tickets if t["state"] != "closed" and not (t["void"] and all(i["state"] in ("void", "done") for i in t["items"]))]
        for c in self.retired:
            if c in self.customers:
                self.customers.remove(c)
            if c in self.ch.npcs:
                self.ch.npcs.remove(c)
            c.rec_free = True
            REAPER.push(c.fig.root)
            self.timers.append([self.rng.uniform(22.0, 65.0), self.spawn_arrival])
        self.retired.clear()
        self.vis_t -= dt
        if self.vis_t <= 0:
            self.vis_t = 0.25
            self._tidy()
            self._sync_vis()

    def spawn_arrival(self) -> None:
        if len([c for c in self.customers if not c.gone]) >= self.arrive_cap:
            return
        free = [r for r in self.ch.extra["jp_seats"] if not r["it"].occupied and r["occ"] is None]
        if not free:
            self.timers.append([20.0, self.spawn_arrival])
            return
        rec = self.rng.choice(free)
        d = 0 if self.rng.random() < 0.5 else 1
        c = JpCustomer(self.g, self.ch, self.rng, self, rec, arrive=d)
        self.customers.append(c)
        self.ch.npcs.append(c)
        c.fig.root.enabled = self.ch.shown

    def _sync_vis(self) -> None:
        V = self.V
        for key, n in (("silver_pl", self.silver), ("rack_pl", self.rack), ("cabL_pl", self.plates["L"]), ("cabR_pl", self.plates["R"])):
            for i, e in enumerate(V.get(key, ())):
                e.enabled = i < n
        for key, n in (("cabL_lamp", self.plates["L"]), ("cabR_lamp", self.plates["R"])):
            e = V.get(key)
            if e is not None:
                e.color = C(0.2, 1.0, 0.3) if n > 4 else (C(1.0, 0.7, 0.1) if n > 0 else C(1.0, 0.1, 0.1))
                if FANCY:
                    e.set_shader_input("emissive", 1.5)

    def tick(self, dt, g) -> bool:
        self.clock += dt
        seat = g.player.seat
        rec = seat.data.get("jp") if (seat is not None and seat in self.ch.inters) else None
        if rec is not self.p_rec:
            self._player_changed(rec)
        for key in ("_tick_doors", "_tick_belt", "_tick_decor", "_tick_ambience", "_player_logic", "_tick_pipeline", "_tick_wash"):
            try:
                getattr(self, key)(dt)
            except Exception:
                log.exception("restaurant %s failed", key)
                setattr(self, key, lambda dt_: None)
        for tm in list(self.timers):
            tm[0] -= dt
            if tm[0] <= 0:
                self.timers.remove(tm)
                try:
                    tm[1]()
                except Exception:
                    log.exception("restaurant timer failed")
        return True


def _japan_npcs(self) -> None:
    """Called from the (patched) _v3_npcs hook: staff + diners for SAKURA-YA."""
    ch, g = self.ch, self.w.g
    rest = JpRestaurant(g, ch)
    ch.extra["jp_rest"] = rest
    rng = seeded(self.seed, "japan-npc", ch.idx)
    xo, z0 = ch.xoff, ch.z0
    rest.chefL = JpChef(g, ch, (xo - 9.6, z0 + 13.0), rng, rest, "ramen", 0)                    # the dedicated left-wing bar chef
    rest.chefs_mid = [JpChef(g, ch, (xo - 9.6, z0 + 8.6), rng, rest, "mid", 1),                 # the two core chefs behind the left side bar
                      JpChef(g, ch, (xo - 9.6, z0 + 18.2), rng, rest, "mid", 2)]
    rest.chefR = JpChef(g, ch, (xo + JP_L["chefR"][0], z0 + JP_L["chefR"][1]), rng, rest, "sushi", 0)
    rest.servers = [JpServer(g, ch, (xo + 0.0, z0 + 3.0), rng, rest, "server"), JpServer(g, ch, (xo - 5.5, z0 + 15.0), rng, rest, "server"),
                    JpServer(g, ch, (xo + 5.5, z0 + 9.0), rng, rest, "server")]
    rest.hiro = JpHiro(g, ch, (xo + JP_L["rack_stand"][0], z0 + JP_L["rack_stand"][1]), rng, rest)
    rest.cleaner = JpCleaner(g, ch, (xo + JP_L["cleaner"][0], z0 + JP_L["cleaner"][1]), rng, rest)
    ch.npcs += [rest.chefL, rest.chefR] + rest.chefs_mid + rest.servers + [rest.hiro, rest.cleaner]
    seats = ch.extra["jp_seats"]
    for rec in seats:
        s, c = math.sin(math.radians(rec["yaw"])), math.cos(math.radians(rec["yaw"]))
        if rec["kind"] == "table":
            rec["clear_spot"] = rec["spot"]
            rec["pile"] = (rec["slot"][0], rec["slot"][1] + (0.34 if rec["slot"][1] > rec["tz"] else -0.34))
        elif rec["kind"] == "ramen":
            rec["clear_spot"] = (rec["x"] + 0.9, rec["z"])
            rec["pile"] = (rec["slot"][0], rec["slot"][1] + 0.34)
        else:
            rec["clear_spot"] = (rec["x"] - 0.9, rec["z"])
            rec["pile"] = (rec["slot"][0] - 0.12, rec["slot"][1] + 0.34)
    for rec in seats:
        pr = {"table": 0.22, "ramen": 0.26, "belt": 0.55}[rec["kind"]]
        if rng.random() < pr and not rec["it"].occupied:
            c = JpCustomer(g, ch, rng, rest, rec)
            rest.customers.append(c)
            ch.npcs.append(c)
    rest.arrive_cap = max(8, len(rest.customers) + 2)
    ch.tickers.append(rest.tick)

    def warm():                                         # synthesize the restaurant sounds off the render thread
        for nm in ("door_slide", "door_bell", "irasshai", "chop", "sizzle", "boil", "mw_hum", "mw_beep", "mw_ding", "mw_door", "plate", "slurp",
                   "bell_table", "whoosh", "pour", "pat", "murmur", "belt_hum", "oishii", "arigato", "itadaki", "gochisou", "dish_water",
                   "dish_scrub", "dish_clink", "dish_stack", "tray_set") + tuple(f"koto{k}" for k in range(14)):
            try:
                g.sfx._arr(nm, g.sfx.sr)
            except Exception:
                pass
    if g.sfx.enabled and not getattr(g.sfx, "_jp_warm", False):
        g.sfx._jp_warm = True
        threading.Thread(target=warm, name="jp-sfx-warm", daemon=True).start()


# ---- order menu -----------------------------------------------------------------------------------------------------
class JpMenuUI(UiBase):
    """Sakura-ya menu: W/S browse, Enter adds a dish, Backspace removes the last, Space sends the order."""
    mouse_free = False

    def __init__(self, g, pr) -> None:
        super().__init__(g)
        self.pr, self.sel, self.basket, self.msg = pr, 0, [], ""
        self.rows_data = []
        last = None
        for sec, key, price, how in JP_MENU:
            if sec != last:
                self.rows_data.append(("hdr", sec, 0))
                last = sec
            self.rows_data.append(("item", key, price))
        self.items = [i for i, r in enumerate(self.rows_data) if r[0] == "item"]
        self.quad((0, 0), (1.30, 0.90), C(1.0, 0.75, 0.45, 0.35), z=0.22)
        self.quad((0, 0), (1.28, 0.88), C(0.10, 0.04, 0.05, 0.96), z=0.21)
        self.text("SAKURA-YA   -   MENU   (okonomi)", (-0.60, 0.40), 1.1, C(1.0, 0.85, 0.55, 1))
        self.wallet = self.text("", (0.20, 0.40), 0.85, C(1.0, 0.9, 0.5, 1))
        self.rows = [self.text("", (-0.60, 0.33 - 0.034 * i), 0.78) for i in range(17)]
        self.bk = self.text("", (-0.60, -0.28), 0.82, C(1.0, 0.95, 0.75, 1))
        self.text("W/S browse    Enter add dish    Backspace remove    Space SEND ORDER    E / Esc not now", (-0.60, -0.40), 0.62, C(0.8, 0.75, 0.7, 0.95))
        self.refresh()

    def total(self) -> int:
        return sum(JP_PRICE[k] for k in self.basket)

    def refresh(self) -> None:
        g = self.g
        self.wallet.text = f"{g.inv.coins} c   {self.msg}"
        cur = self.items[self.sel]
        top = max(0, min(cur - 8, len(self.rows_data) - len(self.rows)))
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= len(self.rows_data):
                row.text = ""
                continue
            kind, key, price = self.rows_data[j]
            if kind == "hdr":
                row.text = f"--  {key}  --"
                row.color = C(1.0, 0.65, 0.4, 1)
            else:
                n = self.basket.count(key)
                row.text = f"{'>' if j == cur else ' '} {ITEMS[key]['name']:<20} {price:>2} c   {ITEMS[key]['desc'][:46]}{'   x' + str(n) if n else ''}"
                row.color = C(1.0, 0.95, 0.6, 1) if j == cur else C(0.88, 0.95, 0.9, 0.95)
        names = ", ".join(f"{ITEMS[k]['name']}" for k in self.basket) or "(nothing yet)"
        self.bk.text = f"YOUR ORDER  {self.total()} c:  {names}"[:130]

    def key(self, k: str) -> None:
        g, pr = self.g, self.pr
        n = len(self.items)
        if k in ("e", "escape"):
            pr["state"] = "no"
            g.close_ui()
            return
        if k in ("w", "up arrow"):
            self.sel = (self.sel - 1) % n
        elif k in ("s", "down arrow"):
            self.sel = (self.sel + 1) % n
        elif k in ("enter", "return"):
            if len(self.basket) >= 6:
                self.msg = "that's plenty for one round"
            else:
                self.basket.append(self.rows_data[self.items[self.sel]][1])
                g.sfx.play("tick", 0.6)
        elif k == "backspace" and self.basket:
            self.basket.pop()
        elif k in ("space", "f"):
            if not self.basket:
                self.msg = "add something first"
            elif g.inv.coins < self.total():
                self.msg = "not enough coins"
            else:
                g.inv.coins -= self.total()
                g.inv.dirty = True
                g.sfx.play("coin")
                rest = g._jp_rest()
                if rest is not None:
                    rest.player_order(pr, self.basket)
                g.hud.note(f"Order sent:  {len(self.basket)} dish(es), {self.total()} c.")
                g.close_ui()
                return
        g.sfx.play("tick", 0.4)
        self.refresh()

    def close(self) -> None:
        if self.pr.get("state") == "menu":
            self.pr["state"] = "no"
        super().close()


# ---- phone music ------------------------------------------------------------------------------------------------------
def _sfx_set_phone(self, arr: np.ndarray, gain: float = 0.9) -> None:
    with self._lock:
        self._phone, self._phone_paused = [arr, 0, float(gain)], False


def _sfx_phone_stop(self) -> None:
    with self._lock:
        self._phone = None


def _sfx_phone_pause(self, b: bool) -> None:
    self._phone_paused = bool(b)


def _sfx_phone_active(self) -> bool:
    return getattr(self, "_phone", None) is not None


def _make_cb(orig):
    def _cb(self, outdata, frames, time_info, status) -> None:
        orig(self, outdata, frames, time_info, status)
        ph = getattr(self, "_phone", None)
        if ph is None or getattr(self, "_phone_paused", False):
            return
        arr, pos, gain = ph
        m = min(frames, arr.shape[1] - pos)
        if m > 0:
            seg = arr[:, pos:pos + m].T.astype(np.float32)
            a = float(getattr(self, "phone_lp", 1.0))
            if a < 0.995:
                zi = getattr(self, "_phone_zi", None)
                if zi is None:
                    zi = np.zeros((1, 2))
                seg, self._phone_zi = signal.lfilter([a], [1.0, -(1.0 - a)], seg, axis=0, zi=zi)
                seg = seg.astype(np.float32)
            else:
                self._phone_zi = None
            lr = getattr(self, "phone_lr", (1.0, 1.0))
            outdata[:m] += seg * np.array(lr, np.float32) * gain
            np.clip(outdata, -1.0, 1.0, out=outdata)
        ph[1] = pos + max(m, 0)
        if ph[1] >= arr.shape[1]:
            self._phone = None
    return _cb


class PhoneMusic:
    """Plays tracks you downloaded (Listening Post D / phone Library) through the SFX stream and shuffles on."""
    def __init__(self, g) -> None:
        self.g, self.cur, self.loading, self.paused, self.held = g, None, False, False, False
        self.token, self.hist = 0, []

    def title(self, tid=None) -> str:
        tid = self.cur if tid is None else tid
        try:
            return self.g.covers.title(tid).replace("_", " ")[:38]
        except Exception:
            return f"track {tid}"

    def playlist(self) -> list:
        tr = self.g.stack.indexer.tracks if self.g.stack.indexer is not None else {}
        return [t for t in self.g.inv.downloads if t in tr]

    @property
    def active(self) -> bool:
        return self.cur is not None and not self.paused and not self.held

    def play(self, tid: int, quiet: bool = False) -> None:
        g = self.g
        if not g.sfx.enabled or librosa is None or g.stack.indexer is None:
            g.hud.note("No audio output available for the phone.", good=False)
            return
        info = g.stack.indexer.tracks.get(tid)
        if info is None:
            return
        self.cur, self.paused, self.loading = tid, False, True
        self.token += 1
        tok, sr = self.token, g.sfx.sr
        g.sfx.phone_stop()
        self.hist = (self.hist + [tid])[-12:]

        def work():
            try:
                y, _ = librosa.load(str(info.path), sr=sr, mono=False, duration=600.0)
                y = np.stack([y, y]) if y.ndim == 1 else y[:2]
                if tok == self.token:
                    g.sfx.set_phone(np.ascontiguousarray(y.astype(np.float16)), 0.9)
                    g.sfx.phone_pause(self.held or self.paused)
                    self.loading = False
            except Exception as exc:
                log.warning("phone playback failed: %s", exc)
                if tok == self.token:
                    self.cur, self.loading = None, False
        threading.Thread(target=work, name="phone-music", daemon=True).start()
        if not quiet:
            g.hud.note(f"Phone  >  {self.title(tid)}      [M] pause   [N] next (shuffle)", 4.0)
        g.award("playlist")

    def pick(self, exclude=None) -> Optional[int]:
        pl = self.playlist()
        if not pl:
            return None
        cand = [t for t in pl if t != exclude and t not in self.hist[-max(1, len(pl) // 2):]] or [t for t in pl if t != exclude] or pl
        return random.choice(cand)

    def toggle(self) -> None:
        g = self.g
        if self.cur is None:
            t = self.pick()
            if t is None:
                g.hud.note("No downloaded music.  Press D at a Listening Post, or open the phone's MUSIC page.", good=False)
            else:
                self.play(t)
            return
        self.paused = not self.paused
        g.sfx.phone_pause(self.paused or self.held)
        g.hud.note(f"Phone: {'paused' if self.paused else 'playing'}  -  {self.title()}")

    def next(self) -> None:
        t = self.pick(self.cur)
        if t is None:
            self.g.hud.note("No downloaded music.", good=False)
        else:
            self.play(t)

    def stop(self) -> None:
        self.cur, self.loading, self.paused = None, False, False
        self.token += 1
        self.g.sfx.phone_stop()

    def hold(self, b: bool) -> None:
        self.held = b
        self.g.sfx.phone_pause(self.paused or b)

    def update(self, dt: float) -> None:
        if self.cur is not None and not self.loading and not self.g.sfx.phone_active and not self.paused:
            nxt = self.pick(self.cur)
            self.cur = None
            if nxt is not None:
                self.play(nxt)


# ---- the phone: map / garage / music ----------------------------------------------------------------------------------
_PhoneUI_v3 = PhoneUI


class PhoneUI(_PhoneUI_v3):      # noqa: F811
    PAGES = ("MAP & ORDER", "GARAGE", "MUSIC")

    def __init__(self, g) -> None:
        self.page, self.sel, self.lib, self.pmsg, self.hint0, self.hint1 = 0, 0, False, "", None, None
        super().__init__(g)
        self.title = self.text("", (-0.26, 0.405), 0.8, C(1.0, 0.9, 0.5, 1))
        for e in self.ents:
            tx = getattr(e, "text", "")
            if isinstance(tx, str) and tx.startswith("[1] order"):
                self.hint0 = e
            elif isinstance(tx, str) and tx.startswith("P / E"):
                self.hint1 = e
        self.redraw()

    # -- helpers
    def near_karts(self) -> list:
        g, p = self.g, self.g.player
        ks = [k for k in g.karts.values() if getattr(k, "level", "mall") == g.world.level and k is not p.kart
              and math.hypot(k.x - p.x, k.z - p.z) < 14.0]
        return sorted(ks, key=lambda k: math.hypot(k.x - p.x, k.z - p.z))

    def stored(self) -> list:
        return self.g.inv.flags.setdefault("pkarts", [])

    def redraw(self) -> None:
        pg = getattr(self, "page", 0)
        if getattr(self, "title", None) is not None:
            self.title.text = f"<  {self.PAGES[pg]}  >      A/D page"
        if pg == 0:
            if self.hint0 is not None:
                self.hint0.text = "[1] order water  2 c    [2] order donut  3 c"
                self.hint1.text = "P / E / Esc close"
            return super().redraw()
        g, p = self.g, self.g.player
        S = 320
        im = Image.new("RGBA", (S, S), (8, 14, 14, 255))
        d = ImageDraw.Draw(im)
        f, f2 = _font(15), _font(12)
        if pg == 1:
            near, st = self.near_karts(), self.stored()
            self.sel = max(0, min(self.sel, max(0, len(near) - 1)))
            d.text((10, 8), f"POCKET GARAGE   {len(st)}/3 stored", fill=(255, 220, 120, 255), font=f)
            d.text((10, 36), "NEARBY KARTS", fill=(120, 255, 200, 255), font=f2)
            if not near:
                d.text((14, 56), "no kart within 14 m", fill=(150, 160, 170, 255), font=f2)
            for i, k in enumerate(near[:5]):
                y = 56 + 22 * i
                if i == self.sel:
                    d.rectangle([8, y - 2, S - 8, y + 17], fill=(40, 80, 70, 255))
                col = tuple(int(255 * c) for c in (getattr(k, "paint", None) or (0.8, 0.12, 0.14)))
                d.rectangle([14, y + 1, 30, y + 14], fill=col + (255,))
                d.text((38, y), f"{math.hypot(k.x - p.x, k.z - p.z):4.1f} m   fuel {k.fuel:3.0f}%", fill=(235, 250, 240, 255), font=f2)
            d.text((10, 176), "STORED IN PHONE", fill=(120, 255, 200, 255), font=f2)
            for i in range(3):
                y = 198 + 24 * i
                if i < len(st):
                    col = tuple(int(255 * c) for c in (st[i].get("paint") or (0.8, 0.12, 0.14)))
                    d.rectangle([14, y + 1, 30, y + 14], fill=col + (255,))
                    d.text((38, y), f"[{i + 1}]  summon   fuel {st[i]['fuel']:3.0f}%", fill=(255, 235, 190, 255), font=f2)
                else:
                    d.text((38, y), f"[{i + 1}]  empty", fill=(110, 120, 125, 255), font=f2)
            self.info.text = "W/S choose   Enter / K store nearest-selected kart"
            self.info2.text = self.pmsg or "1 / 2 / 3  summon a stored kart beside you"
            if self.hint0 is not None:
                self.hint0.text = "Enter / K store    1-3 summon"
                self.hint1.text = "A/D page    P / E / Esc close"
        else:
            pm = g.phone
            lst = self.library() if self.lib else pm.playlist()
            self.sel = max(0, min(self.sel, max(0, len(lst) - 1)))
            d.text((10, 8), "LIBRARY (download)" if self.lib else "MY MUSIC  (downloaded)", fill=(255, 220, 120, 255), font=f)
            now = pm.title() if pm.cur is not None else "nothing playing"
            d.text((10, 32), ("|| " if pm.paused else "> ") + now + ("  (loading)" if pm.loading else ""), fill=(120, 255, 200, 255), font=f2)
            top = max(0, min(self.sel - 5, len(lst) - 10))
            for i in range(10):
                j = top + i
                if j >= len(lst):
                    break
                y = 58 + 24 * i
                if j == self.sel:
                    d.rectangle([8, y - 2, S - 8, y + 19], fill=(40, 80, 70, 255))
                tag = "*" if (lst[j] == pm.cur) else " "
                done = "  [on phone]" if (self.lib and lst[j] in g.inv.downloads) else ""
                d.text((12, y), f"{tag}{j + 1:2d} {pm.title(lst[j])[:30]}{done}", fill=(235, 250, 240, 255), font=f2)
            if not lst:
                d.text((14, 70), "Nothing here yet. Press L for the library," if not self.lib else "No unlocked tracks yet.", fill=(150, 160, 170, 255), font=f2)
                d.text((14, 90), "or press D at a Listening Post.", fill=(150, 160, 170, 255), font=f2)
            self.info.text = "Enter play/download   Space pause   N next   L library"
            self.info2.text = self.pmsg or "X remove   (M / N work anywhere while the phone is in your bag)"
            if self.hint0 is not None:
                self.hint0.text = "Enter play / download   Space pause   N next"
                self.hint1.text = "L library    X remove    A/D page    P/E close"
        self.screen.texture = tex_from_pil(im)

    def library(self) -> list:
        g = self.g
        tr = sorted(g.stack.indexer.tracks) if g.stack.indexer is not None else []
        return [t for t in tr if t in g.inv.unlocked or stable_hash(g.args.seed, "free", t) % 4 == 0]

    def key(self, k: str) -> None:
        g = self.g
        if k in ("left arrow", "a") or k in ("right arrow", "d"):
            self.page = (self.page + (-1 if k in ("left arrow", "a") else 1)) % 3
            self.sel, self.pmsg = 0, ""
            g.sfx.play("tick", 0.5)
            self.redraw()
            return
        if self.page == 0:
            return super().key(k)
        if k in ("p", "e", "escape", "g"):
            g.close_ui()
            return
        if self.page == 1:
            near, st = self.near_karts(), self.stored()
            if k in ("w", "up arrow") and near:
                self.sel = (self.sel - 1) % len(near)
            elif k in ("s", "down arrow") and near:
                self.sel = (self.sel + 1) % len(near)
            elif k in ("enter", "return", "k"):
                if len(st) >= 3:
                    self.pmsg = "storage full (3)"
                elif not near:
                    self.pmsg = "no kart nearby"
                else:
                    kt = near[min(self.sel, len(near) - 1)]
                    st.append(dict(paint=list(getattr(kt, "paint", None) or []) or None, fuel=float(kt.fuel)))
                    g.karts.pop(kt.key, None)
                    g._drop_kart(kt)
                    g.inv.dirty = True
                    g.sfx.play("chime", 0.7)
                    g.sfx.play("pickup", 0.6)
                    self.pmsg = "kart folded into your phone"
                    g.award("pocketkart")
            elif k in ("1", "2", "3"):
                i = int(k) - 1
                if i >= len(st):
                    self.pmsg = "that slot is empty"
                elif g.summon_kart(st[i]):
                    st.pop(i)
                    g.inv.dirty = True
                    self.pmsg = "kart unfolded beside you"
                else:
                    self.pmsg = "no room to unfold it here"
        else:
            pm = g.phone
            lst = self.library() if self.lib else pm.playlist()
            if k in ("w", "up arrow") and lst:
                self.sel = (self.sel - 1) % len(lst)
            elif k in ("s", "down arrow") and lst:
                self.sel = (self.sel + 1) % len(lst)
            elif k == "l":
                self.lib, self.sel = not self.lib, 0
            elif k in ("enter", "return") and lst:
                tid = lst[min(self.sel, len(lst) - 1)]
                if self.lib:
                    if tid in g.inv.downloads:
                        self.pmsg = "already on your phone"
                    else:
                        g.inv.downloads.append(tid)
                        g.inv.dirty = True
                        g.sfx.play("chime", 0.6)
                        self.pmsg = "downloaded"
                else:
                    pm.play(tid, quiet=True, src=None)
                    self.pmsg = "playing"
            elif k == "space":
                pm.toggle()
            elif k == "n":
                pm.next()
            elif k in ("x", "delete") and not self.lib and lst:
                tid = lst[min(self.sel, len(lst) - 1)]
                g.inv.downloads.remove(tid)
                g.inv.dirty = True
                if pm.cur == tid:
                    pm.stop()
                self.pmsg = "removed"
        g.sfx.play("tick", 0.4)
        self.redraw()


# ---- Game additions ----------------------------------------------------------------------------------------------------
def _jp_rest(self):
    ch = self.world.at(self.player.z)
    return ch.extra.get("jp_rest") if ch is not None else None


def _jp_rec_of_seat(self):
    s = self.player.seat
    d = getattr(s, "data", None) if s is not None else None
    return d.get("jp") if isinstance(d, dict) else None


def summon_kart(self, rec: dict) -> bool:
    p, w = self.player, self.world
    yaw = p.yaw_body
    for dist in (2.8, 2.2, 3.6, 1.8):
        nx, nz = p.x + math.sin(math.radians(yaw)) * dist, p.z + math.cos(math.radians(yaw)) * dist
        rx, rz, _g, bumped = w.resolve(nx, p.y, nz, GoKart.R, step=0.25)
        if not bumped and math.hypot(rx - nx, rz - nz) < 0.05:
            key = f"{w.level}:phone:{int(time.time() * 1000) % 10 ** 9}"
            k = GoKart(key, nx, nz, yaw, fuel=float(rec.get("fuel", 30.0)))
            k.level = w.level
            if rec.get("paint"):
                k.paint = tuple(rec["paint"])
            self.karts[key] = k
            k.build()
            self.sfx.play("thud", 0.6)
            return True
    return False


def _v4_init(self) -> None:
    _orig_v3_init(self)
    self.phone = PhoneMusic(self)
    self.jp_prompt, self.jp_eat, self.pbub = None, None, None


def _v4_update(self, dt: float, kind: str) -> None:
    _orig_update_v3(self, dt, kind)
    p = self.player
    if getattr(self.bridge, "game", None) is None:
        self.bridge.game = self
    self.phone.update(dt)
    pr = self.jp_prompt
    if pr is not None and (p.seat is None or pr["state"] not in ("open", "menu")):
        if pr["state"] in ("open", "menu") and p.seat is None:
            pr["state"] = "gone"
        self.jp_prompt = None
    if isinstance(self.ui, JpMenuUI) and (self.jp_prompt is None or p.seat is None):
        self.close_ui()
    if self.pbub is not None:
        self.pbub.tick(dt, True)
    ea = self.jp_eat
    if ea is not None and (p.seat is None or ea["s"] not in (self.world.at(p.z).served if self.world.at(p.z) else [])):
        p.fig.anim = None
        if getattr(p.fig, "chop", None) is not None:
            p.fig.chop.enabled = False
        self.jp_eat = None


def _jp_start_eat(self, s: dict) -> None:
    p, fig = self.player, self.player.fig
    if getattr(fig, "chop", None) is None:
        root = ur.Entity(parent=fig.elb[1])
        for dx in (0.014, -0.014):
            _fm(fig, root, "cube", (dx, -0.42, 0.15), (0.012, 0.012, 0.26), (0.62, 0.45, 0.28))
        fig.chop = root
    fig.chop.enabled = True
    if self.pbub is None:
        self.pbub = Bubble(fig)
    key = s["key"]
    n = 5 if key.startswith("ramen") else 3
    st = dict(t=0.0, n=n, s=s, key=key, slurp=-1)
    self.jp_eat = st
    self.pbub.say("Itadakimasu!", 2.2)
    self.sfx.play("itadaki", 0.55)
    per = 2.5

    def anim(f, dt):
        st["t"] += dt
        e = st["t"]
        i, ph = int(e // per), (e % per) / per
        if i >= n:
            self._jp_finish_eat(st)
            return False
        if ph < 0.30:
            f.arms(dt, -30, -40, -78, -22, 0, -6, 14, 0, 11.0, "sit")
        elif ph < 0.45:
            f.arms(dt, -30, -40, -62, -45, 0, -8, 12, 0, 11.0, "sit")
        elif ph < 0.65:
            f.arms(dt, -30, -40, -52, -132, 0, -18, -4, 0, 11.0, "sit")
            if key.startswith("ramen") and st["slurp"] != i and ph > 0.5:
                st["slurp"] = i
                self.sfx.play("slurp", 0.6)
        else:
            f.arms(dt, -30, -40, -48, -128, 0, -18, 6 * math.sin(e * 16), 0, 11.0, "sit")
        return True
    fig.anim = anim


def _jp_finish_eat(self, st: dict) -> None:
    p = self.player
    ch = self.world.at(p.z)
    s = st["s"]
    if ch is not None and s in ch.served:
        ch.served.remove(s)
        ur.destroy(s["ent"])
    self.vitals.eat(st["key"])
    self.sfx.play("gulp")
    self.hud.note(f"{ITEMS[st['key']]['name']}  -  {ITEMS[st['key']]['desc']}")
    if getattr(p.fig, "chop", None) is not None:
        p.fig.chop.enabled = False
    if self.pbub is not None:
        self.pbub.say(random.choice(("Oishii!", "Gochisousama deshita!", "Umai!")), 2.6)
        self.sfx.play("oishii", 0.5)
    self.jp_eat = None


def _use_held_v4(self) -> None:
    p = self.player
    s = self._served_near() if p.seat is not None else None
    if s is not None and s["key"] in JP_KEYS:
        if self.jp_eat is None:
            self._jp_start_eat(s)
        return
    _orig_use_held(self)


def _input_v4(self, key: str) -> bool:
    p = self.player
    pr = self.jp_prompt
    if pr is not None and pr["state"] == "open" and key in ("y", "n"):
        if key == "y":
            pr["state"] = "menu"
            self.open_ui(JpMenuUI(self, pr))
        else:
            pr["state"] = "no"
        self.sfx.play("tick", 0.6)
        return True
    if p.seat is not None and p.kart is None:
        rec = self._jp_rec_of_seat()
        rest = self._jp_rest()
        if rec is not None and rest is not None:
            if key == "r" and rec["kind"] in ("table", "ramen"):
                rest.player_ring(rec)
                return True
            if key == "e" and rec["kind"] == "belt" and self.activity is None:
                rest.player_take_plate()
                return True
    if key in ("m", "n") and self.inv.n("phone") > 0 and not (key == "n" and pr is not None):
        if key == "m":
            self.phone.toggle()
        else:
            self.phone.next()
        return True
    return _orig_v3_input(self, key)


def _hint_v4(self) -> str:
    base = _orig_v3_hint(self)
    if self.ui is not None:
        return base
    out = [base] if base else []
    pr = self.jp_prompt
    if pr is not None and pr["state"] == "open":
        out.append(f"{pr['who'].name.upper()} asks if you're ready:   [Y] Yes, order   [N] Not yet   (R rings the bell)")
    elif self.player.seat is not None and self.player.kart is None:
        rec = self._jp_rec_of_seat()
        if rec is not None:
            out.append("[R] ring the bell" if rec["kind"] in ("table", "ramen") else "[E] take a plate off the belt   (move to stand)")
    if self.inv.n("phone") > 0 and self.phone.cur is not None:
        out.append(f"phone: {self.phone.title()}  {'(paused)' if self.phone.paused else ''}  [M] play/pause  [N] next  [X] close")
    return "   |   ".join(out)


_bridge_update_v3 = AudioBridge.update


def _bridge_update_v4(self, *a, **k):
    g = getattr(self, "game", None)
    if g is not None and g.phone.active:
        self.duck_target = 0.0                         # the phone is playing: the orchestra steps aside (SFX stay)
    return _bridge_update_v3(self, *a, **k)


def _v4_install() -> None:
    global _orig_v3_init, _orig_update_v3, _orig_v3_input, _orig_v3_hint, _orig_use_held
    for n in ("_jp_ell", "_jp_stool", "_jp_shoji", "_jp_wave_poster", "_jp_building_shell", "_jp_door", "_jp_ramen", "_jp_belt", "_jp_tables",
              "_jp_decor", "_build_japan"):
        setattr(ChunkBuilder, n, getattr(V4Builders, n))
    for nm in ("_litter", "_features", "_v3_features", "_npcs"):
        def mk_skip(orig):
            def w(self, *a, **k):
                if self.ch.kind in ("japan", "jpcourt"):
                    return None
                return orig(self, *a, **k)
            return w
        setattr(ChunkBuilder, nm, mk_skip(getattr(ChunkBuilder, nm)))
    _on = ChunkBuilder._v3_npcs
    def _v3npcs(self):
        if self.ch.kind == "jpcourt":     # NEW: forecourt chunks get no random mall NPCs
            return
        if self.ch.kind == "japan":       # unchanged: the restaurant spawns its staff and diners
            _japan_npcs(self)
            return
        _on(self)                         # unchanged: every other chunk gets the normal NPCs
    ChunkBuilder._v3_npcs = _v3npcs
    Figure.arms = _arms
    SfxBus._cb = _make_cb(SfxBus._cb)
    SfxBus.set_phone, SfxBus.phone_stop, SfxBus.phone_pause = _sfx_set_phone, _sfx_phone_stop, _sfx_phone_pause
    SfxBus.phone_active = property(_sfx_phone_active)
    AudioBridge.update = _bridge_update_v4
    _orig_v3_init, _orig_update_v3, _orig_v3_input = Game._v3_init, Game._update_v3, Game._v3_input
    _orig_v3_hint, _orig_use_held = Game._v3_hint, Game.use_held_or_served
    Game._v3_init, Game._update_v3, Game._v3_input = _v4_init, _v4_update, _input_v4
    Game._v3_hint, Game.use_held_or_served = _hint_v4, _use_held_v4
    Game._jp_rest, Game._jp_rec_of_seat, Game.summon_kart = _jp_rest, _jp_rec_of_seat, summon_kart
    Game._jp_start_eat, Game._jp_finish_eat = _jp_start_eat, _jp_finish_eat
    # listening booth: D = download the highlighted track to the phone
    _ki_init, _ki_key, _ki_close = KioskUI.__init__, KioskUI.key, KioskUI.close

    def ki_init(self, g, seat):
        _ki_init(self, g, seat)
        self.text("D  download highlighted track to your phone", (-0.56, -0.40), 0.7, C(1.0, 0.85, 0.5, 0.95))
        g.phone.hold(True)

    def ki_key(self, k):
        if k == "d":
            g, n = self.g, len(self.tids)
            if n:
                tid = self.tids[self.sel]
                if not self.is_open(tid):
                    self.msg = "locked - insert a cassette (I) first"
                elif g.inv.n("phone") <= 0:
                    self.msg = "you need a phone to download (the Backrooms cafe sells one)"
                elif tid in g.inv.downloads:
                    self.msg = "already on your phone"
                else:
                    g.inv.downloads.append(tid)
                    g.inv.dirty = True
                    g.sfx.play("chime", 0.7)
                    self.msg = "downloaded to your phone  (M plays, N skips)"
                self.refresh()
            return
        _ki_key(self, k)

    def ki_close(self):
        _ki_close(self)
        self.g.phone.hold(False)
    KioskUI.__init__, KioskUI.key, KioskUI.close = ki_init, ki_key, ki_close


_v4_install()


# ============================================================================
# v5 EXPANSION  -  Sakura-ya kitchen pipeline, kart upgrades & Eco-Kart, audio jack, CCTV rework, phone GPS
# (bolted on the classes above; see _v5_install at the end of this block)
# ============================================================================
PALETTES.update({
    "jp_chef_m": dict(coat=(0.40, 0.08, 0.10), legs=(0.08, 0.08, 0.12), skin=(0.90, 0.76, 0.64), em=0.05),
    "jp_hiro": dict(coat=(0.20, 0.50, 0.30), legs=(0.12, 0.14, 0.12), skin=(0.90, 0.78, 0.66), em=0.06),
    "jp_cleaner": dict(coat=(0.25, 0.55, 0.60), legs=(0.10, 0.12, 0.16), skin=(0.92, 0.80, 0.70), em=0.06),
})
BADGES.update({"dishes": ("Clean Plate Club", "Washed dishes at Sakura-ya for tips."), "ecokart": ("Zero Emissions", "Owned a battery-powered Eco-Kart."),
               "flex": ("Flex", "Earned a kart-value bonus by storing it 2000 m from its origin.")})
ITEMS["phone"]["hold"] = True
ITEMS["phone"]["desc"] = "P: map & order, kart garage (lock/select), music, GPS teleport. K summons your selected kart, Z mutes mall ambience, Q scans a kart (hold it with H)."
ITEMS["jack"] = dict(name="Audio Jack Wire", kind="tool", thirst=0, hunger=0, fx=None, hold=True, col=(30, 30, 36),
                     desc="Hold it (H), then T beside a Listening Post and T beside your Eco-Kart (10 m max). Press J inside the Eco-Kart.")


def money(n) -> str:
    return f"${int(n):,}"


# ---- sound effects ---------------------------------------------------------------------------------------------------
def _v5_synth(name: str, sr: int, seed: int = 0):
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    t = lambda d: np.arange(int(d * sr)) / sr
    if name == "dish_water":                     # tap running into a basin: smooth band noise + soft droplets
        x = t(2.2)
        m = _bp(rng.standard_normal(x.size), 1200, 5200, sr) * (0.55 + 0.25 * np.sin(2 * np.pi * 3.1 * x + 1.0) + 0.1 * np.sin(2 * np.pi * 7.3 * x))
        m += _lp(rng.standard_normal(x.size), 400, sr, 2) * 0.25
        for _ in range(14):
            t0 = rng.uniform(0.1, 2.0)
            f = rng.uniform(900, 2200)
            xs = np.clip(x - t0, 0, None)
            m += np.sin(2 * np.pi * np.cumsum(f * (1 + 1.6 * np.exp(-xs * 35))) / sr) * np.exp(-xs * 38) * 0.16 * (x >= t0)
        return _st(m * np.minimum(1, np.minimum(x / 0.25, (2.2 - x) / 0.4)) * 0.30)
    if name == "dish_scrub":                     # sponge on porcelain: rhythmic squeaky friction + bubbles
        x = t(1.8)
        fr = _bp(rng.standard_normal(x.size), 2400, 7800, sr) * (0.5 + 0.5 * np.sin(2 * np.pi * 4.2 * x)) ** 2
        sq = np.sin(2 * np.pi * np.cumsum(2900 + 700 * np.sin(2 * np.pi * 4.2 * x)) / sr) * (0.5 + 0.5 * np.sin(2 * np.pi * 4.2 * x)) ** 6 * 0.12
        m = fr * 0.5 + sq
        for _ in range(18):
            t0 = rng.uniform(0, 1.7)
            xs = np.clip(x - t0, 0, None)
            m += np.sin(2 * np.pi * rng.uniform(2500, 4200) * xs) * np.exp(-xs * 90) * 0.12 * (x >= t0)
        return _st(m * np.minimum(1, np.minimum(x / 0.1, (1.8 - x) / 0.3)) * 0.32)
    if name == "dish_clink":                     # ceramic ringing
        x = t(0.7)
        m = sum(np.sin(2 * np.pi * f * x) * np.exp(-x * d) * a for f, d, a in ((2480, 22, 0.5), (3720, 30, 0.35), (5610, 45, 0.2), (1320, 14, 0.25)))
        m += rng.standard_normal(x.size) * np.exp(-x * 500) * 0.2
        return _st(m * 0.55)
    if name == "dish_stack":                     # a stack set down: several quick clacks and a low ring
        x = t(0.9)
        m = np.zeros(x.size)
        for k, t0 in enumerate((0.0, 0.07, 0.15, 0.22)):
            xs = np.clip(x - t0, 0, None)
            f = 2200 + 400 * k
            m += (np.sin(2 * np.pi * f * xs) * np.exp(-xs * 38) * 0.4 + rng.standard_normal(x.size) * np.exp(-xs * 500) * 0.25) * (x >= t0)
        m += np.sin(2 * np.pi * 180 * x) * np.exp(-x * 14) * 0.35
        return _st(_lp(m, 8000, sr, 2) * 0.6)
    if name == "tray_set":
        x = t(0.5)
        m = np.sin(2 * np.pi * 150 * x) * np.exp(-x * 20) * 0.7 + _lp(rng.standard_normal(x.size), 1500, sr, 2) * np.exp(-x * 40) * 0.5
        m += np.sin(2 * np.pi * 2600 * x) * np.exp(-x * 30) * 0.15
        return _st(m * 0.7)
    if name == "gochisou":
        m = _voice(sr, [("o", ""), ("i", "sh"), ("o", ""), ("u", ""), ("a", ""), ("a", "m")], 165.0, 0.14, 0.10, 21)
        return _st(m * 0.55)
    if name == "jack_plug":
        x = t(0.35)
        m = np.sin(2 * np.pi * 900 * x) * np.exp(-x * 40) * 0.5 + _lp(rng.standard_normal(x.size), 3500, sr, 2) * np.exp(-x * 90) * 0.6
        return _st(m * 0.7)
    if name == "gps_warp":
        x = t(1.1)
        f = 180 + 1600 * (x / 1.1) ** 2
        m = np.sin(np.cumsum(2 * np.pi * f / sr)) * np.sin(np.pi * x / 1.1) ** 1.5 * 0.4 + _bp(rng.standard_normal(x.size), 1500, 6000, sr) * np.sin(np.pi * x / 1.1) * 0.25
        return _st(m * 0.7)
    if name == "overheat":
        x = t(0.9)
        m = (np.sin(2 * np.pi * 880 * x) * ((x % 0.3) < 0.14)) * 0.35
        return _st(m)
    if name == "cam_beep":
        x = t(0.2)
        return _st(np.sin(2 * np.pi * 1760 * x) * np.exp(-x * 20) * 0.25)
    return None


_synth_sfx_v4 = synth_sfx


def synth_sfx(name: str, sr: int, seed: int = 0) -> np.ndarray:      # noqa: F811  (adds the v5 sounds)
    a = _v5_synth(name, sr, seed)
    return a if a is not None else _synth_sfx_v4(name, sr, seed)


# ---- Sakura-ya washing area, plate cabinets, numbered trays ---------------------------------------------------------------
class V5Builders:
    def _jp_wash(self):
        ch, z0, xo = self.ch, self.ch.z0, self.ch.xoff
        L, V = JP_L, ch.extra["jp_vis"]
        self.par = ch.detail
        steel, dark = C(0.78, 0.80, 0.84), C(0.12, 0.12, 0.14)
        # main sink counter (west wall) with basin, faucet and soap
        sx, sz = L["wash_sink"]
        self.block(sx, 0.48, z0 + sz, 0.9, 0.96, 2.4, "metal", color=steel)
        self.box(sx, 0.97, z0 + sz, 0.8, 0.03, 2.3, "metal", color=C(0.88, 0.9, 0.93))
        self.box(sx + 0.02, 0.93, z0 + sz - 0.2, 0.62, 0.06, 1.0, color=C(0.35, 0.38, 0.42))
        self.box(sx - 0.25, 1.22, z0 + sz - 0.2, 0.05, 0.5, 0.05, "metal", color=C(0.9, 0.92, 0.95))
        self.box(sx - 0.1, 1.45, z0 + sz - 0.2, 0.3, 0.05, 0.05, "metal", color=C(0.9, 0.92, 0.95))
        self.quad(-11.87, 2.4, z0 + sz, 1.6, 0.5, "+x", jp_sign("洗い場", "WASHING AREA", (200, 240, 255), (10, 40, 52), 384, 96), emissive=1.1, double=False)
        # silver table: dirty plates are dropped here by the waiters; the player can wash from it too
        tx, tz = L["silver"]
        self.block(tx, 0.45, z0 + tz, 0.9, 0.9, 1.6, "metal", color=C(0.86, 0.88, 0.92))
        self.box(tx, 0.915, z0 + tz, 0.96, 0.03, 1.66, "metal", color=C(0.93, 0.95, 0.98), emissive=0.12)
        self.quad(tx + 0.46, 0.55, z0 + tz, 0.9, 0.18, "+x", jp_sign("銀のテーブル", "DIRTY DISHES", (230, 240, 250), (40, 40, 52), 256, 64), emissive=0.6, double=False)
        # player's wash basin
        bx, bz = tx, 1.3
        self.block(bx, 0.46, z0 + bz, 0.9, 0.92, 0.9, "metal", color=steel)
        self.box(bx, 0.93, z0 + bz, 0.7, 0.05, 0.7, color=C(0.3, 0.45, 0.55), emissive=0.2)
        self.box(bx, 1.2, z0 + bz - 0.3, 0.05, 0.5, 0.05, "metal", color=C(0.9, 0.92, 0.95))
        self.box(bx, 1.42, z0 + bz - 0.2, 0.05, 0.05, 0.3, "metal", color=C(0.9, 0.92, 0.95))
        self.quad(tx + 0.46, 1.5, z0 + 2.2, 1.6, 0.4, "+x", jp_sign("皿洗い  E", "WASH A DISH  -  EARN $1-4", (255, 240, 200), (60, 20, 20), 384, 64), emissive=1.0, double=False)
        # drying rack
        rx, rz = L["rack"]
        self.block(rx, 0.5, z0 + rz, 0.9, 1.0, 0.9, "metal", color=C(0.6, 0.62, 0.66))
        for dz in (-0.3, 0.0, 0.3):
            self.box(rx, 1.15, z0 + rz + dz, 0.8, 0.02, 0.03, "metal", color=C(0.9, 0.9, 0.92))
        self.quad(rx - 0.0, 1.9, z0 + 4.76, 0.9, 0.3, "-z", jp_sign("乾燥棚", "DRYING RACK", (255, 255, 255), (20, 60, 80), 256, 64), emissive=0.9, double=False)
        # visual plate pools (enabled by count)
        self.par = ch.detail
        V["silver_pl"] = [self.box(tx + (i % 3 - 1) * 0.26, 0.96 + 0.035 * (i // 3), z0 + tz - 0.45 + 0.45 * ((i % 3)), 0.24, 0.03, 0.24, color=C(0.82, 0.8, 0.72)) for i in range(JP_SILVER_CAP)]
        V["rack_pl"] = [self.box(rx - 0.35 + 0.07 * (i % 11), 1.3, z0 + rz - 0.15 + 0.3 * (i // 11), 0.02, 0.26, 0.24, color=C(0.97, 0.97, 0.95), emissive=0.2) for i in range(JP_RACK_CAP)]
        for k in ("silver_pl", "rack_pl"):
            for e in V[k]:
                e.enabled = False
        # plate cabinets: left wing (on the back counter) and right wing (beside the belt chef)
        cx, cz = L["cab_L"]
        self.box(cx, 1.5, z0 + cz, 0.55, 1.0, 0.85, "wood", color=C(0.38, 0.20, 0.12))
        self.box(cx + 0.29, 1.5, z0 + cz, 0.02, 0.9, 0.78, "wood", color=C(0.55, 0.32, 0.18))
        V["cabL_lamp"] = self.ball(cx + 0.3, 2.05, z0 + cz, 0.05, color=C(0.2, 1.0, 0.3), emissive=1.5)
        self.quad(cx + 0.31, 1.75, z0 + cz, 0.7, 0.18, "+x", jp_sign("左 皿", "LEFT WING PLATES", (255, 240, 200), (50, 20, 10), 256, 64), emissive=0.8, double=False)
        V["cabL_pl"] = [self.box(cx + 0.31, 1.05 + 0.042 * i, z0 + cz, 0.05, 0.02, 0.3, color=C(0.97, 0.97, 0.95), emissive=0.2) for i in range(22)]
        rcx, rcz = L["cab_R"]
        self.block(rcx, 0.9, z0 + rcz, 0.9, 1.8, 0.9, "wood", color=C(0.38, 0.20, 0.12))
        self.box(rcx - 0.46, 1.2, z0 + rcz, 0.02, 1.2, 0.8, "wood", color=C(0.55, 0.32, 0.18))
        V["cabR_lamp"] = self.ball(rcx - 0.47, 1.9, z0 + rcz, 0.06, color=C(0.2, 1.0, 0.3), emissive=1.5)
        self.quad(rcx - 0.47, 1.6, z0 + rcz, 0.7, 0.18, "-x", jp_sign("右 皿", "RIGHT WING PLATES", (255, 240, 200), (50, 20, 10), 256, 64), emissive=0.8, double=False)
        V["cabR_pl"] = [self.box(rcx - 0.47, 0.35 + 0.07 * i, z0 + rcz, 0.05, 0.03, 0.34, color=C(0.97, 0.97, 0.95), emissive=0.2) for i in range(14)]
        for k in ("cabL_pl", "cabR_pl"):
            for e in V[k]:
                e.enabled = False
        # numbered trays on the pass counter, each with a "complete" lamp
        V["trays"] = {}
        for tid in range(6):
            px, pz = jp_tray_slot(tid)
            self.box(px, 1.115, z0 + pz, 0.40, 0.03, 0.52, "wood", color=C(0.10, 0.06, 0.05))
            self.quad(px, 1.16, z0 + pz - 0.27, 0.22, 0.08, "-z", BANK.sign(f"TABLE {tid + 1}", (255, 240, 200), (60, 20, 14), 192, 56), emissive=0.9, double=False)
            V["trays"][tid] = self.ball(px - 0.15, 1.17, z0 + pz + 0.22, 0.035, color=C(0.15, 0.10, 0.10), emissive=0.0)
        # interactable: wash a dish for tips
        wx, wz = L["wash_spot"]
        self.inter("jpwash", wx, 0.0, z0 + wz, 2.0, "Wash a dirty dish from the silver table  (earns $1-4)", 0.0, {})

    def _v5_cctv(self):
        """Physical security cameras (exactly two per room) and 'Smile, you're on camera' posters in every room the security desk can see."""
        ch, z0, xo, hw = self.ch, self.ch.z0, self.ch.xoff, self.ch.hw
        self.par = ch.detail
        leds = []
        for (cx, cy, cz, yaw, pitch, tag) in cam_poses(xo, hw, ch.h, z0):
            lx = cx - xo
            sgn = -1.0 if lx < 0 else 1.0
            self.box(sgn * (hw - 0.14), cy, cz, 0.28, 0.06, 0.12, "metal", color=C(0.2, 0.2, 0.22))             # wall plate
            self.box(lx, cy + 0.02, cz, 0.12, 0.05, 0.12, "metal", color=C(0.25, 0.26, 0.3))                      # arm
            self.box(lx, cy - 0.06, cz + 0.12 * (1 if yaw == 0 else -1), 0.20, 0.17, 0.42, color=C(0.82, 0.84, 0.88), rot=(pitch, yaw, 0))
            self.ball(lx, cy - 0.07, cz + 0.36 * (1 if yaw == 0 else -1), 0.07, color=C(0.02, 0.02, 0.04), emissive=0.3)
            leds.append(self.ball(lx + 0.09, cy + 0.03, cz + 0.2 * (1 if yaw == 0 else -1), 0.025, color=C(1.0, 0.1, 0.1), emissive=2.0))
        t0 = [0.0]

        def blink(dt, g, leds=leds, t0=t0):
            t0[0] += dt
            on = (t0[0] % 1.4) < 0.18
            for e in leds:
                e.color = C(1.0, 0.1, 0.1) if on else C(0.18, 0.02, 0.02)
        ch.tickers.append(blink)
        rng = seeded(self.seed, "cctvposter", ch.idx)
        tex = _cctv_poster()
        for _ in range(rng.choice((0, 1, 1, 2))):
            side = rng.choice((-1, 1))
            zz = rng.uniform(3.0, CH - 3.0)
            wall = xo + side * hw
            blocked = any(c[4] > 1.2 and c[2] - 0.7 < z0 + zz < c[3] + 0.7 and ((c[1] > wall - 1.1) if side > 0 else (c[0] < wall + 1.1)) for c in ch.colliders)
            if blocked or ch.kind == "japan" and abs(zz - 12) < 11:
                continue
            self.quad(side * (hw - 0.07), 2.15, z0 + zz, 0.8, 1.05, "-x" if side > 0 else "+x", tex, emissive=0.5, double=False)


def cam_poses(xo: float, hw: float, h: float, z0: float) -> list:
    """The two security cameras of one room: a (+z)-looking one by the left wall and a (-z)-looking one by the right wall."""
    y = min(h, 6.0) - 0.8
    return [(xo - (hw - 0.7), y, z0 + 1.5, 0.0, 14.0, "+Z"), (xo + (hw - 0.7), y, z0 + CH - 1.5, 180.0, 14.0, "-Z")]


def cctv_covered(seed: int, idx: int) -> bool:
    """True when a security room lies within 5 chunks of this one (10 chunks centred on the room, the room itself excluded)."""
    return chunk_kind(seed, idx) != "security" and any(chunk_kind(seed, j) == "security" for j in range(idx - 5, idx + 6) if j != idx)


def _cctv_poster():
    key = "cctv_poster"
    if key not in BANK._tex:
        im = Image.new("RGBA", (256, 336), (250, 214, 40, 255))
        d = ImageDraw.Draw(im)
        d.rectangle([6, 6, 249, 329], outline=(20, 20, 20, 255), width=6)
        d.rectangle([58, 70, 198, 150], fill=(25, 25, 30, 255))
        d.rectangle([84, 54, 130, 70], fill=(25, 25, 30, 255))
        d.ellipse([100, 82, 156, 138], fill=(90, 130, 170, 255), outline=(240, 240, 240, 255), width=5)
        d.ellipse([118, 100, 138, 120], fill=(15, 20, 30, 255))
        d.ellipse([174, 78, 188, 92], fill=(230, 40, 40, 255))
        d.arc([88, 176, 168, 236], 20, 160, fill=(20, 20, 20, 255), width=7)
        d.ellipse([100, 168, 112, 182], fill=(20, 20, 20, 255))
        d.ellipse([144, 168, 156, 182], fill=(20, 20, 20, 255))
        f1, sz2 = _font(42), 23
        while d.textlength("YOU'RE ON CAMERA", font=_font(sz2)) > 226 and sz2 > 10:
            sz2 -= 1
        f2 = _font(sz2)
        d.text((128 - d.textlength("SMILE!", font=f1) / 2, 246), "SMILE!", fill=(20, 20, 20, 255), font=f1)
        d.text((128 - d.textlength("YOU'RE ON CAMERA", font=f2) / 2, 298), "YOU'RE ON CAMERA", fill=(20, 20, 20, 255), font=f2)
        BANK._tex[key] = BANK.to_texture(im)
    return BANK._tex[key]


# ---- kart records: plates, neon, solar, headlight, seat, eco ----------------------------------------------------------------
PLATE_TIERS = (("Silver", 1000), ("Gold", 2000), ("Diamond", 6000), ("Platinum", 7800), ("Obsidian", 10140))     # +30% per tier after Diamond
PLATE_BASIC, PLATE_CUSTOM, NEON_PRICE = 300, 4000, 500
NEON_COLORS = {"Cyan": (0.10, 1.0, 1.0), "Magenta": (1.0, 0.10, 0.85), "Lime": (0.5, 1.0, 0.1), "Amber": (1.0, 0.7, 0.1), "Violet": (0.6, 0.25, 1.0),
               "Red": (1.0, 0.12, 0.12), "Electric Blue": (0.15, 0.4, 1.0), "White": (1.0, 1.0, 1.0), "Hot Pink": (1.0, 0.3, 0.6), "Mint": (0.2, 1.0, 0.7)}
SOLAR_BASE, SOLAR_UP, SOLAR_MAX = 5000, 1200, 5            # level 1 = module (+1 fuel/s); each upgrade +1/s; level 5 = 5 fuel/s
LIGHT_PRICE, SEAT_PRICE, ECO_UP, ECO_DOWN, JACK_PRICE = 900, 4000, 8750, 4520, 150
ECO_CAP, ECO_ACC, ECO_COAST, ECO_HEAT_S = 25.75, 11.0, 0.0195, 25.0    # ~55 mph cruise, 55->45 mph coast takes ~10 s longer than the standard kart
BONUS_Z, BONUS_FLAT = 2000.0, 500
MPS_MPH = 2.2369
PLATE_FRAME = {0: ((190, 195, 205), (40, 44, 52)), 1: ((235, 190, 60), (50, 34, 8)), 2: ((150, 225, 250), (10, 40, 70)),
               3: ((240, 245, 255), (60, 40, 110)), 4: ((30, 20, 44), (200, 120, 255))}


def new_kart_rec() -> dict:
    return dict(id="%08x" % random.getrandbits(32), plate_tier=None, plate_text=None, plate_custom=False, neon=None, neon_on=True, solar=0,
                light=False, light_on=True, seat=False, eco=False, miles=0.0, origin=None, earned=0, locked=False, paint=None, fuel=30.0, lib=[], pos=None)


def random_plate() -> str:
    ch = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return "".join(random.choice(ch) for _ in range(3)) + "-" + "".join(random.choice(ch) for _ in range(3))


def kart_value(r: dict) -> int:
    v = 2000
    if r.get("plate_tier") is not None:
        v += int(PLATE_TIERS[r["plate_tier"]][1] * 0.8)
    if r.get("plate_text"):
        v += int((PLATE_CUSTOM if r.get("plate_custom") else PLATE_BASIC) * 0.8)
    if r.get("neon"):
        v += int(NEON_PRICE * 0.8)
    if r.get("solar"):
        v += int((SOLAR_BASE + SOLAR_UP * (r["solar"] - 1)) * 0.7)
    if r.get("light"):
        v += int(LIGHT_PRICE * 0.8)
    if r.get("seat"):
        v += int(SEAT_PRICE * 0.7)
    if r.get("eco"):
        v += int(ECO_UP * 0.8)
    return v + int(r.get("earned", 0))


def regen_rate(r: dict) -> float:
    return float(r.get("solar", 0))                         # +1 fuel/s per solar level (max 5)


def all_upgrades(r: dict) -> bool:
    return (r.get("plate_tier") is not None and r.get("plate_custom") and r.get("neon") and r.get("solar", 0) >= SOLAR_MAX
            and r.get("light") and r.get("seat"))


def plate_label(r: dict) -> str:
    if r.get("plate_tier") is None:
        return "NO PLATE"
    return f"{r.get('plate_text') or '-- --'}  [{PLATE_TIERS[r['plate_tier']][0]}]"


def plate_texture(r: dict):
    tier, text = r.get("plate_tier") or 0, (r.get("plate_text") or "").upper()
    key = f"plate:{tier}:{text}"
    if key not in BANK._tex:
        fg, bg = PLATE_FRAME[tier]
        W, H = 440, 220
        im = Image.new("RGBA", (W, H), fg + (255,))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([10, 10, W - 10, H - 10], radius=22, fill=bg + (255,), outline=(255, 255, 255, 255), width=4)
        if tier == 2:
            for k in range(14):
                x0 = 30 + (k * 53) % (W - 60)
                y0 = 24 + (k * 37) % (H - 60)
                d.polygon([(x0, y0 - 8), (x0 + 6, y0), (x0, y0 + 8), (x0 - 6, y0)], fill=(255, 255, 255, 140))
        if text:
            size, f = 120, None
            while size > 14:                                   # auto-size so even 11 characters fit the plate exactly
                f = _font(size)
                if d.textlength(text, font=f) <= W - 70:
                    break
                size -= 3
            tw = d.textlength(text, font=f)
            d.text(((W - tw) / 2, (H - size) / 2 - size * 0.12), text, fill=(255, 255, 255, 255) if tier != 4 else fg + (255,), font=f)
        else:
            d.text((W / 2 - 80, H / 2 - 24), "-- --", fill=(255, 255, 255, 200), font=_font(60))
        BANK._tex[key] = BANK.to_texture(im)
    return BANK._tex[key]


def kart_export(k) -> dict:
    r = dict(k.rec)
    r["paint"] = list(getattr(k, "paint", None) or []) or None
    r["fuel"] = float(k.fuel)
    r["lib"] = list(r.get("lib", []))
    r["pos"] = (float(k.x), float(k.z), float(k.yaw), getattr(k, "level", "mall"))
    return r


def kart_from_rec(r: dict, key: str, x: float, z: float, yaw: float, level: str):
    k = GoKart(key, x, z, yaw, fuel=float(r.get("fuel", 30.0)))
    k.level = level
    k.rec = dict(new_kart_rec(), **{a: b for a, b in r.items() if a in new_kart_rec()})
    k.rec["lib"] = list(k.rec.get("lib", []))
    if r.get("paint"):
        k.paint = tuple(r["paint"])
    return k


def reset_origin(k) -> None:
    k.rec["origin"] = float(k.z)


def store_bonus(k) -> int:
    """Value bonus when a kart is stored (K): needs the kart to be >= 2000 m (z+ or z-) from its origin; then the origin moves to here."""
    o = k.rec.get("origin")
    d = abs(k.z - (k.z if o is None else o))
    bonus = 0
    if d >= BONUS_Z:
        bonus = BONUS_FLAT + int((d - BONUS_Z) / 1609.344 * 200)
        k.rec["earned"] = int(k.rec.get("earned", 0)) + bonus
    reset_origin(k)
    return bonus


# ---- the GoKart class: extra state, visuals and new physics (spliced into GoKart by _v5_install) ------------------------------
def _gk_init_extra(self) -> None:
    self.rec = new_kart_rec()
    self.rec["fuel"] = self.fuel
    self.heat, self.overheated, self.passengers, self.glow, self.dress_root, self.dress_sig = 0.0, False, [], [], None, None
    self.rec["origin"] = float(self.z)


def _gk_engine_vol(self) -> float:
    if self.rec["eco"]:
        return 0.0 if self.overheated else 0.32
    return 0.6 if self.fuel > 0 else 0.0


def _gk_dress(self) -> None:
    """(Re)build everything the craftsman adds: plate, neon, solar panel, headlight lens, seat pad, eco battery pack."""
    if self.root is None:
        return
    r = self.rec
    sig = (r["plate_tier"], r["plate_text"], r["neon"], r["neon_on"], r["solar"], r["light"], r["light_on"], r["seat"], r["eco"])
    if sig == self.dress_sig and self.dress_root is not None:
        return
    self.dress_sig = sig
    if self.dress_root is not None:
        ur.destroy(self.dress_root)
    root = self.dress_root = ur.Entity(parent=self.root)

    def part(model, pos, sc, col, em=0.0, rot=(0, 0, 0), tex=None):
        e = mk(root, model, pos, sc, tex, color=col, emissive=em, lm=0.0, rot=rot, double=(model == "quad"))
        return e
    neon = NEON_COLORS.get(r["neon"]) if (r["neon"] and r["neon_on"]) else None
    for e in self.glow:                                         # underglow follows the neon colour
        e.color = C(*(neon or (0.10, 0.95, 1.0)))
    if r["plate_tier"] is not None:
        part("quad", (0, 0.36, -1.435), (0.62, 0.31, 1), C(1, 1, 1), 0.35, (0, 180, 0), plate_texture(r))
        if neon:
            part("cube", (0, 0.36, -1.42), (0.70, 0.38, 0.02), C(*neon), 1.8)
            part("cube", (0, 0.36, -1.425), (0.66, 0.34, 0.02), C(0.02, 0.02, 0.03), 0.0)
    if neon:
        for s in (-1, 1):
            part("cube", (s * 0.67, 0.31, -0.05), (0.02, 0.05, 0.9), C(*neon), 1.8)
    if r["solar"]:
        part("cube", (0, 0.745, -1.18), (0.70, 0.025, 0.36), C(0.05, 0.10, 0.28), 0.3, (-12, 0, 0))
        for i in range(min(5, r["solar"])):
            part("cube", (-0.28 + 0.14 * i, 0.76, -1.18), (0.11, 0.012, 0.30), C(0.2, 0.45, 0.95), 0.9, (-12, 0, 0))
    if r["light"]:
        for s in (-1, 1):
            part("cube", (s * 0.29, 0.345, 1.52), (0.16, 0.08, 0.03), C(1.0, 1.0, 0.9), 2.6 if r["light_on"] else 0.4)
    if r["seat"]:
        part("cube", (0, 1.0, -1.08), (0.52, 0.06, 0.30), C(0.08, 0.08, 0.1), 0.0)
        part("cube", (0, 1.035, -1.08), (0.50, 0.012, 0.08), C(0.9, 0.15, 0.15), 0.3)
    if r["eco"]:
        part("cube", (0, 0.66, -0.92), (0.50, 0.14, 0.44), C(0.08, 0.50, 0.30), 0.6)
        part("cube", (0, 0.74, -0.92), (0.42, 0.02, 0.34), C(0.2, 1.0, 0.6), 1.4)
        for s in (-1, 1):
            part("cube", (s * 0.14, 0.34, -1.43), (0.08, 0.08, 0.04), C(0.2, 1.0, 0.7), 2.0)
            part("cube", (s * 0.68, 0.42, 0.28), (0.02, 0.04, 0.5), C(0.2, 1.0, 0.7), 1.4)


def _gk_seat_world(self) -> tuple:
    s, c = math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw))
    return self.x - s * 1.08, self.y + 1.0, self.z - c * 1.08


GK_DRIVE_SRC = '''
    def drive(self, dt: float, w: "World", hk, boost_mult: float = 1.0) -> dict:
        eco = self.rec["eco"]
        fwd = hk["w"] + hk["up arrow"] - hk["s"] - hk["down arrow"]
        steer = hk["d"] + hk["right arrow"] - hk["a"] - hk["left arrow"]
        steer = max(-1, min(1, steer))
        want_drift = bool(hk["space"])
        throttle = fwd > 0
        engine = (not self.overheated) if eco else self.fuel > 0.0
        top = ECO_CAP if eco else self.MAXV * boost_mult
        yr = math.radians(self.yaw)
        vf0 = self.vx * math.sin(yr) + self.vz * math.cos(yr)
        sf = min(1.0, abs(vf0) / 4.0)
        drifting = want_drift and abs(vf0) > 6.0 and steer != 0
        rate = 88.0 * (1.0 - 0.35 * min(1.0, abs(vf0) / top)) * (1.35 if drifting else 1.0)
        self.yaw += steer * rate * sf * (1.0 if vf0 >= 0 else -1.0) * dt
        self.steer_vis += (steer - self.steer_vis) * min(1.0, 10 * dt)
        yr = math.radians(self.yaw)
        hx, hz, rx, rz = math.sin(yr), math.cos(yr), math.cos(yr), -math.sin(yr)
        vf = self.vx * hx + self.vz * hz
        vl = self.vx * rx + self.vz * rz
        burn = 0.12
        if engine and throttle:
            vf += (ECO_ACC if eco else self.ACC) * dt * max(0.0, 1.0 - vf / top)
            burn = 1.7
        elif fwd < 0:
            vf = vf - 22.0 * dt if vf > 0.5 else max(-4.0, vf - 7.0 * dt) if engine else vf - 8.0 * dt
        if eco:
            vf -= vf * ECO_COAST * dt                            # momentum retention: almost no coasting drag
            if self.overheated:
                vf -= vf * 0.30 * dt                             # overheated: the motor cuts out and the kart slows down
            sp01 = min(1.0, abs(vf) / ECO_CAP)
            if throttle and not self.overheated:
                self.heat = min(1.0, self.heat + dt / ECO_HEAT_S)
                if self.heat >= 1.0:
                    self.overheated = True
                    self.hot_event = True
            elif not (self.overheated and throttle):             # cooling needs the throttle released; faster the faster you roll
                self.heat = max(0.0, self.heat - dt * (0.05 + 0.10 * sp01))
                if self.overheated and self.heat < 0.4:
                    self.overheated = False
        else:
            vf -= vf * 0.12 * dt
            if fwd == 0:
                vf *= math.exp(-0.55 * dt)
        vl *= math.exp(-(1.8 if drifting else 9.0) * dt)
        if drifting:
            self.charge += dt
        else:
            if self.charge > 0.8 and vf > 5:
                vf += min(3.5, self.charge * 2.0)
            self.charge = 0.0
        if not eco:
            self.fuel = max(0.0, self.fuel - burn * dt)
        self.skid = min(1.0, abs(vl) / 3.5) if abs(vf) > 3 else 0.0
        self.vx, self.vz = hx * vf + rx * vl, hz * vf + rz * vl
        nx, nz = self.x + self.vx * dt, self.z + self.vz * dt
        nx, nz, ground, bumped = w.resolve(nx, self.y, nz, self.R, step=0.25)
        if bumped:
            self.vx *= 0.35
            self.vz *= 0.35
        xo, hw = w.xo_at(nz), w.hw_at(nz)
        lim = hw - 0.7
        if abs(nx - xo) > lim:
            nx = xo + math.copysign(lim, nx - xo)
            self.vx *= 0.4
        self.x, self.z = nx, nz
        self.y += (ground - self.y) * min(1.0, 18 * dt)
        self.speed = math.hypot(self.vx, self.vz)
        self.rec["miles"] += self.speed * dt / 1609.344
        for wh in self.wheels:
            wh.rotation_y = self.steer_vis * 20 if wh.z > 0 else 0
        self.sync()
        return dict(drifting=drifting, charge=self.charge, skid=self.skid, rpm=min(1.0, self.speed / top + (0.25 if throttle else 0.0)))
'''


# ---- craftsman v5: tabs browsed with the left/right arrows ----------------------------------------------------------------------
class CraftUI(UiBase):
    TABS = ("PAINT & FUEL", "PLATES", "NEON", "POWER", "LIGHTS & SEATS", "ECO-KART")

    def __init__(self, g, npc) -> None:
        super().__init__(g)
        self.npc, self.sel, self.tab, self.msg, self.typing = npc, 0, 0, "", None
        self.names = list(KART_COLORS)
        self.quad((0, 0), (1.2, 0.74), C(0.9, 0.55, 0.2, 0.35), z=0.21)
        self.quad((0, 0), (1.18, 0.72), C(0.05, 0.04, 0.03, 0.95))
        self.title = self.text("", (-0.55, 0.31), 1.0, C(1.0, 0.85, 0.5, 1))
        self.wallet = self.text("", (-0.55, 0.265), 0.8, C(0.8, 1.0, 0.8, 1))
        self.rows = [self.text("", (-0.55, 0.20 - 0.045 * i), 0.78) for i in range(11)]
        self.foot = self.text("", (-0.55, -0.335), 0.6, C(0.8, 0.75, 0.6, 0.9))
        self.refresh()

    def kart(self):
        g = self.g
        k = g.last_kart
        if k is None or getattr(k, "level", "mall") != g.world.level:
            k = min((k_ for k_ in g.karts.values() if getattr(k_, "level", "mall") == g.world.level), key=lambda k_: math.hypot(k_.x - g.player.x, k_.z - g.player.z), default=None)
        return k

    def tabs(self, r) -> list:
        t = list(self.TABS)
        if not all_upgrades(r):
            t.pop()                                         # the Eco-Kart tab appears only when every upgrade is installed
        return t

    def build(self, r) -> list:
        """Rows: (label, price or None, action key)."""
        npc, t = self.npc, self.tab
        out = []
        if t == 0:
            out = [(f"repaint:  {nm}", npc.paint_price, ("paint", nm)) for nm in self.names] + [("refuel to full" + ("  (solar makes this obsolete)" if r["solar"] else ""), npc.fuel_price, ("fuel", None))]
        elif t == 1:
            for i, (nm, pr) in enumerate(PLATE_TIERS):
                out.append((f"{nm} plate" + ("   [installed]" if r["plate_tier"] == i else ""), pr, ("tier", i)))
            out.append(("characters: basic random  XXX-XXX" + (f"   [{r['plate_text']}]" if r["plate_text"] and not r["plate_custom"] else ""), PLATE_BASIC, ("basic", None)))
            out.append(("characters: custom, up to 11  XXXXXXXXXXX" + (f"   [{r['plate_text']}]" if r["plate_custom"] else ""), PLATE_CUSTOM, ("custom", None)))
        elif t == 2:
            out = [(f"neon:  {nm}" + ("   [installed]" if r["neon"] == nm else ""), NEON_PRICE, ("neon", nm)) for nm in NEON_COLORS]
            out.append((f"neon lights: {'ON' if r['neon_on'] else 'OFF'}  (toggle, free)", None, ("neon_toggle", None)))
        elif t == 3:
            if not r["solar"]:
                out.append(("Solar panel / regeneration module  +1 fuel/s   (replaces fuel canisters)", SOLAR_BASE, ("solar", None)))
            elif r["solar"] < SOLAR_MAX:
                out.append((f"Upgrade regeneration  +{r['solar']} -> +{r['solar'] + 1} fuel/s", SOLAR_UP, ("solar", None)))
            else:
                out.append(("Regeneration maxed:  +5 fuel/s", None, ("none", None)))
        elif t == 4:
            ok = r["solar"] >= SOLAR_MAX and r["plate_custom"] and r["neon"]
            out.append(("Path-finding headlight" + ("   [installed]" if r["light"] else ("" if ok else "   (needs max solar + custom plate + neon)")), LIGHT_PRICE, ("light", None)))
            out.append(("Secondary seating (spoiler seat)" + ("   [installed]" if r["seat"] else ""), SEAT_PRICE, ("seat", None)))
        else:
            if r["eco"]:
                out.append(("Downgrade to the standard kart (keeps the other upgrades)", ECO_DOWN, ("eco_down", None)))
            else:
                out.append(("Upgrade to Eco-Kart: battery, heat bar, long glide, audio jack", ECO_UP, ("eco_up", None)))
            out.append(("Audio jack wire (10 m, booth <-> Eco-Kart)", JACK_PRICE, ("jack", None)))
        return out

    def refresh(self) -> None:
        g = self.g
        k = self.kart()
        r = k.rec if k is not None else new_kart_rec()
        tabs = self.tabs(r)
        self.tab = min(self.tab, len(tabs) - 1)
        self.title.text = f"{self.npc.name.upper()}  -  < {tabs[self.tab]} >   ({self.tab + 1}/{len(tabs)})   left/right arrows: browse"
        self.wallet.text = f"you have {money(g.inv.coins)}   value {money(kart_value(r))}   {self.msg}"
        self.data = self.build(r)
        self.sel = max(0, min(self.sel, len(self.data) - 1))
        top = max(0, min(self.sel - 5, len(self.data) - len(self.rows)))
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= len(self.data):
                row.text = ""
                continue
            lab, pr, _a = self.data[j]
            row.text = f"{'>' if j == self.sel else ' '} {lab:<62} {money(pr) if pr else ''}"
            row.color = C(1.0, 0.95, 0.6, 1) if j == self.sel else C(0.9, 0.85, 0.75, 0.95)
        if self.typing is not None:
            self.rows[0].text = f"type your plate (max 11):  {self.typing}_"
            self.foot.text = "letters & digits   Backspace delete   Enter confirm   Esc cancel"
        else:
            self.foot.text = "W/S choose   Enter buy   E / Esc leave   (lock your kart in the phone garage so upgrades are never lost)" if not (k and k.rec["locked"]) else "W/S choose   Enter buy   E / Esc leave   (kart is locked: saved in cache)"

    def pay(self, price) -> bool:
        g = self.g
        if g.inv.coins < price:
            self.msg = "not enough money"
            return False
        g.inv.coins -= price
        g.inv.dirty = True
        g.sfx.play("coin")
        return True

    def key(self, k: str) -> None:
        g = self.g
        kart = self.kart()
        if self.typing is not None:
            if k in ("enter", "return"):
                if self.typing and kart is not None and self.pay(PLATE_CUSTOM):
                    kart.rec["plate_text"], kart.rec["plate_custom"] = self.typing, True
                    if kart.rec["plate_tier"] is None:
                        kart.rec["plate_tier"] = 0
                    kart.dress()
                    self.msg = f"custom plate {self.typing} installed"
                self.typing = None
            elif k == "escape":
                self.typing = None
            elif k == "backspace":
                self.typing = self.typing[:-1]
            else:
                ch = key_to_char(k)
                if ch and (ch.isalnum()) and len(self.typing) < 11:
                    self.typing += ch.upper()
            self.refresh()
            return
        if k in ("e", "escape", "g"):
            g.close_ui()
            return
        if k == "left arrow" or k == "right arrow":
            r = kart.rec if kart is not None else new_kart_rec()
            n = len(self.tabs(r))
            self.tab = (self.tab + (1 if k == "right arrow" else -1)) % n
            self.sel = 0
            g.sfx.play("tick", 0.5)
        elif k in ("w", "up arrow") and self.data:
            self.sel = (self.sel - 1) % len(self.data)
        elif k in ("s", "down arrow") and self.data:
            self.sel = (self.sel + 1) % len(self.data)
        elif k in ("enter", "return"):
            self.buy(kart)
        self.refresh()

    def buy(self, kart) -> None:
        g = self.g
        lab, price, (act, arg) = self.data[self.sel]
        if act == "none":
            return
        if kart is None or math.hypot(kart.x - g.player.x, kart.z - g.player.z) > 16.0:
            self.msg = "bring your kart closer"
            return
        if kart.passengers and act in ("eco_up", "eco_down"):
            self.msg = "a passenger is seated - ask them to stand first"
            return
        r = kart.rec
        kart.build()
        if act == "neon_toggle":
            r["neon_on"] = not r["neon_on"]
        elif act == "tier":
            if r["plate_tier"] == arg:
                self.msg = "already installed"
            elif self.pay(price):
                r["plate_tier"] = arg
                self.msg = f"{PLATE_TIERS[arg][0]} plate fitted"
        elif act == "basic":
            if self.pay(price):
                r["plate_text"], r["plate_custom"] = random_plate(), False
                if r["plate_tier"] is None:
                    r["plate_tier"] = 0
                self.msg = f"plate {r['plate_text']}"
        elif act == "custom":
            self.typing = ""
            return
        elif act == "neon":
            if r["neon"] == arg:
                self.msg = "already installed"
            elif self.pay(price):
                r["neon"], r["neon_on"] = arg, True
                self.msg = f"{arg} neon"
        elif act == "solar":
            if self.pay(price):
                r["solar"] += 1
                self.msg = f"regeneration +{r['solar']} fuel/s"
        elif act == "light":
            if r["light"]:
                self.msg = "already installed"
            elif not (r["solar"] >= SOLAR_MAX and r["plate_custom"] and r["neon"]):
                self.msg = "max the solar panel, custom plate and neon first"
            elif self.pay(price):
                r["light"] = True
                self.msg = "headlight installed   (H toggles it while driving)"
        elif act == "seat":
            if r["seat"]:
                self.msg = "already installed"
            elif self.pay(price):
                r["seat"] = True
                self.msg = "spoiler seat installed   (B boards a passenger)"
        elif act == "eco_up":
            if self.pay(price):
                r["eco"] = True
                kart.heat, kart.overheated = 0.0, False
                if g.inv.n("jack") <= 0:
                    g.inv.add("jack")
                g.award("ecokart")
                self.msg = "Eco-Kart!  (an audio jack wire is in your bag)"
        elif act == "eco_down":
            if self.pay(price):
                r["eco"] = False
                kart.heat, kart.overheated = 0.0, False
                self.msg = "back to the standard kart"
        elif act == "jack":
            if self.pay(price):
                g.inv.add("jack")
                self.msg = "audio jack wire added to your bag"
        elif act == "paint":
            if self.pay(price):
                kart.recolor(KART_COLORS[arg])
                self.msg = f"{arg} - looks good"
                g.award("paint")
        elif act == "fuel":
            if self.pay(price):
                kart.fuel = 100.0
                self.msg = "tank filled"
        kart.dress()
        g.inv.dirty = True


# ---- phone v5: garage with lock + selected kart, GPS teleport, ambience mute ---------------------------------------------------------
_PhoneUI_v4 = PhoneUI


class PhoneUI(_PhoneUI_v4):      # noqa: F811
    PAGES = ("MAP & ORDER", "GARAGE", "MUSIC", "GPS")

    def __init__(self, g) -> None:
        self.gps_d = 500.0
        super().__init__(g)

    def entries(self) -> list:
        g, p = self.g, self.g.player
        out = [("slot", i, r) for i, r in enumerate(self.stored())]
        near = self.near_karts()
        out += [("near", k, k.rec) for k in near]
        out += [("far", k, k.rec) for k in g.karts.values() if k not in near and k is not p.kart and k.rec.get("locked") and getattr(k, "level", "mall") == g.world.level]
        return out

    def redraw(self) -> None:
        pg = getattr(self, "page", 0)
        if pg == 0:
            super().redraw()
            if self.hint0 is not None:
                self.hint1.text = f"P / E / Esc close    Z: mall ambience {'MUTED' if getattr(self.g, 'ambient_muted', False) else 'on'} (toggle)"
            return
        if pg == 2:
            return super().redraw()
        if getattr(self, "title", None) is not None:
            self.title.text = f"<  {self.PAGES[pg]}  >      A/D page"
        g, p = self.g, self.g.player
        S = 320
        im = Image.new("RGBA", (S, S), (8, 14, 14, 255))
        d = ImageDraw.Draw(im)
        f, f2 = _font(15), _font(12)
        if pg == 1:
            es = self.entries()
            self.sel = max(0, min(self.sel, max(0, len(es) - 1)))
            sel_id = g.inv.flags.get("ksel")
            d.text((10, 8), f"POCKET GARAGE   {len(self.stored())}/3 stored", fill=(255, 220, 120, 255), font=f)
            for i, (kind, ref, r) in enumerate(es[:12]):
                y = 34 + 21 * i
                if i == self.sel:
                    d.rectangle([6, y - 2, S - 6, y + 17], fill=(40, 80, 70, 255))
                col = tuple(int(255 * c) for c in (r.get("paint") or (0.8, 0.12, 0.14)))
                d.rectangle([10, y + 1, 22, y + 14], fill=col + (255,))
                tag = f"S{ref + 1}" if kind == "slot" else ("near" if kind == "near" else "world")
                star = "*" if r.get("id") == sel_id else " "
                lock = "LOCK" if r.get("locked") else "    "
                neon = f" neon:{r['neon'][:5]}{'' if r.get('neon_on', True) else '(off)'}" if r.get("neon") else ""
                d.text((28, y), f"{star}{tag:<5}{lock} {plate_label(r)[:24]}", fill=(235, 250, 240, 255), font=f2)
                d.text((28, y + 9), f"", fill=(150, 160, 170, 255), font=f2)
            if not es:
                d.text((14, 50), "no karts yet: bring one within 14 m", fill=(150, 160, 170, 255), font=f2)
            sel = es[self.sel] if es else None
            if sel is not None:
                r = sel[2]
                d.text((10, 292), f"value {money(kart_value(r))}   {'ECO ' if r.get('eco') else ''}neon:{r.get('neon') or '-'}" + ("" if r.get("neon_on", True) else "(off)"), fill=(255, 235, 190, 255), font=f2)
            self.info.text = "W/S choose  Enter = SELECT (K summons it)  L lock"
            self.info2.text = self.pmsg or "K store a nearby kart (gets a bonus 2000 m from its origin)"
            if self.hint0 is not None:
                self.hint0.text = "Enter select   K store nearby   L lock/unlock"
                self.hint1.text = "A/D page    P / E / Esc close    (K outside the phone: summon)"
        else:                                                   # GPS
            dm = (g.world.level == "mall")
            fwd = self.gps_fwd()
            tz = p.z + fwd * self.gps_d
            idx, tidx = g.world.chunk_index(p.z), g.world.chunk_index(tz)
            d.text((10, 8), "GPS  -  FRONT COORDINATES", fill=(255, 220, 120, 255), font=f)
            d.text((10, 36), f"YOU    z = {p.z:9.1f}   chunk {idx}", fill=(120, 255, 200, 255), font=f2)
            d.text((10, 56), f"FRONT  z = {tz:9.1f}   chunk {tidx}", fill=(255, 235, 150, 255), font=f)
            kinds = [zone_info(g.args.seed, i)[0] for i in range(min(idx, tidx), max(idx, tidx) + 1)] if dm else []
            n = min(len(kinds), 14)
            step = max(1, len(kinds) // 14) if kinds else 1
            for i in range(n):
                kd = kinds[i * step]
                fg = ZONE_BG.get(kd, (40, 50, 60))
                col = tuple(min(255, int(c * 2.2) + 30) for c in fg)
                y = 96 + 14 * i
                d.rectangle([14, y, 110, y + 12], fill=col + (255,))
                d.text((118, y - 1), ZONES.get(kd, {}).get("title", kd).split("  ")[0][:22], fill=(220, 235, 230, 255), font=f2)
            d.text((10, 296), f"{fwd * self.gps_d:+.0f} m ahead   (W/S +-100 m)", fill=(160, 180, 190, 255), font=f2)
            self.info.text = f"[T] TELEPORT to z = {tz:.0f}   cost {money(1000)}" if dm else "GPS teleport works in the mall only"
            self.info2.text = self.pmsg or f"you have {money(g.inv.coins)}   teleporting resets every kart's origin"
            if self.hint0 is not None:
                self.hint0.text = "T teleport    W/S distance"
                self.hint1.text = "A/D page    P / E / Esc close"
        self.screen.texture = tex_from_pil(im)

    def key(self, k: str) -> None:
        g = self.g
        if k in ("left arrow", "a", "right arrow", "d"):
            self.page = (self.page + (-1 if k in ("left arrow", "a") else 1)) % 4
            self.sel, self.pmsg = 0, ""
            g.sfx.play("tick", 0.5)
            self.redraw()
            return
        if k == "z":
            g.toggle_ambient()
            self.redraw()
            return
        if self.page in (0, 2):
            return _PhoneUI_v4.key(self, k)
        if k in ("p", "e", "escape", "g"):
            g.close_ui()
            return
        if self.page == 3:
            if k in ("w", "up arrow"):
                self.gps_d = min(5000.0, self.gps_d + 100.0)
            elif k in ("s", "down arrow"):
                self.gps_d = max(100.0, self.gps_d - 100.0)
            elif k == "t":
                fwd = self.gps_fwd()
                self.pmsg = g.gps_teleport(g.player.z + fwd * self.gps_d)
            self.redraw()
            return
        es = self.entries()
        st = self.stored()
        if k in ("w", "up arrow") and es:
            self.sel = (self.sel - 1) % len(es)
        elif k in ("s", "down arrow") and es:
            self.sel = (self.sel + 1) % len(es)
        elif es:
            kind, ref, r = es[min(self.sel, len(es) - 1)]
            if k in ("enter", "return"):
                g.inv.flags["ksel"] = r["id"]
                g.inv.dirty = True
                self.pmsg = "selected: press K anywhere to summon it"
            elif k == "l":
                r["locked"] = not r.get("locked")
                g.inv.dirty = True
                self.pmsg = "LOCKED: saved in cache, never lost" if r["locked"] else "unlocked"
            elif k == "k":
                if kind != "near":
                    self.pmsg = "select a nearby kart to store it"
                elif len(st) >= 3:
                    self.pmsg = "storage full (3)"
                elif ref.passengers:
                    self.pmsg = "a passenger is seated: ask them to stand first (B)"
                else:
                    bonus = store_bonus(ref)
                    rec = kart_export(ref)
                    rec["pos"] = None
                    st.append(rec)
                    g.karts.pop(ref.key, None)
                    g._drop_kart(ref)
                    g.inv.dirty = True
                    g.sfx.play("chime", 0.7)
                    g.sfx.play("pickup", 0.6)
                    g.award("pocketkart")
                    self.pmsg = f"kart folded into your phone" + (f"   +{money(bonus)} value!" if bonus else "")
                    if bonus:
                        g.award("flex")
        g.sfx.play("tick", 0.4)
        self.redraw()


# ---- kart value inspector --------------------------------------------------------------------------------------------------------------
class KartInspectUI(UiBase):
    def __init__(self, g, k) -> None:
        super().__init__(g)
        r = k.rec
        self.quad((0, 0), (0.96, 0.78), C(0.4, 0.8, 1.0, 0.35), z=0.21)
        self.quad((0, 0), (0.94, 0.76), C(0.02, 0.03, 0.05, 0.96))
        self.text("KART INSPECTOR   -   phone scan", (-0.44, 0.32), 1.0, C(0.6, 0.95, 1.0, 1))
        self.quad((-0.22, 0.2), (0.34, 0.17), C(1, 1, 1, 1), plate_texture(r), 0.15)
        d = abs(k.z - (k.z if r.get("origin") is None else r["origin"]))
        lines = [f"PLATE      {r.get('plate_text') or '-'}   ({PLATE_TIERS[r['plate_tier']][0] if r.get('plate_tier') is not None else 'no tier'}{', custom' if r.get('plate_custom') else ''})",
                 f"VALUE      {money(kart_value(r))}      (earned bonuses {money(r.get('earned', 0))})",
                 f"TYPE       {'Eco-Kart (battery, heat bar)' if r.get('eco') else 'Standard kart (fuel)'}",
                 f"SOLAR      {('level %d  -  +%d fuel/s' % (r['solar'], r['solar'])) if r.get('solar') else 'none'}",
                 f"NEON       {(r['neon'] + (' (on)' if r.get('neon_on', True) else ' (off)')) if r.get('neon') else 'none'}",
                 f"HEADLIGHT  {'installed' if r.get('light') else 'none'}      SEAT  {'spoiler seat' if r.get('seat') else 'none'}",
                 f"MILES      {r.get('miles', 0.0):.2f} mi        origin distance  {d:.0f} / {BONUS_Z:.0f} m",
                 f"STATUS     {'LOCKED (cached)' if r.get('locked') else 'unlocked'}      music library {len(r.get('lib', []))} tracks"]
        for i, ln in enumerate(lines):
            self.text(ln, (-0.44, 0.07 - 0.052 * i), 0.8, C(0.85, 1.0, 0.92, 0.97))
        self.text("Q / E / Esc close", (-0.44, -0.34), 0.65, C(0.6, 0.75, 0.85, 0.9))

    def key(self, k: str) -> None:
        if k in ("q", "e", "escape", "p"):
            self.g.close_ui()


# ---- Eco-Kart dashboard: audio jack (listening booth) / phone library / flex value ---------------------------------------------------
class CarUI(UiBase):
    def __init__(self, g, kart) -> None:
        super().__init__(g)
        self.k, self.sel, self.msg, self.src = kart, 0, "", 0
        self.sources = []
        j = g.jack
        if j and j.get("booth") is not None and j.get("kart") is kart:
            self.sources.append("booth")
        if g.held == "phone" and g.inv.n("phone") > 0:
            self.sources.append("phone")
        self.quad((0, 0), (1.2, 0.8), C(0.1, 0.9, 0.6, 0.35), z=0.21)
        self.quad((0, 0), (1.18, 0.78), C(0.015, 0.03, 0.03, 0.95))
        self.title = self.text("", (-0.56, 0.34), 1.0)
        self.status = self.text("", (-0.56, 0.29), 0.8, C(1.0, 0.8, 0.4, 0.95))
        self.rows = [self.text("", (-0.56, 0.23 - 0.042 * i), 0.82) for i in range(12)]
        self.foot = self.text("", (-0.56, -0.355), 0.62, C(0.6, 0.8, 0.7, 0.9))
        self.refresh()

    def tracks(self) -> list:
        g = self.g
        if not self.sources:
            return []
        s = self.sources[self.src]
        if s == "phone":
            return g.phone.playlist()
        tr = sorted(g.stack.indexer.tracks) if g.stack.indexer is not None else []
        return [t for t in tr if t in g.inv.unlocked or stable_hash(g.args.seed, "free", t) % 4 == 0]

    def refresh(self) -> None:
        g, k = self.g, self.k
        r = k.rec
        if not self.sources:
            d = abs(k.z - (k.z if r.get("origin") is None else r["origin"]))
            self.title.text = "ECO-KART  -  FLEX DASHBOARD"
            self.status.text = "nothing connected  (hold the phone out, or wire the car to a Listening Post with the audio jack)"
            lines = [f"TOTAL MILES DRIVEN      {r.get('miles', 0.0):.2f} mi", f"FLEX VALUE              {money(kart_value(r))}",
                     f"PLATE                   {plate_label(r)}", f"DISTANCE FROM ORIGIN    {d:.0f} m   (bonus at {BONUS_Z:.0f} m, paid when you store the kart with K)",
                     f"BONUS ON STORE          {money(BONUS_FLAT)} flat + miles beyond 2000 m"]
            for i, row in enumerate(self.rows):
                row.text = lines[i] if i < len(lines) else ""
                row.color = C(0.8, 1.0, 0.9, 0.97)
            self.foot.text = "J / E / Esc close"
            return
        s = self.sources[self.src]
        lst = self.tracks()
        self.sel = max(0, min(self.sel, max(0, len(lst) - 1)))
        pm = g.phone
        self.title.text = ("ECO-KART  -  LISTENING POST PLAYLIST (wired)" if s == "booth" else "ECO-KART  -  PHONE LIBRARY") + (f"      [{self.src + 1}/{len(self.sources)}] Tab switches" if len(self.sources) > 1 else "")
        now = pm.title() if pm.cur is not None else "nothing playing"
        self.status.text = f"{'|| ' if pm.paused else '> '}{now}   {self.msg}"
        top = max(0, min(self.sel - 5, len(lst) - len(self.rows)))
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= len(lst):
                row.text = ""
                continue
            tid = lst[j]
            tag = "  [in car]" if (s == "booth" and tid in k.rec["lib"]) else ""
            row.text = f"{'>' if j == self.sel else ' '} {j + 1:3d}  {pm.title(tid)[:46]}{tag}{'  *' if tid == pm.cur else ''}"
            row.color = C(1.0, 0.95, 0.6, 1) if j == self.sel else C(0.8, 1.0, 0.9, 0.95)
        if not lst:
            self.rows[0].text = "(empty)"
        self.foot.text = ("W/S select   Enter play   D download to car   X delete from car   Space pause   N next   J/E close" if s == "booth"
                          else "W/S select   Enter play   Space pause   N next   J/E close")

    def key(self, k: str) -> None:
        g = self.g
        if k in ("j", "e", "escape"):
            g.close_ui()
            return
        if not self.sources:
            return
        lst = self.tracks()
        if k in ("w", "up arrow") and lst:
            self.sel = (self.sel - 1) % len(lst)
        elif k in ("s", "down arrow") and lst:
            self.sel = (self.sel + 1) % len(lst)
        elif k == "tab" and len(self.sources) > 1:
            self.src, self.sel = (self.src + 1) % len(self.sources), 0
        elif k in ("enter", "return") and lst:
            g.phone.play(lst[min(self.sel, len(lst) - 1)], quiet=True)
        elif k == "space":
            g.phone.toggle()
        elif k == "n":
            g.phone.next()
        elif k == "d" and self.sources[self.src] == "booth" and lst:
            tid = lst[min(self.sel, len(lst) - 1)]
            if tid in self.k.rec["lib"]:
                self.msg = "already in the car"
            else:
                self.k.rec["lib"].append(tid)
                g.inv.dirty = True
                g.sfx.play("chime", 0.6)
                self.msg = "downloaded to the car"
        elif k in ("x", "delete") and self.sources[self.src] == "booth" and lst:
            tid = lst[min(self.sel, len(lst) - 1)]
            if tid in self.k.rec["lib"]:
                self.k.rec["lib"].remove(tid)
                g.inv.dirty = True
                self.msg = "deleted from the car"
        g.sfx.play("tick", 0.4)
        self.refresh()

    def update(self, dt: float) -> None:
        if self.g.grid.out and self.sources:
            self.g.close_ui()
            self.g.hud.note("Blackout: the car music GUI went dark.", good=False)


# ---- kart HUD (fuel / heat bar + mph) ------------------------------------------------------------------------------------------------------
class KartHud:
    def __init__(self) -> None:
        mk_q = lambda pos, sc, col: ur.Entity(parent=ur.camera.ui, model="quad", position=pos, scale=sc, color=col, z=0.5)
        self.back = mk_q((0, -0.38), (0.44, 0.030), C(0, 0, 0, 0.6))
        self.bar = mk_q((0, -0.38), (0.42, 0.018), C(1, 0.6, 0.1, 1))
        self.bar.origin = (-0.5, 0)
        self.bar.x = -0.21
        self.txt = ur.Text(text="", position=(-0.21, -0.345), scale=0.9, color=C(1, 1, 1, 1))
        self.show(False)

    def show(self, b: bool) -> None:
        for e in (self.back, self.bar, self.txt):
            e.enabled = b

    def update(self, k) -> None:
        self.show(True)
        mph = k.speed * MPS_MPH
        if k.rec["eco"]:
            self.bar.scale_x = 0.42 * max(0.01, k.heat)
            self.bar.color = C(0.2 + 0.8 * k.heat, 1.0 - 0.9 * k.heat, 0.3, 1)
            self.txt.text = f"{mph:3.0f} mph   HEAT {k.heat * 100:3.0f}%" + ("   OVERHEATED - release throttle" if k.overheated else "")
        else:
            self.bar.scale_x = 0.42 * max(0.01, k.fuel / 100.0)
            self.bar.color = C(1.0, 0.6, 0.1, 1)
            self.txt.text = f"{mph:3.0f} mph   FUEL {k.fuel:3.0f}%" + (f"   +{k.rec['solar']}/s" if k.rec["solar"] else "")


# ---- CCTV: black TVs, one live desk terminal, 10 chunks of coverage -----------------------------------------------------------------------
class CctvUI(UiBase):
    """Clean, unfiltered live feed - only from the desk computer."""
    def __init__(self, g) -> None:
        super().__init__(g)
        self.idx = 0
        self.quad((0, 0), (1.9, 1.2), C(0, 0, 0, 1), z=0.3)
        self.screen = self.quad((0, 0.02), (1.28, 0.96), C(1, 1, 1, 1), None, 0.25)
        self.lab = self.text("", (-0.62, 0.43), 1.0, C(0.8, 1.0, 0.9, 1))
        self.text("A/D switch camera   E / Esc leave", (-0.62, -0.47), 0.75, C(0.6, 0.8, 0.7, 0.9))
        g.cctv_ui = self
        if g.feeds and g.feeds[0].ok:
            set_raw_texture(self.screen, g.feeds[0].tex)
        else:
            self.screen.color = C(0.02, 0.02, 0.02, 1)
        self.refresh()

    def refresh(self) -> None:
        pts = self.g.cctv_points()
        if pts:
            self.idx %= len(pts)
            self.g.feeds[0].point(*pts[self.idx])
            self.lab.text = pts[self.idx][5]
        else:
            self.lab.text = "NO SIGNAL"

    def key(self, k: str) -> None:
        if k in ("e", "escape"):
            self.g.close_ui()
        elif k in ("d", "right arrow", "a", "left arrow"):
            self.idx += 1 if k in ("d", "right arrow") else -1
            self.g.sfx.play("tick")
            self.refresh()

    def update(self, dt: float) -> None:
        self.refresh()

    def close(self) -> None:
        self.g.cctv_ui = None
        super().close()


def _cctv_points(self) -> list:
    sidx = getattr(self, "sec_idx", None)
    if sidx is None:
        sidx = self.world.chunk_index(self.player.z)
    pts, n = [], 0
    for i in list(range(sidx - 5, sidx)) + list(range(sidx + 1, sidx + 6)):     # 10 chunks centred on the security room
        kind, part, xo, hw, h = self.world.geom(i)
        title = ZONES[kind]["title"].split("  ")[0][:22]
        loaded = i in self.world.chunks
        for (x, y, z, yaw, pitch, tag) in cam_poses(xo, hw, h, i * CH):          # exactly two cameras per room
            n += 1
            fz = 0.5 if yaw == 0 else -0.5
            pts.append((x, y - 0.08, z + fz, yaw, pitch, f"CAM {n:02d}  {title}  ({tag})" + ("" if loaded else "   [loading chunk...]")))
    return pts


def _use_cctv(self, it) -> None:
    if not self.feeds:
        self.feeds = [FeedCam("cctvA", 448, 336)]
    self.sec_idx = self.world.chunk_index(it.z)
    self.open_ui(CctvUI(self))


def _update_cctv(self, dt: float) -> None:
    active = self.cctv_ui is not None                          # the TVs stay black; only the desk terminal renders a camera
    for fc in self.feeds:
        was = getattr(fc, "on", False)
        fc.on = active
        if was != active:
            fc.active(active)
    w = self.world
    w.ahead = w.behind = 5 if active else 3                    # stream the whole coverage window while the terminal is open


# ---- Game additions ---------------------------------------------------------------------------------------------------------------------
def _toggle_ambient(self) -> None:
    self.ambient_muted = not self.ambient_muted
    self.inv.flags["muted"] = self.ambient_muted
    self.inv.dirty = True
    if not self.ambient_muted:
        self.bridge.duck_target = 1.0
    self.sfx.play("tick", 0.6)
    self.hud.note("Phone: mall ambience muted   (Z to restore)" if self.ambient_muted else "Phone: mall ambience restored")


def summon_kart_v5(self, rec: dict) -> bool:
    p, w = self.player, self.world
    yaw = p.yaw_body
    for dist in (2.8, 2.2, 3.6, 1.8):
        nx, nz = p.x + math.sin(math.radians(yaw)) * dist, p.z + math.cos(math.radians(yaw)) * dist
        rx, rz, _g, bumped = w.resolve(nx, p.y, nz, GoKart.R, step=0.25)
        if not bumped and math.hypot(rx - nx, rz - nz) < 0.05:
            key = f"{w.level}:k{rec.get('id', int(time.time() * 1000) % 10 ** 9)}"
            k = kart_from_rec(rec, key, nx, nz, yaw, w.level)
            reset_origin(k)
            self.karts[key] = k
            k.build()
            self.sfx.play("thud", 0.6)
            return True
    return False


def summon_selected(self) -> None:
    p, w = self.player, self.world
    if p.kart is not None:
        self.hud.note("Get out of the kart first.", good=False)
        return
    sel0 = self.inv.flags.get("ksel")
    st0 = self.inv.flags.setdefault("pkarts", [])
    near = next((k for k in self.karts.values()
                 if sel0 and k.rec["id"] == sel0 and getattr(k, "level", "mall") == w.level
                 and math.hypot(k.x - p.x, k.z - p.z) < 14.0), None)
    if near is not None:                                   # selected kart is close: store it
        if self.clock - getattr(self, "_summon_t", -99.0) < 4.0:
            return
        if len(st0) >= 3:
            self.hud.note("Phone storage is full (3).", good=False)
        elif near.passengers:
            self.hud.note("A passenger is seated: ask them to stand first (B).", good=False)
        else:
            bonus = store_bonus(near)
            rec = kart_export(near)
            rec["pos"] = None
            st0.append(rec)
            self.karts.pop(near.key, None)
            if self.last_kart is near:
                self.last_kart = None
            self._drop_kart(near)
            self.inv.dirty = True
            self.sfx.play("chime", 0.7)
            self.sfx.play("pickup", 0.6)
            self.award("pocketkart")
            self.hud.note("Kart folded into your phone" + (f"   +{money(bonus)} value!" if bonus else "") + "   (K again to summon)")
            if bonus:
                self.award("flex")
        return
    st = self.inv.flags.setdefault("pkarts", [])
    sel = self.inv.flags.get("ksel")
    for i, r in enumerate(st):
        if r.get("id") == sel or (sel is None and self.last_kart is None and i == 0):
            if self.summon_kart(r):
                st.pop(i)
                self.inv.dirty = True
                self.hud.note("Your kart unfolds beside you.")
            else:
                self.hud.note("No room to unfold it here.", good=False)
            return
    k = next((k for k in self.karts.values() if k.rec["id"] == sel), None) if sel else None
    if k is None and sel is None:
        k = self.last_kart if (self.last_kart is not None and self.last_kart.key in self.karts) else None
    if k is None:
        self.hud.note("No kart selected.  Open the phone (P) > GARAGE and press Enter on a kart.", good=False)
        return
    yaw = p.yaw_body
    for dist in (2.8, 2.2, 3.6, 1.8):
        nx, nz = p.x + math.sin(math.radians(yaw)) * dist, p.z + math.cos(math.radians(yaw)) * dist
        rx, rz, _g, bumped = w.resolve(nx, p.y, nz, GoKart.R, step=0.25)
        if not bumped and math.hypot(rx - nx, rz - nz) < 0.05:
            k.level = w.level
            k.x, k.z, k.yaw, k.vx, k.vz, k.y = nx, nz, yaw, 0.0, 0.0, p.y
            reset_origin(k)                                    # summoning moves the origin too: no value farming
            if k.root is None:
                k.build()
            k.sync()
            self.sfx.play("thud", 0.6)
            self.hud.note("Your kart rolls up beside you.")
            return
    self.hud.note("No room to summon it here.", good=False)


def gps_teleport(self, tz: float) -> str:
    p, w = self.player, self.world
    if w.level != "mall":
        return "GPS only works in the mall"
    if self.inv.coins < 1000:
        return "teleport costs $1,000"
    if p.seat is not None:
        p.stand()
    self.inv.coins -= 1000
    self.inv.dirty = True
    w.stream(tz, budget=99)
    xo = w.xo_at(tz)
    spot = None
    for dx in (0.0, 1.5, -1.5, 3.0, -3.0, 4.5, -4.5):
        for dz in (0.0, 1.0, -1.0, 2.0):
            rx, rz, gnd, bumped = w.resolve(xo + dx, 0.0, tz + dz, Player.R)
            if not bumped:
                spot = (rx, rz, gnd)
                break
        if spot:
            break
    rx, rz, gnd = spot or (xo, tz, 0.0)
    k = p.kart
    p.x, p.z, p.y, p.vx, p.vz = rx, rz, gnd, 0.0, 0.0
    if k is not None:
        k.x, k.z, k.vx, k.vz, k.y = rx, rz, 0.0, 0.0, gnd
        k.sync()
    for kk in self.karts.values():                              # new origin everywhere: teleport + store must not farm the 2000 m bonus
        if kk is k:
            reset_origin(kk)
        elif getattr(kk, "level", "mall") == w.level:
            reset_origin(kk)
    self.sfx.play("gps_warp", 0.8)
    self.fx.flash_a = 0.8
    self.world.stream(p.z, budget=99)
    return f"teleported to z = {tz:.0f}"


def _target_kart(self, max_d: float = 8.0, cos_min: float = 0.88):
    p, best, bd = self.player, None, 1e9
    for k in self.karts.values():
        if getattr(k, "level", "mall") != self.world.level or k is p.kart:
            continue
        d = math.hypot(k.x - p.x, k.z - p.z)
        if d < max_d and self.view_cos(k.x, k.z) > cos_min and d < bd:
            best, bd = k, d
    return best


def _near_kiosk(self, rad: float = 3.8):
    p = self.player
    best, bd = None, rad
    for ch in self.world.near_chunks(p.z, 1):
        for it in ch.inters:
            if it.kind == "kiosk":
                d = math.hypot(it.x - p.x, it.z - p.z)
                if d < bd:
                    best, bd = it, d
    return best


def _use_jack(self) -> None:
    p = self.player
    j = self.jack or dict(booth=None, kart=None, ent=None)
    kiosk = self.near_kiosk()
    ek = next((k for k in self.karts.values() if k.rec["eco"] and getattr(k, "level", "mall") == self.world.level
               and math.hypot(k.x - p.x, k.z - p.z) < 3.8), None)
    if kiosk is not None and j["booth"] is None:
        j["booth"] = kiosk
        self.sfx.play("jack_plug", 0.8)
        self.hud.note("Audio jack plugged into the Listening Post.  Now plug the other end into your Eco-Kart (T beside it).")
    elif ek is not None and j["kart"] is None:
        j["kart"] = ek
        self.sfx.play("jack_plug", 0.8)
        self.hud.note("Audio jack plugged into the Eco-Kart.  Sit in it and press J." if j["booth"] is not None else "Plugged into the Eco-Kart.  Now plug the other end into a Listening Post.")
    elif j["booth"] is not None or j["kart"] is not None:
        self._jack_cut(j, "Audio jack unplugged.")
        return
    else:
        self.hud.note("Stand beside a Listening Post or an Eco-Kart (within ~4 m) and press T.", good=False)
    if j["booth"] is not None and j["kart"] is not None:
        if math.hypot(j["booth"].x - j["kart"].x, j["booth"].z - j["kart"].z) > 10.0:
            self._jack_cut(j, "Too far: the wire only reaches 10 m.")
            return
        if j["ent"] is None:
            j["ent"] = ur.Entity(model="cube", color=C(0.02, 0.02, 0.03), scale=(0.035, 0.035, 1.0))
    self.jack = j


def _jack_cut(self, j, msg) -> None:
    if j.get("ent") is not None:
        ur.destroy(j["ent"])
    self.jack = None
    self.hud.note(msg, good=False)


def _update_jack(self, dt) -> None:
    j = self.jack
    if not j or j["ent"] is None:
        return
    b, k = j["booth"], j["kart"]
    d = math.hypot(b.x - k.x, b.z - k.z)
    if d > 10.0:
        self._jack_cut(j, "The wire pulled free: 10 m is the limit.")
        return
    a, c = ur.Vec3(b.x, 0.9, b.z), ur.Vec3(k.x, 0.6, k.z)
    j["ent"].position = (a + c) / 2
    j["ent"].look_at(c)
    j["ent"].scale_z = max(0.1, d)


def _open_car_ui(self) -> None:
    k = self.player.kart
    if self.grid.out and (self.jack or self.held == "phone"):
        self.hud.note("Blackout: the car music GUI will not load.", good=False)
        return
    self.open_ui(CarUI(self, k))


def _toggle_passenger(self) -> None:
    p = self.player
    k = p.kart or min((k for k in self.karts.values() if k.rec["seat"] and getattr(k, "level", "mall") == self.world.level and math.hypot(k.x - p.x, k.z - p.z) < 4.0),
                      key=lambda k: math.hypot(k.x - p.x, k.z - p.z), default=None)
    if k is None or not k.rec["seat"]:
        self.hud.note("No spoiler seat here (craftsman: LIGHTS & SEATS).", good=False)
        return
    if k.passengers:
        who = k.passengers[0]
        self.dismount_passenger(k, who)
        return
    if p.kart is None:                                          # on foot beside a parked kart: sit on its spoiler yourself
        k.passengers.append(("player", p))
        p.rider = k
        self.hud.note("You sit on the spoiler.  Move to hop off.")
        return
    best, bd = None, 5.0
    for npc in self.nearby_npcs():
        if type(npc) is Npc and not npc.gone and npc.state != "sit" and getattr(npc, "ride", None) is None:
            d = math.hypot(npc.x - k.x, npc.z - k.z)
            if d < bd:
                best, bd = npc, d
    if best is None:
        self.hud.note("Nobody close enough to invite.", good=False)
        return
    best.ride = k
    k.passengers.append(("npc", best))
    self.hud.speak(best, random.choice(("Oh! Sure. Hold on tight, I guess?", "Never sat on a spoiler. Let's go.", "Is this allowed? ...Okay.")))


def _dismount_passenger(self, k, who) -> None:
    kind, ref = who
    if who in k.passengers:
        k.passengers.remove(who)
    yr = math.radians(k.yaw)
    sx, sz = k.x - math.sin(yr) * 1.6 + math.cos(yr) * 1.2, k.z - math.cos(yr) * 1.6 - math.sin(yr) * 1.2
    rx, rz, gnd, _ = self.world.resolve(sx, 0.0, sz, Player.R)
    if kind == "npc":
        ref.ride, ref.x, ref.z, ref.y, ref.state, ref.timer = None, rx, rz, gnd, "idle", 2.0
        ref.fig.place(ref.x, ref.y, ref.z, ref.yaw)
    else:
        ref.rider = None
        ref.x, ref.z, ref.y = rx, rz, gnd
    self.hud.note("The passenger hops off.")


def _update_passengers(self, dt) -> None:
    for k in self.karts.values():
        if not k.passengers:
            continue
        sx, sy, sz = k.seat_world()
        for who in list(k.passengers):
            kind, ref = who
            if kind == "npc":
                c = self.world.at(k.z)
                if c is not None and ref not in c.npcs:           # the rider changes chunk with the kart
                    old = getattr(ref, "ch", None)
                    if old is not None and ref in old.npcs:
                        old.npcs.remove(ref)
                    c.npcs.append(ref)
                    ref.ch = c
                ref.x, ref.z, ref.y = sx, sz, sy - 0.72
                ref.fig.place(sx, sy - 0.72, sz, k.yaw)
                ref.fig.pose(dt, 0, True)
                ref.fig.root.enabled = k.root is not None and k.root.enabled
            elif self.player.rider is not k:
                k.passengers.remove(who)


def _player_update_v5(self, dt, w, yaw, jump):
    k = getattr(self, "rider", None)
    if k is not None:
        hk = ur.held_keys
        if any(hk[x] for x in ("w", "a", "s", "d", "space", "up arrow", "down arrow")) or k.root is None:
            who = next((x for x in k.passengers if x[0] == "player"), None)
            self.g.dismount_passenger(k, who) if who else setattr(self, "rider", None)
            return
        sx, sy, sz = k.seat_world()
        self.x, self.z, self.y, self.vx, self.vz = sx, sz, sy - 0.72, k.vx, k.vz
        self.fig.place(sx, sy - 0.72, sz, k.yaw)
        self.fig.pose(dt, 0, True)
        return
    return _orig_player_v4(self, dt, w, yaw, jump)


def _npc_update_v5(self, dt, w):
    if getattr(self, "ride", None) is not None:
        return
    return _orig_npc_update(self, dt, w)


def _kart_prompt(self, k) -> str:
    if k.rec["eco"]:
        return f"[E] Get out  [J] car dashboard / music  [H] light  [O] neon  [B] passenger   {k.speed * MPS_MPH:.0f} mph   heat {k.heat * 100:.0f}%" + ("  OVERHEATED" if k.overheated else "")
    return f"[E] Get out  [R] Refuel  [H] light  [O] neon  [B] passenger   fuel {k.fuel:.0f}%   {k.speed * MPS_MPH:.0f} mph"


def _kcache_sync(self) -> None:
    """Locked karts live in cache memory (the save file): their full record + position, so leaving the game never loses them."""
    fl = self.inv.flags
    rows = [kart_export(k) for k in self.karts.values() if k.rec.get("locked")]
    if json_dumps(rows) != json_dumps(fl.get("kcache")):
        fl["kcache"] = rows
        self.inv.dirty = True


def json_dumps(o) -> str:
    import json
    return json.dumps(o, sort_keys=True, default=str)


def _v5_init(self) -> None:
    _orig_v4_init(self)
    self.ambient_muted = bool(self.inv.flags.get("muted", False))
    self.jack, self.kart_hud, self._v5_ready, self._kc_t, self._sec_t = None, None, False, 0.0, 0.0
    self.sec_idx = None


def _v5_update(self, dt: float, kind: str) -> None:
    _orig_update_v4(self, dt, kind)
    p = self.player
    if not self._v5_ready:
        self._v5_ready = True
        fl = self.inv.flags
        fl["pkarts"] = [dict(new_kart_rec(), **r) for r in fl.get("pkarts", [])]
        for r in fl.get("kcache", []):                         # respawn cached (locked) karts where they were left
            pos = r.get("pos") or (0.0, 0.0, 0.0, "mall")
            key = f"cache:{r['id']}"
            if key not in self.karts:
                k = kart_from_rec(r, key, pos[0], pos[1], pos[2], pos[3])
                self.karts[key] = k
    for k in self.karts.values():                              # solar regeneration: +1 fuel/s per level (up to +5), also while parked
        if k.rec["solar"] and not k.rec["eco"]:
            k.fuel = min(100.0, k.fuel + regen_rate(k.rec) * dt)
        if k.root is not None:
            k.dress()
        if getattr(k, "hot_event", False):
            k.hot_event = False
            if k is p.kart:
                self.sfx.play("overheat", 0.7)
                self.hud.note("Overheating!  Release the throttle to cool down.", good=False)
    self._update_passengers(dt)
    self._update_jack(dt)
    self._kc_t -= dt
    if self._kc_t <= 0:
        self._kc_t = 3.0
        self._kcache_sync()
    if p.kart is not None:
        if self.kart_hud is None:
            self.kart_hud = KartHud()
        self.kart_hud.update(p.kart)
    elif self.kart_hud is not None:
        self.kart_hud.show(False)


def _use_held_v5(self) -> None:
    if self.held == "jack":
        self.use_jack()
        return
    if self.held == "phone":
        self.hud.note("Phone out.  P open   Z mute ambience   K summon kart   Q scan a kart you're looking at   J in an Eco-Kart for its music.", 4.0)
        return
    _orig_use_held_v4(self)


def _input_v5(self, key: str) -> bool:
    p = self.player
    has_phone = self.inv.n("phone") > 0
    if key == "k" and has_phone:
        self.summon_selected()
        return True
    if key == "z" and has_phone:
        self.toggle_ambient()
        return True
    if key == "q" and self.held == "phone":
        k = self.target_kart()
        if k is None:
            self.hud.note("Look at a kart (within 8 m) while holding the phone.", good=False)
        else:
            self.sfx.play("chime", 0.5)
            self.open_ui(KartInspectUI(self, k))
        return True
    if key == "j" and p.kart is not None and p.kart.rec["eco"]:
        self.open_car_ui()
        return True
    if key == "b" and (p.kart is not None or self.target is None or self.target.kind == "kart"):
        self.toggle_passenger()
        return True
    if key == "h" and p.kart is not None and p.kart.rec["light"]:
        p.kart.rec["light_on"] = not p.kart.rec["light_on"]
        self.sfx.play("click", 0.7)
        return True
    if key == "o" and p.kart is not None and p.kart.rec["neon"]:
        p.kart.rec["neon_on"] = not p.kart.rec["neon_on"]
        self.sfx.play("click", 0.7)
        return True
    if key == "x" and has_phone and self.phone.cur is not None:
        self.phone.stop()
        self.hud.note("Phone music stopped.   (M starts a new shuffle)")
        return True
    return _input_v4(self, key)


def _hint_v5(self) -> str:
    base = _hint_v4(self)
    if self.ui is not None or self.player.kart is not None:
        return base
    out = [base] if base else []
    if self.inv.n("phone") > 0:
        out.append("[K] summon kart  [Z] mute ambience" + ("  [Q] scan kart" if self.held == "phone" else ""))
    if self.held == "jack":
        j = self.jack
        out.append("jack: T beside a Listening Post, then T beside your Eco-Kart" + (f"  (booth {'ok' if j and j['booth'] else '-'} / kart {'ok' if j and j['kart'] else '-'})" if j else ""))
    return "   |   ".join(out)


_hold_orig = None


def _hold_v5(self, key: str) -> None:
    _hold_orig(self, key)
    if key in ("phone", "jack") and self.hold_ent is not None:
        for c in list(self.hold_ent.children):
            ur.destroy(c)
        if key == "phone":
            ur.Entity(parent=self.hold_ent, model="cube", scale=(0.085, 0.16, 0.012), color=C(0.05, 0.06, 0.09))
            ur.Entity(parent=self.hold_ent, model="cube", scale=(0.074, 0.14, 0.004), position=(0, 0, -0.008), color=C(0.3, 0.8, 0.9))
        else:
            ur.Entity(parent=self.hold_ent, model="cube", scale=(0.02, 0.02, 0.22), color=C(0.03, 0.03, 0.04))
            ur.Entity(parent=self.hold_ent, model="cube", scale=(0.03, 0.03, 0.04), position=(0, 0, 0.12), color=C(0.85, 0.85, 0.9))


_push_orig = None


def _push_look_v5(self, px, py, pz, level):
    _push_orig(self, px, py, pz, level)
    p = getattr(self, "player", None)
    k = getattr(p, "kart", None) if p is not None else None
    if k is not None and k.rec["light"] and k.rec["light_on"] and FANCY:       # path-finding headlight: a forward beam through the same spot-light uniform as the torch
        s, c = math.sin(math.radians(k.yaw)), math.cos(math.radians(k.yaw))
        ur.scene.set_shader_input("fl_k", 1.35)
        ur.scene.set_shader_input("fl_pos", ur.Vec3(k.x + s * 1.3, k.y + 0.55, k.z + c * 1.3))
        ur.scene.set_shader_input("fl_dir", ur.Vec3(s * 0.99, -0.05, c * 0.99))


def _bridge_update_v5(self, *a, **kw):
    g = getattr(self, "game", None)
    if g is not None:
        quiet = (g.phone.active and (g.phone.src is None or getattr(g, "car_gain", 1.0) > 0.35)) or getattr(g, "ambient_muted", False) or isinstance(g.ui, KioskUI)
        self.duck_target = 0.0 if quiet else 1.0
    return _bridge_update_v4_ref(self, *a, **kw)


def _v5_install() -> None:
    global _orig_v4_init, _orig_update_v4, _orig_use_held_v4, _orig_player_v4, _orig_npc_update, _hold_orig, _push_orig, _bridge_update_v4_ref
    import textwrap
    # kart
    GoKart.__orig_init = GoKart.__init__

    def gk_init(self, *a, **k):
        GoKart.__orig_init(self, *a, **k)
        _gk_init_extra(self)
    GoKart.__init__ = gk_init
    GoKart.engine_vol, GoKart.dress, GoKart.seat_world = _gk_engine_vol, _gk_dress, _gk_seat_world
    ns: dict = {}
    exec(textwrap.dedent(GK_DRIVE_SRC), globals(), ns)
    GoKart.drive = ns["drive"]
    _gb = GoKart.build

    def gk_build(self):
        had = self.root is not None
        _gb(self)
        if not had:
            self.dress_root, self.dress_sig = None, None
            self.dress()
    GoKart.build = gk_build
    # restaurant / world builders
    ChunkBuilder._jp_wash, ChunkBuilder._v5_cctv = V5Builders._jp_wash, V5Builders._v5_cctv
    _bj = ChunkBuilder._build_japan

    def build_japan(self):
        _bj(self)
        self._jp_wash()
    ChunkBuilder._build_japan = build_japan
    _vf = ChunkBuilder._v3_features

    def v3_features(self):
        _vf(self)
        if self.w.level == "mall" and cctv_covered(self.seed, self.ch.idx):
            try:
                self._v5_cctv()
            except Exception:
                log.exception("cctv props failed for chunk %s", self.ch.idx)
    ChunkBuilder._v3_features = v3_features
    # game
    Game.toggle_ambient, Game.summon_kart, Game.summon_selected, Game.gps_teleport = _toggle_ambient, summon_kart_v5, summon_selected, gps_teleport
    Game.target_kart, Game.near_kiosk, Game.use_jack, Game._jack_cut, Game._update_jack = _target_kart, _near_kiosk, _use_jack, _jack_cut, _update_jack
    Game.open_car_ui, Game.toggle_passenger, Game.dismount_passenger, Game._update_passengers = _open_car_ui, _toggle_passenger, _dismount_passenger, _update_passengers
    Game._kcache_sync, Game._kart_prompt = _kcache_sync, _kart_prompt
    Game.cctv_points, Game._update_cctv, Game._use_cctv = _cctv_points, _update_cctv, _use_cctv
    _orig_v4_init, _orig_update_v4, _orig_use_held_v4 = Game._v3_init, Game._update_v3, Game.use_held_or_served
    Game._v3_init, Game._update_v3, Game.use_held_or_served = _v5_init, _v5_update, _use_held_v5
    global _input_v4, _hint_v4
    _input_v4, _hint_v4 = Game._v3_input, Game._v3_hint
    Game._v3_input, Game._v3_hint = _input_v5, _hint_v5
    _hold_orig, Game.hold = Game.hold, _hold_v5
    _push_orig, Game._push_look = Game._push_look, _push_look_v5
    _orig_player_v4, Player.update = Player.update, _player_update_v5
    Player.rider = None
    _orig_npc_update, Npc.update = Npc.update, _npc_update_v5
    Npc.ride = None
    _bridge_update_v4_ref, AudioBridge.update = AudioBridge.update, _bridge_update_v5
    # dishwashing interaction
    def _use_jpwash(self, it):
        rest = self._jp_rest()
        if rest is not None:
            rest.player_wash()
            self.award("dishes")
    Game._use_jpwash = _use_jpwash
    # eating: an emptied plate goes onto the pile beside the seat instead of vanishing
    _fin = Game._jp_finish_eat

    def jp_finish_eat(self, st):
        s, rest = st["s"], self._jp_rest()
        if rest is not None and s.get("rec") is not None:
            ch = self.world.at(self.player.z)
            if ch is not None and s in ch.served:
                ch.served.remove(s)
            ent = s["ent"]
            s["ent"] = None
            self.vitals.eat(st["key"])
            self.sfx.play("gulp")
            self.hud.note(f"{ITEMS[st['key']]['name']}  -  {ITEMS[st['key']]['desc']}")
            if getattr(self.player.fig, "chop", None) is not None:
                self.player.fig.chop.enabled = False
            if self.pbub is not None:
                self.pbub.say(random.choice(("Oishii!", "Gochisousama deshita!", "Umai!")), 2.6)
                self.sfx.play("oishii", 0.5)
            self.jp_eat = None
            rest.finish_dish(s["rec"], st["key"], ent)
            return
        _fin(self, st)
    Game._jp_finish_eat = jp_finish_eat

# ---- v6: car music (library from the booth, positional playback from the car) ----------------------------------
_KEEP = object()
PhoneMusic.src = None
_pm_play, _pm_stop, _pm_playlist = PhoneMusic.play, PhoneMusic.stop, PhoneMusic.playlist


def _pm_play2(self, tid, quiet=False, src=_KEEP):
    if src is not _KEEP:
        self.src = src
    _pm_play(self, tid, quiet)


def _pm_stop2(self):
    self.src = None
    _pm_stop(self)


def _pm_playlist2(self):
    if self.src is not None:
        tr = self.g.stack.indexer.tracks if self.g.stack.indexer is not None else {}
        return [t for t in self.src.rec["lib"] if t in tr]
    return _pm_playlist(self)


PhoneMusic.play, PhoneMusic.stop, PhoneMusic.playlist = _pm_play2, _pm_stop2, _pm_playlist2

_prev_ki_key = KioskUI.key


def _ki_key_v6(self, k):
    if k in ("d", "x"):
        g, j = self.g, self.g.jack
        car = j["kart"] if (j and j.get("booth") is self.seat and j.get("kart") is not None) else None
        if car is None:
            if k == "x":
                self.msg = "wire the audio jack between this post and your Eco-Kart to manage its tracks"
                self.refresh()
                return
            return _prev_ki_key(self, k)                       # no car wired: D downloads to the phone
        if not self.tids:
            return
        tid = self.tids[self.sel]
        if g.grid.out:
            self.msg = "blackout: no transmission to the car"
        elif math.hypot(car.x - self.seat.x, car.z - self.seat.z) > 10.0:
            self.msg = "the car is out of wire range (10 m)"
        elif k == "d":
            if not self.is_open(tid):
                self.msg = "locked - insert a cassette (I) first"
            elif tid in car.rec["lib"]:
                self.msg = "already in the car"
            else:
                car.rec["lib"].append(tid)
                g.inv.dirty = True
                g.sfx.play("chime", 0.7)
                self.msg = "sent to the car  (J inside the car to play)"
        else:
            if tid in car.rec["lib"]:
                car.rec["lib"].remove(tid)
                g.inv.dirty = True
                if g.phone.src is car and g.phone.cur == tid:
                    g.phone.stop()
                self.msg = "removed from the car"
            else:
                self.msg = "that track isn't in the car"
        self.refresh()
        return
    _prev_ki_key(self, k)


KioskUI.key = _ki_key_v6


class CarUI(UiBase):                                           # replaces the v5 CarUI
    def __init__(self, g, kart) -> None:
        super().__init__(g)
        self.k, self.sel, self.msg = kart, 0, ""
        self.quad((0, 0), (1.2, 0.8), C(0.1, 0.9, 0.6, 0.35), z=0.21)
        self.quad((0, 0), (1.18, 0.78), C(0.015, 0.03, 0.03, 0.95))
        self.title = self.text("ECO-KART  -  CAR MUSIC", (-0.56, 0.34), 1.0)
        self.status = self.text("", (-0.56, 0.29), 0.8, C(1.0, 0.8, 0.4, 0.95))
        self.rows = [self.text("", (-0.56, 0.23 - 0.042 * i), 0.82) for i in range(12)]
        self.text("W/S select   Enter play   Space pause   N next   X stop   J / E close", (-0.56, -0.355), 0.62, C(0.6, 0.8, 0.7, 0.9))
        self.refresh()

    def lib(self) -> list:
        tr = self.g.stack.indexer.tracks if self.g.stack.indexer is not None else {}
        return [t for t in self.k.rec["lib"] if t in tr]

    def refresh(self) -> None:
        g, pm = self.g, self.g.phone
        lst = self.lib()
        self.sel = max(0, min(self.sel, max(0, len(lst) - 1)))
        mine = pm.src is self.k and pm.cur is not None
        self.status.text = (("|| " if pm.paused else "> ") + pm.title() if mine else "nothing playing from the car") + f"   {self.msg}"
        top = max(0, min(self.sel - 5, len(lst) - len(self.rows)))
        for i, row in enumerate(self.rows):
            j = top + i
            if j >= len(lst):
                row.text = ""
                continue
            row.text = f"{'>' if j == self.sel else ' '} {j + 1:3d}  {pm.title(lst[j])[:46]}{'  *' if mine and lst[j] == pm.cur else ''}"
            row.color = C(1.0, 0.95, 0.6, 1) if j == self.sel else C(0.8, 1.0, 0.9, 0.95)
        if not lst:
            self.rows[0].text = "(empty)  wire the jack from a Listening Post to this car, then press D on a track there"

    def key(self, k: str) -> None:
        g, pm = self.g, self.g.phone
        if k in ("j", "e", "escape"):
            g.close_ui()
            return
        lst = self.lib()
        if k in ("w", "up arrow") and lst:
            self.sel = (self.sel - 1) % len(lst)
        elif k in ("s", "down arrow") and lst:
            self.sel = (self.sel + 1) % len(lst)
        elif k in ("enter", "return") and lst:
            pm.play(lst[min(self.sel, len(lst) - 1)], quiet=True, src=self.k)
        elif k == "space" and pm.src is self.k:
            pm.toggle()
        elif k == "n" and pm.src is self.k:
            pm.next()
        elif k == "x" and pm.src is self.k:
            pm.stop()
        g.sfx.play("tick", 0.4)
        self.refresh()

    def update(self, dt: float) -> None:
        if self.g.grid.out:
            self.g.close_ui()
            self.g.hud.note("Blackout: the car music GUI went dark (the music keeps playing).", good=False)


def _open_car_ui_v6(self) -> None:
    if self.grid.out:
        self.hud.note("Blackout: the car music GUI will not load.", good=False)
        return
    self.open_ui(CarUI(self, self.player.kart))


def _update_car_audio(self, dt: float) -> None:
    pm, p, sfx = self.phone, self.player, self.sfx
    k = pm.src
    if k is None:
        sfx.phone_lr, sfx.phone_lp, self.car_gain = (1.0, 1.0), 1.0, 1.0
        return
    if self.karts.get(k.key) is not k:                         # the car was stored / removed
        pm.stop()
        return
    if p.kart is k:
        tg, pan, fc = 1.0, 0.0, 16000.0
    else:
        dx, dz = k.x - p.x, k.z - p.z
        d = math.hypot(dx, dz)
        if getattr(k, "level", "mall") != self.world.level:
            d = 999.0
        self._car_t = getattr(self, "_car_t", 0.0) - dt
        if self._car_t <= 0:                                   # wall check at ~7 Hz
            self._car_t = 0.15
            self._car_blocked = False
            if d < 45.0:
                w = self.world
                rows = [c[:4] for c in w.colliders_near(p.z) + w.colliders_near(k.z) if c[4] >= 2.0 and (c[1] - c[0]) >= 0.4]
                if rows:
                    self._car_blocked = World._blocked(p.x, p.z, k.x, k.z, np.array(rows, dtype=float))
        blocked = getattr(self, "_car_blocked", False)
        tg = 0.0 if d > 45.0 else 1.0 / (1.0 + (d / 7.0) ** 2)
        fc = max(500.0, 14000.0 / (1.0 + d / 5.0))
        if blocked:
            tg *= 0.4
            fc *= 0.35
        yr = math.radians(self.yaw)
        pan = (dx * math.cos(yr) - dz * math.sin(yr)) / (d + 1e-6) * min(1.0, d / 3.0)
    s = min(1.0, 6.0 * dt)
    self.car_gain = getattr(self, "car_gain", 1.0)
    self.car_gain += (tg - self.car_gain) * s
    self._car_pan = getattr(self, "_car_pan", 0.0) + (pan - getattr(self, "_car_pan", 0.0)) * s
    self._car_fc = getattr(self, "_car_fc", 16000.0) + (fc - getattr(self, "_car_fc", 16000.0)) * s
    th = (max(-1.0, min(1.0, self._car_pan)) + 1.0) * math.pi / 4.0
    sfx.phone_lr = (math.cos(th) * 1.4142 * self.car_gain, math.sin(th) * 1.4142 * self.car_gain)
    sfx.phone_lp = 1.0 if self._car_fc > 15000 else 1.0 - math.exp(-2.0 * math.pi * self._car_fc / sfx.sr)


_prev_update_v6 = Game._update_v3


def _update_v6(self, dt: float, kind: str) -> None:
    _prev_update_v6(self, dt, kind)
    self._update_car_audio(dt)


Game.open_car_ui, Game._update_car_audio, Game._update_v3 = _open_car_ui_v6, _update_car_audio, _update_v6

_v5_install()


# ---- v7: random tapes, Lost & Found tapes + cash, rare errand NPCs ---------------------------------------------
BADGES["errand"] = ("Errand Runner", "Filled a stranger's shopping list.")
PALETTES["dealer_q"] = dict(coat=(0.12, 0.38, 0.34), legs=(0.10, 0.10, 0.14), skin=(0.86, 0.72, 0.64), em=0.25)
PALETTES["dealer_t"] = dict(coat=(0.36, 0.16, 0.46), legs=(0.10, 0.08, 0.14), skin=(0.86, 0.72, 0.64), em=0.25)
TAPE_DUPE_ODDS, TAPE_ALLOWN_ODDS, TAPE_PRICE = 100, 10, 325
DEALER_ODDS, DEALER_TRADER_ODDS, QUEST_REWARD, QUEST_NEED = 10, 5, 500, 4
QUEST_KEYS = {k for k, _ in SHOP_STOCK if k != "fuel"} | {k for k, _ in CINEMA_STOCK}


def tape_pools(self) -> tuple:
    ids = self.covers.ids()
    lockable = [t for t in ids if t < 0 or stable_hash(self.args.seed, "free", t) % 4 != 0]
    owned = set(self.inv.cassettes) | set(self.inv.unlocked)      # inventory AND booth count
    return [t for t in lockable if t not in owned], [t for t in lockable if t in owned]


def roll_tape(self):
    missing, have = self.tape_pools()
    if missing:
        if have and random.randrange(TAPE_DUPE_ODDS) == 0:
            return random.choice(have)
        return random.choice(missing)
    pool = have
    if pool and random.randrange(TAPE_ALLOWN_ODDS) == 0:
        return random.choice(pool)
    return None


def grant_tape(self) -> str:
    tid = self.roll_tape()
    self.inv.dirty = True
    if tid is None:
        self.inv.coins += 25
        self.sfx.play("coin")
        return "A blank tape. You trade it in: +25 c"
    self.inv.cassettes.append(tid)
    self.sfx.play("pickup")
    return f"Cassette: {self.covers.title(tid).replace('_', ' ')[:34]}  -  bring it to a Listening Post"


def _use_cassette_v7(self, it) -> None:
    self.inv.collected.add(it.data["key"])
    self._remove_inter(it)
    self.hud.note(self.grant_tape())


Game.tape_pools, Game.roll_tape, Game.grant_tape, Game._use_cassette = tape_pools, roll_tape, grant_tape, _use_cassette_v7

# ---- Lost & Found: T takes the tape (once per desk), 1/11 chance of 200-250 cash -------------------------------
_ar_init, _ar_key = ArchiveUI.__init__, ArchiveUI.key


def _ar_init_v7(self, g, it):
    _ar_init(self, g, it)
    self.tkey = f"laf:{it.data.get('tag', 0)}"
    taken = self.tkey in g.inv.collected
    self.tnote = self.text("The shelf is empty." if taken else "T  take the tape from the shelf", (0.05, -0.21), 0.62, C(1.0, 0.85, 0.5, 0.95))


def _ar_key_v7(self, k):
    g = self.g
    if k == "t":
        if self.tkey in g.inv.collected:
            self.tnote.text = "The shelf is empty."
            return
        g.inv.collected.add(self.tkey)
        msg = g.grant_tape()
        if random.randrange(11) == 0:
            n = random.randint(200, 250)
            g.inv.coins += n
            g.sfx.play("coin")
            msg += f"   +{n} c  (an envelope was taped under the shelf)"
        g.hud.note(msg, 5.0)
        self.tnote.text = "The shelf is empty."
        return
    _ar_key(self, k)


ArchiveUI.__init__, ArchiveUI.key = _ar_init_v7, _ar_key_v7


# ---- rare NPC: errand collector (4/5) or tape dealer (1/5) in cafe / anchor / restroom, 1/10 per room -------------
class DealerNpc(Actor):
    def __init__(self, g, ch, pos, rng, variant: str) -> None:
        Actor.__init__(self, g, ch, pos, rng, "errand collector" if variant == "quest" else "tape dealer", "talk",
                       "dealer_q" if variant == "quest" else "dealer_t")
        self.variant, self.offer, self.leave_t = variant, False, None
        self.qkey = f"dealer:{ch.idx}"

    def progress(self) -> tuple:
        inv = self.g.inv
        return len(inv.fruits), sum(inv.n(k) for k in QUEST_KEYS)

    def on_talk(self) -> None:
        g, inv = self.g, self.g.inv
        if self.leave_t is not None:
            return
        if self.variant == "trader":
            missing, _ = g.tape_pools()
            if not missing:
                g.hud.speak(self, "You already own every tape I could sell. Impressive. Annoying.")
            elif not self.offer:
                self.offer = True
                g.hud.speak(self, f"One tape, picked at random, never one you own. {TAPE_PRICE} c.  Talk to me again to buy it.")
            elif inv.coins < TAPE_PRICE:
                g.hud.speak(self, f"Come back with {TAPE_PRICE}. The tape will wait. Probably.")
            else:
                inv.coins -= TAPE_PRICE
                tid = random.choice(missing)
                inv.cassettes.append(tid)
                inv.dirty = True
                self.offer = False
                g.sfx.play("coin")
                g.sfx.play("pickup")
                g.hud.speak(self, "Pleasure doing business.")
                g.hud.note(f"Cassette: {g.covers.title(tid).replace('_', ' ')[:34]}  (-{TAPE_PRICE} c)")
            return
        fruit, items = self.progress()
        if fruit < 1 or items < QUEST_NEED:
            g.hud.speak(self, f"Errands. One orchard fruit from the atrium tree, plus {QUEST_NEED} treats from the cafe or cinema counters. "
                              f"You have fruit {min(fruit, 1)}/1, treats {min(items, QUEST_NEED)}/{QUEST_NEED}.  {QUEST_REWARD} c for me.")
            return
        inv.fruits.pop(0)
        need = QUEST_NEED
        for k in sorted(QUEST_KEYS):
            while need and inv.n(k) > 0:
                inv.add(k, -1)
                need -= 1
        if g.held and inv.n(g.held) <= 0 and g.held != "phone":
            g.drop_held(True)
        inv.coins += QUEST_REWARD
        inv.collected.add(self.qkey)
        inv.dirty = True
        g.sfx.play("coin")
        g.award("errand")
        g.hud.speak(self, "Perfect. That's everything. I was never here.")
        g.hud.note(f"+{QUEST_REWARD} c")
        self.leave_t = 3.0

    def update(self, dt: float, w) -> None:
        if self.leave_t is not None:
            self.leave_t -= dt
            if self.leave_t <= 0 and not self.gone:
                self.gone = True
                if self in self.ch.npcs:
                    self.ch.npcs.remove(self)
                REAPER.push(self.fig.root)
        if self.gone:
            return
        Actor.update(self, dt, w)


_prev_v3npcs = ChunkBuilder._v3_npcs


def _v3npcs_v7(self):
    _prev_v3npcs(self)
    ch, g = self.ch, self.w.g
    if self.w.level != "mall" or ch.kind not in ("cafe", "anchor", "restroom") or ch.part != 0 or abs(ch.idx) <= 1:
        return
    rng = seeded(self.seed, "dealer", ch.idx)
    if rng.randrange(DEALER_ODDS) != 0:
        return
    variant = "trader" if rng.randrange(DEALER_TRADER_ODDS) == 0 else "quest"
    if variant == "quest" and f"dealer:{ch.idx}" in g.inv.collected:
        return                                              # already completed: he stays gone
    for _ in range(20):
        x, z = rng.uniform(-2.5, 2.5), ch.z0 + rng.uniform(3.0, CH - 3.0)
        if self._clear(x, z, 0.8):
            ch.npcs.append(DealerNpc(g, ch, (ch.xoff + x, z), rng, variant))
            return


ChunkBuilder._v3_npcs = _v3npcs_v7

# ---- v8: gym rework, endurance (max stamina), cooldowns ---------------------------------------------------------
STAM_BASE, STAM_STEP, STAM_CAP = 100.0, 6.0, 1320.0     # 1320 = a full-stamina sprint covers ~500 m
GYM_PAY, LIFT_REST, TREAD_REST, REP_GAP = 25, 5.0, 5.0, 0.9


def stam_max(self) -> float:
    return float(self.inv.flags.get("stam_max", STAM_BASE))


def gym_reward(self, what: str) -> None:
    inv = self.inv
    inv.coins += GYM_PAY
    inv.flags["stam_max"] = min(STAM_CAP, self.stam_max() + STAM_STEP)
    inv.dirty = True
    self.sfx.play("coin")
    self.hud.note(f"{what} done: +{GYM_PAY} c   endurance {self.stam_max():.1f}/{STAM_CAP:.0f}   (full sprint ~{7.2 * self.stam_max() / 19.0:.0f} m)", 5.0)


def _update_vitals_v8(self, dt: float) -> None:
    v, p = self.vitals, self.player
    end = max(1.0, self.stam_max() / STAM_BASE)
    before, was = v.stamina, v.exhausted
    v.update(dt, p.running and p.speed > 1.0)
    if end > 1.0 and v.stamina < before:                 # running drains stamina `end` times slower
        v.stamina = before - (before - v.stamina) / end
        v.exhausted = True if v.stamina <= 0 else (False if v.stamina > 30 else was)


def _activity_press_v8(self) -> None:
    a = self.activity
    if a["kind"] != "lift":
        return _prev_press(self)                          # treadmill: E steps off, as before
    if self.clock - a.get("last", -9.0) < REP_GAP:
        self.hud.note("Slow down - control the rep.", good=False)
        return
    a["last"] = self.clock
    a["reps"] += 1
    a["lift"] = 0.6
    self.inv.flags["reps"] = int(self.inv.flags.get("reps", 0)) + 1
    self.vitals.stamina = max(0.0, self.vitals.stamina - 4.0)
    self.sfx.play("weights", 0.8)
    if self.inv.flags["reps"] >= 30:
        self.award("iron")
    if a["reps"] >= 10:                                   # a set of 10 pays, then the station needs a rest
        self.activity = None
        self._lift_until = self.clock + LIFT_REST
        self.gym_reward("Set")
    elif self.vitals.stamina <= 0:
        self.activity = None
        self.hud.note("You're spent.", good=False)


def _use_weights_v8(self, it) -> None:
    left = getattr(self, "_lift_until", 0.0) - self.clock
    if left > 0:
        self.hud.note(f"Rest your arms: {left:.0f}s until the next paid set.", good=False)
        return
    _prev_weights(self, it)


def _use_treadmill_v8(self, it) -> None:
    left = getattr(self, "_tread_until", 0.0) - self.clock
    if not it.data["on"] and left > 0:
        self.hud.note(f"Catch your breath: {left:.0f}s until the next paid run.", good=False)
        return
    _prev_tread(self, it)
    a = self.activity
    if a is not None and a["kind"] == "treadmill":
        a["total"] = 10 ** 6                              # switches off the old $1 per 30 s payout
        self.hud.note("Run a full 60 s to get paid.   E steps off.")


def _activity_player_v8(self, dt: float) -> None:
    _prev_aplayer(self, dt)
    a = self.activity
    if a is not None and a["kind"] == "treadmill" and a["t"] >= 60.0:
        a["it"].data["on"] = False
        self.activity = None
        self._tread_until = self.clock + TREAD_REST
        self.gym_reward("Run")


_prev_press, _prev_weights, _prev_tread, _prev_aplayer = Game._activity_press, Game._use_weights, Game._use_treadmill, Game._activity_player
Game.stam_max, Game.gym_reward, Game._update_vitals = stam_max, gym_reward, _update_vitals_v8
Game._activity_press, Game._use_weights, Game._use_treadmill, Game._activity_player = _activity_press_v8, _use_weights_v8, _use_treadmill_v8, _activity_player_v8

# ---- GPS direction: R cycles facing / +Z / -Z ------------------------------------------------------------------
def _gps_fwd(self) -> float:
    d = getattr(self, "gps_dir", 0)
    if d:
        return float(d)
    return 1.0 if math.cos(math.radians(self.g.yaw)) >= 0 else -1.0


_prev_pk = PhoneUI.key


def _pk_v8(self, k):
    if self.page == 3 and k == "r":
        self.gps_dir = {0: 1, 1: -1, -1: 0}[getattr(self, "gps_dir", 0)]
        self.pmsg = {0: "direction: whichever way you face", 1: "direction: +Z (forward)", -1: "direction: -Z (backward)"}[self.gps_dir]
        self.redraw()
        return
    _prev_pk(self, k)


PhoneUI.gps_fwd, PhoneUI.key = _gps_fwd, _pk_v8



# ---- v9: reception, waiting line, mochi rounds, bar-chef bell queue, staff pacing --------------------------------
PALETTES["jp_host"] = dict(coat=(0.62, 0.12, 0.18), legs=(0.10, 0.08, 0.12), skin=(0.92, 0.80, 0.70), em=0.08)
MOCHI = (("Sakura", (1.0, 0.78, 0.86)), ("Matcha", (0.55, 0.82, 0.45)), ("Red Bean", (0.62, 0.22, 0.24)),
         ("Yuzu", (1.0, 0.88, 0.35)), ("Black Sesame", (0.22, 0.22, 0.28)), ("Strawberry", (1.0, 0.50, 0.58)))
CATNAME = dict(table="a table", ramen="the ramen bar", belt="the kaiten belt")
LINE_TIMEOUT, MOCHI_GAP, LEAVE_ODDS, NUDGE_MIN_WAIT, SEAT_MIN_STAY = 300.0, 60.0, 6, 3, 100.0


class V9Builders:
    def _jp_reception(self):
        ch, z0, xo = self.ch, self.ch.z0, self.ch.xoff
        self.par = ch.detail
        red = C(1.0, 0.08, 0.08)
        ch.extra["jp_wait"], ch.extra["jp_zone"] = [[], []], []
        for d in (0, 1):
            mz = (lambda z: z) if d == 0 else (lambda z: 24.0 - z)
            za, zb = sorted((mz(2.2), mz(5.0)))
            x0, x1 = 0.3, 4.1
            ch.extra["jp_zone"].append((xo + x0 - 0.4, xo + x1 + 0.4, z0 + za - 0.4, z0 + zb + 0.4))
            for (cx, cz, sx, sz) in (((x0 + x1) / 2, za, x1 - x0, 0.07), ((x0 + x1) / 2, zb, x1 - x0, 0.07),
                                     (x0, (za + zb) / 2, 0.07, zb - za), (x1, (za + zb) / 2, 0.07, zb - za)):
                self.box(cx, 0.03, z0 + cz, sx, 0.012, sz, color=red, emissive=1.4)          # the red line
            for zc_ in (2.9, 4.4):
                self.block(3.3, 0.22, z0 + mz(zc_), 0.5, 0.44, 1.3, "wood", color=C(0.34, 0.15, 0.10))
                self.box(3.5, 0.6, z0 + mz(zc_), 0.1, 0.4, 1.3, "wood", color=C(0.30, 0.13, 0.09))
            spots = ch.extra["jp_wait"][d]
            for zs in (2.55, 3.25, 4.05, 4.75):                                             # bench seats
                it = self.inter("sit", 3.25, 0.5, z0 + mz(zs), 1.3, "Sit and wait", -90.0)
                spots.append(dict(x=it.x, z=it.z, sit=True, it=it, y=0.5, yaw=-90.0, taken=None))
            for sx_, sz_ in ((1.0, 2.8), (1.9, 2.8), (1.0, 3.6), (1.9, 3.6), (1.0, 4.4), (1.9, 4.4)):   # standing spots
                spots.append(dict(x=xo + sx_, z=z0 + mz(sz_), sit=False, it=None, y=0.0, yaw=90.0, taken=None))
            if d == 1:                                                                      # south reception counter
                self.block(4.0, 0.55, z0 + 22.3, 1.3, 1.1, 0.7, "wood", color=C(0.30, 0.15, 0.09))
                self.box(4.0, 1.12, z0 + 22.3, 1.4, 0.05, 0.8, "wood", color=C(0.62, 0.40, 0.22))
            self.block(4.55, 0.5, z0 + mz(0.78), 0.6, 1.0, 0.4, "wood", color=C(0.22, 0.10, 0.07))   # mochi cabinet
            self.quad(4.55, 1.2, z0 + mz(1.02), 0.5, 0.2, "+z" if d == 0 else "-z",
                      BANK.sign("MOCHI", (255, 214, 190), (60, 14, 18), 192, 72), emissive=0.9)
            self.quad(4.0, 2.4, z0 + mz(1.7), 1.6, 0.4, "+z" if d == 0 else "-z",
                      jp_sign("受付", "RECEPTION", (255, 226, 190), (60, 10, 14), 384, 96), emissive=1.0)


ChunkBuilder._jp_reception = V9Builders._jp_reception
_bj9 = ChunkBuilder._build_japan


def _build_japan_v9(self):
    _bj9(self)
    self._jp_reception()


ChunkBuilder._build_japan = _build_japan_v9


# ---- pacing: walk speed and work tempo per role ---------------------------------------------------------------------
def _stat(self):
    c = getattr(self.rest, "crowd", 0.0)
    if isinstance(self, JpCleaner):
        return 1.0, 0.6                                     # slow enough that you can out-wash him
    if isinstance(self, JpChef):
        p, t = {"ramen": (1.0, 1.0), "mid": (1.25, 0.88), "sushi": (1.15, 0.8)}[self.which]
    elif isinstance(self, JpServer):
        p, t = 1.3, 0.85
    elif isinstance(self, JpHost):
        p, t = 1.25, 1.0
    elif isinstance(self, JpHiro):
        p, t = 1.0, 0.9
    else:
        return 1.0, 1.0
    return p * (1 + 0.3 * c), t * (1 + 0.3 * c)             # rush: everybody hustles


_walk_prev, _upd_prev = JpActor.walk, JpActor.update


def _walk_v9(self, x, z, speed=1.7):
    return _walk_prev(self, x, z, speed * _stat(self)[0])


def _upd_v9(self, dt, w):
    return _upd_prev(self, dt * _stat(self)[1], w)


JpActor.walk, JpActor.update = _walk_v9, _upd_v9


# ---- host / reception ------------------------------------------------------------------------------------------------
class JpWaitUI(UiBase):
    def __init__(self, g, host) -> None:
        super().__init__(g)
        self.host = host
        rest = g._jp_rest()
        n = sum(1 for e in rest.waitlist if e["seat"] is None) if rest else 0
        self.quad((0, 0), (0.92, 0.46), C(1.0, 0.4, 0.3, 0.35), z=0.22)
        self.quad((0, 0), (0.90, 0.44), C(0.10, 0.04, 0.05, 0.96), z=0.21)
        self.text("SAKURA-YA  -  WAITING LIST", (-0.42, 0.16), 1.0, C(1.0, 0.85, 0.55, 1))
        self.text(f"We're full right now.  {n} guest(s) ahead of you.", (-0.42, 0.08), 0.8)
        self.text("[1] a table     [2] a ramen bar stool     [3] a kaiten belt stool", (-0.42, 0.0), 0.8, C(1.0, 0.95, 0.6, 1))
        self.text("Stay inside the RED LINE on the floor. If you leave it, you drop off the list.", (-0.42, -0.08), 0.68, C(1.0, 0.6, 0.5, 1))
        self.text("N / E / Esc: no thanks", (-0.42, -0.16), 0.65, C(0.7, 0.7, 0.75, 0.9))

    def key(self, k: str) -> None:
        g = self.g
        if k in ("1", "2", "3"):
            rest = g._jp_rest()
            cat = ("table", "ramen", "belt")[int(k) - 1]
            msg = rest.player_join(cat, self.host) if rest else "no restaurant here"
            g.hud.note(msg, 4.0)
            g.close_ui()
        elif k in ("n", "e", "escape"):
            g.close_ui()


class JpHost(JpActor):
    def __init__(self, g, ch, pos, rng, rest, door) -> None:
        JpActor.__init__(self, g, ch, pos, rng, "host", "jp_host", "server", rest, "talk")
        self.door, self.post = door, pos
        self.post_yaw = 0.0 if door == 0 else 180.0
        self.yaw = self.post_yaw
        self.rq, self.next_round, self.last_fresh = None, 20.0, -99.0
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def on_talk(self) -> None:
        g, rest = self.g, self.rest
        e = rest.p_entry
        if e is not None and not e["dead"]:
            if e["seat"] is None:
                order = [x for x in rest.waitlist if x["seat"] is None]
                g.hud.speak(self, f"You're number {order.index(e) + 1} for {CATNAME[e['cat']]}. Please stay inside the red line.")
            else:
                g.hud.speak(self, f"Your seat is ready: {rest.seat_name(e['seat'])}. Douzo!")
            return
        fc = rest.free_counts()
        if sum(fc.values()) > 0:
            g.hud.speak(self, f"Free now: tables {fc['table']}, ramen bar {fc['ramen']}, belt {fc['belt']}. Sit anywhere you like. Mochi comes around every minute!")
        else:
            g.hud.speak(self, "I'm so sorry, we're completely full. Would you like to join the waiting list?")
            g.open_ui(JpWaitUI(g, self))

    def brain(self):
        rest = self.rest
        while True:
            job = rest.host_job(self)
            if job is None:
                yield from self.go(self.post[0], self.post[1], 1.6)
                yield from self.turn(self.post_yaw, 0.3)
                yield from self.hold(self.rng.uniform(1.0, 2.5), None)
            else:
                yield from self.serve_mochi(job[1])

    def serve_mochi(self, rec):
        rest, g = self.rest, self.g
        flav = rest.next_flavor()
        yield from self.go(self.post[0], self.post[1], 1.8)
        yield from self.turn(self.post_yaw, 0.2)
        yield from self.hold(0.7, "place", 0.7)
        self.set_carry("mochi")
        jp_say(g, self.x, self.z, "plate", 0.5, 14.0)
        cs = rec["clear_spot"]
        yield from self.go(cs[0], cs[1], 1.9)
        if not rest.seat_ok(rec):
            self.set_carry(None)
            return
        yield from self.turn(math.degrees(math.atan2(rec["slot"][0] - self.x, rec["slot"][1] - self.z)), 0.25)
        yield from self.hold(0.7, "place", 0.7)
        self.set_carry(None)
        rest.give_mochi(rec, flav)
        self.say(f"Mochi, douzo!  ({flav[0]})", 3.0)
        jp_say(g, self.x, self.z, "plate", 0.7)


# ---- restaurant: seats, line, hosts, mochi, nudges ----------------------------------------------------------------------
def _r_free_seats(self, cat=None):
    return [r for r in self.ch.extra["jp_seats"] if r["occ"] is None and not r["it"].occupied and r.get("resv") is None
            and (cat in (None, "any") or r["kind"] == cat)]


def _r_free_counts(self):
    return {c: len(self.free_seats(c)) for c in ("table", "ramen", "belt")}


def _r_seat_name(self, rec):
    if rec["kind"] == "table":
        return f"table {rec['tid'] + 1}"
    return f"{'ramen bar' if rec['kind'] == 'ramen' else 'belt'} stool {rec['sid'] + 1}"


def _r_unresv(self, rec):
    rec["resv"] = None
    rec["it"].label = "Sit"


def _r_in_zone(self, door, x, z):
    x0, x1, za, zb = self.ch.extra["jp_zone"][door]
    return x0 <= x <= x1 and za <= z <= zb


def _r_join_line(self, who, cat, door):
    spots = self.ch.extra["jp_wait"][door]
    used = sum(1 for sp in spots if sp["taken"] is not None) + sum(1 for e in self.waitlist if e["who"] == "player" and e["door"] == door)
    if used >= len(spots):
        return None
    e = dict(who=who, cat=cat, door=door, t0=self.clock, seat=None, spot=None, dead=False, t_res=0.0)
    self.waitlist.append(e)
    return e


def _r_player_join(self, cat, host):
    if self.p_entry is not None:
        return "You're already on the list."
    e = self.join_line("player", cat, host.door)
    if e is None:
        return "The waiting area is full."
    self.p_entry = e
    n = sum(1 for x in self.waitlist if x["seat"] is None)
    return f"On the list for {CATNAME[cat]}  (#{n}).  Stay inside the red line."


def _r_take_wait_spot(self, e):
    for sp in self.ch.extra["jp_wait"][e["door"]]:
        if sp["taken"] is None and not (sp["it"] is not None and sp["it"].occupied):
            sp["taken"], e["spot"] = e, sp
            if sp["it"] is not None:
                sp["it"].occupied = True
            return sp
    return None


def _r_release_spot(self, e):
    sp = e.get("spot")
    if sp is not None:
        sp["taken"] = None
        if sp["it"] is not None:
            sp["it"].occupied = False
        e["spot"] = None


def _r_remove_entry(self, e):
    if e in self.waitlist:
        self.waitlist.remove(e)
    self.release_spot(e)
    if e is self.p_entry:
        self.p_entry = None


def _r_drop_entry(self, e):
    e["dead"] = True
    rec = e.get("seat")
    if rec is not None and rec.get("resv") is e:
        self.unresv(rec)
    self.remove_entry(e)


def _r_assign_seats(self):
    for e in list(self.waitlist):
        if e["seat"] is not None or e["dead"]:
            continue
        cands = self.free_seats(e["cat"])
        if not cands:
            continue
        edge = self.z0 if e["door"] == 0 else self.z0 + CH
        rec = min(cands, key=lambda r: abs(r["z"] - edge))
        rec["resv"], e["seat"], e["t_res"] = e, rec, self.clock
        mine = e["who"] == "player"
        rec["it"].label = "Sit  (reserved for you)" if mine else "Reserved - waiting list"
        h = next((h for h in self.hosts if h.door == e["door"] and not h.dead), None)
        if mine:
            self.g.hud.note(f"Your seat is ready: {self.seat_name(rec)}.  Walk there and press E  (75 s).", 6.0)
            self.g.sfx.play("chime", 0.7)
            if h is not None:
                h.say(f"Your seat is ready: {self.seat_name(rec)}! Douzo!", 4.0)
        elif h is not None:
            h.say("Next guest! Your seat is ready, douzo!", 3.0)


def _r_nudge(self):
    waiting = [e for e in self.waitlist if e["seat"] is None and not e["dead"]]
    if len(waiting) < NUDGE_MIN_WAIT:                         # only a real rush asks people to leave
        return
    for cat in ("table", "ramen", "belt"):
        need = sum(1 for e in waiting if e["cat"] == cat)
        if not need or self.free_seats(cat):
            continue
        pend = sum(1 for c in self.customers if c.nudged and not c.gone and c.srec.get("kind") == cat and c.srec.get("occ") is c)
        if pend >= need:
            continue
        ok = [c for c in self.customers if not c.gone and not c.dead and not c.nudged and not c.srec.get("dummy")
              and c.srec.get("occ") is c and c.srec["kind"] == cat and c.state_tag == "idle" and c.meals >= 1
              and c.ordered is None and c.hips == "sit" and self.clock - c.seat_t > SEAT_MIN_STAY]
        if not ok:
            continue
        c = min(ok, key=lambda c: c.seat_t)
        c.nudged = True
        h = min((h for h in self.hosts if not h.dead), key=lambda h: math.hypot(h.x - c.x, h.z - c.z), default=None)
        if h is not None:
            h.say("Sorry to rush you - we have guests waiting. Thank you for dining with us!", 4.0)
        c.say("Ah, of course. Gochisousama!", 3.0)


def _r_player_line(self):
    e, g = self.p_entry, self.g
    if e is None:
        return
    p = g.player
    if e["dead"]:
        self.p_entry = None
    elif e["seat"] is None:
        if not self.in_zone(e["door"], p.x, p.z):
            self.drop_entry(e)
            g.hud.note("You left the red line - you're off the waiting list.", good=False)
    elif self.clock - e["t_res"] > 75.0:
        self.drop_entry(e)
        g.hud.note("Your reserved seat was given to someone else.", good=False)


def _r_seat_ok(self, rec):
    if self.g.player.seat is rec["it"]:
        ok = True
    else:
        o = rec["occ"]
        ok = o is not None and not o.dead and not o.gone and o.hips == "sit" and not o.srec.get("dummy")
    return ok and not any(s.get("rec") is rec and s["key"] == "mochi" for s in self.ch.served)


def _r_next_flavor(self):
    self.flav_i += 1
    return MOCHI[(self.flav_i - 1) % len(MOCHI)]


def _r_host_job(self, host):
    mid = self.z0 + 12.0
    mine = [r for r in self.ch.extra["jp_seats"] if (r["z"] < mid) == (host.door == 0) and self.seat_ok(r)]
    if not mine:
        host.rq = None
        return None
    fresh = [r for r in mine if r.get("fresh")]
    if fresh and self.clock - host.last_fresh > 6.0:          # someone just sat down: mochi first
        r = min(fresh, key=lambda r: math.hypot(r["x"] - host.x, r["z"] - host.z))
        r["fresh"], host.last_fresh = False, self.clock
        return ("mochi", r)
    if host.rq is None:
        if self.clock < host.next_round:
            return None
        host.rq = sorted(mine, key=lambda r: r["z"] if host.door == 0 else -r["z"])
    while host.rq:
        r = host.rq.pop(0)
        if self.seat_ok(r) and self.clock - r.get("mochi_t", -99.0) > 40.0:
            return ("mochi", r)
    host.rq, host.next_round = None, self.clock + MOCHI_GAP   # round finished: next one in a minute
    return None


def _r_give_mochi(self, rec, flav):
    x, z = rec["slot"]
    base = {"table": 0.80, "ramen": 1.10, "belt": 1.08}[rec["kind"]]
    root = ur.Entity(parent=self.ch.detail, position=(x - self.xo, base, z + 0.26))
    mk(root, "cube", (0, 0.01, 0), (0.22, 0.02, 0.22), color=C(0.97, 0.97, 0.95), **self.kw)
    mk(root, "sphere", (0, 0.07, 0), (0.12, 0.09, 0.12), color=C(*flav[1]), emissive=0.25, **self.kw)
    rec["mochi_t"] = self.clock
    if self.g.player.seat is rec["it"]:
        self.ch.served.append(dict(key="mochi", ent=root, x=x, z=z + 0.26, rec=rec, t=self.clock))
        self.g.hud.note(f"{flav[0]} mochi from the host.  Press T to eat it, U to bag it.", 4.5)
        return
    cust = rec["occ"]

    def eat():
        if cust is not None and not cust.gone and self.rng.random() < 0.5:
            cust.say("Mmm, mochi!", 2.4)
        self.finish_dish(rec, "mochi", root)
    self.timers.append([self.rng.uniform(7.0, 13.0), eat])


def _r_spawn_arrival(self):
    alive = [c for c in self.customers if not getattr(c, "gone", False)]
    if len(alive) + len(self.waitlist) >= 58:
        return
    d = 0 if self.rng.random() < 0.5 else 1
    queue = any(e["seat"] is None and not e["dead"] for e in self.waitlist)
    free = [] if queue else self.free_seats()
    if free:
        rec = self.rng.choice(free)
    else:                                                      # truly no seat: this guest has to line up (or leave)
        cat = self.rng.choices(("table", "ramen", "belt"), (0.45, 0.30, 0.25))[0]
        rec = dict(kind=cat, cat=cat, dummy=True, x=self.xo, z=self.z0 + JP_L["doors_z"][d], y=0.0, yaw=0.0,
                   it=SimpleNamespace(occupied=False, label="Sit"), occ=None, slot=(0.0, 0.0), pile=(0.0, 0.0),
                   clear_spot=(0.0, 0.0), bell=(0.0, 0.0))
    c = JpCustomer(self.g, self.ch, self.rng, self, rec, arrive=d)
    self.customers.append(c)
    self.ch.npcs.append(c)
    c.fig.root.enabled = self.ch.shown


def _r_retire(self, cust):
    cust.gone = True
    if getattr(cust, "entry", None) is not None:
        self.remove_entry(cust.entry)
    if getattr(cust, "no_replace", False):                     # never sat down: no replacement guest
        if cust in self.customers:
            self.customers.remove(cust)
        if cust in self.ch.npcs:
            self.ch.npcs.remove(cust)
        REAPER.push(cust.fig.root)
    else:
        self.retired.append(cust)


def _r_v9_tick(self, dt):
    for e in list(self.waitlist):
        w = e["who"]
        if w != "player" and (getattr(w, "dead", False) or getattr(w, "gone", False)):
            self.drop_entry(e)
    self.assign_t -= dt
    if self.assign_t <= 0:
        self.assign_t = 0.8
        self.assign_seats()
    self.nudge_t -= dt
    if self.nudge_t <= 0:
        self.nudge_t = 3.0
        self.nudge()
    self.player_line()
    waiting = sum(1 for e in self.waitlist if e["seat"] is None and not e["dead"])
    self.crowd = min(1.0, waiting / 4.0)
    self.rush = max(self.rush, self.crowd)
    self.next_wave -= dt
    if self.next_wave <= 0:                                    # a rush-hour wave
        self.next_wave = self.rng.uniform(240.0, 480.0)
        self.burst += self.rng.randint(12, 18)
    self.arr_t -= dt
    if self.arr_t <= 0:
        if self.burst > 0:
            self.burst -= 1
            self.arr_t = self.rng.uniform(2.0, 4.0)
        else:
            self.arr_t = self.rng.uniform(30.0, 70.0)
        self.spawn_arrival()


_rest_init, _tp_prev, _pc_prev, _staff_prev = JpRestaurant.__init__, JpRestaurant._tick_pipeline, JpRestaurant._player_changed, JpRestaurant.staff


def _rest_init_v9(self, g, ch):
    _rest_init(self, g, ch)
    self.plates = {"L": 10, "R": 6}                            # tighter plate supply: the kitchens can run dry
    self.waitlist, self.hosts, self.crowd, self.p_entry = [], [], 0.0, None
    self.burst, self.arr_t, self.next_wave = self.rng.choice((0, 14, 20)), 4.0, self.rng.uniform(150.0, 400.0)
    self.flav_i, self.assign_t, self.nudge_t = 0, 0.0, 2.0


def _tp_v9(self, dt):
    _tp_prev(self, dt)
    try:
        self.v9_tick(dt)
    except Exception:
        log.exception("restaurant v9 tick failed")


def _pc_v9(self, rec):
    _pc_prev(self, rec)
    if rec is None:
        return
    rec["fresh"] = True
    e = rec.get("resv")
    if e is not None and e["who"] == "player":
        self.remove_entry(e)
        self.unresv(rec)
    elif self.p_entry is not None:
        self.drop_entry(self.p_entry)


def _staff_v9(self):
    return _staff_prev(self) + [h for h in self.hosts if not h.dead]


for _n, _f in (("free_seats", _r_free_seats), ("free_counts", _r_free_counts), ("seat_name", _r_seat_name), ("unresv", _r_unresv),
               ("in_zone", _r_in_zone), ("join_line", _r_join_line), ("player_join", _r_player_join), ("take_wait_spot", _r_take_wait_spot),
               ("release_spot", _r_release_spot), ("remove_entry", _r_remove_entry), ("drop_entry", _r_drop_entry),
               ("assign_seats", _r_assign_seats), ("nudge", _r_nudge), ("player_line", _r_player_line), ("seat_ok", _r_seat_ok),
               ("next_flavor", _r_next_flavor), ("host_job", _r_host_job), ("give_mochi", _r_give_mochi),
               ("spawn_arrival", _r_spawn_arrival), ("retire", _r_retire), ("v9_tick", _r_v9_tick)):
    setattr(JpRestaurant, _n, _f)
JpRestaurant.__init__, JpRestaurant._tick_pipeline, JpRestaurant._player_changed, JpRestaurant.staff = _rest_init_v9, _tp_v9, _pc_v9, _staff_v9


# ---- bell queue: first to ring is first served; bar chef scans every 10 s; mid chefs only cook ---------------------------
def _r_counter_wait(self, cust, rec):
    if not any(r["cust"] is cust and r["state"] in ("waiting", "claimed") for r in self.counter_reqs):
        self.counter_reqs.append(dict(state="waiting", rec=rec, cust=cust, t=self.clock))


def _r_player_call_chef(self, rec):
    if not any(r["cust"] is None and r["state"] in ("waiting", "claimed") for r in self.counter_reqs):
        self.counter_reqs.append(dict(state="waiting", rec=rec, cust=None, t=self.clock))


def _r_next_ask(self, chef):
    c = [r for r in self.counter_reqs if r["state"] == "waiting" and self._req_ok(r)]
    self.counter_reqs = [r for r in self.counter_reqs if r["state"] in ("waiting", "claimed")]
    if not c:
        return None
    r = min(c, key=lambda r: r.get("t", 0.0))                  # ring order, not seat order
    r["state"] = "claimed"
    return r


def _r_pool_of(self, chef):
    if chef.which == "ramen" and not any(it["state"] == "queued" and not it["tk"]["void"] for it in self.bar_items):
        return self.mid_items                                  # nobody at the bar needs food: the bar chef helps the kitchen
    return self.bar_items if chef.which == "ramen" else self.mid_items


JpRestaurant.counter_wait, JpRestaurant.player_call_chef = _r_counter_wait, _r_player_call_chef
JpRestaurant.next_ask, JpRestaurant.pool_of = _r_next_ask, _r_pool_of


def _chef_has_work(self):
    if self.which == "sushi":
        return False
    r = self.rest
    if r.pool_pending(self):
        return True
    return self.which == "ramen" and r.asks_pending() and r.clock >= getattr(self, "next_scan", 0.0)


def _chef_brain(self):
    rest = self.rest
    if self.which == "sushi":
        yield from self.sushi_loop()
        return
    self.next_scan, self.pend = rest.clock + 3.0, None
    while True:
        if self.which == "ramen" and self.pend is None and rest.clock >= self.next_scan:
            self.next_scan = rest.clock + 10.0                 # scan the bar every 10 s
            self.pend = rest.next_ask(self)
        if self.pend is not None and not rest.bar_in_progress():
            a, self.pend = self.pend, None
            yield from self.ask_counter(a)
            self.next_scan = min(self.next_scan, rest.clock + 1.5)
            continue
        it = rest.claim_item(self)
        if it is not None:
            yield from self.cook_item(it)
            continue
        if self.pend is not None:
            a, self.pend = self.pend, None
            yield from self.ask_counter(a)
            continue
        if rest.pool_pending(self):
            yield from self.wait_blocked()
        else:
            yield from self.idle_routine()


JpChef.has_work, JpChef.brain = _chef_has_work, _chef_brain


# ---- customers: line-up, 1/6 leave chance, polite nudges ------------------------------------------------------------------
_cu_init = JpCustomer.__init__


def _cu_init_v9(self, g, ch, rng, rest, rec, arrive=None):
    _cu_init(self, g, ch, rng, rest, rec, arrive)
    self.nudged, self.entry, self.seat_t = False, None, rest.clock
    self.cat = rec.get("cat", rec["kind"])
    self.no_replace = bool(rec.get("dummy"))


def _cu_sat(self):
    self.seat_t, self.no_replace = self.rest.clock, False
    self.srec["fresh"] = True


def _cu_idle(self, t, anim=None):
    e, nxt = 0.0, self.rng.uniform(3.0, 8.0)
    while e < t:
        if self.nudged and self.state_tag == "idle":
            return
        e += self.dt
        nxt -= self.dt
        if nxt <= 0:
            nxt = self.rng.uniform(5.0, 12.0)
            if self.dist() < 16 and self.rng.random() < 0.18:
                self.say(self.rng.choice(JP_WAIT if self.state_tag == "queue" else JP_CHAT), 3.0)
        sway = math.sin(self.rest.clock * 0.6) * 4.0
        if anim == "hand":
            self.pose(rs=-165, re=-10, rz=-8 + 12 * math.sin(e * 7), hx=0, hy=None)
        elif self.hips == "sit":
            self.pose(ls=-30, le=-40, rs=-30, re=-40, hx=sway * 0.3)
        else:
            self.pose()
        yield


def _cu_exit_door(self, d):
    rest, rng = self.rest, self.rng
    inward = 1.0 if d == 0 else -1.0
    self.hips, self.y = None, 0.0
    yield from self.walk(self.ch.xoff + rng.uniform(-0.4, 0.4), self.ch.z0 + JP_L["doors_z"][d] + inward * 1.6, 1.5)
    rest.npc_door(d, False, self)
    yield from self.walk(self.ch.xoff + rng.uniform(-0.4, 0.4), self.ch.z0 + JP_L["doors_z"][d] - inward * 2.8, 1.5)
    rest.retire(self)


def _cu_line_up(self):
    rest, rng, d = self.rest, self.rng, self.arrive
    inward = 1.0 if d == 0 else -1.0
    rest.npc_door(d, True, self)
    yield from self.walk(self.ch.xoff + rng.uniform(-0.4, 0.4), self.ch.z0 + JP_L["doors_z"][d] + inward * 1.6, 1.5)
    if rng.random() < 0.40:                                    # some guests simply leave when it's full
        self.say(rng.choice(("Full? Maybe another time.", "So busy... I'll come back later.", "No seats? Ah, never mind.")), 3.2)
        yield from self.idle(1.4)
        yield from self.exit_door(d)
        return False
    e = rest.join_line(self, self.cat, d)
    sp = rest.take_wait_spot(e) if e is not None else None
    if sp is None:
        if e is not None:
            rest.drop_entry(e)
        self.say("Too crowded. Next time!", 2.8)
        yield from self.idle(1.0)
        yield from self.exit_door(d)
        return False
    self.entry = e
    self.say("Table for one, please!" if self.cat == "table" else "Is there room at the bar?", 2.8)
    yield from self.walk(sp["x"], sp["z"], 1.4)
    self.hips, self.y, self.yaw = ("sit" if sp["sit"] else None), (sp["y"] - 0.72 if sp["sit"] else 0.0), sp["yaw"]
    self.fig.place(self.x, self.y, self.z, self.yaw)
    self.state_tag = "queue"
    while e["seat"] is None:
        if e["dead"] or rest.clock - e["t0"] > LINE_TIMEOUT:
            if not e["dead"]:
                rest.drop_entry(e)
            self.say("I'll go to a different restaurant. Thank you anyway!", 3.6)
            yield from self.idle(1.8)
            yield from self.exit_door(d)
            return False
        yield from self.idle(0.3)
    rec = e["seat"]
    rest.remove_entry(e)
    rest.unresv(rec)
    rec["occ"], rec["it"].occupied, self.srec = self, True, rec
    self.hips, self.y = None, 0.0
    ex, ez = self.exit_point()
    yield from self.go(ex, ez, 1.7)
    yield from self.turn(rec["yaw"], 0.3)
    self.x, self.z, self.y, self.yaw, self.hips = rec["x"], rec["z"], rec["y"] - 0.72, rec["yaw"], "sit"
    self.fig.place(self.x, self.y, self.z, self.yaw)
    self.state_tag = "idle"
    return True


def _cu_brain(self):
    rest, rng = self.rest, self.rng
    first = self.arrive is None
    if self.arrive is not None:
        if self.srec.get("dummy"):
            ok = yield from self.line_up()
            if not ok:
                return
        else:
            yield from self.enter_and_sit()
    self.sat()
    if self.arrive is not None:
        yield from self.idle(rng.uniform(4.0, 10.0))
    while True:
        rec = self.srec
        if first:
            yield from self.idle(rng.uniform(10.0, 80.0) if rec["kind"] == "table" else rng.uniform(2.0, 30.0))
        else:
            yield from self.idle(max(0.0, self.reorder_at - rest.clock) + rng.uniform(1.0, 18.0))
            if not self.nudged:
                self.say(rng.choice(JP_REORDER), 2.8)
        first = False
        if self.nudged:
            yield from self.leave()
            return
        gave_up = False
        if rec["kind"] == "table":
            gave_up = yield from self.order_table()
        elif rec["kind"] == "ramen":
            gave_up = yield from self.order_counter()
        else:
            yield from self.order_belt()
        self.state_tag = "idle"
        self.reorder_at = rest.clock + rng.uniform(32.0, 58.0)
        if gave_up and rng.random() < 0.8:
            yield from self.leave(annoyed=True)
            return
        if self.nudged:
            yield from self.leave()
            return
        if self.meals >= 1:
            if rng.randrange(LEAVE_ODDS) == 0:                 # 1 in 6 leave after a meal
                yield from self.leave()
                return
            self.say(rng.choice(JP_STAY), 3.0)


JpCustomer.__init__, JpCustomer.sat, JpCustomer.idle = _cu_init_v9, _cu_sat, _cu_idle
JpCustomer.exit_door, JpCustomer.line_up, JpCustomer.brain = _cu_exit_door, _cu_line_up, _cu_brain

# ---- hosts spawn with the restaurant; reserved seats are protected -----------------------------------------------------------
_japan_npcs_prev = _japan_npcs


def _japan_npcs(self) -> None:                                 # noqa: F811  (the v4 hook looks this name up at call time)
    _japan_npcs_prev(self)
    ch, g = self.ch, self.w.g
    rest = ch.extra.get("jp_rest")
    if rest is None:
        return
    rng = seeded(self.seed, "jp-host", ch.idx)
    for d in (0, 1):
        h = JpHost(g, ch, (ch.xoff + 3.8, ch.z0 + (0.98 if d == 0 else 23.02)), rng, rest, d)
        rest.hosts.append(h)
        ch.npcs.append(h)


_use_sit_prev = Game._use_sit


def _use_sit_v9(self, it):
    rec = it.data.get("jp") if isinstance(it.data, dict) else None
    if rec is not None:
        e = rec.get("resv")
        if e is not None and e["who"] != "player":
            self.hud.note("That seat is reserved for a guest from the waiting list.", good=False)
            return
    _use_sit_prev(self, it)


Game._use_sit = _use_sit_v9

# ---- v10: 3 plates per chair, Hiro helps wash in a shortage, cleaners hustle in rush -------------------------------
PLATES_PER_SEAT = 3
_rest_init_v9_prev = JpRestaurant.__init__


def _rest_init_v10(self, g, ch):
    _rest_init_v9_prev(self, g, ch)
    seats = ch.extra.get("jp_seats", [])
    nL = sum(1 for r in seats if r["kind"] != "belt")          # ramen bar + middle tables are served from the left cabinet
    nR = sum(1 for r in seats if r["kind"] == "belt")          # kaiten stools from the right cabinet
    self.plates = {"L": PLATES_PER_SEAT * nL, "R": PLATES_PER_SEAT * nR}
    self.plate_cap = dict(self.plates)


def _r_plate_short(self):
    cap = self.plate_cap
    return self.plates["L"] <= 0.25 * cap["L"] or self.plates["R"] <= 0.25 * cap["R"]


def _r_hiro_target(self):
    if self.rack <= 0:
        return None
    cap = self.plate_cap
    fl, fr = self.plates["L"] / max(1, cap["L"]), self.plates["R"] / max(1, cap["R"])
    if self.rack >= 4 or min(fl, fr) <= 0.25 or self.clock - self.last_deliver > 25.0:
        return "L" if fl <= fr else "R"
    return None


JpRestaurant.__init__ = _rest_init_v10
JpRestaurant.plate_short, JpRestaurant.hiro_target = _r_plate_short, _r_hiro_target


def _hurry(self):
    """0..1: how hard this worker should hustle (crowd, kitchen rush, or a plate shortage)."""
    r = self.rest
    return max(getattr(r, "crowd", 0.0), getattr(r, "rush", 0.0), 1.0 if r.plate_short() else 0.0)


_stat_v9 = _stat


def _stat(self):                                            # noqa: F811  (the v9 walkers look this name up at call time)
    if isinstance(self, JpCleaner):
        k = _hurry(self)
        return 1.0 * (1 + 0.5 * k), 0.6 * (1 + 1.0 * k)     # about 0.6 -> 1.2 work tempo
    if isinstance(self, JpHiro):
        k = _hurry(self)
        return 1.0 * (1 + 0.6 * k), 0.9 * (1 + 0.6 * k)
    return _stat_v9(self)


def jp_wash_batch(self, n):
    """Shared by the dishwasher and Hiro: scrub n dirty plates at the sink, stack them on the rack."""
    rest, L = self.rest, JP_L
    if n <= 0:
        return
    yield from self.walk(L["cleaner"][0] + self.ch.xoff, L["cleaner"][1] + self.ch.z0, 1.5)
    yield from self.turn(90.0)
    yield from self.hold(0.6, "place", 0.6)
    yield from self.walk(L["cleaner_sink"][0] + self.ch.xoff, L["cleaner_sink"][1] + self.ch.z0, 1.2)
    yield from self.turn(-90.0)
    for _ in range(n):
        e, nxt = 0.0, 0.0
        while e < 2.8:
            e += self.dt
            if e >= nxt:
                jp_say(self.g, self.x, self.z, "dish_water" if int(nxt / 2.0) % 2 == 0 else "dish_scrub", 0.8, 20.0)
                nxt += 1.7
            _an(self, "wash", e, 2.8)
            yield
        jp_say(self.g, self.x, self.z, "dish_clink", 0.6, 16.0)
    yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0, 1.3)
    yield from self.turn(-90.0)
    yield from self.hold(0.8, "place", 0.8)
    rest.rack_add(n)
    jp_say(self.g, self.x, self.z, "dish_stack", 0.7, 18.0)


def _cleaner_brain(self):
    rest, L = self.rest, JP_L
    while True:
        if rest.silver > 0 and rest.rack < JP_RACK_CAP:
            batch = 3 + int(4 * _hurry(self))                # bigger armfuls when it's busy
            yield from jp_wash_batch(self, rest.silver_take(batch))
        else:
            yield from self.walk(L["cleaner"][0] + self.ch.xoff, L["cleaner"][1] + self.ch.z0, 1.2)
            yield from self.turn(-90.0)
            yield from self.hold(self.rng.uniform(2.0, 4.0), "wipe")


def _hiro_brain(self):
    rest, L = self.rest, JP_L
    while True:
        wing = rest.hiro_target()
        if wing is None:
            if rest.plate_short() and rest.silver > 0 and rest.rack < JP_RACK_CAP:      # big shortage and nothing to deliver: wash
                if self.rng.random() < 0.3:
                    self.say("Plates are low - I'll wash!", 2.8)
                yield from jp_wash_batch(self, rest.silver_take(4))
                continue
            yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0)
            yield from self.turn(-90.0)
            yield from self.hold(self.rng.uniform(2.0, 4.0), "wipe" if self.rng.random() < 0.5 else None, brk=False)
            continue
        yield from self.walk(L["rack_stand"][0] + self.ch.xoff, L["rack_stand"][1] + self.ch.z0, 1.9)
        yield from self.turn(-90.0)
        n = rest.rack_take(12 if rest.plate_short() else 8)
        if n <= 0:
            continue
        yield from self.hold(0.9, "place", 0.9)
        if self.carry_ent is not None:
            ur.destroy(self.carry_ent)
        root = ur.Entity(parent=self.fig.root, position=(0, 1.02, 0.42))
        for i in range(min(n, 8)):
            mk(root, "cube", (0, 0.03 * i, 0), (0.30, 0.025, 0.30), color=C(0.97, 0.97, 0.95), **self.fig.k)
        self.carry_ent = root
        if self.rng.random() < 0.4:
            self.say(self.rng.choice(JP_HIRO), 2.8)
        yield from self.path(self.route_pts(wing))
        yield from self.turn(-90.0 if wing == "L" else 90.0)
        yield from self.hold(1.0, "place", 1.0)
        self.set_carry(None)
        rest.plates[wing] += n
        rest.last_deliver = rest.clock
        jp_say(self.g, self.x, self.z, "dish_stack", 0.8, 18.0)
        yield from self.path(list(reversed(self.route_pts(wing)[:-1])))


JpCleaner.brain, JpHiro.brain = _cleaner_brain, _hiro_brain

# ---- v10b: SAKURA-YA is a 3-chunk site: north forecourt / restaurant / south forecourt, a street lane on each side -----
JP_BW, JP_OW = 12.0, 17.0           # restaurant half-width, outer street-wall half-width


def jp_lane_floor(self):
    z0 = self.ch.z0
    cx = JP_BW + 0.5 + (JP_OW - JP_BW - 0.5) / 2
    for s in (-1, 1):
        self.box(s * cx, 0.012, z0 + CH / 2, JP_OW - JP_BW - 0.5, 0.02, CH, "asphalt", tile=(2, 8))
        self.merged([(s * cx, 0.026, z0 + 1.5 + 3.0 * k, 0.12, 0.01, 1.4) for k in range(8)], "white", color=C(0.9, 0.85, 0.5))


def jp_street_walls(self):
    z0, H = self.ch.z0, self.ch.h
    zc = z0 + CH / 2
    for s in (-1, 1):
        self.box(s * (JP_BW + 0.25), H / 2, zc, 0.5, H, CH, "panel", tile=(CH / 4, H / 4), color=C(0.30, 0.17, 0.11))   # the building's own wall
        self.solid(s * JP_BW, s * (JP_BW + 0.6), z0 - 0.1, z0 + CH + 0.1)
        self.box(s * (JP_OW + 0.25), H / 2, zc, 0.5, H, CH, "panel", tile=(CH / 4, H / 4), color=C(0.22, 0.13, 0.09))   # outer wall of the lane
        self.solid(s * JP_OW, s * (JP_OW + 0.6), z0 - 0.1, z0 + CH + 0.1)
    jp_lane_floor(self)


def _build_jpcourt(self):
    ch, z0, H, xo, HW = self.ch, self.ch.z0, self.ch.h, self.ch.xoff, self.ch.hw
    north = (ch.part == 0)                                   # north court: restaurant is at +z; south court: at -z
    zt = (lambda t: z0 + CH - t) if north else (lambda t: z0 + t)       # t = distance from the restaurant end of this chunk
    toward, away = (0.0, 180.0) if north else (180.0, 0.0)            # yaw facing the restaurant / facing the arrivals
    far_face = "-z" if north else "+z"
    wallc = C(0.30, 0.17, 0.11)
    self._shell("terrazzo", "panel", wallc, ceil_tex="panel")
    jp_lane_floor(self)
    # end wall toward the corridor, with an opening exactly as wide as the neighbouring chunk
    nidx = ch.idx - 1 if north else ch.idx + 1
    _nk, _np, nxo, nhw, _nh = self.w.geom(nidx)
    ol, orr = max(-HW, nxo - nhw - xo), min(HW, nxo + nhw - xo)
    ez = zt(CH - 0.2)
    if ol + HW > 0.2:
        self.block((-HW + ol) / 2, H / 2, ez, ol + HW, H, 0.4, "panel", color=wallc)
    if HW - orr > 0.2:
        self.block((orr + HW) / 2, H / 2, ez, HW - orr, H, 0.4, "panel", color=wallc)
    mid, wid = (ol + orr) / 2, max(0.5, orr - ol)
    self.box(mid, H - 0.5, ez, wid, 1.0, 0.4, "panel", color=wallc)
    self.quad(mid, H - 0.5, ez + (-0.22 if north else 0.22), min(wid, 5.0), 0.8, far_face,
              jp_sign("さくら屋", "SAKURA-YA", (255, 226, 190), (70, 12, 14), 640, 160, "sushi / ramen  -  straight ahead"), emissive=1.2, double=False)
    self.par = ch.detail
    # benches (sit spots), standing spots, reception desk with the host's tablet
    spots = []
    for bx0 in (-6.25, 6.25):
        self.block(bx0, 0.22, zt(8.5), 5.4, 0.44, 0.55, "wood", color=C(0.34, 0.15, 0.10))
        self.box(bx0, 0.78, zt(8.85), 5.4, 0.6, 0.08, "wood", color=C(0.28, 0.12, 0.08))
    for bx in (-8.2, -6.9, -5.6, -4.3, 4.3, 5.6, 6.9, 8.2):
        it = self.inter("sit", bx, 0.5, zt(8.5), 1.3, "Sit and wait", toward)
        spots.append(dict(x=it.x, z=it.z, sit=True, it=it, y=0.5, yaw=toward, taken=None))
    for sx in (-9.0, -7.5, -6.0, -4.5, -3.0, -1.5, 0.0, 1.5):
        spots.append(dict(x=xo + sx, z=zt(5.0), sit=False, it=None, y=0.0, yaw=toward, taken=None))
    dx, dt = 3.4, 5.0
    self.block(dx, 0.55, zt(dt), 2.2, 1.1, 0.8, "wood", color=C(0.30, 0.15, 0.09))
    self.box(dx, 1.12, zt(dt), 2.4, 0.05, 0.9, "wood", color=C(0.62, 0.40, 0.22))
    self.box(dx - 0.5, 1.17, zt(dt), 0.5, 0.04, 0.34, color=C(0.10, 0.10, 0.14))
    self.box(dx - 0.5, 1.20, zt(dt), 0.44, 0.01, 0.28, color=C(0.3, 0.9, 0.5), emissive=0.9)
    for px in (dx - 1.0, dx + 1.0):
        self.box(px, 1.5, zt(dt + 0.3), 0.07, 3.0, 0.07, "wood", color=C(0.16, 0.08, 0.05))
    self.quad(dx, 2.7, zt(dt + 0.3), 2.2, 0.55, far_face, jp_sign("受付", "RECEPTION", (255, 226, 190), (60, 10, 14), 384, 96), emissive=1.0, double=False)
    # sakura planters by the facade, bollards along both lanes
    for tx in (-9.5, 9.5):
        self.block(tx, 0.3, zt(2.2), 0.8, 0.6, 0.8, "wood", color=C(0.2, 0.1, 0.07))
        self.box(tx, 1.2, zt(2.2), 0.14, 1.4, 0.14, color=C(0.22, 0.12, 0.08))
        for ddx, ddy, ddz, r in ((0, 2.2, 0, 0.7), (0.4, 2.0, 0.2, 0.5), (-0.4, 2.05, -0.2, 0.5)):
            self.ball(tx + ddx, ddy, zt(2.2) + ddz, r, color=C(1.0, 0.74, 0.82), emissive=0.3)
    for t in (3.0, 6.0, 9.0, 12.0, 15.0, 18.0):
        for s in (-1, 1):
            self.block(s * 12.1, 0.35, zt(t), 0.6, 0.7, 0.6, "wood", color=C(0.25, 0.12, 0.08))
    self.inter("jpdoor", 0.0, 1.2, zt(1.3), 1.8, "Door  -  show your ticket", toward, data={"d": 0 if north else 1})    
    ch.extra["jp_court"] = dict(part=ch.part, north=north, idx=ch.idx, spots=spots, toward=toward, host_yaw=away,
                                desk=(xo + dx, zt(dt)), host=(xo + dx, zt(dt - 0.9)), door_xz=(xo, zt(-0.3)))

    kw = dict(lm=ch.lm, lamp=ch.lamp, xc=xo)                                   # mochi tray + kettle on the desk (shown according to stock)
    tray = ur.Entity(parent=ch.detail, position=(dx + 0.6, 1.16, zt(dt)))
    mk(tray, "cube", (0, 0, 0), (0.62, 0.03, 0.36), color=C(0.16, 0.08, 0.05), **kw)
    balls = [mk(tray, "sphere", (-0.24 + 0.12 * (i % 5), 0.06, -0.08 + 0.16 * (i // 5)), (0.1, 0.07, 0.1),
                color=C(*MOCHI[i % len(MOCHI)][1]), emissive=0.2, **kw) for i in range(10)]
    kettle = ur.Entity(parent=ch.detail, position=(dx + 0.05, 1.15, zt(dt)))
    mk(kettle, "sphere", (0, 0.14, 0), (0.28, 0.26, 0.28), color=C(0.55, 0.57, 0.62), **kw)
    mk(kettle, "cube", (0.18, 0.2, 0), (0.16, 0.04, 0.04), color=C(0.55, 0.57, 0.62), **kw)
    mk(kettle, "cube", (0, 0.31, 0), (0.18, 0.03, 0.03), color=C(0.2, 0.2, 0.22), **kw)
    ch.extra["jp_court"]["vis"] = dict(balls=balls, kettle=kettle)
   


ChunkBuilder._build_jpcourt = _build_jpcourt

_cc_prev = cctv_covered


def cctv_covered(seed: int, idx: int) -> bool:                  # noqa: F811  (no wall cameras in the forecourts)
    return _cc_prev(seed, idx) and chunk_kind(seed, idx) != "jpcourt"



# ---- v10c: outside host + tickets, locked doors during a rush, waiter admission ---------------------------------------
JP_TK_LETTER = dict(table="M", ramen="B", belt="S")          # ticket letters: Middle table / Bar (ramen) / Sushi belt
JP_FORCE_LOCK = False                                        # set True to test the locked door without waiting for a rush
ITEMS["jp_ticket"] = dict(name="Waiting Ticket", kind="tool", thirst=0, hunger=0, fx=None, col=(250, 232, 205),
                          desc="Your place in the Sakura-ya line. Show it at the door when the host calls your number.")
ChunkBuilder._jp_reception = lambda self: None               # the indoor reception is gone
JpRestaurant.host_job = lambda self, host: None              # mochi rounds come back (outside) in Part 3

_rest_init_v10_prev = JpRestaurant.__init__


def _rest_init_v10b(self, g, ch):
    _rest_init_v10_prev(self, g, ch)
    self.admit_q, self.admit_t, self.admit_e = [], [0.0, 0.0], [None, None]
    self.tk_no, self.lock_note_t = dict(table=0, ramen=0, belt=0), 0.0
    g.inv.counts["jp_ticket"] = 0                            # a ticket only belongs to the restaurant that issued it


JpRestaurant.__init__ = _rest_init_v10b


# ---- court geometry (works even before the forecourt chunk is loaded) ---------------------------------------------------
def _r_court(self, d):
    c = self.g.world.chunks.get(self.ch.idx - 1 if d == 0 else self.ch.idx + 1)
    return c.extra.get("jp_court") if c is not None else None


def _r_court_pts(self, d):
    ch, xo = self.ch, self.ch.xoff
    zt = (lambda t: ch.z0 - t) if d == 0 else (lambda t: ch.z0 + CH + t)       # t = distance outward from the building
    return dict(post=(xo + 3.4, zt(4.1)), front=(xo + 3.4, zt(6.6)), far=(xo, zt(22.0)), door_out=(xo, zt(1.3)))


# ---- door lock + admission ----------------------------------------------------------------------------------------------
def _r_door_locked(self, d):
    ae = self.admit_e[d]
    if ae is not None and ae["who"] == "player" and ae is self.p_entry:
        return False                                          # a waiter is letting YOU in
    return JP_FORCE_LOCK or self.rush >= 0.55 or not self.free_seats()


def _r_request_admit(self, e):
    if not e["dead"] and not e["admit"] and e not in self.admit_q:
        self.admit_q.append(e)


def _r_admit_open(self, d, e):
    e["admit"] = True
    self.admit_t[d], self.admit_e[d] = 14.0, e
    self.door_t[d] = max(self.door_t[d], 14.0)
    jp_say(self.g, self.xo, self.ch.extra["jp_doors"][d]["z"], "door_slide", 0.9, 30.0)


def _r_admit_close(self, d):
    self.admit_t[d] = min(self.admit_t[d], 0.8)
    self.door_t[d] = min(self.door_t[d], 0.8)


_pop_prev, _shw_prev, _td_prev = JpRestaurant.pop_job, JpRestaurant.server_has_work, JpRestaurant._tick_doors


def _r_pop_job_v10(self, srv):
    while self.admit_q:                                       # a waiter picks this up as soon as the current task is done
        e = self.admit_q.pop(0)
        if not e["dead"] and not e["admit"]:
            return dict(type="admit", e=e)
    return _pop_prev(self, srv)


def _r_server_has_work_v10(self):
    return bool(self.admit_q) or _shw_prev(self)


def _r_tick_doors_v10(self, dt):
    _td_prev(self, dt)
    g, p, ch = self.g, self.g.player, self.ch
    self.lock_note_t -= dt
    for i, d in enumerate(ch.extra.get("jp_doors", ())):
        inward = 1.0 if d["side"] == "-z" else -1.0
        pin = (p.z - d["z"]) * inward > 0.0                   # the player is inside the restaurant
        self.admit_t[i] = max(0.0, self.admit_t[i] - dt)
        ae = self.admit_e[i]
        if ae is not None and (self.admit_t[i] <= 0.0 or ae["dead"]):
            self.admit_e[i] = ae = None
        if ae is not None and ae["who"] == "player" and pin:
            ae["in"] = True
        locked = self.door_locked(i)
        lc = d.get("lock_coll")
        if lc is None:
            lc = d["lock_coll"] = (-d["gap"] + self.xo, d["gap"] + self.xo, d["z"] - 0.45, d["z"] + 0.45, 3.0)
        if locked and not pin:                                # blocks people OUTSIDE only, even while the panels are open for a guest
            if lc not in ch.colliders:
                ch.colliders.append(lc)
            if self.door_t[i] <= 0.0:                         # no guest being admitted: keep the doors shut
                d["open"] = 0.0
                for pv, s in d["panels"]:
                    pv.x = s * d["gap"] / 2
                if d["coll"] not in ch.colliders:
                    ch.colliders.append(d["coll"])
            if self.lock_note_t <= 0.0 and abs(p.x - self.xo) < 3.0 and abs(p.z - d["z"]) < 2.6:
                self.lock_note_t = 9.0
                g.hud.note("Locked during the rush - get a ticket from the host.", 3.5, good=False)
        elif lc in ch.colliders:
            ch.colliders.remove(lc)


for _n, _f in (("court", _r_court), ("court_pts", _r_court_pts), ("door_locked", _r_door_locked), ("request_admit", _r_request_admit),
               ("admit_open", _r_admit_open), ("admit_close", _r_admit_close), ("pop_job", _r_pop_job_v10),
               ("server_has_work", _r_server_has_work_v10), ("_tick_doors", _r_tick_doors_v10)):
    setattr(JpRestaurant, _n, _f)


def _srv_do_admit(self, job):
    rest, e = self.rest, job["e"]
    d = e["door"]
    dz = rest.ch.extra["jp_doors"][d]["z"]
    yield from self.go(rest.xo + 0.4, dz + (1.1 if d == 0 else -1.1), 1.8)
    yield from self.turn(180.0 if d == 0 else 0.0, 0.3)
    if e["dead"]:
        return
    rest.admit_open(d, e)
    self.shout(f"Ticket {e['code']}!  Douzo, this way please!", "irasshai")
    t = 0.0
    while t < 20.0 and not e["dead"] and not e["in"]:
        t += self.dt
        self.pose(rs=-24.0 if t < 3.0 else 0.0)
        yield
    rest.admit_close(d)
    yield from self.hold(0.6, None)


def _server_brain_v10(self):
    rest = self.rest
    while True:
        job = rest.pop_job(self)
        if job is None:
            self.job = None
            yield from self.patrol()
            continue
        self.job = job
        t = job["type"]
        if t == "ask":
            yield from self.do_ask(job)
        elif t == "tray":
            yield from self.do_tray(job)
        elif t == "clear":
            yield from self.do_clear(job)
        elif t == "admit":
            yield from self.do_admit(job)
        self.job = None


JpServer.do_admit, JpServer.brain = _srv_do_admit, _server_brain_v10


# ---- the waiting list: tickets, outside spots, seat assignment ------------------------------------------------------------
def _r_join_line(self, who, cat, door):
    court = self.court(door)
    if court is None:
        return None
    spots = court["spots"]
    used = sum(1 for sp in spots if sp["taken"] is not None) + sum(1 for e in self.waitlist if e["who"] == "player" and e["door"] == door)
    if used >= len(spots):
        return None
    self.tk_no[cat] = self.tk_no[cat] % 99 + 1
    e = {"who": who, "cat": cat, "door": door, "t0": self.clock, "seat": None, "spot": None, "dead": False, "t_res": 0.0,
         "code": f"{JP_TK_LETTER[cat]}{self.tk_no[cat]:02d}", "called": False, "admit": False, "in": False}
    self.waitlist.append(e)
    return e


def _r_take_wait_spot(self, e):
    court = self.court(e["door"])
    for sp in (court["spots"] if court else ()):
        if sp["taken"] is None and not (sp["it"] is not None and sp["it"].occupied):
            sp["taken"], e["spot"] = e, sp
            if sp["it"] is not None:
                sp["it"].occupied = True
            return sp
    return None


def _r_assign_seats(self):
    for e in list(self.waitlist):
        if e["seat"] is not None or e["dead"]:
            continue
        cands = self.free_seats(e["cat"])
        if not cands:
            continue
        edge = self.z0 if e["door"] == 0 else self.z0 + CH
        rec = min(cands, key=lambda r: abs(r["z"] - edge))
        rec["resv"], e["seat"], e["t_res"] = e, rec, self.clock
        rec["it"].label = "Sit  (reserved for you)" if e["who"] == "player" else "Reserved - waiting list"


def _r_player_join(self, cat, host):
    if self.p_entry is not None:
        return "You already have a ticket."
    e = self.join_line("player", cat, host.door)
    if e is None:
        return "The waiting area is full."
    self.p_entry = e
    ITEMS["jp_ticket"]["name"] = f"Ticket {e['code']}"
    self.g.inv.add("jp_ticket", 1 - self.g.inv.n("jp_ticket"))
    ahead = sum(1 for x in self.waitlist if x["seat"] is None and not x["dead"] and x["cat"] == cat and x is not e)
    return f"Ticket {e['code']} for {CATNAME[cat]}  -  {ahead} ahead.  Wait anywhere nearby; the host will call you."


def _r_player_line(self):
    e, g = self.p_entry, self.g
    if e is None:
        return
    if e["dead"]:
        self.p_entry = None
        g.inv.counts["jp_ticket"] = 0
    elif e["seat"] is not None and not e["in"] and self.clock - e["t_res"] > 150.0:
        self.drop_entry(e)
        g.hud.note("Your ticket was skipped - the seat went to someone else.", good=False)


def _r_hand_ticket(self, e):
    h = next((h for h in self.hosts if h.door == e["door"] and not h.dead), None)
    if h is not None:
        h.say(f"Ticket {e['code']} - please wait nearby, I'll call you!", 3.2)


def _r_host_call_job(self, host):
    c = [e for e in self.waitlist if e["door"] == host.door and e["seat"] is not None and not e["dead"] and not e["called"]]
    return min(c, key=lambda e: e["t_res"]) if c else None


_re_prev = JpRestaurant.remove_entry


def _r_remove_entry_v10(self, e):
    _re_prev(self, e)
    if e.get("who") == "player":
        self.g.inv.counts["jp_ticket"] = 0
        self.g.inv.dirty = True


for _n, _f in (("join_line", _r_join_line), ("take_wait_spot", _r_take_wait_spot), ("assign_seats", _r_assign_seats),
               ("player_join", _r_player_join), ("player_line", _r_player_line), ("hand_ticket", _r_hand_ticket),
               ("host_call_job", _r_host_call_job), ("remove_entry", _r_remove_entry_v10), ("in_zone", lambda self, d, x, z: True)):
    setattr(JpRestaurant, _n, _f)


# ---- the host ---------------------------------------------------------------------------------------------------------------
class JpTicketUI(UiBase):
    def __init__(self, g, host) -> None:
        super().__init__(g)
        self.host = host
        rest = g._jp_rest_near()
        n = sum(1 for e in rest.waitlist if e["seat"] is None and not e["dead"]) if rest else 0
        lock = "doors locked (rush / no free seats)" if (rest and rest.door_locked(host.door)) else "doors open"
        self.quad((0, 0), (0.92, 0.46), C(1.0, 0.4, 0.3, 0.35), z=0.22)
        self.quad((0, 0), (0.90, 0.44), C(0.10, 0.04, 0.05, 0.96), z=0.21)
        self.text("SAKURA-YA  -  RECEPTION", (-0.42, 0.16), 1.0, C(1.0, 0.85, 0.55, 1))
        self.text(f"{n} guest(s) waiting  -  {lock}.", (-0.42, 0.08), 0.8)
        self.text("[1] middle area table (M)    [2] ramen bar (B)    [3] sushi belt (S)", (-0.42, 0.0), 0.8, C(1.0, 0.95, 0.6, 1))
        self.text("You get a ticket. Wait anywhere nearby; at your call, press E at the door and show it.", (-0.42, -0.08), 0.66, C(1.0, 0.6, 0.5, 1))
        self.text("N / E / Esc: no thanks", (-0.42, -0.16), 0.65, C(0.7, 0.7, 0.75, 0.9))

    def key(self, k: str) -> None:
        g = self.g
        if k in ("1", "2", "3"):
            rest = g._jp_rest_near()
            cat = ("table", "ramen", "belt")[int(k) - 1]
            g.hud.note(rest.player_join(cat, self.host) if rest else "no restaurant here", 5.0)
            g.close_ui()
        elif k in ("n", "e", "escape"):
            g.close_ui()


class JpHost2(JpHost):
    def __init__(self, g, ch, pos, rng, rest, door) -> None:
        JpHost.__init__(self, g, ch, pos, rng, rest, door)
        self.post_yaw = 180.0 if door == 0 else 0.0           # faces the arrivals, not the restaurant
        self.yaw = self.post_yaw
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def on_talk(self) -> None:
        g, rest = self.g, self.rest
        e = rest.p_entry
        if e is not None and not e["dead"]:
            if e["seat"] is None:
                ahead = sum(1 for x in rest.waitlist if x["seat"] is None and not x["dead"] and x["cat"] == e["cat"] and x["t0"] < e["t0"])
                g.hud.speak(self, f"Ticket {e['code']}: {ahead} ahead of you for {CATNAME[e['cat']]}. Please wait nearby!")
            elif not e["called"]:
                g.hud.speak(self, f"Ticket {e['code']} - your seat is ready, one second while I update the list!")
            else:
                g.hud.speak(self, f"Ticket {e['code']}! Show it at the door - a waiter will let you in.")
            return
        queue = any(x["seat"] is None and not x["dead"] for x in rest.waitlist if x["door"] == self.door)
        if not rest.door_locked(self.door) and not queue and sum(rest.free_counts().values()) > 0:
            g.hud.speak(self, "We have seats! Just walk in and sit anywhere you like.")
            return
        g.hud.speak(self, "Welcome to Sakura-ya! Would you like a ticket? Which seats - the middle area, the bar, or the sushi belt?")
        g.open_ui(JpTicketUI(g, self))

    def brain(self):
        rest = self.rest
        while True:
            e = rest.host_call_job(self)
            if e is not None:
                yield from self.call_ticket(e)
                continue
            yield from self.go(self.post[0], self.post[1], 1.6)
            yield from self.turn(self.post_yaw, 0.3)
            yield from self.hold(self.rng.uniform(1.5, 3.0), None)

    def call_ticket(self, e):
        rest, g = self.rest, self.g
        yield from self.go(self.post[0], self.post[1], 1.6)
        yield from self.turn(self.post_yaw, 0.3)
        t = 0.0
        while t < 2.4:                                        # typing the update on the tablet
            t += self.dt
            _an(self, "write", t, 2.4)
            yield
        if e["dead"]:
            return
        e["called"] = True
        self.say(f"Ticket {e['code']}!  {rest.seat_name(e['seat'])} is ready.", 4.0)
        if e["who"] == "player":
            g.hud.note(f"Ticket {e['code']} called!  Go to the door (E) and show it.", 6.0)
            g.sfx.play("chime", 0.7)


_japan_npcs_prev2 = _japan_npcs


def _japan_npcs(self) -> None:                                 # noqa: F811  (the v4 hook looks this name up at call time)
    _japan_npcs_prev2(self)
    ch = self.ch
    rest = ch.extra.get("jp_rest")
    if rest is None:
        return
    for h in list(rest.hosts):                                 # drop the old indoor hosts
        h.dead = True
        if h in ch.npcs:
            ch.npcs.remove(h)
        REAPER.push(h.fig.root)
    rest.hosts.clear()
    rng = seeded(self.seed, "jp-host2", ch.idx)
    for d in (0, 1):
        h = JpHost2(self.w.g, ch, rest.court_pts(d)["post"], rng, rest, d)
        rest.hosts.append(h)
        ch.npcs.append(h)


# ---- player: the door and the lookup that works from the forecourt ------------------------------------------------------
def _jp_rest_near(self):
    i = self.world.chunk_index(self.player.z)
    for k in (i, i - 1, i + 1):
        c = self.world.chunks.get(k)
        if c is not None and c.extra.get("jp_rest") is not None:
            return c.extra["jp_rest"]
    return None


def _use_jpdoor(self, it):
    rest = self._jp_rest_near()
    if rest is None:
        return
    d, e = it.data["d"], rest.p_entry
    if not rest.door_locked(d):
        self.hud.note("The door is open - come on in!", 3.0)
    elif e is None or e["dead"]:
        self.hud.note("Locked during the rush. Get a ticket from the host first.", 4.5, good=False)
    elif e["seat"] is None:
        ahead = sum(1 for x in rest.waitlist if x["seat"] is None and not x["dead"] and x["cat"] == e["cat"] and x["t0"] < e["t0"])
        self.hud.note(f"Ticket {e['code']}: not called yet - {ahead} ahead of you.", 4.5)
    elif not e["called"]:
        self.hud.note(f"Ticket {e['code']}: the host is updating the list...", 3.5)
    elif not e["admit"]:
        rest.request_admit(e)
        self.hud.note(f"You show ticket {e['code']}. A waiter is on the way.", 4.0)


Game._jp_rest_near, Game._use_jpdoor = _jp_rest_near, _use_jpdoor


# ---- customers: arrive from the far end of the forecourt, take a ticket, wait, get called, get let in ----------------------
_cu_init_v9_prev = JpCustomer.__init__


def _cu_init_v10(self, g, ch, rng, rest, rec, arrive=None):
    _cu_init_v9_prev(self, g, ch, rng, rest, rec, arrive)
    if arrive is not None:
        fx, fz = rest.court_pts(arrive)["far"]
        self.x, self.z = fx + rng.uniform(-1.2, 1.2), fz
        self.fig.place(self.x, self.y, self.z, self.yaw)


_enter_prev = JpCustomer.enter_and_sit


def _cu_enter_v10(self):
    ox, oz = self.rest.court_pts(self.arrive)["door_out"]
    yield from self.walk(ox + self.rng.uniform(-0.5, 0.5), oz, 1.5)
    yield from _enter_prev(self)


def _cu_leave_far(self, d):
    self.hips, self.y, self.state_tag = None, 0.0, "leaving"
    fx, fz = self.rest.court_pts(d)["far"]
    yield from self.walk(fx, fz, 1.5)
    self.rest.retire(self)


def _cu_line_up_v10(self):
    rest, rng, d = self.rest, self.rng, self.arrive
    cp = rest.court_pts(d)
    inward = 1.0 if d == 0 else -1.0
    dz = rest.ch.extra["jp_doors"][d]["z"]
    yield from self.walk(cp["front"][0] + rng.uniform(-0.3, 0.3), cp["front"][1], 1.5)
    if rng.random() < 0.30:
        self.say(rng.choice(("Full? Maybe another time.", "So busy... I'll come back later.", "A line? Ah, never mind.")), 3.2)
        yield from self.idle(1.4)
        yield from self.leave_far(d)
        return False
    e = rest.join_line(self, self.cat, d)
    sp = rest.take_wait_spot(e) if e is not None else None
    if sp is None:
        if e is not None:
            rest.drop_entry(e)
        self.say("Too crowded. Next time!", 2.8)
        yield from self.idle(1.0)
        yield from self.leave_far(d)
        return False
    self.entry = e
    self.say("Table for one, please!" if self.cat == "table" else "Is there room at the bar?", 2.8)
    yield from self.idle(1.4)
    rest.hand_ticket(e)
    yield from self.walk(sp["x"], sp["z"], 1.4)
    self.hips, self.y, self.yaw = ("sit" if sp["sit"] else None), (sp["y"] - 0.72 if sp["sit"] else 0.0), sp["yaw"]
    self.fig.place(self.x, self.y, self.z, self.yaw)
    self.state_tag = "queue"
    while not e["called"]:                                    # waiting for the host to call the ticket
        if e["dead"] or rest.clock - e["t0"] > LINE_TIMEOUT:
            if not e["dead"]:
                rest.drop_entry(e)
            self.say("I'll go to a different restaurant. Thank you anyway!", 3.6)
            yield from self.idle(1.8)
            yield from self.leave_far(d)
            return False
        yield from self.idle(0.3)
    self.hips, self.y = None, 0.0                             # called: stand up, go to the door and show the ticket
    rest.release_spot(e)
    self.say(f"That's us - {e['code']}!", 2.4)
    yield from self.go(cp["door_out"][0] + rng.uniform(-0.6, 0.6), cp["door_out"][1], 1.6)
    yield from self.turn(0.0 if d == 0 else 180.0, 0.3)
    rest.request_admit(e)
    t0 = rest.clock
    while not e["admit"]:
        if e["dead"] or rest.clock - t0 > 90.0:
            rest.drop_entry(e)
            yield from self.leave_far(d)
            return False
        yield from self.idle(0.3)
    yield from self.walk(rest.xo + rng.uniform(-0.4, 0.4), dz + inward * 1.8, 1.5)
    e["in"] = True
    rec = e["seat"]
    rest.remove_entry(e)
    rest.unresv(rec)
    rec["occ"], rec["it"].occupied, self.srec = self, True, rec
    ex, ez = self.exit_point()
    yield from self.go(ex, ez, 1.7)
    yield from self.turn(rec["yaw"], 0.3)
    self.x, self.z, self.y, self.yaw, self.hips = rec["x"], rec["z"], rec["y"] - 0.72, rec["yaw"], "sit"
    self.fig.place(self.x, self.y, self.z, self.yaw)
    self.state_tag = "idle"
    return True


JpCustomer.__init__, JpCustomer.enter_and_sit, JpCustomer.leave_far, JpCustomer.line_up = _cu_init_v10, _cu_enter_v10, _cu_leave_far, _cu_line_up_v10


def _r_spawn_arrival_v10(self):
    alive = [c for c in self.customers if not getattr(c, "gone", False)]
    if len(alive) + len(self.waitlist) >= 58:
        return
    ds = [d for d in (0, 1) if self.court(d) is not None]     # guests only arrive on a side whose forecourt is loaded
    if not ds:
        return
    d = self.rng.choice(ds)
    queue = self.door_locked(d) or any(e["seat"] is None and not e["dead"] and e["door"] == d for e in self.waitlist)
    free = [] if queue else self.free_seats()
    if free:
        rec = self.rng.choice(free)
    else:
        cat = self.rng.choices(("table", "ramen", "belt"), (0.45, 0.30, 0.25))[0]
        rec = dict(kind=cat, cat=cat, dummy=True, x=self.xo, z=self.z0 + JP_L["doors_z"][d], y=0.0, yaw=0.0,
                   it=SimpleNamespace(occupied=False, label="Sit"), occ=None, slot=(0.0, 0.0), pile=(0.0, 0.0),
                   clear_spot=(0.0, 0.0), bell=(0.0, 0.0))
    c = JpCustomer(self.g, self.ch, self.rng, self, rec, arrive=d)
    self.customers.append(c)
    self.ch.npcs.append(c)
    c.fig.root.enabled = self.ch.shown


JpRestaurant.spawn_arrival = _r_spawn_arrival_v10

# ---- v10d: free mochi + tea outside, the storage counter, restock / brew jobs ------------------------------------------------
MOCHI_BATCH, POT_CAP, KETTLE_CUPS, COURT_MOCHI, ROUND_GAP = 30, 15, 5, 30, 90.0


def _jp_mochi_counter(self):                                  # replaces the (now empty) indoor reception: the ONE mochi & tea counter
    ch, z0, xo = self.ch, self.ch.z0, self.ch.xoff
    self.par = ch.detail
    kw = dict(lm=ch.lm, lamp=ch.lamp, xc=xo)
    self.block(5.2, 0.5, z0 + 1.0, 2.4, 1.0, 0.6, "wood", color=C(0.30, 0.15, 0.09))
    self.box(5.2, 1.03, z0 + 1.0, 2.5, 0.05, 0.7, "wood", color=C(0.62, 0.40, 0.22))
    self.quad(5.2, 2.2, z0 + 0.55, 2.0, 0.5, "+z", jp_sign("餅と御茶", "MOCHI & TEA", (255, 226, 190), (60, 10, 14), 384, 96), emissive=1.0, double=False)
    pot = ur.Entity(parent=ch.detail, position=(4.5, 1.06, z0 + 1.0))                    # the big teapot
    mk(pot, "sphere", (0, 0.2, 0), (0.55, 0.42, 0.55), color=C(0.20, 0.24, 0.30), **kw)
    mk(pot, "cube", (0, 0.45, 0), (0.28, 0.06, 0.28), color=C(0.12, 0.14, 0.18), **kw)
    mk(pot, "cube", (0.34, 0.26, 0), (0.28, 0.06, 0.06), color=C(0.20, 0.24, 0.30), **kw)
    cab = ur.Entity(parent=ch.detail, position=(5.9, 1.06, z0 + 1.0))                    # the mochi storage
    mk(cab, "cube", (0, 0.02, 0), (0.9, 0.04, 0.5), color=C(0.16, 0.08, 0.05), **kw)
    balls = [mk(cab, "sphere", (-0.36 + 0.18 * (i % 5), 0.1, -0.1 + 0.2 * (i // 5)), (0.13, 0.1, 0.13),
                color=C(*MOCHI[i % len(MOCHI)][1]), emissive=0.2, **kw) for i in range(10)]
    ch.extra["jp_mstore"] = dict(stand=(xo + 5.0, z0 + 1.9), balls=balls)


ChunkBuilder._jp_reception = _jp_mochi_counter

_rest_init_v10b_prev = JpRestaurant.__init__


def _rest_init_v10d(self, g, ch):
    _rest_init_v10b_prev(self, g, ch)
    self.mochi_store, self.mochi_making = 2 * MOCHI_BATCH, False          # 60 mochi in storage
    self.pot_tea, self.brewing = POT_CAP, False                           # the big pot: 15 cups
    self.cstock = [dict(mochi=COURT_MOCHI, tea=KETTLE_CUPS), dict(mochi=COURT_MOCHI, tea=KETTLE_CUPS)]
    self.restock_req, self.restock_cool, self.restock_t = [0, 0], [0.0, 0.0], [0.0, 0.0]      # req: 0 none, 1 asked, 2 a waiter is on it


JpRestaurant.__init__ = _rest_init_v10d


def _r_mochi_batch_needed(self):
    return (not self.mochi_making) and self.mochi_store <= MOCHI_BATCH


def _r_open_for_staff(self, d, secs):
    self.door_t[d] = max(self.door_t[d], secs)
    jp_say(self.g, self.xo, self.ch.extra["jp_doors"][d]["z"], "door_slide", 0.8, 30.0)


def _r_mt_tick(self, dt):
    for d in (0, 1):
        c = self.cstock[d]
        if self.restock_req[d] == 2 and self.clock - self.restock_t[d] > 120.0:      # a waiter got stuck: ask again
            self.restock_req[d] = 0
        if self.restock_req[d] == 0 and self.clock >= self.restock_cool[d] and (c["mochi"] <= 1 or c["tea"] <= 0):
            self.restock_req[d] = 1
        cc = self.court(d)
        vis = cc.get("vis") if cc else None
        if vis:
            n = min(len(vis["balls"]), (c["mochi"] + 2) // 3)
            for i, b in enumerate(vis["balls"]):
                if b.enabled != (i < n):
                    b.enabled = i < n
            if vis["kettle"].enabled != (c["tea"] > 0):
                vis["kettle"].enabled = c["tea"] > 0
    ms = self.ch.extra.get("jp_mstore")
    if ms:
        n = min(len(ms["balls"]), (self.mochi_store + 5) // 6)
        for i, b in enumerate(ms["balls"]):
            if b.enabled != (i < n):
                b.enabled = i < n


def _r_host_round_job(self, host):
    if self.clock < host.next_round:
        return None
    npcs = [cu for cu in self.customers
            if getattr(cu, "entry", None) is not None and not cu.entry["dead"] and cu.entry["door"] == host.door
            and not getattr(cu, "gone", False) and cu.state_tag == "queue" and self.clock - getattr(cu, "offer_t", -999.0) > 60.0]
    pe, p = self.p_entry, self.g.player
    player = (pe is not None and not pe["dead"] and pe["door"] == host.door and not pe["in"]
              and math.hypot(p.x - host.post[0], p.z - host.post[1]) < 18.0 and self.clock - pe.get("offer_t", -999.0) > 60.0)
    if not npcs and not player:
        return None
    c = self.cstock[host.door]
    if c["mochi"] < 2 or c["tea"] < 1:                        # nothing left to hand out: a waiter is already being asked to restock
        host.next_round = self.clock + 12.0
        return None
    return dict(npcs=npcs, player=player)


_tp_prev3, _pop_prev3, _shw_prev3 = JpRestaurant._tick_pipeline, JpRestaurant.pop_job, JpRestaurant.server_has_work


def _tp_v10d(self, dt):
    _tp_prev3(self, dt)
    try:
        self.mt_tick(dt)
    except Exception:
        log.exception("mochi/tea tick failed")


def _r_pop_job_v10d(self, srv):
    if self.pot_tea <= 0 and not self.brewing:                # tea hit 0: the next free waiter MUST brew
        self.brewing = True
        return dict(type="brew")
    for d in (0, 1):
        if self.restock_req[d] == 1:
            self.restock_req[d], self.restock_t[d] = 2, self.clock
            return dict(type="restock", d=d)
    return _pop_prev3(self, srv)


def _r_server_has_work_v10d(self):
    return (self.pot_tea <= 0 and not self.brewing) or 1 in self.restock_req or _shw_prev3(self)


for _n, _f in (("mochi_batch_needed", _r_mochi_batch_needed), ("open_for_staff", _r_open_for_staff), ("mt_tick", _r_mt_tick),
               ("host_round_job", _r_host_round_job), ("pop_job", _r_pop_job_v10d), ("server_has_work", _r_server_has_work_v10d),
               ("_tick_pipeline", _tp_v10d)):
    setattr(JpRestaurant, _n, _f)


# ---- waiters: brew the tea, restock a forecourt -------------------------------------------------------------------------------
def _srv_do_brew(self, job):
    rest = self.rest
    cx, cz = rest.ch.extra["jp_mstore"]["stand"]
    yield from self.go(cx - 0.5, cz, 1.8)
    yield from self.turn(180.0, 0.3)
    jp_say(self.g, self.x, self.z, "pour", 0.7, 16.0)
    t = 0.0
    while t < 6.0:
        t += self.dt
        _an(self, "pour", t, 6.0)
        yield
    rest.pot_tea, rest.brewing = POT_CAP, False
    self.say("Fresh tea is ready!", 2.4)


def _srv_do_restock(self, job):
    rest, d = self.rest, job["d"]
    c = rest.cstock[d]
    cx, cz = rest.ch.extra["jp_mstore"]["stand"]
    yield from self.go(cx, cz, 1.8)
    yield from self.turn(180.0, 0.3)
    yield from self.hold(1.0, "place", 1.0)
    if rest.pot_tea < KETTLE_CUPS and not rest.brewing:        # pot nearly dry: brew first
        rest.brewing = True
        yield from self.do_brew(None)
    w = 0.0
    while rest.pot_tea < KETTLE_CUPS and w < 25.0:             # somebody else is brewing: wait for it
        w += self.dt
        yield
    take = max(0, min(COURT_MOCHI - c["mochi"], rest.mochi_store))
    rest.mochi_store -= take
    rest.pot_tea = max(0, rest.pot_tea - KETTLE_CUPS)
    self.set_carry("mochi")
    cp, inward = rest.court_pts(d), (1.0 if d == 0 else -1.0)
    dz = rest.ch.extra["jp_doors"][d]["z"]
    rest.open_for_staff(d, 8.0)
    yield from self.go(rest.xo + 0.3, dz + inward * 1.2, 1.8)
    yield from self.walk(rest.xo + 0.3, dz - inward * 1.4, 1.8)
    yield from self.walk(cp["post"][0] - 1.4, cp["post"][1], 1.8)                 # beside the reception desk
    yield from self.hold(1.4, "place", 1.4)
    self.set_carry(None)
    c["mochi"], c["tea"] = c["mochi"] + take, KETTLE_CUPS
    rest.restock_req[d], rest.restock_cool[d] = 0, rest.clock + 25.0
    self.say("Fresh mochi and tea!", 2.8)
    rest.open_for_staff(d, 8.0)
    yield from self.walk(rest.xo + 0.3, dz - inward * 1.4, 1.8)
    yield from self.walk(rest.xo + 0.3, dz + inward * 1.2, 1.8)


def _server_brain_v10d(self):
    rest = self.rest
    while True:
        job = rest.pop_job(self)
        if job is None:
            self.job = None
            yield from self.patrol()
            continue
        self.job = job
        t = job["type"]
        if t == "ask":
            yield from self.do_ask(job)
        elif t == "tray":
            yield from self.do_tray(job)
        elif t == "clear":
            yield from self.do_clear(job)
        elif t == "admit":
            yield from self.do_admit(job)
        elif t == "brew":
            yield from self.do_brew(job)
        elif t == "restock":
            yield from self.do_restock(job)
        self.job = None


JpServer.do_brew, JpServer.do_restock, JpServer.brain = _srv_do_brew, _srv_do_restock, _server_brain_v10d


# ---- the main (ramen bar) chef bakes a batch of 30 mochi when the storage is down to 30 ---------------------------------------
def _chef_make_mochi(self):
    rest = self.rest
    rest.mochi_making = True
    try:
        yield from self.to_station("board")
        self.show_prop("knife")
        e, nxt = 0.0, 0.0
        while e < 14.0:
            e += self.dt
            if e >= nxt:
                jp_say(self.g, self.x, self.z, "chop", 0.7, 16.0)
                nxt += 0.9
            _an(self, "chop", e, 14.0)
            yield
        self.show_prop(None)
        rest.release("board", self)
        self.cur_st = None
        rest.mochi_store += MOCHI_BATCH
        self.say("A fresh batch of mochi!", 2.8)
    finally:
        rest.mochi_making = False


_chef_hw_prev = JpChef.has_work


def _chef_has_work_v10d(self):
    return _chef_hw_prev(self) or (self.which == "ramen" and self.rest.mochi_batch_needed())


def _chef_brain_v10d(self):
    rest = self.rest
    if self.which == "sushi":
        yield from self.sushi_loop()
        return
    self.next_scan, self.pend = rest.clock + 3.0, None
    while True:
        if self.which == "ramen" and self.pend is None and rest.clock >= self.next_scan:
            self.next_scan = rest.clock + 10.0                 # scan the bar every 10 s
            self.pend = rest.next_ask(self)
        if self.pend is not None and not rest.bar_in_progress():
            a, self.pend = self.pend, None
            yield from self.ask_counter(a)
            self.next_scan = min(self.next_scan, rest.clock + 1.5)
            continue
        it = rest.claim_item(self)
        if it is not None:
            yield from self.cook_item(it)
            continue
        if self.pend is not None:
            a, self.pend = self.pend, None
            yield from self.ask_counter(a)
            continue
        if self.which == "ramen" and rest.mochi_batch_needed():          # NEW: nothing to cook -> bake mochi
            yield from self.make_mochi()
            continue
        if rest.pool_pending(self):
            yield from self.wait_blocked()
        else:
            yield from self.idle_routine()


JpChef.make_mochi, JpChef.has_work, JpChef.brain = _chef_make_mochi, _chef_has_work_v10d, _chef_brain_v10d


# ---- the host's mochi + tea round ------------------------------------------------------------------------------------------------
class JpOfferUI(UiBase):
    def __init__(self, g, host) -> None:
        super().__init__(g)
        self.host, self.done = host, False
        self.quad((0, -0.28), (0.74, 0.2), C(1.0, 0.4, 0.3, 0.35), z=0.22)
        self.quad((0, -0.28), (0.72, 0.18), C(0.10, 0.04, 0.05, 0.96), z=0.21)
        self.text("Free while you wait:  2 mochi + a hot green tea", (-0.33, -0.24), 0.8, C(1.0, 0.85, 0.55, 1))
        self.text("[Y / E] yes please       [N / Esc] no thanks", (-0.33, -0.31), 0.72)

    def key(self, k: str) -> None:
        g, rest = self.g, self.host.rest
        if k in ("y", "e"):
            c = rest.cstock[self.host.door]
            if c["mochi"] >= 2 and c["tea"] >= 1:
                c["mochi"] -= 2
                c["tea"] -= 1
                g.inv.add("mochi", 2)
                g.inv.add("green_tea", 1)
                g.hud.note("2 mochi and a green tea - on the house. They're in your inventory.", 4.5)
            else:
                g.hud.note("Sorry - we just ran out!", 3.0, good=False)
        elif k not in ("n", "escape"):
            return
        self.done = True
        g.close_ui()


def _host_do_round(self, job):
    rest, g, c = self.rest, self.g, self.rest.cstock[self.door]
    yield from self.go(self.post[0], self.post[1], 1.6)
    yield from self.hold(0.8, "place", 0.8)
    self.set_carry("mochi")
    self.say("Mochi and tea while you wait!", 3.0)
    targets = sorted(job["npcs"], key=lambda cu: math.hypot(cu.x - self.x, cu.z - self.z))
    if job["player"]:
        targets.append("player")
    for t in targets:
        if rest.host_call_job(self) is not None:               # a called ticket comes first
            self.set_carry(None)
            self.next_round = rest.clock + 5.0
            return False
        if c["mochi"] < 2 or c["tea"] < 1:
            break
        who = g.player if t == "player" else t
        if t != "player" and (getattr(t, "gone", False) or t.state_tag != "queue"):
            continue
        ux, uz = self.x - who.x, self.z - who.z
        n = math.hypot(ux, uz) or 1.0
        yield from self.go(who.x + ux / n * 1.0, who.z + uz / n * 1.0, 1.6)
        yield from self.turn(math.degrees(math.atan2(who.x - self.x, who.z - self.z)), 0.25)
        if t == "player":
            if rest.p_entry is None:
                continue
            rest.p_entry["offer_t"] = rest.clock
            g.hud.speak(self, "Mochi and tea while you wait? It's on the house!")
            ui = JpOfferUI(g, self)
            g.open_ui(ui)
            w = 0.0
            while not ui.done and w < 14.0:
                w += self.dt
                yield
            if not ui.done and g.ui is ui:
                g.close_ui()
        else:
            t.offer_t = rest.clock
            self.say(self.rng.choice(("Mochi and tea?", "Something sweet while you wait?", "Free mochi and tea, douzo!")), 2.8)
            yield from self.hold(1.2, "place", 1.2)
            if self.rng.random() < 0.8:
                c["mochi"] -= 2
                c["tea"] -= 1
                t.say(self.rng.choice(("Arigato!", "Oh, mochi! Thank you.", "How kind - oishii!")), 3.0)
                jp_say(g, self.x, self.z, "plate", 0.6, 14.0)
            else:
                t.say("No thank you, I'm fine.", 2.4)
    self.set_carry(None)
    return True


def _host_brain_v10d(self):
    rest = self.rest
    while True:
        e = rest.host_call_job(self)
        if e is not None:
            yield from self.call_ticket(e)
            continue
        job = rest.host_round_job(self)
        if job is not None:
            done = yield from self.do_round(job)
            if done:
                self.next_round = rest.clock + ROUND_GAP          # 1:30 until the next round
            continue
        yield from self.go(self.post[0], self.post[1], 1.6)
        yield from self.turn(self.post_yaw, 0.3)
        yield from self.hold(self.rng.uniform(1.5, 3.0), None)


JpHost2.do_round, JpHost2.brain = _host_do_round, _host_brain_v10d

# ---- v10e: paid mochi is fetched by a WAITER from the mochi counter (random flavour) -----------------------------------------
_nt_prev = JpRestaurant.new_ticket


def _r_new_ticket_v10e(self, rec, cust, items, dest, prio=False):
    tk = _nt_prev(self, rec, cust, items, dest, prio)
    for it in tk["items"]:
        if it["key"] == "mochi":
            it["nb"], it["not_before"] = it["not_before"], 1e12      # chefs never pick it up; a waiter will, once "nb" has passed
    return tk


def _r_sweet_ok(self, it, now):
    tk = it["tk"]
    if it["key"] != "mochi" or it["state"] != "queued" or tk["void"] or it.get("nb", 1e12) > now:
        return False
    return not (tk["dest"] == "mid" and self.tray_owner.get(tk["tid"]) not in (None, tk))      # that table's tray is busy with another order


def _r_sweet_ready(self):
    now = self.clock
    return self.plates["L"] - self.plate_res["L"] >= 1 and any(self._sweet_ok(it, now) for it in self.mid_items + self.bar_items)


def _r_claim_sweet(self, srv):
    now = self.clock
    if self.plates["L"] - self.plate_res["L"] < 1:
        return None
    c = [it for it in self.mid_items + self.bar_items if self._sweet_ok(it, now)]
    if not c:
        return None
    it = min(c, key=lambda i: i["tk"]["t0"])
    self.plate_res["L"] += 1
    it["state"], it["chef"] = "claimed", srv
    tk = it["tk"]
    tk["started"] = True
    if tk["dest"] == "mid":
        self.tray_owner[tk["tid"]] = tk
    return it


_pop_prev4, _shw_prev4 = JpRestaurant.pop_job, JpRestaurant.server_has_work
_pp_prev4, _bip_prev4 = JpRestaurant.pool_pending, JpRestaurant.bar_in_progress


def _r_pop_job_v10e(self, srv):
    urgent = (self.pot_tea <= 0 and not self.brewing) or 1 in self.restock_req or bool(self.admit_q)
    player_ask = any(r["state"] == "new" and r["cust"] is None for r in self.reqs)
    if not urgent and not player_ask and not self.pass_ready:        # trays, brewing, restocking and admitting guests come first
        it = self.claim_sweet(srv)
        if it is not None:
            return dict(type="sweet", it=it)
    return _pop_prev4(self, srv)


def _r_server_has_work_v10e(self):
    return self.sweet_ready() or _shw_prev4(self)


def _r_pool_pending_v10e(self, chef):                               # a queued mochi is not the chefs' business
    return any(it["state"] == "queued" and not it["tk"]["void"] and it["not_before"] < 1e11 for it in self.pool_of(chef))


def _r_bar_in_progress_v10e(self):                                  # ...and it must not stall the bar chef either
    return any(it["state"] in ("queued", "claimed", "cooking") and it["tk"]["started"] and not it["tk"]["void"] and it["key"] != "mochi"
               for it in self.bar_items)


for _n, _f in (("new_ticket", _r_new_ticket_v10e), ("_sweet_ok", _r_sweet_ok), ("sweet_ready", _r_sweet_ready), ("claim_sweet", _r_claim_sweet),
               ("pop_job", _r_pop_job_v10e), ("server_has_work", _r_server_has_work_v10e), ("pool_pending", _r_pool_pending_v10e),
               ("bar_in_progress", _r_bar_in_progress_v10e)):
    setattr(JpRestaurant, _n, _f)


# ---- the waiter's mochi job: counter -> flavour -> the same hand-over a chef does ---------------------------------------------
def _srv_do_sweet(self, job):
    rest, it = self.rest, job["it"]
    tk = it["tk"]
    it["state"] = "cooking"
    cx, cz = rest.ch.extra["jp_mstore"]["stand"]
    yield from self.go(cx + 0.8, cz, 1.8)                                       # in front of the mochi cabinet
    yield from self.turn(180.0, 0.3)
    yield from self.hold(1.0, "place", 1.0)
    w = 0.0
    while rest.mochi_store <= 0 and w < 40.0:                                   # storage empty: the main chef is baking a batch
        w += self.dt
        yield
    rest.mochi_store = max(0, rest.mochi_store - 1)
    it["flavor"] = self.rng.choice(MOCHI)
    self.say(f"{it['flavor'][0]} mochi!", 2.0)
    rest.take_plate("L")
    jp_say(self.g, self.x, self.z, "plate", 0.5, 16.0)
    if tk["dest"] == "mid":
        self.set_carry("mochi")
        sx, sz = jp_tray_slot(tk["tid"])
        yield from self.walk(JP_L["tray_chef_x"] + self.ch.xoff, self.ch.z0 + sz, 1.6)
        yield from self.turn(90.0)
        yield from self.hold(0.8, "place", 0.8)
        self.set_carry(None)
        rest.tray_put(tk, it)
        jp_say(self.g, self.x, self.z, "plate", 0.8)
    else:
        tk["plated"].append(it)
        it["state"] = "done"
        if len(tk["plated"]) < len(tk["items"]):                                # a chef is still cooking the rest: he delivers the tray
            return
        self.set_tray([i["key"] for i in tk["items"]], "BAR")
        rec = tk["rec"]
        yield from self.walk(rec["stand"][0], rec["stand"][1], 1.6)
        yield from self.turn(90.0)
        yield from self.hold(0.9, "place", 0.9)
        self.set_carry(None)
        rest.put_counter_tray(tk)
        jp_say(self.g, self.x, self.z, "plate", 0.8)
        self.say("Dozo! Enjoy!", 2.6)


def _server_brain_v10e(self):
    rest = self.rest
    while True:
        job = rest.pop_job(self)
        if job is None:
            self.job = None
            yield from self.patrol()
            continue
        self.job = job
        t = job["type"]
        if t == "ask":
            yield from self.do_ask(job)
        elif t == "tray":
            yield from self.do_tray(job)
        elif t == "clear":
            yield from self.do_clear(job)
        elif t == "admit":
            yield from self.do_admit(job)
        elif t == "brew":
            yield from self.do_brew(job)
        elif t == "restock":
            yield from self.do_restock(job)
        elif t == "sweet":
            yield from self.do_sweet(job)
        self.job = None


JpServer.do_sweet, JpServer.brain = _srv_do_sweet, _server_brain_v10e


# ---- the flavour shows up as the colour of the mochi ---------------------------------------------------------------------------
def _recolor(ent, flav):
    for c in list(ent.children)[1:]:
        c.color = C(*flav[1])


_trayput_prev, _dishat_prev = JpRestaurant.tray_put, JpRestaurant._dish_at
_ptt_prev, _pct_prev = JpRestaurant.put_table_tray, JpRestaurant.put_counter_tray


def _r_tray_put_v10e(self, tk, it):
    _trayput_prev(self, tk, it)
    if it["key"] == "mochi" and it.get("flavor") and it.get("ent") is not None:
        _recolor(it["ent"], it["flavor"])


def _r_dish_at_v10e(self, key, x, y, z):
    ent = _dishat_prev(self, key, x, y, z)
    fl = getattr(self, "cur_flav", None)
    if key == "mochi" and fl:
        _recolor(ent, fl)
    return ent


def _flav_wrap(prev):
    def w(self, tk):
        fl = next((it.get("flavor") for it in tk["items"] if it["key"] == "mochi" and it.get("flavor")), None)
        self.cur_flav = fl
        try:
            prev(self, tk)
        finally:
            self.cur_flav = None
        if fl and tk["cust"] is None:
            self.g.hud.note(f"{fl[0]} mochi", 3.0)
    return w


JpRestaurant.tray_put, JpRestaurant._dish_at = _r_tray_put_v10e, _r_dish_at_v10e
JpRestaurant.put_table_tray, JpRestaurant.put_counter_tray = _flav_wrap(_ptt_prev), _flav_wrap(_pct_prev)

# ---- v10f: TAKEAWAY: east-lane annex, kiosk, black-robe chef, monitor, bags ------------------------------------------------
JP_TQ_LETTER, JP_TQ_PRIO = dict(take="T", drive="D", deliv="E"), dict(drive=0, take=1, deliv=2)
PALETTES["jp_black"] = dict(coat=(0.03, 0.03, 0.04), legs=(0.03, 0.03, 0.04), skin=(0.90, 0.76, 0.64), em=0.02)
ITEMS["jp_tkt"] = dict(name="Takeaway Ticket", kind="tool", thirst=0, hunger=0, fx=None, col=(250, 232, 205),
                       desc="Show it at the takeaway window to collect your bag. No ticket, no bag.")
ITEMS["jp_bag"] = dict(name="Takeaway Bag", kind="tool", thirst=0, hunger=0, fx=None, col=(190, 140, 85),
                       desc="Warm and rustly. Select it here and press O to open it.")


def jp_monitor_tex(prep, ready):
    im = Image.new("RGBA", (768, 240), (6, 12, 16, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([3, 3, 764, 236], outline=(255, 226, 190, 255), width=3)
    d.line([(384, 14), (384, 226)], fill=(255, 226, 190, 120), width=2)
    f, f2 = _font(28), _font(22)
    d.text((16, 8), "PREPARING", fill=(255, 190, 90, 255), font=f)
    d.text((400, 8), "READY  -  ON THE COUNTER", fill=(120, 255, 150, 255), font=f)
    for i, ln in enumerate(prep[:5]):
        d.text((16, 48 + i * 36), ln, fill=(255, 225, 170, 255), font=f2)
    for i, ln in enumerate(ready[:5]):
        d.text((400, 48 + i * 36), ln, fill=(190, 255, 205, 255), font=f2)
    return tex_from_pil(im)


def _jp_takeaway(self):
    ch, z0, xo = self.ch, self.ch.z0, self.ch.xoff
    self.par = ch.detail
    X0, X1, Z0, Z1 = 12.5, 14.0, z0 + 9.0, z0 + 15.0                       # the annex footprint (against the building's east wall)
    mid, wood, dark = (Z0 + Z1) / 2, C(0.30, 0.15, 0.09), C(0.07, 0.07, 0.09)
    self.box((X0 + X1) / 2, 0.02, mid, X1 - X0, 0.03, Z1 - Z0, "wood", color=C(0.5, 0.4, 0.3))
    self.box((X0 + X1) / 2, 3.2, mid, X1 - X0 + 0.3, 0.15, Z1 - Z0 + 0.2, "panel", color=dark)                       # roof
    self.block((X0 + X1) / 2, 1.6, Z1, X1 - X0, 3.2, 0.15, "panel", color=wood)                                    # south wall
    self.block((X0 + 13.1) / 2, 1.6, Z0, 13.1 - X0, 3.2, 0.15, "panel", color=wood)                                # north wall (gap = staff gate)
    self.solid(13.1, X1, Z0 - 0.1, Z0 + 0.1)                                                                        # invisible gate: staff only
    self.block(13.95, 0.55, mid, 0.4, 1.1, Z1 - Z0, "wood", color=C(0.55, 0.32, 0.18))                              # serving counter
    self.solid(13.75, 14.15, Z0, Z1)
    self.box(13.95, 2.8, mid, 0.3, 0.8, Z1 - Z0, "panel", color=wood)                                               # header above the window
    self.block(12.95, 0.5, Z0 + 0.75, 0.8, 1.0, 0.8, "panel", color=C(0.45, 0.45, 0.5))                             # compact kitchen: stove,
    self.box(12.95, 1.02, Z0 + 0.75, 0.6, 0.03, 0.6, color=C(1.0, 0.45, 0.2), emissive=0.6)
    self.block(13.2, 0.45, Z1 - 0.6, 1.2, 0.9, 0.6, "wood", color=C(0.5, 0.3, 0.18))                                # prep table,
    self.box(12.6, 1.9, mid, 0.15, 0.1, 4.0, "wood", color=wood)                                                    # shelf, lamp
    self.box((X0 + X1) / 2, 3.0, mid, 0.5, 0.05, 1.2, color=C(1.0, 0.9, 0.7), emissive=1.2)
    self.quad(14.12, 2.8, mid, 2.8, 0.7, "+x", jp_sign("持ち帰り", "TAKEAWAY", (255, 226, 190), (60, 10, 14), 512, 128,
                                                        "kiosk: order   /   window: collect"), emissive=1.1, double=False)
    for pz in (mid - 1.6, mid + 1.6):
        self.box(14.0, 3.7, pz, 0.08, 1.0, 0.08, color=dark)
    mon = self.quad(14.1, 3.9, mid, 3.4, 1.0, "+x", jp_monitor_tex([], []), emissive=1.3, double=False)           # the order monitor
    zk = Z0 - 1.4                                                                                                   # the kiosk
    self.block(14.45, 0.65, zk, 0.4, 1.3, 0.5, "panel", color=dark)
    self.quad(14.66, 1.05, zk, 0.4, 0.5, "+x", jp_sign("注文", "ORDER", (255, 226, 190), (10, 60, 40), 256, 160), emissive=1.2, double=False)
    self.inter("jpkiosk", 14.95, 1.1, zk, 1.7, "Takeaway kiosk  -  order", FACE_YAW["-x"])
    self.inter("jpbag", 14.4, 1.1, mid, 2.4, "Takeaway counter  -  collect your bag (ticket)", FACE_YAW["-x"])
    self.block(16.65, 0.22, mid, 0.4, 0.44, 3.6, "wood", color=C(0.34, 0.15, 0.10))                                 # bench against the OPPOSITE wall
    self.box(16.9, 0.75, mid, 0.08, 0.6, 3.6, "wood", color=C(0.28, 0.12, 0.08))
    seats = []
    for dz in (-1.2, 0.0, 1.2):
        it = self.inter("sit", 16.6, 0.5, mid + dz, 1.3, "Sit and wait for your order", FACE_YAW["-x"])
        seats.append(dict(x=it.x, z=it.z, it=it, taken=None))
    ch.extra["jp_tk"] = dict(chef_home=(xo + 13.3, mid), gate=(xo + 13.55, Z0), counter=(xo + 14.4, mid), kiosk_stand=(xo + 14.95, zk),
                             seats=seats, mon=mon, sig=None,
                             bag_slots=[dict(x=xo + 13.95, z=mid + dz, ent=None) for dz in (-1.2, -0.4, 0.4, 1.2)])


_jp_rec_prev = ChunkBuilder._jp_reception


def _jp_rec_v10f(self):
    _jp_rec_prev(self)                  # the mochi & tea counter
    self._jp_takeaway()


ChunkBuilder._jp_takeaway, ChunkBuilder._jp_reception = _jp_takeaway, _jp_rec_v10f

_rest_init_v10d_prev = JpRestaurant.__init__


def _rest_init_v10f(self, g, ch):
    _rest_init_v10d_prev(self, g, ch)
    self.tq, self.tq_no, self.tk_npcs, self.tk_chef = [], dict(take=0, drive=0, deliv=0), [], None
    g.inv.counts["jp_tkt"] = 0


JpRestaurant.__init__ = _rest_init_v10f


# ---- the order book ------------------------------------------------------------------------------------------------------------
def _r_takeaway_order(self, src, items, owner):
    self.tq_no[src] = self.tq_no[src] % 99 + 1
    o = dict(code=f"{JP_TQ_LETTER[src]}{self.tq_no[src]:02d}", src=src, prio=JP_TQ_PRIO[src], items=list(items), state="queued",
             t0=self.clock, owner=owner, bag=None, slot=None)
    self.tq.append(o)
    return o


def _r_pick_takeaway(self):
    q = [o for o in self.tq if o["state"] == "queued"]
    return min(q, key=lambda o: (o["prio"], o["t0"])) if q else None


def _r_spawn_bag(self, o):
    tk = self.ch.extra["jp_tk"]
    i = next((i for i, s in enumerate(tk["bag_slots"]) if s["ent"] is None), None)
    if i is None:
        return                                                                  # counter full: the order is still collectable, just not drawn
    s = tk["bag_slots"][i]
    ent = ur.Entity(parent=self.ch.detail, position=(s["x"] - self.xo, 1.12, s["z"]))
    mk(ent, "cube", (0, 0.15, 0), (0.30, 0.30, 0.22), color=C(0.72, 0.55, 0.34), **self.kw)
    mk(ent, "cube", (0, 0.33, 0), (0.26, 0.05, 0.18), color=C(0.60, 0.45, 0.28), **self.kw)
    mk(ent, "cube", (0, 0.42, 0), (0.18, 0.02, 0.02), color=C(0.35, 0.25, 0.15), **self.kw)
    s["ent"], o["bag"], o["slot"] = ent, ent, i


def _r_take_bag(self, o, who):
    o["state"] = "done"
    if o["bag"] is not None:
        ur.destroy(o["bag"])
        o["bag"] = None
    tk = self.ch.extra.get("jp_tk")
    if tk and o.get("slot") is not None:
        tk["bag_slots"][o["slot"]]["ent"] = None


def _r_retire_tk(self, c):
    c.gone = True
    if c in self.tk_npcs:
        self.tk_npcs.remove(c)
    if c in self.ch.npcs:
        self.ch.npcs.remove(c)
    REAPER.push(c.fig.root)


def _r_spawn_takeaway(self):
    ds = [d for d in (0, 1) if self.court(d) is not None]
    if not ds:
        return
    c = JpTakeCust(self.g, self.ch, self.rng, self, self.rng.choice(ds))
    self.tk_npcs.append(c)
    self.ch.npcs.append(c)
    c.fig.root.enabled = self.ch.shown


_tp_prev4 = JpRestaurant._tick_pipeline
_sa_prev = JpRestaurant.spawn_arrival


def _r_tk_tick(self, dt):
    tk = self.ch.extra.get("jp_tk")
    if tk is None:
        return
    self.tq = [o for o in self.tq if o["state"] != "done"]
    prep = [f"{o['code']}  " + ", ".join(ITEMS[k]["name"] for k in o["items"])[:30]
            for o in sorted(self.tq, key=lambda o: (o["prio"], o["t0"])) if o["state"] in ("queued", "cooking")]
    ready = [o["code"] for o in self.tq if o["state"] == "ready"]
    sig = (tuple(prep), tuple(ready))
    if sig != tk["sig"]:                                                       # redraw the monitor only when something changed
        tk["sig"] = sig
        tk["mon"].texture = jp_monitor_tex(prep, ready)


def _tp_v10f(self, dt):
    _tp_prev4(self, dt)
    try:
        self.tk_tick(dt)
    except Exception:
        log.exception("takeaway tick failed")


def _r_spawn_arrival_v10f(self):
    if self.rng.random() < 0.25 and len(self.tk_npcs) < 3 and any(self.court(d) for d in (0, 1)):          # 1 in 4 wants takeaway
        self.spawn_takeaway()
        return
    _sa_prev(self)


for _n, _f in (("takeaway_order", _r_takeaway_order), ("pick_takeaway", _r_pick_takeaway), ("spawn_bag", _r_spawn_bag),
               ("take_bag", _r_take_bag), ("retire_tk", _r_retire_tk), ("spawn_takeaway", _r_spawn_takeaway), ("tk_tick", _r_tk_tick),
               ("_tick_pipeline", _tp_v10f), ("spawn_arrival", _r_spawn_arrival_v10f)):
    setattr(JpRestaurant, _n, _f)


# ---- the black-robe takeaway chef --------------------------------------------------------------------------------------------------
class JpTakeChef(JpActor):
    def __init__(self, g, ch, pos, rng, rest) -> None:
        super().__init__(g, ch, pos, rng, "takeaway chef", "jp_black", "ramen", rest, "talk")
        e1 = self.fig.elb[1]
        self.props = dict(
            knife=self.mkprop(e1, [("cube", (0, -0.42, 0.12), (0.014, 0.06, 0.24), (0.86, 0.88, 0.92), 0.2), ("cube", (0, -0.42, -0.04), (0.024, 0.03, 0.1), (0.1, 0.06, 0.04), 0)]),
            ladle=self.mkprop(e1, [("cube", (0, -0.42, 0.16), (0.016, 0.016, 0.34), (0.8, 0.82, 0.86), 0.1), ("sphere", (0, -0.42, 0.34), (0.1, 0.06, 0.1), (0.8, 0.82, 0.86), 0.1)]),
            pan=self.mkprop(e1, [("cube", (0, -0.42, 0.14), (0.02, 0.02, 0.28), (0.1, 0.1, 0.12), 0), ("cube", (0, -0.42, 0.36), (0.26, 0.03, 0.26), (0.08, 0.08, 0.1), 0)]),
        )
        self.yaw = 90.0
        self.fig.place(self.x, self.y, self.z, self.yaw)

    def brain(self):
        rest, tk = self.rest, self.ch.extra["jp_tk"]
        while True:
            o = rest.pick_takeaway()                                           # drive-through first, then takeaway, then delivery
            if o is not None:
                yield from self.make_order(o)
                continue
            yield from self.walk(tk["chef_home"][0], tk["chef_home"][1], 1.6)
            yield from self.turn(90.0, 0.3)
            yield from self.hold(self.rng.uniform(1.5, 3.0), None)

    def make_order(self, o):
        rest, tk = self.rest, self.ch.extra["jp_tk"]
        o["state"] = "cooking"
        yield from self.walk(tk["chef_home"][0], tk["chef_home"][1], 1.6)
        yield from self.turn(90.0, 0.3)
        for key in o["items"]:
            if key in ("mochi", "green_tea"):                                  # not made here: he walks in and takes it from the counter
                yield from self.fetch(key)
                continue
            for st, anim, dur, sfx, gap in JpChef.STEPS[JP_HOW[key]]:
                self.show_prop({"chop": "knife", "ladle": "ladle", "fry": "pan", "flip": "pan"}.get(anim))
                e, nxt, dd = 0.0, 0.0, dur * 0.8
                while e < dd:
                    e += self.dt
                    if sfx and e >= nxt:
                        jp_say(self.g, self.x, self.z, sfx, 0.7, 16.0)
                        nxt += max(0.4, gap)
                    _an(self, anim, e, dd)
                    yield
            self.show_prop(None)
        yield from self.hold(1.4, "place", 1.4)                                # pack the bag
        o["state"], o["t_ready"] = "ready", rest.clock
        rest.spawn_bag(o)
        self.say(f"Order {o['code']}!", 3.0)

    def fetch(self, key):
        rest, tk = self.rest, self.ch.extra["jp_tk"]
        xo, z0 = self.ch.xoff, self.ch.z0
        self.say("One moment - fetching it from the counter!", 2.4)
        yield from self.walk(tk["gate"][0], tk["gate"][1], 1.8)                 # out of the staff gate, up the lane, in at the main door
        yield from self.walk(xo + 13.6, z0 - 1.4, 1.9)
        rest.open_for_staff(0, 10.0)
        yield from self.walk(xo + 0.4, z0 - 1.4, 1.9)
        yield from self.walk(xo + 0.4, z0 + 1.3, 1.9)
        cx, cz = rest.ch.extra["jp_mstore"]["stand"]
        yield from self.walk(cx + 0.4, cz, 1.9)
        yield from self.turn(180.0, 0.3)
        yield from self.hold(1.0, "place", 1.0)
        if key == "mochi":
            rest.mochi_store = max(0, rest.mochi_store - 1)
        else:
            rest.pot_tea = max(0, rest.pot_tea - 1)
        self.set_carry(key)
        yield from self.walk(xo + 0.4, z0 + 1.3, 1.9)
        rest.open_for_staff(0, 10.0)
        yield from self.walk(xo + 0.4, z0 - 1.4, 1.9)
        yield from self.walk(xo + 13.6, z0 - 1.4, 1.9)
        yield from self.walk(tk["gate"][0], tk["gate"][1], 1.8)
        yield from self.walk(tk["chef_home"][0], tk["chef_home"][1], 1.8)
        yield from self.turn(90.0, 0.3)
        self.set_carry(None)


_japan_npcs_prev3 = _japan_npcs


def _japan_npcs(self) -> None:                                                  # noqa: F811
    _japan_npcs_prev3(self)
    rest, tk = self.ch.extra.get("jp_rest"), self.ch.extra.get("jp_tk")
    if rest is None or tk is None:
        return
    rest.tk_chef = JpTakeChef(self.w.g, self.ch, tk["chef_home"], seeded(self.seed, "jp-tkchef", self.ch.idx), rest)
    self.ch.npcs.append(rest.tk_chef)


# ---- takeaway customers (NPC) --------------------------------------------------------------------------------------------------------
class JpTakeCust(JpActor):
    def __init__(self, g, ch, rng, rest, d) -> None:
        self.d = d
        fz = ch.z0 - 21.0 if d == 0 else ch.z0 + CH + 21.0
        super().__init__(g, ch, (ch.xoff + 15.3 + rng.uniform(-0.6, 0.6), fz), rng, rng.choice(("diner", "guest")),
                         f"jp_c{rng.randrange(5)}", "guest", rest, "talk")
        self.yaw = 0.0 if d == 0 else 180.0
        self.fig.place(self.x, self.y, self.z, self.yaw)
        self.state_tag = "takeaway"

    def brain(self):
        rest, rng, tk = self.rest, self.rng, self.ch.extra["jp_tk"]
        kx, kz = tk["kiosk_stand"]
        yield from self.walk(kx, kz, 1.5)
        yield from self.turn(-90.0, 0.3)
        yield from self.hold(rng.uniform(2.5, 4.5), None)                       # ordering at the kiosk
        keys = [k for s, k, p, h in JP_MENU]
        o = rest.takeaway_order("take", rng.sample(keys, rng.choice((1, 2, 2, 3))), self)
        self.say(f"Takeaway, please  -  {o['code']}", 2.8)
        seat = next((s for s in tk["seats"] if s["taken"] is None and not s["it"].occupied), None)
        if seat is not None and rng.random() < 0.75:
            seat["taken"] = self
            yield from self.walk(seat["x"], seat["z"], 1.4)
            self.hips, self.y, self.yaw = "sit", 0.5 - 0.72, -90.0
            self.fig.place(self.x, self.y, self.z, self.yaw)
        else:
            yield from self.walk(kx + 0.5, tk["counter"][1] + rng.uniform(-1.0, 1.0), 1.4)
        while o["state"] != "ready" and rest.clock - o["t0"] < 420.0:
            yield from self.idle(0.4)
        self.hips, self.y = None, 0.0
        if seat is not None and seat["taken"] is self:
            seat["taken"] = None
        if o["state"] == "ready":
            cx, cz = tk["counter"]
            yield from self.walk(cx + 0.3, cz + rng.uniform(-0.8, 0.8), 1.6)
            yield from self.turn(-90.0, 0.3)
            yield from self.hold(1.0, "place", 1.0)
            rest.take_bag(o, self)
            self.say(rng.choice(("Arigato!", "Smells great!", "Thank you!")), 2.4)
        else:
            o["state"] = "done"
        fz = self.ch.z0 - 21.0 if self.d == 0 else self.ch.z0 + CH + 21.0
        yield from self.walk(self.ch.xoff + 15.3, fz, 1.6)
        rest.retire_tk(self)


# ---- the player: kiosk, counter, opening the bag ----------------------------------------------------------------------------------
class JpTakeawayUI(JpMenuUI):
    def __init__(self, g, rest) -> None:
        JpMenuUI.__init__(self, g, dict(state="menu"))
        self.rest = rest
        self.text("TAKEAWAY KIOSK  -  up to 5 dishes, nothing to wash: it all goes in a bag", (-0.60, 0.365), 0.66, C(0.7, 1.0, 0.8, 1))

    def key(self, k: str) -> None:
        g = self.g
        if k in ("enter", "return") and len(self.basket) >= 5:
            self.msg = "5 dishes is the limit"
            self.refresh()
            return
        if k in ("space", "f"):
            if not self.basket:
                self.msg = "add something first"
            elif g.inv.coins < self.total():
                self.msg = "not enough coins"
            else:
                g.inv.coins -= self.total()
                g.inv.dirty = True
                g.sfx.play("coin")
                o = self.rest.takeaway_order("take", self.basket, "player")
                ITEMS["jp_tkt"]["name"] = f"Takeaway Ticket {o['code']}"
                g.inv.add("jp_tkt", 1)
                g.hud.note(f"Ticket {o['code']}: watch the monitor, then collect your bag at the window.", 5.0)
                g.close_ui()
                return
            self.refresh()
            return
        JpMenuUI.key(self, k)


def _use_jpkiosk(self, it):
    rest = self._jp_rest_near()
    if rest is not None and rest.tk_chef is not None:
        self.open_ui(JpTakeawayUI(self, rest))


def _use_jpbag(self, it):
    rest = self._jp_rest_near()
    if rest is None:
        return
    ready = [o for o in rest.tq if o["state"] == "ready" and o["owner"] == "player"]
    waiting = [o for o in rest.tq if o["state"] in ("queued", "cooking") and o["owner"] == "player"]
    if ready and self.inv.n("jp_tkt") > 0:
        o = ready[0]
        rest.take_bag(o, "player")
        self.inv.add("jp_tkt", -1)
        self.inv.add("jp_bag", 1)
        if not hasattr(self, "jp_bags"):
            self.jp_bags = []
        self.jp_bags.append(list(o["items"]))
        self.sfx.play("coin", 0.5)
        self.hud.note(f"You take bag {o['code']}.  Backpack (I): select it and press O to open.", 5.0)
    elif ready:
        self.hud.note("Bags only go to people with their ticket.", 4.0, good=False)
    elif waiting:
        self.hud.note("Your order is still being prepared - watch the monitor.", 4.0)
    else:
        self.hud.note("Nothing to collect. Order at the kiosk first.", 3.5)


Game._use_jpkiosk, Game._use_jpbag = _use_jpkiosk, _use_jpbag

_inv_key_prev = InventoryUI.key


def _inv_key_v10f(self, k):
    if k == "o" and getattr(self, "es", None) and self.es[self.sel][0] == "jp_bag":
        g = self.g
        g.inv.add("jp_bag", -1)
        bags = getattr(g, "jp_bags", [])
        if bags:
            items = bags.pop(0)
            for key in items:
                g.inv.add(key, 1)
            g.sfx.play("tick", 0.7)
            g.hud.note("You open the bag: " + ", ".join(ITEMS[x]["name"] for x in items), 5.0)
        else:
            g.hud.note("The bag is empty - someone got there first.", good=False)
        self.refresh()
        return
    _inv_key_prev(self, k)


InventoryUI.key = _inv_key_v10f

# ---- v10g: DRIVE-THROUGH: NPC karts, ordering from your own kart, badges -------------------------------------------------------
DRIVEBY_ODDS = 10                                    # 1 in 10 arrivals is a drive-by (set to 1 to test)
JP_KART_COLS = ((0.15, 0.45, 0.90), (0.10, 0.70, 0.30), (0.90, 0.75, 0.10), (0.80, 0.20, 0.60), (0.95, 0.50, 0.10), (0.90, 0.90, 0.92))
JP_KLOCK = [False]                                   # True while the kiosk menu is open from a kart: the kart ignores the keys
BADGES["driveby"] = ("Window Service", "Watched a driver collect an order at the Sakura-ya drive-through.")
BADGES["drivethru"] = ("Drive-Thru Regular", "Ordered and collected at the Sakura-ya window without leaving your kart.")


class _NoKeys(dict):
    def __missing__(self, k):
        return 0


_gk_drive_prev = GoKart.drive


def _gk_drive_v10g(self, dt, w, hk, boost_mult=1.0):
    return _gk_drive_prev(self, dt, w, _NoKeys() if JP_KLOCK[0] else hk, boost_mult)


GoKart.drive = _gk_drive_v10g


class JpDriveBy:
    """A drive-through customer: a basic kart, a driver at the wheel, one order."""

    def __init__(self, rest) -> None:
        g, ch = rest.g, rest.ch
        self.rest, self.g, self.ch, self.rng = rest, g, ch, rest.rng
        zs = ch.z0 - 2 * CH + 8.0                                                  # two chunks up the road
        self.k = GoKart(f"jpdb{id(self) & 0xffff}", g.world.xo_at(zs), zs, 0.0, fuel=60.0)
        self.k.paint = self.rng.choice(JP_KART_COLS)
        self.k.build()
        pal = PALETTES[f"jp_c{self.rng.randrange(5)}"]
        self.fig = Figure(dict(coat=pal["coat"], legs=pal["legs"], skin=pal["skin"]), 0.0, ZONES["corridor"]["lamp"], em=0.18)
        xo, z0 = ch.xoff, ch.z0
        self.wps = [(g.world.xo_at(z0 - 26.0), z0 - 26.0), (xo + 15.4, z0 - 14.0), (xo + 15.4, z0 + 7.6)]   # corridor -> forecourt lane -> kiosk
        self.wi, self.state, self.t, self.speed, self.order, self.gone = 0, "approach", 0.0, 0.0, None, False
        self.k.yaw = math.degrees(math.atan2(self.wps[0][0] - self.k.x, self.wps[0][1] - self.k.z))

    def drive_to(self, tx, tz, vmax, dt, final=False):
        k = self.k
        dx, dz = tx - k.x, tz - k.z
        d = math.hypot(dx, dz)
        if d < (0.35 if final else 1.2):
            return True
        dy = wrap180(math.degrees(math.atan2(dx, dz)) - k.yaw)
        k.yaw += dy * min(1.0, 5.0 * dt)
        k.steer_vis = max(-1.0, min(1.0, dy / 30.0))
        tv = min(vmax, 1.5 + d * 1.1) if final else vmax                         # brake into the stop
        self.speed += (tv - self.speed) * min(1.0, 2.5 * dt)
        yr = math.radians(k.yaw)
        k.vx, k.vz = math.sin(yr) * self.speed, math.cos(yr) * self.speed
        k.x, k.z = k.x + k.vx * dt, k.z + k.vz * dt
        k.speed = self.speed
        for wh in k.wheels:
            wh.rotation_y = k.steer_vis * 20 if wh.z > 0 else 0
        return False

    def update(self, dt) -> None:
        if self.gone:
            return
        k, rest, ch = self.k, self.rest, self.ch
        z0, xo = ch.z0, ch.xoff
        if self.state == "approach":
            wp, last = self.wps[self.wi], self.wi == len(self.wps) - 1
            if self.drive_to(wp[0], wp[1], 6.0 if last else 9.0, dt, final=last):
                if last:
                    self.state, self.t, self.speed, k.vx, k.vz, k.speed = "kiosk", 0.0, 0.0, 0.0, 0.0, 0.0
                    k.yaw = 0.0
                else:
                    self.wi += 1
        elif self.state == "kiosk":                                               # orders at the kiosk
            self.t += dt
            if self.t > 3.5:
                keys = [x for s, x, p, h in JP_MENU]
                self.order = rest.takeaway_order("drive", self.rng.sample(keys, self.rng.choice((1, 2, 3))), self)
                jp_say(self.g, k.x, k.z, "coin", 0.6, 16.0)
                self.state, self.t = "wait", 0.0
        elif self.state == "wait":                                                # ...and waits right there
            self.t += dt
            if self.order["state"] == "ready":
                self.state = "to_window"
            elif self.t > 420.0:
                self.order["state"], self.state = "done", "leave"
        elif self.state == "to_window":
            if self.drive_to(xo + 15.4, z0 + 12.0, 4.0, dt, final=True):
                self.state, self.t, self.speed, k.vx, k.vz, k.speed = "collect", 0.0, 0.0, 0.0, 0.0, 0.0
        elif self.state == "collect":
            self.t += dt
            if self.t > 2.2:
                rest.take_bag(self.order, self)
                p = self.g.player
                if math.hypot(p.x - k.x, p.z - k.z) < 30.0:
                    self.g.award("driveby")
                self.state = "leave"
        elif self.state == "leave":                                                # drives away and disappears
            self.drive_to(xo + 15.4, z0 + CH + 26.0, 10.0, dt)
            if k.z > z0 + CH + 16.0:
                self.finish()
                return
        self.fig.place(k.x, k.y + 0.52 - 0.72 + 0.1, k.z, k.yaw)
        self.fig.pose(dt, 0, True)
        k.sync()

    def finish(self) -> None:
        if self.gone:
            return
        self.gone = True
        if self.order is not None and self.order["state"] != "done":
            self.rest.take_bag(self.order, self)
        for ent in (self.k.root, self.fig.root):
            if ent is not None:
                REAPER.push(ent)
        self.k.root = None


_rest_init_v10f_prev = JpRestaurant.__init__


def _rest_init_v10g(self, g, ch):
    _rest_init_v10f_prev(self, g, ch)
    self.dk = []


JpRestaurant.__init__ = _rest_init_v10g

_tp_prev5, _sa_prev2, _tko_prev = JpRestaurant._tick_pipeline, JpRestaurant.spawn_arrival, JpRestaurant.takeaway_order


def _tp_v10g(self, dt):
    _tp_prev5(self, dt)
    for d in list(self.dk):
        try:
            d.update(dt)
        except Exception:
            log.exception("drive-by failed")
            d.finish()
    self.dk = [d for d in self.dk if not d.gone]
    for o in list(self.tq):                                                        # orders whose owner is gone: clear the bag
        ow = o["owner"]
        if ow is not None and ow != "player" and getattr(ow, "gone", False) and o["state"] != "done":
            self.take_bag(o, None)


def _r_spawn_arrival_v10g(self):
    w = self.g.world
    if (self.rng.randrange(DRIVEBY_ODDS) == 0 and not self.dk and self.tk_chef is not None
            and self.court(0) is not None and w.chunks.get(self.ch.idx - 2) is not None):
        self.dk.append(JpDriveBy(self))
        return
    _sa_prev2(self)


def _r_tko_v10g(self, src, items, owner):
    if owner == "player" and src == "take" and self.g.player.kart is not None:     # ordering from your kart = a drive-through order
        src = "drive"
    return _tko_prev(self, src, items, owner)


JpRestaurant._tick_pipeline, JpRestaurant.spawn_arrival, JpRestaurant.takeaway_order = _tp_v10g, _r_spawn_arrival_v10g, _r_tko_v10g

_unload_prev = Game.on_unload


def _on_unload_v10g(self, ch):
    rest = ch.extra.get("jp_rest")
    if rest is not None:
        for d in list(getattr(rest, "dk", ())):                                    # the karts live in the scene, not in the chunk
            d.finish()
    _unload_prev(self, ch)


Game.on_unload = _on_unload_v10g


# ---- you, in your own kart ----------------------------------------------------------------------------------------------------------
def _jp_drive_target(self):
    p = self.player
    best, bd = None, 1e9
    for ch in self.world.near_chunks(p.z, 1):
        for it in ch.inters:
            if it.kind in ("jpkiosk", "jpbag"):
                d = math.hypot(it.x - p.x, it.z - p.z)
                if d < it.radius + 0.9 and d < bd:
                    best, bd = it, d
    return best


_interact_prev = Game.interact


def _interact_v10g(self):
    p = self.player
    if p.kart is not None and p.kart.speed < 3.5 and self.ui is None:
        it = self._jp_drive_target()
        if it is not None:
            getattr(self, "_use_" + it.kind)(it)
            return
    _interact_prev(self)


_kp_prev, _ub_prev = Game._kart_prompt, Game._use_jpbag


def _kart_prompt_v10g(self, k):
    base = _kp_prev(self, k)
    if k.speed < 3.5:
        it = self._jp_drive_target()
        if it is not None:
            return f"[E] {it.label}     {base}"
    return base


def _use_jpkiosk_v10g(self, it):
    rest = self._jp_rest_near()
    if rest is not None and rest.tk_chef is not None:
        if self.player.kart is not None:
            JP_KLOCK[0] = True
        self.open_ui(JpTakeawayUI(self, rest))


def _use_jpbag_v10g(self, it):
    n0 = self.inv.n("jp_bag")
    _ub_prev(self, it)
    if self.inv.n("jp_bag") > n0 and self.player.kart is not None:
        self.award("drivethru")


_tui_close_prev = JpTakeawayUI.close


def _tui_close_v10g(self):
    JP_KLOCK[0] = False
    _tui_close_prev(self)


Game._jp_drive_target, Game.interact, Game._kart_prompt = _jp_drive_target, _interact_v10g, _kart_prompt_v10g
Game._use_jpkiosk, Game._use_jpbag = _use_jpkiosk_v10g, _use_jpbag_v10g
JpTakeawayUI.close = _tui_close_v10g

# ============================================================================
# CLI / ENTRY POINT
# ============================================================================
def add_game_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("game")
    g.add_argument("--audio-only", action="store_true", help="skip the 3D mall: run the original headless orchestra CLI")
    g.add_argument("--silent", action="store_true", help="3D mall without audio")
    g.add_argument("--seed", type=int, default=1337, help="world seed (same seed = same mall)")
    g.add_argument("--render-distance", type=int, default=3, help="(fixed: 3 chunks ahead, 3 behind)")
    g.add_argument("--no-shader", action="store_true", help="use Ursina's default shader + fog (if the custom GLSL fails)")
    g.add_argument("--flip-art", action="store_true", help="flip generated textures vertically (if signs/posters look upside down)")
    g.add_argument("--sens", type=float, default=60.0, help="mouse sensitivity")
    g.add_argument("--fullscreen", action="store_true")
    g.add_argument("--no-save", action="store_true", help="do not read/write the save file (~/.ethereal_mall/save_<seed>.json)")
    g.add_argument("--video-dir", default="", help="folder searched for the cinema easter-egg video (default: this script's folder)")
    g.add_argument("--no-blackouts", action="store_true", help="disable random power failures")
    g.add_argument("--rare-odds", type=int, default=30, help="a rare room (the Japanese izakaya) appears with chance 1/N per block of chunks (1 = always, for testing)")


def run_game(args: argparse.Namespace) -> int:
    global RARE_ODDS
    RARE_ODDS = max(1, int(args.rare_odds))
    _load_ursina()
    stack = AudioStack(args)
    stack.prepare()
    log.info("Audio: %s", stack.status)
    game = None
    try:
        game = Game(args, stack)
        game.run()
    finally:
        if game is not None:
            game.shutdown()
        stack.stop()
    return 0



def main(argv=None) -> int:
    multiprocessing.freeze_support()
    args = parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.audio_only or args.list_devices:
            return run_audio_only(args)
        return run_game(args)
    except EtherealError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
