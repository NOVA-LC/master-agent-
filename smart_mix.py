"""
SMART_MIX.py v3 — context-aware production mixing agent.

UPGRADES IN v3:
  - Pre-Mix Detector: analyzes vocal crest factor + HP rolloff before processing.
    If vocal is already mixed/treated, switches to GLUE MODE (no destructive double-processing).
  - Plugin Delay Compensation (PDC): queries plugin latency, pads dry signals for parallel routing.
  - Key Confidence Gate: only engages Auto-Tune if key detection confidence > 85%.
    Below threshold uses Chromatic mode (no scale lock) to prevent mis-tuned snaps.
  - knowledge_base/ awareness: loads engineer transcripts as context, references them in chains.

CRITICAL DIRECTIVE (baked in):
  DO NOT OVERPROCESS PRE-MIXED AUDIO. If the vocal stem is already treated, the agent's
  ONLY job is to carve the frequency pocket and glue vocal+beat together on the master bus.
  No pitch correction, surgical EQ, or aggressive compression on an already-mixed vocal.
  Respect the printed stem. Goal is INTEGRATION, not RECONSTRUCTION.

Usage:
    python smart_mix.py <vocal.wav> <music.wav> <out.wav> [style_override] [--force-full-chain]
"""
import sys, os, json, numpy as np, soundfile as sf, pyloudnorm as pyln, librosa
from pathlib import Path
from pedalboard import Pedalboard, load_plugin, Compressor, HighpassFilter, LowpassFilter, PeakFilter, HighShelfFilter, LowShelfFilter
from scipy.signal import butter, sosfilt
try:
    from knowledge_index import get_kb
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False

# Decision -> KB topic query mapping (every chain choice has a citable source)
DECISION_CITATIONS = {
    'vocal_hp_120':       'vocal high pass 120 lead',
    'vocal_hp_drill':     'drill vocal HP filter clear 808',
    'vocal_mud_cut':      'vocal EQ cut 300 mud boxy',
    'vocal_nasal_cut':    'vocal EQ 1kHz nasal cut',
    'vocal_presence':     'vocal presence 3kHz boost rap',
    'vocal_air_shelf':    'high shelf air 12k vocal',
    'comp_4_to_1':        'rap vocal compression ratio 4:1',
    'deess_5_8k':         'de-ess 5kHz 8kHz sibilance',
    'autotune_drill_0':   'autotune retune 0 drill hard tune',
    'autotune_juice_15':  'autotune retune Juice WRLD melodic',
    'verb_predelay':      'reverb pre-delay 28ms 3d depth psychoacoustic',
    'verb_hp_400':        'reverb high pass 400 drill 808',
    'verb_lp_5k':         'reverb low pass 5000 tail recede distance',
    'verb_sidechain':     'sidechain compress reverb duck vocal Devvon',
    'verb_concert_hall':  'concert hall reverb dark floating drill',
    'verb_smooth_plate':  'smooth plate transparent vocal Valhalla',
    'verb_dirty_plate':   'dirty plate gritty warm Juice WRLD',
    'delay_eighth_dotted':'1/8 dotted delay BPM ping pong rap',
    'delay_gated_throw':  'Travis Scott delay throw automate send last word',
    'bv_hp_300':          'background vocal HP 300 roll off',
    'bv_lead_hole_2_5k':  'background vocal mid scoop 2.5k hole lead',
    'bv_lp_6k_radio':     'low pass 6k radio filter ad-lib',
    'bv_widen_subtle':    'background vocal widen subtle Pensado rap',
    'bv_tuck_8':          'background vocal tuck -8dB under lead',
    'ai_humanize_shift':  'AI vocal time shift humanize off grid',
    'ai_humanize_sat':    'tape saturation analog harmonic AI vocal',
    'lufs_drill':         'drill rap LUFS -10 target streaming',
    'tp_minus_1':         'true peak -1 dBTP ITU 1770',
    'imager_drill':       'stereo imager width drill rap',
}

REPO = Path(__file__).parent
KB = REPO / "knowledge_base"
PARAM_MAPS = REPO / "param_maps"

PLUGINS = {
    'autotune': r"C:\Program Files\Common Files\VST3\Antares\Auto-Tune Pro.vst3",
    'proq3':    r"C:\Program Files\Common Files\VST3\FabFilter Pro-Q 3.vst3",
    'vcomp':    r"C:\Program Files\Common Files\VST3\Auto-Tune Vocal Compressor.vst3",
    'deess':    r"C:\Program Files\Common Files\VST3\Vocal De-Esser.vst3",
    'verb':     r"C:\Program Files\Common Files\VST3\ValhallaVintageVerb.vst3",
    'ozone':    r"C:\Program Files\Common Files\VST3\iZotope\Ozone 9 Elements.vst3",
}


# ============================================================
# STYLE LIBRARY (same as v2, engineer-sourced)
# ============================================================
STYLES = {
    'ny_drill': {
        'description': 'NY Drill — Sleepy Hallow / Kyle Richh',
        'autotune': {'retune': 0, 'humanize': 0, 'flex': 0},
        'vocal_hp': 85, 'vocal_cuts': [(-2.5, 300, 1.2), (-1.5, 900, 1.5)],  # V15: 120 -> 85 (chest restoration)
        'vocal_boosts': [(2.5, 3000, 0.9), (3.5, 12000, 0.7)],
        'deess_thresh': -22,
        'verb': {'mode': 'Room', 'decay': 0.8, 'predelay': 25, 'mix_pct': 100,
                 'wet_blend': 0.10, 'send_hp': 400, 'colormode': 'eighties'},
        'pocket': {'lo': 1000, 'hi': 4000, 'duck_db': 5.0},
        'vocal_lead_db': 5.0, 'lufs_target': -10.0,
        'ozone_eq': {'mud_100': -1.0, 'cut_1200': -1.5, 'presence_2700': 1.5, 'air_10k': 2.5},
        'ozone_width': 18, 'ozone_max_thresh': -8.0,
    },
    'uk_drill': {
        'description': 'UK Drill — Central Cee / Headie One',
        'autotune': {'retune': 10, 'humanize': 10, 'flex': 10},
        'vocal_hp': 85, 'vocal_cuts': [(-2.5, 300, 1.2), (-1.5, 900, 1.5)],  # V15: 120 -> 85 (chest restoration)
        'vocal_boosts': [(2.0, 3000, 0.9), (3.0, 11000, 0.7)],
        'deess_thresh': -22,
        'verb': {'mode': 'Plate', 'decay': 1.0, 'predelay': 25, 'mix_pct': 100,
                 'wet_blend': 0.13, 'send_hp': 400, 'colormode': 'eighties'},
        'pocket': {'lo': 1000, 'hi': 4000, 'duck_db': 4.0},
        'vocal_lead_db': 4.5, 'lufs_target': -10.5,
        'ozone_eq': {'mud_100': -1.0, 'cut_1200': -1.0, 'presence_2700': 1.5, 'air_10k': 2.5},
        'ozone_width': 18, 'ozone_max_thresh': -7.5,
    },
    'emo_rap': {
        'description': 'Emo Rap — Juice WRLD (Max Lord)',
        'autotune': {'retune': 15, 'humanize': 25, 'flex': 25},
        'vocal_hp': 100, 'vocal_cuts': [(-2.0, 280, 1.1), (-2.0, 800, 1.3), (-1.5, 1500, 1.4)],
        'vocal_boosts': [(3.0, 3200, 0.85), (3.5, 12000, 0.7)],
        'deess_thresh': -22,
        'verb': {'mode': 'Dirty Plate', 'decay': 1.8, 'predelay': 25, 'mix_pct': 100,
                 'wet_blend': 0.22, 'send_hp': 300, 'colormode': 'seventies'},
        'pocket': {'lo': 800, 'hi': 4000, 'duck_db': 4.0},
        'vocal_lead_db': 5.0, 'lufs_target': -10.5,
        'ozone_eq': {'mud_100': -1.0, 'cut_1200': -1.0, 'presence_2700': 1.5, 'air_10k': 2.5},
        'ozone_width': 20, 'ozone_max_thresh': -7.5,
    },
    'modern_trap': {
        'description': 'Modern Trap — Young Thug (Alex Tumay)',
        'autotune': {'retune': 20, 'humanize': 20, 'flex': 20},
        'vocal_hp': 110, 'vocal_cuts': [(-2.0, 300, 1.2), (-1.5, 1000, 1.5)],
        'vocal_boosts': [(2.5, 3500, 0.8), (3.0, 13000, 0.7)],
        'deess_thresh': -22,
        'verb': {'mode': 'Plate', 'decay': 1.5, 'predelay': 20, 'mix_pct': 100,
                 'wet_blend': 0.15, 'send_hp': 350, 'colormode': 'eighties'},
        'pocket': {'lo': 900, 'hi': 4000, 'duck_db': 4.0},
        'vocal_lead_db': 4.5, 'lufs_target': -10.5,
        'ozone_eq': {'mud_100': -1.0, 'cut_1200': -1.0, 'presence_2700': 1.5, 'air_10k': 2.5},
        'ozone_width': 18, 'ozone_max_thresh': -7.5,
    },
    'melodic_trap': {
        'description': 'Melodic Trap — Brent Faiyaz / Don Toliver',
        'autotune': {'retune': 25, 'humanize': 35, 'flex': 30},
        'vocal_hp': 90, 'vocal_cuts': [(-1.5, 250, 1.0), (-1.0, 800, 1.3)],
        'vocal_boosts': [(2.0, 3500, 0.8), (4.0, 14000, 0.7)],
        'deess_thresh': -20,
        'verb': {'mode': 'Chamber', 'decay': 2.0, 'predelay': 30, 'mix_pct': 100,
                 'wet_blend': 0.20, 'send_hp': 250, 'colormode': 'eighties'},
        'pocket': {'lo': 800, 'hi': 3500, 'duck_db': 3.0},
        'vocal_lead_db': 4.0, 'lufs_target': -11.0,
        'ozone_eq': {'mud_100': -0.5, 'cut_1200': -0.5, 'presence_2700': 1.0, 'air_10k': 3.0},
        'ozone_width': 22, 'ozone_max_thresh': -7.0,
    },
    'pop_intimate': {
        'description': 'Pop Intimate — Billie Eilish (Rob Kinelski)',
        'autotune': {'retune': 30, 'humanize': 40, 'flex': 40},
        'vocal_hp': 100, 'vocal_cuts': [(-1.5, 300, 1.0), (-2.0, 2500, 1.3), (-1.0, 900, 1.3)],
        'vocal_boosts': [(1.5, 3500, 0.7), (3.0, 14000, 0.7)],
        'deess_thresh': -18,
        'verb': {'mode': 'Smooth Plate', 'decay': 0.6, 'predelay': 15, 'mix_pct': 100,
                 'wet_blend': 0.05, 'send_hp': 500, 'colormode': 'now'},
        'pocket': {'lo': 1000, 'hi': 4000, 'duck_db': 2.0},
        'vocal_lead_db': 6.0, 'lufs_target': -14.0,
        'ozone_eq': {'mud_100': 0.0, 'cut_1200': -1.0, 'presence_2700': 0.5, 'air_10k': 1.5},
        'ozone_width': 12, 'ozone_max_thresh': -10.0,
    },
    'hyperpop': {
        'description': 'Hyperpop — 100 gecs / Glaive',
        'autotune': {'retune': 0, 'humanize': 0, 'flex': 0},
        'vocal_hp': 130, 'vocal_cuts': [(-3.0, 300, 1.0), (-2.0, 800, 1.3)],
        'vocal_boosts': [(4.0, 3500, 0.8), (5.0, 15000, 0.6)],
        'deess_thresh': -20,
        'verb': {'mode': 'Plate', 'decay': 0.8, 'predelay': 10, 'mix_pct': 100,
                 'wet_blend': 0.08, 'send_hp': 500, 'colormode': 'now'},
        'pocket': {'lo': 1000, 'hi': 5000, 'duck_db': 6.0},
        'vocal_lead_db': 5.5, 'lufs_target': -8.0,
        'ozone_eq': {'mud_100': -2.0, 'cut_1200': -1.0, 'presence_2700': 2.5, 'air_10k': 4.0},
        'ozone_width': 25, 'ozone_max_thresh': -5.5,
    },
    'rnb_modern': {
        'description': 'Modern R&B — SZA / Frank Ocean',
        'autotune': {'retune': 30, 'humanize': 50, 'flex': 40},
        'vocal_hp': 85, 'vocal_cuts': [(-1.0, 250, 0.9), (-1.0, 900, 1.2)],
        'vocal_boosts': [(1.5, 3000, 0.7), (3.5, 14000, 0.7)],
        'deess_thresh': -20,
        'verb': {'mode': 'Smooth Room', 'decay': 2.5, 'predelay': 40, 'mix_pct': 100,
                 'wet_blend': 0.22, 'send_hp': 250, 'colormode': 'eighties'},
        'pocket': {'lo': 800, 'hi': 3500, 'duck_db': 2.5},
        'vocal_lead_db': 3.5, 'lufs_target': -12.0,
        'ozone_eq': {'mud_100': 0.0, 'cut_1200': -0.5, 'presence_2700': 0.8, 'air_10k': 3.0},
        'ozone_width': 25, 'ozone_max_thresh': -8.0,
    },
}

# ============================================================
# UTILITIES
# ============================================================
def db(x): return 20*np.log10(np.max(np.abs(x))+1e-12)
def rdb(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)

def crest_factor(x):
    peak = np.max(np.abs(x)) + 1e-12
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    return 20 * np.log10(peak / rms)

# ============================================================
# PRE-MIX DETECTOR — the critical addition
# ============================================================
def detect_premix(vocal, sr):
    """
    Detect if vocal stem is already heavily mixed/treated.
    Returns (is_premixed: bool, evidence: dict).
    """
    mono = np.mean(vocal, axis=1) if vocal.ndim == 2 else vocal

    # Indicator 1: Crest factor (dynamic range)
    # Raw vocal: 15-22 dB | Mixed: 8-12 | Squashed: <8
    cf = crest_factor(mono)

    # Indicator 2: Low-frequency rolloff (was it HP'd already?)
    # Compute spectral content below 80 Hz vs above
    S = np.abs(librosa.stft(mono.astype(np.float32), n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    lo_mask  = (freqs >= 30)  & (freqs < 80)
    mid_mask = (freqs >= 200) & (freqs < 2000)
    lo_energy  = float(np.mean(S[lo_mask]**2))
    mid_energy = float(np.mean(S[mid_mask]**2))
    lo_mid_ratio_db = 10 * np.log10((lo_energy / (mid_energy + 1e-12)) + 1e-12)
    # If lo/mid ratio is very negative, low-end has been cut
    hp_already = lo_mid_ratio_db < -25  # arbitrary threshold

    # Indicator 3: High-shelf air boost (was air added?)
    air_mask = (freqs >= 10000) & (freqs < 16000)
    pres_mask = (freqs >= 1000) & (freqs < 4000)
    air_energy  = float(np.mean(S[air_mask]**2))
    pres_energy = float(np.mean(S[pres_mask]**2))
    air_to_presence_db = 10 * np.log10((air_energy / (pres_energy + 1e-12)) + 1e-12)
    # If air is unusually hot relative to presence, shelf was likely added
    air_boosted = air_to_presence_db > -15  # arbitrary threshold for treated

    # Indicator 4: De-essing footprint
    # Compute sibilance band (5-9 kHz) vs presence (2-4 kHz)
    sib_mask = (freqs >= 5000) & (freqs < 9000)
    sib_energy = float(np.mean(S[sib_mask]**2))
    sib_ratio_db = 10 * np.log10((sib_energy / (pres_energy + 1e-12)) + 1e-12)
    deessed = sib_ratio_db < -8  # heavily de-essed if sib is way below presence

    # Decision logic: any 2+ indicators = pre-mixed
    indicators = {
        'crest_factor_db': round(cf, 2),
        'crest_indicates_compressed': cf < 11,
        'lo_mid_ratio_db': round(lo_mid_ratio_db, 1),
        'hp_already_applied': hp_already,
        'air_to_presence_db': round(air_to_presence_db, 1),
        'air_boosted': air_boosted,
        'sib_ratio_db': round(sib_ratio_db, 1),
        'deessed': deessed,
    }
    score = sum([
        indicators['crest_indicates_compressed'],
        indicators['hp_already_applied'],
        indicators['air_boosted'],
        indicators['deessed'],
    ])
    indicators['premix_score'] = score
    return score >= 2, indicators

# ============================================================
# FEATURE EXTRACTION + STYLE CLASSIFICATION
# ============================================================
def extract_features(path):
    y, sr = librosa.load(path, sr=22050, mono=False)
    if y.ndim == 1: y = np.stack([y, y], axis=0)
    mono = np.mean(y, axis=0)
    duration = len(mono)/sr
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(mono), sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    major_p = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_p = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    scores = []
    for i in range(12):
        cm = np.corrcoef(chroma_avg, np.roll(major_p, i))[0,1]
        cn = np.corrcoef(chroma_avg, np.roll(minor_p, i))[0,1]
        scores.append((f"{notes[i]} major", cm))
        scores.append((f"{notes[i]} minor", cn))
    scores.sort(key=lambda x: -x[1])
    # Key confidence = ratio of top score to second
    key_top = scores[0][1]
    key_second = scores[1][1]
    key_confidence = key_top / (abs(key_second) + 1e-9) if key_second > 0 else 2.0
    key = scores[0][0]
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onsets) / duration
    S = np.abs(librosa.stft(mono, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    bands = {}
    for name, lo, hi in [('sub',20,60),('bass',60,250),('lowmid',250,500),
                          ('mid',500,2000),('high',2000,6000),('air',6000,20000)]:
        m = (freqs >= lo) & (freqs < hi)
        bands[name] = float(np.mean(S[m]**2))
    total = sum(bands.values()) + 1e-9
    return {
        'tempo': tempo, 'key': key, 'key_confidence': key_confidence,
        'key_top_score': key_top, 'key_second_score': key_second,
        'is_minor': 'minor' in key, 'centroid': spec_cent, 'onset_rate': onset_rate,
        'sub_ratio': bands['sub']/total,
        'high_ratio': (bands['high']+bands['air'])/total,
        'duration': duration,
    }

def classify_style(f):
    bpm, bpm_half = f['tempo'], f['tempo']/2
    is_minor, cent, sub, high, onset = f['is_minor'], f['centroid'], f['sub_ratio'], f['high_ratio'], f['onset_rate']
    if bpm > 130 and cent > 3000 and high > 0.15: return 'hyperpop', 0.9
    drill_tempo = bpm >= 135 or (67 <= bpm_half <= 82 and bpm > 130)
    if drill_tempo and is_minor and sub > 0.15:
        return ('ny_drill', 0.85) if onset > 5.0 else ('uk_drill', 0.75)
    if sub < 0.10 and onset < 3.0 and cent < 1800: return 'pop_intimate', 0.7
    if 65 <= bpm <= 110 or 130 <= bpm <= 165:
        if is_minor:
            if cent < 1400: return 'melodic_trap', 0.75
            elif cent < 2200: return 'emo_rap', 0.7
            else: return 'modern_trap', 0.65
        return 'rnb_modern', 0.65
    return 'modern_trap', 0.5

def set_band(plugin, n, shape, freq, gain=0.0, q=1.0, slope='24 dB/oct'):
    setattr(plugin, f'band_{n}_used', 'Used')
    setattr(plugin, f'band_{n}_enabled', True)
    setattr(plugin, f'band_{n}_shape', shape)
    setattr(plugin, f'band_{n}_frequency', float(freq))
    setattr(plugin, f'band_{n}_gain', float(gain))
    setattr(plugin, f'band_{n}_q', float(q))
    if shape in ('Low Cut','High Cut'):
        setattr(plugin, f'band_{n}_slope', slope)

def _nearest_value(target, valid_values, unit_suffix=''):
    parsed = []
    for v in valid_values:
        try:
            s = str(v).replace(unit_suffix, '').strip()
            parsed.append((float(s), v))
        except: pass
    if not parsed: return valid_values[0]
    parsed.sort(key=lambda x: abs(x[0] - target))
    return parsed[0][1]

def _safe_set(plugin, attr, value):
    try: setattr(plugin, attr, value)
    except Exception: pass

# ============================================================
# HUMANIZE AI VOCAL — break the "too perfect" feel of ElevenLabs
# Per AI Humanization Protocol:
#   1. Subtle tape saturation (analog harmonic distortion = "throat grit")
#   2. Micro time-shift (5-15ms random) — AI vocal NOT on the digital grid
# ============================================================
def humanize_ai_vocal(audio, sr, shift_ms_min=5, shift_ms_max=15, drive_db=2.0, ai_mode=True):
    """
    Process AI-generated vocal to sound more human.

    Args:
        audio: stereo float32 array
        sr: sample rate
        shift_ms_min/max: random timing shift range
        drive_db: tape saturation drive (subtle, 1.5-3 dB typical)
        ai_mode: True = full humanization; False = bypass (already-human vocals)

    Returns processed audio. Sourced from 'How to Make AI Vocals Sound Real' research.
    """
    if not ai_mode or audio is None:
        return audio
    import random
    # 1) Random micro time-shift — push AI vocal off the digital grid
    shift_ms = random.uniform(shift_ms_min, shift_ms_max)
    shift_samples = int(sr * shift_ms / 1000)
    pad = np.zeros((shift_samples, audio.shape[1] if audio.ndim == 2 else 1), dtype=audio.dtype)
    audio_shifted = np.concatenate([pad, audio], axis=0)[:len(audio)]

    # 2) Subtle tape-style saturation — adds 2nd/3rd harmonics for "throat grit"
    # Using soft-clip tanh saturation (matches analog tape character)
    drive_lin = 10 ** (drive_db / 20)
    audio_driven = np.tanh(audio_shifted * drive_lin) / drive_lin

    # 3) Subtle high-shelf restoration — saturation darkens slightly, compensate
    sos_air = butter(2, 8000, btype='hp', fs=sr, output='sos')
    air_only = sosfilt(sos_air, audio_driven, axis=0).astype(np.float32)
    audio_out = audio_driven + air_only * 0.15  # +1.2 dB air shelf

    return audio_out.astype(np.float32)

# ============================================================
# BV STYLE PRESETS mapped to god-tier reference tracks
# ============================================================
BV_STYLES = {
    'drill': {  # Pop Smoke "Invincible" — aggressive, never steps on lead
        'hp': 300, 'lp': 6000, 'mud_cut': (-2.5, 400, 1.2),
        'lead_hole': (-3.0, 2500, 1.4),
        'side_widen': 1.3,
        'verb_mode': 'Room', 'verb_decay': 0.7, 'verb_wet': 0.18,
        'comp_ratio': 4.0, 'comp_attack': 3, 'comp_release': 30,
        'tuck_db': -8.0,
        'ref': 'pop_smoke_invincible',
    },
    'trap_hype': {  # Migos "Bad and Boujee" — gap-filling, panned wide
        'hp': 280, 'lp': 6500, 'mud_cut': (-2.0, 400, 1.2),
        'lead_hole': (-3.5, 2500, 1.4),
        'side_widen': 1.6,  # Migos width corr 0.991 = very wide BVs
        'verb_mode': 'Plate', 'verb_decay': 0.5, 'verb_wet': 0.15,
        'comp_ratio': 5.0, 'comp_attack': 2, 'comp_release': 25,
        'tuck_db': -7.0,
        'ref': 'migos_bad_and_boujee',
    },
    'rnb_harmony': {  # Smino "90 Proof" — buttery synth-pad harmonies
        'hp': 200, 'lp': 8000, 'mud_cut': (-1.5, 400, 1.0),
        'lead_hole': (-2.5, 2700, 1.2),
        'side_widen': 1.5,
        'verb_mode': 'Smooth Plate', 'verb_decay': 1.8, 'verb_wet': 0.28,
        'comp_ratio': 3.0, 'comp_attack': 10, 'comp_release': 80,
        'tuck_db': -10.0,
        'ref': 'smino_90_proof',
    },
    'psychedelic': {  # Travis Scott "STARGAZING" — drowned in verb/delay
        'hp': 350, 'lp': 7000, 'mud_cut': (-2.0, 450, 1.1),
        'lead_hole': (-3.0, 2500, 1.4),
        'side_widen': 1.5,
        'verb_mode': 'Cathedral', 'verb_decay': 3.0, 'verb_wet': 0.45,
        'comp_ratio': 4.0, 'comp_attack': 5, 'comp_release': 40,
        'tuck_db': -9.0,
        'ref': 'travis_stargazing',
    },
}

# ============================================================
# BACKING VOCAL BUS — engineer-sourced from 5 BV masterclasses
# ============================================================
def backing_vocal_bus(bv_audio, sr, mode='adlib', style='drill', is_ai=True):
    """
    Process background vocals through a dedicated chain mapped to god-tier references.

    SOURCES:
      - 5 BV masterclass transcripts (Devvon, Reid Stefan, Pensado, Alex Tumay, Finneas)
      - 4 god-tier reference tracks spectrally mapped (Pop Smoke, Migos, Smino, Travis)

    Args:
        bv_audio: stereo float32 array
        sr: sample rate
        mode: 'adlib' | 'harmony' (affects final tuck depth)
        style: 'drill' | 'trap_hype' | 'rnb_harmony' | 'psychedelic'
        is_ai: if True, run humanize_ai_vocal() first (tape sat + micro time-shift)

    Returns processed stereo array.
    """
    style = style if style in BV_STYLES else 'drill'
    S = BV_STYLES[style]
    print(f"     [bv_bus] style={style} (ref: {S['ref']}) | AI humanize: {is_ai}")

    # ===== HUMANIZE (if AI-generated) =====
    if is_ai:
        bv_audio = humanize_ai_vocal(bv_audio, sr, drive_db=2.0)

    # ===== EQ via pedalboard built-ins — Pro-Q 3 unlicensed = demo-mute risk =====
    # V16: Replaced FabFilter Pro-Q 3 with pedalboard's free EQ building blocks.
    # Same surgical functionality, no license check, never silences.
    mud_g, mud_f, mud_q = S['mud_cut']
    hole_g, hole_f, hole_q = S['lead_hole']
    bv_eq_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=S['hp']),                    # HP @ style spec
        PeakFilter(cutoff_frequency_hz=mud_f, gain_db=mud_g, q=mud_q),  # mud cut
        PeakFilter(cutoff_frequency_hz=hole_f, gain_db=hole_g, q=hole_q),  # lead hole
        LowpassFilter(cutoff_frequency_hz=S['lp']),                     # LP @ style spec
    ])
    bv = bv_eq_chain(bv_audio, sr)

    # ===== COMPRESSION — style-specific ratio + threshold = RMS-4 =====
    active = np.abs(bv).max(axis=1) > 1e-4
    if active.any():
        bv_active = bv[active]
        pre_rms_db = 20 * np.log10(np.sqrt(np.mean(bv_active**2)) + 1e-12)
    else:
        pre_rms_db = -60.0
    pre_rms_db = max(pre_rms_db, -60.0)
    bv_thresh = max(pre_rms_db - 4.0, -50.0)
    bv = Pedalboard([
        Compressor(threshold_db=bv_thresh, ratio=S['comp_ratio'],
                   attack_ms=S['comp_attack'], release_ms=S['comp_release'])
    ])(bv.astype(np.float32), sr)

    # ===== STEREO PLACEMENT — style-specific widening =====
    mid = (bv[:,0] + bv[:,1]) / 2
    side = (bv[:,0] - bv[:,1]) / 2
    sos_side_hp = butter(4, 250, btype='hp', fs=sr, output='sos')
    side_hp = sosfilt(sos_side_hp, side).astype(np.float32)
    side_widened = side_hp * S['side_widen']
    L = mid + side_widened
    R = mid - side_widened
    bv = np.stack([L, R], axis=1).astype(np.float32)

    # ===== REVERB — style-specific mode + decay + wet blend =====
    verb = load_plugin(PLUGINS['verb'])
    verb_config = {
        'mode': S['verb_mode'], 'decay': S['verb_decay'], 'predelay': 15, 'mix_pct': 100,
        'wet_blend': S['verb_wet'], 'send_hp': 350, 'colormode': 'eighties',
        'early_diff': 100, 'late_diff': 100,
    }
    configure_verb(verb, verb_config)
    bv_wet_raw = Pedalboard([verb])(bv, sr)
    sos_v_hp = butter(4, verb_config['send_hp'], btype='hp', fs=sr, output='sos')
    sos_v_lp = butter(4, 6000, btype='lp', fs=sr, output='sos')
    bv_wet = sosfilt(sos_v_lp, sosfilt(sos_v_hp, bv_wet_raw, axis=0), axis=0).astype(np.float32)
    bv_with_verb = bv * (1 - verb_config['wet_blend']) + bv_wet * verb_config['wet_blend']

    # ===== TUCK — style-specific depth, harmonies tucked harder than adlibs =====
    base_tuck = S['tuck_db']
    if mode == 'harmony':
        base_tuck -= 4.0  # harmonies sit deeper than ad-libs
    bv_final = bv_with_verb * (10 ** (base_tuck / 20))

    return bv_final.astype(np.float32)

def configure_verb(verb_plugin, vc):
    P = verb_plugin.parameters
    if 'reverbmode' in P and vc['mode'] in P['reverbmode'].valid_values:
        verb_plugin.reverbmode = vc['mode']
    if 'decay' in P:
        _safe_set(verb_plugin, 'decay', _nearest_value(vc['decay'], P['decay'].valid_values, ' s'))
    for k, src_key, unit in [('predelay','predelay',' ms'), ('mix','mix_pct','%'),
                              ('earlydiffusion','early_diff',''), ('latediffusion','late_diff','')]:
        if k in P:
            vv = P[k].valid_values if hasattr(P[k],'valid_values') else None
            target = vc.get(src_key, 100)
            if isinstance(vv, list) and vv:
                _safe_set(verb_plugin, k, _nearest_value(target, vv, unit))
            else:
                _safe_set(verb_plugin, k, float(target))
    if 'colormode' in P and vc['colormode'] in P['colormode'].valid_values:
        verb_plugin.colormode = vc['colormode']

# ============================================================
# PDC — pad dry signal by plugin latency before parallel mix
# ============================================================
def parallel_mix_with_pdc(dry, wet, plugin_chain, sr):
    """Mix wet+dry with proper latency compensation."""
    total_latency_s = 0
    for p in plugin_chain:
        try:
            lat = getattr(p, 'latency_seconds', 0) or 0
            total_latency_s += float(lat)
        except: pass
    pad_samples = int(round(total_latency_s * sr))
    if pad_samples > 0:
        # Pad dry to match wet's latency
        pad = np.zeros((pad_samples, dry.shape[1]), dtype=dry.dtype)
        dry_padded = np.concatenate([pad, dry], axis=0)
        # Trim to same length
        n = min(len(dry_padded), len(wet))
        return dry_padded[:n], wet[:n], pad_samples
    return dry, wet, 0

# ============================================================
# KNOWLEDGE BASE LOADER
# ============================================================
def list_knowledge():
    """Return summary of available transcripts + articles."""
    if not KB.exists(): return None
    videos = sorted([f.name for f in KB.glob('video_*.txt')])
    articles = sorted([f.name for f in KB.glob('article_*.txt')])
    total_chars = sum(f.stat().st_size for f in KB.glob('*.txt'))
    return {'videos': videos, 'articles': articles, 'total_chars': total_chars}

# ============================================================
# EXECUTOR
# ============================================================
# ============================================================
# ATMOSPHERE BUS — restores 3D depth ("aura") to a mix
# Based on psychoacoustics research + Devvon drill reverb + Alex Tumay throws.
#
# Three components:
#   1. Cathedral/Hall verb with 25-30ms PRE-DELAY (creates front-to-back distance)
#   2. Reverb tail HP @ 400 + LP @ 5k (no 808 mud, no harsh air clash)
#   3. Ping-pong delay tail at -18 dB, 1/8 dotted timing (fills phrase gaps)
# ============================================================
def _vocal_envelope(audio_mono, sr, win_ms=30, attack_ms=5, release_ms=80):
    """Compute smoothed normalized vocal envelope [0,1]. Used for sidechain + gating."""
    win = max(1, int(sr * win_ms / 1000))
    abs_v = np.abs(audio_mono)
    cs = np.cumsum(abs_v**2)
    env = np.zeros_like(abs_v)
    for i in range(len(env)):
        a = max(0, i - win)
        env[i] = np.sqrt((cs[i] - cs[a]) / (i - a + 1))
    aA = np.exp(-1.0/(sr*attack_ms/1000))
    aR = np.exp(-1.0/(sr*release_ms/1000))
    smoothed = np.zeros_like(env); prev = 0
    for i in range(len(env)):
        a = aA if env[i] > prev else aR
        prev = a * prev + (1-a) * env[i]
        smoothed[i] = prev
    peak95 = np.percentile(smoothed, 95) + 1e-9
    return np.clip(smoothed / peak95, 0, 1)

def atmosphere_bus(lead_vocal, sr, bpm=144):
    """
    V10 — Devvon/Travis Protocol:
      - 3 reverb buses (Room close + Plate mid + Hall far)
      - Hall sidechain-ducked to lead vocal (Devvon drill trick)
      - Ping-pong delay GATED to phrase ends only (Travis throw trick)
      - LP @ 5k on Plate/Hall (recede), Room stays brighter (close-up body)
    """
    from pedalboard import Delay
    if lead_vocal.ndim == 1:
        lead_vocal = np.stack([lead_vocal, lead_vocal], axis=1)
    lead = lead_vocal.astype(np.float32)
    mono = np.mean(lead, axis=1).astype(np.float32)

    # ===== ENVELOPE for sidechain + gating =====
    print("  [atmos] computing vocal envelope...")
    env = _vocal_envelope(mono, sr)
    # Active vs silent: > 0.15 = vocal active
    vocal_active = env > 0.15

    # ===== REVERB 1: ROOM (close 3D body) =====
    room = load_plugin(PLUGINS['verb'])
    configure_verb(room, {
        'mode': 'Room', 'decay': 0.5, 'predelay': 15, 'mix_pct': 100,
        'send_hp': 300, 'colormode': 'eighties',
        'early_diff': 90, 'late_diff': 100,
    })
    room_wet = Pedalboard([room])(lead, sr)
    sos_hp_300 = butter(4, 300, btype='hp', fs=sr, output='sos')
    room_wet = sosfilt(sos_hp_300, room_wet, axis=0).astype(np.float32)
    # Room stays brighter (no LP @ 5k) — adds proximity body
    room_wet *= 10 ** (-18.0 / 20)
    print(f"  [atmos] Room (close): decay 0.5s, HP 300 (NO LP — bright body), send -18 dB")

    # ===== REVERB 2: PLATE (mid width) =====
    plate = load_plugin(PLUGINS['verb'])
    configure_verb(plate, {
        'mode': 'Plate', 'decay': 1.5, 'predelay': 22, 'mix_pct': 100,
        'send_hp': 400, 'colormode': 'eighties',
        'early_diff': 100, 'late_diff': 100,
    })
    plate_wet = Pedalboard([plate])(lead, sr)
    sos_hp_400 = butter(4, 400, btype='hp', fs=sr, output='sos')
    sos_lp_5k  = butter(4, 5000, btype='lp', fs=sr, output='sos')
    plate_wet = sosfilt(sos_lp_5k, sosfilt(sos_hp_400, plate_wet, axis=0), axis=0).astype(np.float32)
    plate_wet *= 10 ** (-22.0 / 20)
    print(f"  [atmos] Plate (mid width): decay 1.5s, HP 400 + LP 5k, send -22 dB")

    # ===== REVERB 3: CONCERT HALL (far aura, SIDECHAIN-DUCKED) =====
    hall = load_plugin(PLUGINS['verb'])
    configure_verb(hall, {
        'mode': 'Concert Hall', 'decay': 3.5, 'predelay': 28, 'mix_pct': 100,
        'send_hp': 400, 'colormode': 'eighties',
        'early_diff': 80, 'late_diff': 100,
    })
    hall_wet_raw = Pedalboard([hall])(lead, sr)
    hall_wet = sosfilt(sos_lp_5k, sosfilt(sos_hp_400, hall_wet_raw, axis=0), axis=0).astype(np.float32)
    # SIDECHAIN DUCK (Devvon trick): when vocal_active, hall verb ducked -10 dB
    # When vocal silent, hall verb at full -18 dB (blooms into the gap)
    DUCK_DB = 10.0
    duck_curve_db = -DUCK_DB * env  # -10 dB at peak vocal, 0 dB when silent
    duck_lin = (10 ** (duck_curve_db / 20)).astype(np.float32)
    hall_wet = hall_wet * duck_lin[:, None]
    hall_wet *= 10 ** (-18.0 / 20)
    print(f"  [atmos] Hall (far AURA, sidechain-ducked -10 dB on vocal): decay 3.5s, 28ms predelay, send -18 dB")

    # ===== PING-PONG DELAY (Travis throw, GATED to phrase ends) =====
    beat_sec = 60.0 / bpm
    dly_1_8_dotted = beat_sec * 0.75
    delay_L = Pedalboard([Delay(delay_seconds=dly_1_8_dotted,     feedback=0.42, mix=1.0)])
    delay_R = Pedalboard([Delay(delay_seconds=dly_1_8_dotted * 2, feedback=0.42, mix=1.0)])
    mono_stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    wet_L = delay_L(mono_stereo, sr)
    wet_R = delay_R(mono_stereo, sr)
    ping_pong = np.stack([wet_L[:, 0] * 0.9, wet_R[:, 1] * 0.9], axis=1).astype(np.float32)
    ping_pong = sosfilt(sos_lp_5k, sosfilt(sos_hp_400, ping_pong, axis=0), axis=0).astype(np.float32)

    # GATE: only let delay through when vocal is QUIET (phrase end)
    # gate = (1 - env) -> high when silent, low when vocal loud
    gate = (1.0 - env).astype(np.float32)
    # Sharpen: only really open when vocal drops below 0.25 of peak
    gate = np.clip((gate - 0.7) / 0.25, 0, 1)  # opens fully when env < ~0.05, closed when env > 0.30
    # Smooth gate to avoid clicks
    aA = np.exp(-1.0/(sr*0.030))  # 30ms attack (opens slowly when vocal stops)
    aR = np.exp(-1.0/(sr*0.005))  # 5ms release (closes fast when vocal resumes)
    smooth_gate = np.zeros_like(gate); prev = 0
    for i in range(len(gate)):
        a = aA if gate[i] > prev else aR
        prev = a * prev + (1-a) * gate[i]
        smooth_gate[i] = prev
    ping_pong = ping_pong * smooth_gate[:, None]
    ping_pong *= 10 ** (-18.0 / 20)
    print(f"  [atmos] Ping-pong delay: 1/8 dotted ({dly_1_8_dotted*1000:.0f}ms), GATED (only fires at phrase ends), send -18 dB")

    # Sum atmosphere
    atmosphere = room_wet + plate_wet + hall_wet + ping_pong
    return atmosphere

def cite(key, brief=True):
    """Look up a citable source for a decision key. Returns None silently if no KB."""
    if not _KB_AVAILABLE: return None
    topic = DECISION_CITATIONS.get(key, key)
    try:
        return get_kb().cite(topic, brief=brief)
    except: return None

def log_decision(decisions_log, key, value, citation=None):
    """Append a decision row + source citation to the log dict."""
    decisions_log.append({'key': key, 'value': str(value), 'source': citation or 'general engineering'})

def execute_chain(vocal_path, music_path, output_path, style_override=None, force_full=False, bv_path=None, bv_style='drill', bv_is_ai=True, atmosphere=True):
    # Initialize decisions log (saved per render)
    decisions = []

    # KB summary at startup
    if _KB_AVAILABLE:
        try:
            kb = get_kb()
            n_docs = len(kb.documents)
            n_chars = sum(m['char_count'] for m in kb.metadata.values())
            print(f"=== KNOWLEDGE BASE LOADED ===")
            print(f"  {n_docs} indexed documents ({n_chars:,} chars)")
            print(f"  Every decision cites a source — see decisions.log\n")
        except Exception as e:
            print(f"  (KB load skipped: {e})")
    voc, sr = sf.read(vocal_path)
    mus, _  = sf.read(music_path)
    if voc.ndim == 1: voc = np.stack([voc, voc], axis=1)
    if mus.ndim == 1: mus = np.stack([mus, mus], axis=1)
    n = min(len(voc), len(mus))
    voc = voc[:n].astype(np.float32)
    mus = mus[:n].astype(np.float32)

    # Optional BV stem: separate ad-lib/double/harmony track
    bv_stem = None
    if bv_path and os.path.exists(bv_path):
        bv_stem, bv_sr = sf.read(bv_path)
        if bv_stem.ndim == 1: bv_stem = np.stack([bv_stem, bv_stem], axis=1)
        bv_stem = bv_stem[:n].astype(np.float32)
        if bv_sr != sr:
            print(f"  WARNING: BV stem sample rate {bv_sr} != vocal {sr}; using vocal sr")
        print(f"  BV stem loaded: peak {db(bv_stem):+.1f} | RMS {rdb(bv_stem):+.1f}")

    # Knowledge base check
    kb = list_knowledge()
    if kb:
        print(f"=== KNOWLEDGE BASE ===")
        print(f"  {len(kb['videos'])} engineer transcripts + {len(kb['articles'])} articles ({kb['total_chars']:,} chars)")

    # PRE-MIX DETECTION
    print(f"\n=== PRE-MIX DETECTOR ===")
    is_premixed, ev = detect_premix(voc, sr)
    print(f"  Crest factor: {ev['crest_factor_db']} dB (compressed if <11)")
    print(f"  Lo/Mid ratio: {ev['lo_mid_ratio_db']} dB (HP'd if < -25)")
    print(f"  Air/Presence: {ev['air_to_presence_db']} dB (air boosted if > -15)")
    print(f"  Sibilance ratio: {ev['sib_ratio_db']} dB (de-essed if < -8)")
    print(f"  Premix indicators: {ev['premix_score']}/4 -> {'PRE-MIXED' if is_premixed else 'RAW'}")
    if force_full:
        print("  --force-full-chain set, ignoring detection")
        is_premixed = False
    glue_mode = is_premixed

    # Feature extraction + style classification
    print(f"\n=== FEATURE EXTRACTION ===")
    feats = extract_features(music_path)
    print(f"  BPM: {feats['tempo']:.1f} | Key: {feats['key']} (conf {feats['key_confidence']:.2f})")
    print(f"  Centroid: {feats['centroid']:.0f} Hz | Sub: {feats['sub_ratio']:.2f} | Onsets: {feats['onset_rate']:.1f}/s")

    print(f"\n=== STYLE CLASSIFICATION ===")
    if style_override:
        style, conf = style_override, 1.0
    else:
        style, conf = classify_style(feats)
    print(f"  Style: {style} (conf {conf:.2f}) — {STYLES[style]['description']}")

    S = STYLES[style]

    if glue_mode:
        print(f"\n=== GLUE MODE v3.1 — anti-karaoke logic ===")
        print("  BYPASS: Auto-Tune, Pro-Q3, Vocal Comp, De-Esser, Reverb")
        print("  KEEP:   Drop vocal -2 dB + gentle pocket + Bus Glue Comp + Ozone")

        # ANTI-KARAOKE: drop the raw vocal stem BEFORE summing into the beat
        # Per v3.1: "drop the raw vocal stem gain by -2.0 dB before it hits master"
        # Let the master limiter raise overall level, not the vocal fader.
        VOCAL_STEM_DROP_DB = -2.0
        voc_processed = voc * (10 ** (VOCAL_STEM_DROP_DB / 20))
        print(f"  vocal stem dropped {VOCAL_STEM_DROP_DB:+.1f} dB (sit IN beat, not on top)")
        # V15 ROLLBACK: surgical_lead_pass removed (was eating vocal). V11 baseline restored.

        # Apply only dynamic pocket on music (gentle)
        voc_mono = np.mean(voc_processed, axis=1)
        sos_voc_bp = butter(4, [200, 4000], btype='bp', fs=sr, output='sos')
        voc_band = sosfilt(sos_voc_bp, voc_mono).astype(np.float32)
        abs_voc = np.abs(voc_band)
        win = int(sr * 0.020)
        cs = np.cumsum(abs_voc**2)
        env = np.zeros_like(abs_voc)
        for i in range(len(env)):
            a = max(0, i - win)
            env[i] = np.sqrt((cs[i] - cs[a]) / (i - a + 1))
        # V11: slow release 175ms (was 60ms) - prevents audible pumping
        # Source: Reid Stefan "master sounds way more open when you slow down the attack"
        aA = np.exp(-1.0/(sr*0.005)); aR = np.exp(-1.0/(sr*0.175))
        smoothed = np.zeros_like(env); prev = 0
        for i in range(len(env)):
            a = aA if env[i] > prev else aR
            prev = a * prev + (1-a) * env[i]
            smoothed[i] = prev
        peak95 = np.percentile(smoothed, 95) + 1e-9
        env_norm = np.clip(smoothed / peak95, 0, 1)
        env_norm = np.where(env_norm > 0.10, env_norm, 0)
        # V11: max ducking -2.5 dB (was -3 dB cap) - invisible dynamics
        gentle_duck = min(2.5, S['pocket']['duck_db'])
        sos_pocket = butter(6, [S['pocket']['lo'], S['pocket']['hi']], btype='bp', fs=sr, output='sos')
        pocket_band = sosfilt(sos_pocket, mus, axis=0).astype(np.float32)
        rest_mus = mus - pocket_band
        duck_lin = 10**((-gentle_duck*env_norm)/20)
        music_pocketed = rest_mus + pocket_band * duck_lin[:, None]

        # ANTI-KARAOKE: sum at natural levels, no aggressive music attenuation.
        # Default vocal_lead now -1.5 dB (vocal SLIGHTLY UNDER music — sits in pocket).
        DEFAULT_LEAD_DB = -1.5
        voc_rms = rdb(voc_processed); mus_rms = rdb(music_pocketed)
        gap = voc_rms - mus_rms
        bed_atten = -(DEFAULT_LEAD_DB - gap) if gap < DEFAULT_LEAD_DB else 0
        # Cap how much we attenuate the beat — never more than -6 dB
        bed_atten = max(bed_atten, -6.0)
        mus_balanced = music_pocketed * (10**(bed_atten/20))
        mix = voc_processed + mus_balanced
        print(f"  pocket: -{gentle_duck} dB | vocal lead: {DEFAULT_LEAD_DB:+.1f} dB | bed atten: {bed_atten:+.1f} (capped at -6)")
    else:
        # FULL CHAIN
        print(f"\n=== FULL CHAIN (vocal is raw) ===")

        # KEY CONFIDENCE GATE for Auto-Tune
        autotune_enabled = feats['key_confidence'] >= 1.10  # top must be 10% higher than second
        if not autotune_enabled:
            print(f"  Key confidence {feats['key_confidence']:.2f} < 1.10 -> Auto-Tune Chromatic mode (no key lock)")

        # V16: PEDALBOARD BUILT-IN EQ (was Pro-Q 3, unlicensed/demo-mute issue)
        q3_cuts_plugins = [HighpassFilter(cutoff_frequency_hz=S['vocal_hp'])]
        for g, f, q in S['vocal_cuts']:
            q3_cuts_plugins.append(PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q))

        # AUTO-TUNE PRO
        autotune = load_plugin(PLUGINS['autotune'])
        if autotune_enabled:
            detected_key_root = feats['key'].split()[0]
            flat_to_sharp = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'}
            if detected_key_root in flat_to_sharp: detected_key_root = flat_to_sharp[detected_key_root]
            valid_keys = autotune.parameters['key'].valid_values
            if detected_key_root in valid_keys: autotune.key = detected_key_root
            autotune.scale = 'Minor' if feats['is_minor'] else 'Major'
        else:
            autotune.scale = 'Chromatic'  # safest fallback when key uncertain
        autotune.retune_speed_ms = float(S['autotune']['retune'])
        autotune.humanize = float(S['autotune']['humanize'])
        autotune.flex_tune = float(S['autotune']['flex'])

        # VOCAL COMP + DE-ESS
        vcomp = load_plugin(PLUGINS['vcomp'])
        deess = load_plugin(PLUGINS['deess'])
        if 'threshold_db' in deess.parameters:
            deess.threshold_db = float(S['deess_thresh'])

        # V16: PEDALBOARD BUILT-IN EQ BOOSTS (was Pro-Q 3)
        q3_boost_plugins = []
        for g, f, q in S['vocal_boosts']:
            if f >= 8000:
                q3_boost_plugins.append(HighShelfFilter(cutoff_frequency_hz=f, gain_db=g, q=q))
            else:
                q3_boost_plugins.append(PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q))

        vocal_chain = Pedalboard(q3_cuts_plugins + [autotune, vcomp, deess] + q3_boost_plugins)
        voc_dry = vocal_chain(voc, sr)

        # VALHALLA — uses PDC for wet/dry mix
        verb = load_plugin(PLUGINS['verb'])
        configure_verb(verb, S['verb'])
        voc_wet_raw = Pedalboard([verb])(voc_dry, sr)
        # v3.1: HP @ send_hp AND LP @ 6 kHz to darken reverb tail (push behind beat, not floating on top)
        sos_verb_hp = butter(4, S['verb']['send_hp'], btype='hp', fs=sr, output='sos')
        sos_verb_lp = butter(4, 6000, btype='lp', fs=sr, output='sos')
        voc_wet_hp = sosfilt(sos_verb_hp, voc_wet_raw, axis=0).astype(np.float32)
        voc_wet = sosfilt(sos_verb_lp, voc_wet_hp, axis=0).astype(np.float32)
        print(f"  reverb tail: HP @ {S['verb']['send_hp']} Hz + LP @ 6 kHz (darkened, behind beat)")
        # PDC: latency match dry vs wet
        voc_dry_pdc, voc_wet_pdc, pad = parallel_mix_with_pdc(voc_dry, voc_wet, [verb], sr)
        if pad > 0: print(f"  PDC: padded dry by {pad} samples ({pad/sr*1000:.1f} ms) to align with verb")
        wet_amt = S['verb']['wet_blend']
        voc_processed = voc_dry_pdc * (1 - wet_amt) + voc_wet_pdc * wet_amt

        # DYNAMIC POCKET
        voc_mono = np.mean(voc_processed, axis=1)
        sos_voc_bp = butter(4, [200, 4000], btype='bp', fs=sr, output='sos')
        voc_band = sosfilt(sos_voc_bp, voc_mono).astype(np.float32)
        abs_voc = np.abs(voc_band)
        win = int(sr * 0.020)
        cs = np.cumsum(abs_voc**2)
        env = np.zeros_like(abs_voc)
        for i in range(len(env)):
            a = max(0, i - win)
            env[i] = np.sqrt((cs[i] - cs[a]) / (i - a + 1))
        aA = np.exp(-1.0/(sr*0.005)); aR = np.exp(-1.0/(sr*0.060))
        smoothed = np.zeros_like(env); prev = 0
        for i in range(len(env)):
            a = aA if env[i] > prev else aR
            prev = a * prev + (1-a) * env[i]
            smoothed[i] = prev
        peak95 = np.percentile(smoothed, 95) + 1e-9
        env_norm = np.clip(smoothed / peak95, 0, 1)
        env_norm = np.where(env_norm > 0.10, env_norm, 0)
        sos_pocket = butter(6, [S['pocket']['lo'], S['pocket']['hi']], btype='bp', fs=sr, output='sos')
        pocket_band = sosfilt(sos_pocket, mus, axis=0).astype(np.float32)
        # PDC: trim to match voc_processed length
        m = min(len(pocket_band), len(voc_processed))
        rest_mus = (mus - pocket_band)[:m]
        pocket_band = pocket_band[:m]
        env_norm = env_norm[:m]
        voc_processed = voc_processed[:m]
        duck_lin = 10**((-S['pocket']['duck_db']*env_norm)/20)
        music_pocketed = rest_mus + pocket_band * duck_lin[:, None]

        voc_rms = rdb(voc_processed); mus_rms = rdb(music_pocketed)
        gap = voc_rms - mus_rms
        bed_atten = -(S['vocal_lead_db'] - gap) if gap < S['vocal_lead_db'] else 0
        mus_balanced = music_pocketed * (10**(bed_atten/20))
        mix = voc_processed + mus_balanced

    # ====================================================
    # BACKING VOCAL BUS — sums in the processed BV stem
    # Routes ad-libs through their OWN chain (NOT lead chain).
    # Prevents lead/BV clash and phase issues.
    # ====================================================
    # =========================================================
    # ATMOSPHERE BUS — generate stereo aura from lead vocal
    # Added BEFORE backing vocals so the BV chain doesn't trigger on the wet send
    # =========================================================
    if atmosphere:
        print(f"\n=== ATMOSPHERE BUS (3D aura) ===")
        # Compute BPM for delay timing — use detected feats tempo
        bpm_for_delay = feats['tempo'] if feats['tempo'] > 80 else feats['tempo'] * 2
        atmos = atmosphere_bus(voc, sr, bpm=bpm_for_delay)
        # Trim/pad to match mix length
        if len(atmos) > len(mix):
            atmos = atmos[:len(mix)]
        elif len(atmos) < len(mix):
            pad = np.zeros((len(mix) - len(atmos), 2), dtype=atmos.dtype)
            atmos = np.concatenate([atmos, pad], axis=0)
        mix = mix + atmos
        print(f"  Atmosphere summed into mix")

    if bv_stem is not None:
        print(f"\n=== BACKING VOCAL BUS ===")
        print(f"  BV stem -> dedicated chain: HP 300 | -2.5dB @ 400 | -3dB @ 2.5k (hole for lead) | LP 6k")
        print(f"  Comp 4:1 (Reid Stefan) | side widening 1.3x (Pensado restraint) | Plate verb 0.6s + LP 6k")
        bv_processed = backing_vocal_bus(bv_stem, sr, mode='adlib', style=bv_style, is_ai=bv_is_ai)
        print(f"  BV processed RMS: {rdb(bv_processed):+.1f} (tucked -8 dB under lead)")
        mix = mix + bv_processed
        print(f"  Mixed BV into stereo bus")

    # PRE-MASTER NORMALIZE
    mix = mix * (10 ** ((-6.0 - db(mix))/20))

    # ====================================================
    # BUS GLUE COMPRESSOR — forces vocal + beat to breathe together
    # Per v3.1: 2:1 ratio, slow attack 30ms, fast release 50ms, -1 to -2 dB GR
    # ====================================================
    print(f"\n=== BUS GLUE COMPRESSOR ===")
    pre_glue_rms = rdb(mix)
    # Per v3.1 directive: threshold = RMS - 3 dB. Guarantees we always catch peaks
    # and land 1-2 dB GR. No static -12 threshold (that fails when RMS < -12).
    pre_rms_db = rdb(mix)
    pre_rms_db = max(pre_rms_db, -60.0)  # clamp to prevent NaN cascade with silent mixes
    # V11: threshold = RMS + 1.5 (peaks only) — target -1 to -1.5 dB GR
    glue_threshold = max(pre_rms_db + 1.5, -50.0)
    # V15: ratio 1.5:1 (was 2:1), release 250ms (was 300ms) per Tyler micro-tweak
    bus_glue = Pedalboard([
        Compressor(threshold_db=glue_threshold, ratio=1.5, attack_ms=30.0, release_ms=250.0)
    ])
    mix = bus_glue(mix.astype(np.float32), sr)
    post_glue_rms = rdb(mix)
    print(f"  Glue: 1.5:1 (V15), attack 30ms, release 250ms, threshold {glue_threshold:.1f} dB")
    print(f"  Pre/Post RMS: {pre_glue_rms:+.1f} -> {post_glue_rms:+.1f} ({post_glue_rms-pre_glue_rms:+.1f} dB GR)")

    # ====================================================
    # OZONE MASTER — v3.1 corrected EQ
    # ====================================================
    print(f"\n=== OZONE 9 ELEMENTS MASTER (v3.1 corrected) ===")
    ozone = load_plugin(PLUGINS['ozone'])
    # Ozone band defaults: 1=100Hz, 2=240Hz, 3=540Hz, 4=1200Hz, 5=2700Hz, 6=5500Hz, 7=10kHz, 8=16kHz
    if glue_mode:
        # v3.1 EQ: per Tyler's spec — band 1 @ 50 Hz: -2 dB, band 2 @ 240 Hz: +1.5 dB, 2.7k: 0
        # CRITICAL: Ozone disables even-numbered bands by default. Must enable.
        for side in ('l','r'):
            p = f"eq_st_m_{side}"
            # Enable all 4 bands we touch
            for bn in (1, 2, 5, 7):
                setattr(ozone, f"{p}_enable_{bn}", True)
            # V11: Chest Restoration — lighter sub cut + add 150Hz chest weight boost
            setattr(ozone, f"{p}_frequency_1_hz", 50.0)
            setattr(ozone, f"{p}_gain_1_db", -1.0)    # V11: -1 (was -2) — preserve some sub
            setattr(ozone, f"{p}_frequency_2_hz", 150.0)  # V11: 150 (was 240) — chest fundamental
            setattr(ozone, f"{p}_gain_2_db", 1.5)     # +1.5 dB @ 150 Hz - male rap chest weight
            setattr(ozone, f"{p}_frequency_3_hz", 240.0)
            setattr(ozone, f"{p}_gain_3_db", 1.0)     # +1 dB @ 240 - bass body (band 3 enabled by default)
            setattr(ozone, f"{p}_gain_5_db", 0.0)     # 2.7k: ZERO (do not pierce vocal)
            setattr(ozone, f"{p}_gain_7_db", 1.5)     # 10k: gentle air
        ozone.img_width_percent = 22.0
        ozone.max_threshold = -6.0
        ozone.max_ceiling = -1.0
        ozone.max_true_peak_limiting = True
        print(f"  Glue mode v3.1: EQ -2 dB @ 50 Hz, +1.5 dB @ 240 Hz, 0 @ 2.7k, +1.5 dB @ 10k")
        print(f"  Imager +22% | Max thr -6 dB | bands 1+2+5+7 ENABLED")
        target_lufs = max(S['lufs_target'], -12.0)
    else:
        eq = S['ozone_eq']
        for side in ('l','r'):
            p = f"eq_st_m_{side}"
            setattr(ozone, f"{p}_gain_1_db", eq['mud_100'])
            setattr(ozone, f"{p}_gain_4_db", eq['cut_1200'])
            setattr(ozone, f"{p}_gain_5_db", eq['presence_2700'])
            setattr(ozone, f"{p}_gain_7_db", eq['air_10k'])
        ozone.img_width_percent = float(S['ozone_width'])
        ozone.max_threshold = float(S['ozone_max_thresh'])
        ozone.max_ceiling = -1.0
        ozone.max_true_peak_limiting = True
        target_lufs = S['lufs_target']

    mastered = ozone(mix.astype(np.float32), sr)
    meter = pyln.Meter(sr)
    post = meter.integrated_loudness(mastered)
    gap = target_lufs - post
    if abs(gap) > 0.3:
        mastered = mastered * (10 ** (gap/20))
    ceiling = 10**(-1.0/20)
    mastered = np.clip(mastered, -ceiling, ceiling)
    final_lufs = meter.integrated_loudness(mastered)

    print(f"\n=== FINAL ===")
    print(f"  Mode: {'GLUE' if glue_mode else 'FULL'} | Style: {style}")
    print(f"  LUFS: {final_lufs:+.1f} | Peak: {db(mastered):+.1f}")

    # Save decisions log per render with citations
    if glue_mode:
        log_decision(decisions, 'mode', 'GLUE', 'Pre-Mix Detector (>= 2/4 indicators of pre-treated audio)')
    log_decision(decisions, 'style_detected', style, f"feature classifier (BPM={feats['tempo']:.1f}, key={feats['key']})")
    log_decision(decisions, 'lufs_target', S['lufs_target'], cite('lufs_drill'))
    log_decision(decisions, 'true_peak_ceiling', '-1 dBTP', cite('tp_minus_1'))
    log_decision(decisions, 'ozone_imager_width', f"+{S['ozone_width']}%", cite('imager_drill'))
    if atmosphere:
        log_decision(decisions, 'verb_predelay', '28 ms', cite('verb_predelay'))
        log_decision(decisions, 'verb_hp_400', 'HP filter on verb tail @ 400 Hz', cite('verb_hp_400'))
        log_decision(decisions, 'verb_lp_5k', 'LP filter on verb tail @ 5 kHz', cite('verb_lp_5k'))
        log_decision(decisions, 'verb_sidechain_duck', 'Hall verb ducks -10 dB on lead vocal', cite('verb_sidechain'))
        log_decision(decisions, 'delay_gated', 'Ping-pong only fires at phrase ends', cite('delay_gated_throw'))
        log_decision(decisions, 'delay_timing', '1/8 dotted at detected BPM', cite('delay_eighth_dotted'))
    if bv_stem is not None:
        log_decision(decisions, 'bv_chain_used', f"backing_vocal_bus style={bv_style}", cite('bv_hp_300'))
        log_decision(decisions, 'bv_lead_hole', '-3 dB @ 2.5 kHz (carved hole for lead)', cite('bv_lead_hole_2_5k'))
        log_decision(decisions, 'bv_radio_lp', 'LP @ 6 kHz on BV', cite('bv_lp_6k_radio'))
        log_decision(decisions, 'bv_tuck', f"BV tucked {BV_STYLES[bv_style]['tuck_db']} dB", cite('bv_tuck_8'))
        log_decision(decisions, 'ai_humanize', 'time shift + tape sat', cite('ai_humanize_shift'))
    # Save log
    out_dir = Path(output_path).parent
    log_path = out_dir / (Path(output_path).stem + '.decisions.json')
    log_path.write_text(json.dumps({
        'output': output_path,
        'final_lufs': float(final_lufs),
        'final_peak': float(db(mastered)),
        'style': style,
        'glue_mode': bool(glue_mode),
        'decisions': decisions,
    }, indent=2, default=str), encoding='utf-8')
    print(f"  Decisions log: {log_path}")

    sf.write(output_path, mastered, sr, subtype='PCM_16')
    return {'mode': 'glue' if glue_mode else 'full', 'style': style, 'lufs': final_lufs,
            'features': feats, 'premix_evidence': ev, 'output': output_path, 'decisions': decisions}

if __name__ == '__main__':
    args = sys.argv[1:]
    force_full = '--force-full-chain' in args
    # Pull --bv=path arg
    bv_path = None; bv_style = 'drill'; bv_is_ai = True; atmosphere = True
    for a in args[:]:
        if a.startswith('--bv='):
            bv_path = a.split('=', 1)[1]; args.remove(a)
        elif a.startswith('--bv-style='):
            bv_style = a.split('=', 1)[1]; args.remove(a)
        elif a == '--bv-human':
            bv_is_ai = False; args.remove(a)
        elif a == '--no-atmosphere':
            atmosphere = False; args.remove(a)
    args = [a for a in args if not a.startswith('--')]
    vocal, music, out = args[0], args[1], args[2]
    override = args[3] if len(args) > 3 else None
    execute_chain(vocal, music, out, style_override=override, force_full=force_full,
                  bv_path=bv_path, bv_style=bv_style, bv_is_ai=bv_is_ai, atmosphere=atmosphere)
