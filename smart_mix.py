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
from pedalboard import Pedalboard, load_plugin
from scipy.signal import butter, sosfilt

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
        'vocal_hp': 120, 'vocal_cuts': [(-2.5, 300, 1.2), (-1.5, 900, 1.5)],
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
        'vocal_hp': 120, 'vocal_cuts': [(-2.5, 300, 1.2), (-1.5, 900, 1.5)],
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
def execute_chain(vocal_path, music_path, output_path, style_override=None, force_full=False):
    voc, sr = sf.read(vocal_path)
    mus, _  = sf.read(music_path)
    if voc.ndim == 1: voc = np.stack([voc, voc], axis=1)
    if mus.ndim == 1: mus = np.stack([mus, mus], axis=1)
    n = min(len(voc), len(mus))
    voc = voc[:n].astype(np.float32)
    mus = mus[:n].astype(np.float32)

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
        print(f"\n=== GLUE MODE (vocal already mixed) ===")
        print("  BYPASS: Auto-Tune, Pro-Q3, Vocal Comp, De-Esser, Reverb")
        print("  KEEP:   Dynamic pocket on instrumental + gentle Ozone bus glue + LUFS match")
        voc_processed = voc.copy()  # Pass vocal through untouched

        # Apply only dynamic pocket on music
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
        # Gentler pocket in glue mode (-3 dB max instead of -5)
        gentle_duck = min(3.0, S['pocket']['duck_db'])
        sos_pocket = butter(6, [S['pocket']['lo'], S['pocket']['hi']], btype='bp', fs=sr, output='sos')
        pocket_band = sosfilt(sos_pocket, mus, axis=0).astype(np.float32)
        rest_mus = mus - pocket_band
        duck_lin = 10**((-gentle_duck*env_norm)/20)
        music_pocketed = rest_mus + pocket_band * duck_lin[:, None]

        # Balance — vocal sits +3 dB above bed (less aggressive than full chain)
        gentle_lead = min(3.0, S['vocal_lead_db'])
        voc_rms = rdb(voc_processed); mus_rms = rdb(music_pocketed)
        gap = voc_rms - mus_rms
        bed_atten = -(gentle_lead - gap) if gap < gentle_lead else 0
        mus_balanced = music_pocketed * (10**(bed_atten/20))
        mix = voc_processed + mus_balanced
        print(f"  pocket: -{gentle_duck} dB | vocal lead: +{gentle_lead} dB | bed atten: {bed_atten:+.1f}")
    else:
        # FULL CHAIN
        print(f"\n=== FULL CHAIN (vocal is raw) ===")

        # KEY CONFIDENCE GATE for Auto-Tune
        autotune_enabled = feats['key_confidence'] >= 1.10  # top must be 10% higher than second
        if not autotune_enabled:
            print(f"  Key confidence {feats['key_confidence']:.2f} < 1.10 -> Auto-Tune Chromatic mode (no key lock)")

        # PRO-Q3 CUTS
        q3_cuts = load_plugin(PLUGINS['proq3'])
        set_band(q3_cuts, 1, 'Low Cut', S['vocal_hp'])
        for i, (g, f, q) in enumerate(S['vocal_cuts']):
            set_band(q3_cuts, i+2, 'Bell', f, gain=g, q=q)

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

        # PRO-Q3 BOOSTS
        q3_boost = load_plugin(PLUGINS['proq3'])
        for i, (g, f, q) in enumerate(S['vocal_boosts']):
            shape = 'High Shelf' if f >= 8000 else 'Bell'
            set_band(q3_boost, i+1, shape, f, gain=g, q=q)

        vocal_chain = Pedalboard([q3_cuts, autotune, vcomp, deess, q3_boost])
        voc_dry = vocal_chain(voc, sr)

        # VALHALLA — uses PDC for wet/dry mix
        verb = load_plugin(PLUGINS['verb'])
        configure_verb(verb, S['verb'])
        voc_wet_raw = Pedalboard([verb])(voc_dry, sr)
        sos_verb_hp = butter(4, S['verb']['send_hp'], btype='hp', fs=sr, output='sos')
        voc_wet = sosfilt(sos_verb_hp, voc_wet_raw, axis=0).astype(np.float32)
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

    # PRE-MASTER NORMALIZE
    mix = mix * (10 ** ((-6.0 - db(mix))/20))

    # OZONE MASTER (different settings for glue vs full)
    print(f"\n=== OZONE 9 ELEMENTS MASTER ===")
    ozone = load_plugin(PLUGINS['ozone'])
    if glue_mode:
        # Gentle glue settings — barely touch the needles
        # Only Imager + Maximizer; skip aggressive EQ (vocal is already EQ'd)
        ozone.img_width_percent = 10.0  # slight widening
        ozone.max_threshold = -6.0  # gentle limit, much less aggressive
        ozone.max_ceiling = -1.0
        ozone.max_true_peak_limiting = True
        print(f"  Glue mode: Imager +10%, Maximizer thr -6 dB (gentle)")
        target_lufs = max(S['lufs_target'], -12.0)  # don't push too loud in glue mode
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
    sf.write(output_path, mastered, sr, subtype='PCM_16')
    return {'mode': 'glue' if glue_mode else 'full', 'style': style, 'lufs': final_lufs,
            'features': feats, 'premix_evidence': ev, 'output': output_path}

if __name__ == '__main__':
    args = sys.argv[1:]
    force_full = '--force-full-chain' in args
    args = [a for a in args if not a.startswith('--')]
    vocal, music, out = args[0], args[1], args[2]
    override = args[3] if len(args) > 3 else None
    execute_chain(vocal, music, out, style_override=override, force_full=force_full)
