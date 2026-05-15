"""
Composite Drill Target Rubric + Psychoacoustic Grader.

Grades a mix against the AVERAGE spectral/loudness/width fingerprint of 3 reference tracks:
  - Sleepy Hallow "2055"        (the gold standard — clean drill, perfect vocal balance)
  - Pop Smoke "Dior"             (aggressive Brooklyn drill, 808 + width reference)
  - Juice WRLD "Lucid Dreams"    (emo-rap vocal frequency response, 2-5 kHz body)

PSYCHOACOUSTIC RULES (per Fletcher-Munson + drill genre conventions):

1. SUB DOMINANCE IS NOT A BUG IN DRILL MIXES.
   The 808 is the lead instrument. Vocal sits slightly BEHIND it.
   Do not penalize a mix for having sub at 0 dB ref if that matches drill convention.

2. PERCEIVED LOUDNESS != FLAT FREQUENCY RESPONSE.
   At LUFS -10 to -15, the ear is LESS sensitive to bass and air, MORE sensitive to 2-4 kHz.
   A mix that's spectrally "flat" will sound thin. Drill mixes intentionally tilt down to compensate.

3. VOCAL "MASKING" BY 808 IS A FEATURE.
   The 808 occupying 40-80 Hz creates upward masking on the vocal's lower harmonics —
   this is what makes drill vocals sound "tight" and "in the pocket" rather than floating on top.

4. WIDTH < CENTER FOR LOW FREQUENCIES.
   Sub and kick should be near-mono. Width is for the highs and mids, not the bottom.
"""
import sys, os, json, numpy as np, soundfile as sf, librosa, pyloudnorm as pyln
from pathlib import Path

REFS = [
    (r"C:\Users\tyler\Downloads\refs\Sleepy Hallow - 2055 (Official Video).wav",      "Sleepy Hallow - 2055"),
    (r"C:\Users\tyler\Downloads\refs\Pop Smoke - Dior (Official Audio).wav",          "Pop Smoke - Dior"),
    (r"C:\Users\tyler\Downloads\refs\Juice WRLD ＂Lucid Dreams (Forget Me)＂ (Official Audio).wav", "Juice WRLD - Lucid Dreams"),
]
OUT_RUBRIC = Path(r"C:\Users\tyler\OneDrive\Desktop\master-agent-\composite_target.json")

def analyze(path):
    y, sr = sf.read(path)
    if y.ndim == 1: y = np.stack([y, y], axis=1)
    mono = np.mean(y, axis=1)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y[:min(len(y), sr*120)]))
    peak = float(20*np.log10(np.max(np.abs(y))+1e-12))
    rms = float(20*np.log10(np.sqrt(np.mean(y**2))+1e-12))
    if y.shape[1] == 2:
        m = (y[:,0]+y[:,1])/2; s = (y[:,0]-y[:,1])/2
        width = float(20*np.log10((np.sqrt(np.mean(s**2))+1e-12)/(np.sqrt(np.mean(m**2))+1e-12)))
        corr = float(np.corrcoef(y[:,0], y[:,1])[0,1])
    else:
        width = -99; corr = 1.0
    S = np.abs(librosa.stft(mono.astype(np.float32), n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    bands = {}
    for n, lo, hi in [('sub',20,60),('bass',60,250),('lowmid',250,500),
                       ('mid',500,2000),('high',2000,6000),('air',6000,20000)]:
        m = (freqs >= lo) & (freqs < hi)
        bands[n] = float(10*np.log10(np.mean(S[m]**2)+1e-12))
    # Normalize relative to loudest band
    mv = max(bands.values())
    bands_rel = {k: v - mv for k,v in bands.items()}
    return {'lufs': lufs, 'peak': peak, 'rms': rms, 'width': width, 'corr': corr,
            'bands_rel': bands_rel, 'crest': peak - rms}

def build_composite():
    print("=== BUILDING COMPOSITE TARGET ===")
    profiles = []
    for path, name in REFS:
        if not Path(path).exists():
            print(f"  MISSING: {name}"); continue
        a = analyze(path)
        profiles.append(a)
        print(f"  [{name}]")
        print(f"    LUFS {a['lufs']:+.2f} | Peak {a['peak']:+.2f} | Width {a['width']:+.2f} | Corr {a['corr']:+.3f}")
        for b in ['sub','bass','lowmid','mid','high','air']:
            print(f"    {b:<8} {a['bands_rel'][b]:+6.1f} dB")
    # Average
    composite = {
        'lufs':  float(np.mean([p['lufs'] for p in profiles])),
        'peak':  float(np.mean([p['peak'] for p in profiles])),
        'rms':   float(np.mean([p['rms'] for p in profiles])),
        'width': float(np.mean([p['width'] for p in profiles])),
        'corr':  float(np.mean([p['corr'] for p in profiles])),
        'crest': float(np.mean([p['crest'] for p in profiles])),
        'bands_rel': {b: float(np.mean([p['bands_rel'][b] for p in profiles]))
                       for b in ['sub','bass','lowmid','mid','high','air']},
        # Tolerance ranges (1 stdev)
        'lufs_std':  float(np.std([p['lufs'] for p in profiles])),
        'width_std': float(np.std([p['width'] for p in profiles])),
        'bands_std': {b: float(np.std([p['bands_rel'][b] for p in profiles]))
                       for b in ['sub','bass','lowmid','mid','high','air']},
        'sources': [n for _, n in REFS],
    }
    print(f"\n=== COMPOSITE (avg of {len(profiles)}) ===")
    print(f"  LUFS {composite['lufs']:+.2f} +/- {composite['lufs_std']:.2f}")
    print(f"  Width {composite['width']:+.2f} +/- {composite['width_std']:.2f}")
    for b in ['sub','bass','lowmid','mid','high','air']:
        print(f"  {b:<8} {composite['bands_rel'][b]:+6.1f} +/- {composite['bands_std'][b]:.1f} dB")
    OUT_RUBRIC.write_text(json.dumps(composite, indent=2), encoding='utf-8')
    print(f"\nSaved: {OUT_RUBRIC}")
    return composite

def grade(mix_path, composite):
    """
    Psychoacoustic-aware grading.
    Rules:
      - SUB DOMINANCE: NOT penalized in drill. Only penalized if more than 2 stdev above composite.
      - VOCAL MASKING: 2-5 kHz being slightly recessed is GOOD (vocal sits behind 808).
      - STEREO WIDTH: penalized only if more than 1 stdev below composite mean.
      - LUFS: penalized if outside +/- 1.5 LUFS of composite.
      - TRUE PEAK: penalized if above -1 dBTP (hard rule, ITU spec).
      - CREST FACTOR: NOT penalized as long as it's > 6 dB (not squashed).
    """
    print(f"\n=== GRADING: {Path(mix_path).name} ===")
    m = analyze(mix_path)
    print(f"  Measured: LUFS {m['lufs']:+.2f} | Width {m['width']:+.2f} | Crest {m['crest']:.1f}")
    points = 100
    penalties = []

    # LUFS gate
    lufs_dist = abs(m['lufs'] - composite['lufs'])
    lufs_tol = max(2.0, composite['lufs_std'] * 1.5)
    if lufs_dist > lufs_tol:
        p = min(8, int((lufs_dist - lufs_tol) * 2))
        points -= p
        penalties.append((p, f"LUFS off composite avg by {lufs_dist:.1f} dB (tol +/- {lufs_tol:.1f})"))

    # True peak (hard rule)
    if m['peak'] > -1.0:
        p = int((m['peak'] + 1.0) * 4)
        points -= p
        penalties.append((p, f"Peak {m['peak']:+.1f} > -1 dBTP (ITU spec violation)"))

    # Stereo width — penalize ONLY if too narrow (more mono than composite - 1 stdev)
    width_floor = composite['width'] - max(2.0, composite['width_std'])
    if m['width'] < width_floor:
        p = min(12, int((width_floor - m['width']) * 2))
        points -= p
        penalties.append((p, f"Stereo too narrow: {m['width']:+.1f} < composite floor {width_floor:+.1f}"))

    # Crest factor — penalize ONLY if squashed (< 6 dB)
    if m['crest'] < 6.0:
        p = int((6.0 - m['crest']) * 2)
        points -= p
        penalties.append((p, f"Crest {m['crest']:.1f} dB < 6 dB (squashed)"))

    # Band-by-band with PSYCHOACOUSTIC ADJUSTMENT
    print("  Spectral comparison (composite-relative):")
    for b in ['sub','bass','lowmid','mid','high','air']:
        diff = m['bands_rel'][b] - composite['bands_rel'][b]
        tol = max(3.0, composite['bands_std'][b] * 2)  # 2 stdev tolerance
        flag = ""
        # PSYCHOACOUSTIC OVERRIDE: don't penalize sub dominance in drill
        if b == 'sub' and diff > 0:
            flag = "OK (drill convention: 808 is lead)"
        # Don't penalize mid being slightly low — that's vocal sitting behind 808
        elif b in ('mid','lowmid') and diff < 0 and abs(diff) < tol + 3:
            flag = "OK (vocal masked by 808 = drill pocket)"
        elif abs(diff) > tol:
            p = min(6, int(abs(diff) / 3))
            points -= p
            penalties.append((p, f"{b}: {diff:+.1f} dB from composite (tol +/- {tol:.1f})"))
            flag = f"-{p}"
        else:
            flag = "OK"
        print(f"    {b:<8} {m['bands_rel'][b]:+6.1f} | composite {composite['bands_rel'][b]:+6.1f} | diff {diff:+5.1f} | {flag}")

    points = max(0, min(100, points))
    print(f"\n  PENALTIES:")
    if not penalties:
        print("    (none)")
    else:
        for p, reason in penalties:
            print(f"    -{p}  {reason}")
    print(f"\n  COMPOSITE-AWARE GRADE: {points}/100")
    return points, penalties

if __name__ == '__main__':
    if not OUT_RUBRIC.exists() or '--rebuild' in sys.argv:
        composite = build_composite()
    else:
        composite = json.loads(OUT_RUBRIC.read_text(encoding='utf-8'))
        print(f"Loaded composite from {OUT_RUBRIC.name}")
    # Grade arg
    for arg in sys.argv[1:]:
        if arg.endswith('.wav') and Path(arg).exists():
            grade(arg, composite)
