"""
remove_tag.py — surgical producer tag removal.

THREE METHODS (in order of audio preservation):

METHOD A — LOOP/CHOP (Numpy Splicer): SAFEST
  Trap beats are 4-bar loops. Instead of "erasing" the tag (which destroys audio),
  find a tagless musically-identical section and overwrite the tagged region.
  Uses sample-accurate crossfades (10-20ms) at splice points.

METHOD B — LOCALIZED DEMUCS (Surgical Strike): MEDIUM RISK
  Slice only the tagged 3-5 seconds out of the track. Run Demucs on just that
  slice (--two-stems vocals). Take the "no_vocals" stem. Crossfade back.
  Preserves audio quality on rest of track.

METHOD C — SMUDGE (Reverb/Delay Wash): LAST RESORT
  Drown the tag in heavy reverb + delay so it reads as atmospheric riser, not voice.
  Doesn't actually remove the tag — hides it.

USAGE:
    python remove_tag.py <input.wav> <output.wav> <tag_start_sec> <tag_end_sec> [--method=chop|demucs|smudge] [--bpm=144]
"""
import sys, os, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pedalboard import Pedalboard, load_plugin

VALHALLA = r"C:\Program Files\Common Files\VST3\ValhallaVintageVerb.vst3"

# =========================================================================
# METHOD A — LOOP/CHOP
# =========================================================================
def remove_tag_chop(audio_path, output_path, tag_start, tag_end, bpm=None, replacement_start=None):
    """
    Find a clean section musically-identical to the tagged section, splice it in.

    If replacement_start not provided:
      - Use BPM to compute 4-bar musical phrase length
      - Take replacement from tag_end (next cycle of the loop)
      - If less than 4 bars from end, try further into the song
    """
    print(f"\n=== METHOD A: LOOP/CHOP ===")
    y, sr = sf.read(audio_path)
    if y.ndim == 1: y = np.stack([y, y], axis=1)
    y = y.astype(np.float32)
    n = len(y)
    duration = n / sr

    tag_len_samples = int((tag_end - tag_start) * sr)
    tag_start_s = int(tag_start * sr)
    tag_end_s = int(tag_end * sr)

    if replacement_start is None:
        # Heuristic: trap loops at 4-bar boundaries
        if bpm:
            bar_sec = 60 / bpm * 4
            phrase_sec = bar_sec * 4  # 4-bar phrase
            print(f"  BPM={bpm} -> 4-bar phrase = {phrase_sec:.2f}s")
            replacement_start = tag_end  # next phrase cycle
        else:
            replacement_start = tag_end
        # Check if replacement fits
        if int(replacement_start * sr) + tag_len_samples > n:
            # Take from later in track
            replacement_start = duration - (tag_end - tag_start) - 2.0
            print(f"  Adjusted replacement to t={replacement_start:.2f}s (near end of track)")

    rep_start_s = int(replacement_start * sr)
    rep_end_s = rep_start_s + tag_len_samples
    if rep_end_s > n:
        print(f"  ERROR: replacement section {replacement_start:.2f}-{replacement_start+(tag_end-tag_start):.2f}s "
              f"extends past track end {duration:.2f}s")
        return False

    print(f"  Tag region: {tag_start:.2f}s - {tag_end:.2f}s ({(tag_end-tag_start)*1000:.0f}ms)")
    print(f"  Replacement source: {replacement_start:.2f}s - {replacement_start+(tag_end-tag_start):.2f}s")

    # Copy replacement section
    replacement = y[rep_start_s:rep_end_s].copy()

    # CROSSFADES at splice points (15ms)
    xfade_ms = 15
    xfade_samples = int(sr * xfade_ms / 1000)
    if xfade_samples > tag_len_samples // 4:
        xfade_samples = tag_len_samples // 4

    # Build the output
    out = y.copy()

    # Construct crossfade ramps
    fade_in = np.linspace(0, 1, xfade_samples)[:, None]
    fade_out = np.linspace(1, 0, xfade_samples)[:, None]

    # At tag_start: crossfade FROM the audio just BEFORE the tag TO the replacement
    pre_start = max(0, tag_start_s - xfade_samples)
    # Apply: out[pre_start:tag_start] keeps original
    # Then crossfade region: out[tag_start:tag_start + xfade_samples] = original * fade_out + replacement[:xfade] * fade_in
    # Then full replacement: out[tag_start + xfade : tag_end - xfade] = replacement[xfade : tag_len - xfade]
    # Then crossfade back at tag_end

    # Use simpler approach: zero-fade replacement edges
    rep_with_fades = replacement.copy()
    if xfade_samples > 0:
        rep_with_fades[:xfade_samples] *= fade_in
        rep_with_fades[-xfade_samples:] *= fade_out

    # Fade out original audio that we're about to overwrite (only at edges)
    orig_fade_region = out[tag_start_s:tag_end_s].copy()
    if xfade_samples > 0:
        # First xfade_samples: fade-out (will be crossed with rep fade-in)
        orig_fade_region[:xfade_samples] *= fade_out
        # Last xfade_samples: fade-in (will be crossed with rep fade-out)
        orig_fade_region[-xfade_samples:] *= fade_in
        # Middle: zero (will be entirely replaced)
        orig_fade_region[xfade_samples:-xfade_samples] *= 0
    else:
        orig_fade_region *= 0

    # Combine: original_with_holes + replacement_with_fades
    out[tag_start_s:tag_end_s] = orig_fade_region + rep_with_fades
    print(f"  Crossfades: {xfade_ms}ms ({xfade_samples} samples) at each splice point")

    # Save
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out *= 0.99 / peak
    sf.write(output_path, out, sr, subtype='PCM_16')
    print(f"  Saved: {output_path}")
    return True

# =========================================================================
# METHOD B — LOCALIZED DEMUCS
# =========================================================================
def remove_tag_demucs(audio_path, output_path, tag_start, tag_end, pad_seconds=0.5):
    """
    Slice the tagged region (+ padding), run Demucs on it ONLY, swap in the no-vocals stem.
    """
    print(f"\n=== METHOD B: LOCALIZED DEMUCS ===")
    y, sr = sf.read(audio_path)
    if y.ndim == 1: y = np.stack([y, y], axis=1)
    y = y.astype(np.float32)
    n = len(y)

    # Pad the slice for Demucs context
    slice_start = max(0, int((tag_start - pad_seconds) * sr))
    slice_end = min(n, int((tag_end + pad_seconds) * sr))
    slice_audio = y[slice_start:slice_end]
    print(f"  Slicing: {slice_start/sr:.2f}s - {slice_end/sr:.2f}s ({(slice_end-slice_start)/sr:.2f}s slice)")

    # Save slice to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        slice_path = os.path.join(tmpdir, 'slice.wav')
        sf.write(slice_path, slice_audio, sr, subtype='PCM_16')

        # Set up FFmpeg DLL path for torchcodec
        env = os.environ.copy()
        ff_bin = r"C:\Users\tyler\tools\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
        env['PATH'] = ff_bin + os.pathsep + env.get('PATH', '')

        # Run Demucs with --two-stems vocals (faster than 4-stem)
        out_demucs = os.path.join(tmpdir, 'demucs_out')
        os.makedirs(out_demucs, exist_ok=True)
        print(f"  Running Demucs on {(slice_end-slice_start)/sr:.2f}s slice (two-stems mode)...")
        cmd = [sys.executable, '-m', 'demucs', '--two-stems', 'vocals',
               '-n', 'htdemucs', '--out', out_demucs, slice_path]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if result.returncode != 0:
            print(f"  Demucs error: {result.stderr[:300]}")
            return False

        # Find the no_vocals.wav stem
        no_vox = None
        for p in Path(out_demucs).rglob('no_vocals.wav'):
            no_vox = p
            break
        if not no_vox:
            print(f"  No no_vocals.wav produced")
            return False

        clean_slice, slice_sr = sf.read(no_vox)
        if clean_slice.ndim == 1: clean_slice = np.stack([clean_slice, clean_slice], axis=1)
        if slice_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(slice_sr, sr)
            clean_slice = resample_poly(clean_slice, sr//g, slice_sr//g, axis=0)
        clean_slice = clean_slice[:slice_end-slice_start].astype(np.float32)

    # Crossfade clean_slice back into original
    xfade_samples = int(sr * 0.015)  # 15ms

    out = y.copy()
    out_section_start = slice_start
    out_section_end = slice_end

    # Apply crossfades on the clean_slice edges
    fade_in = np.linspace(0, 1, xfade_samples)[:, None]
    fade_out = np.linspace(1, 0, xfade_samples)[:, None]
    if len(clean_slice) >= 2 * xfade_samples:
        clean_slice[:xfade_samples] *= fade_in
        clean_slice[-xfade_samples:] *= fade_out
        out_orig = out[out_section_start:out_section_end].copy()
        out_orig[:xfade_samples] *= fade_out
        out_orig[-xfade_samples:] *= fade_in
        out_orig[xfade_samples:-xfade_samples] *= 0
        out[out_section_start:out_section_end] = out_orig + clean_slice
    else:
        out[out_section_start:out_section_end] = clean_slice

    print(f"  Crossfade: 15ms on each edge of the cleaned slice")
    peak = np.max(np.abs(out))
    if peak > 0.99: out *= 0.99 / peak
    sf.write(output_path, out, sr, subtype='PCM_16')
    print(f"  Saved: {output_path}")
    return True

# =========================================================================
# METHOD C — SMUDGE
# =========================================================================
def remove_tag_smudge(audio_path, output_path, tag_start, tag_end):
    """
    Apply heavy reverb + delay over the tag region. Doesn't remove, but disguises.
    """
    print(f"\n=== METHOD C: SMUDGE (reverb wash) ===")
    y, sr = sf.read(audio_path)
    if y.ndim == 1: y = np.stack([y, y], axis=1)
    y = y.astype(np.float32)
    tag_start_s = int(tag_start * sr)
    tag_end_s = int(tag_end * sr)

    # Extract tagged region
    tagged_region = y[tag_start_s:tag_end_s].copy()
    print(f"  Smudging {tag_start:.2f}s - {tag_end:.2f}s ({(tag_end-tag_start)*1000:.0f}ms)")

    # Apply heavy reverb to disguise tag
    verb = load_plugin(VALHALLA)
    try: verb.reverbmode = 'Cathedral'
    except: pass
    try: verb.decay = '4.00 s'
    except: pass
    try: verb.predelay = '20 ms'
    except: pass
    try: verb.mix = '85%'
    except: pass
    try: verb.colormode = 'eighties'
    except: pass

    smudged = Pedalboard([verb])(tagged_region, sr)
    # HP smudge so it doesn't muddy the low end
    sos = butter(4, 400, btype='hp', fs=sr, output='sos')
    smudged = sosfilt(sos, smudged, axis=0).astype(np.float32)

    # Crossfade smudged region back in
    xfade_samples = int(sr * 0.020)
    fade_in = np.linspace(0, 1, xfade_samples)[:, None]
    fade_out = np.linspace(1, 0, xfade_samples)[:, None]
    if len(smudged) >= 2 * xfade_samples:
        smudged[:xfade_samples] *= fade_in
        smudged[-xfade_samples:] *= fade_out

    out = y.copy()
    # Replace with smudge (don't sum — that would still have the tag)
    out[tag_start_s:tag_end_s] = smudged * 0.5  # tame the smudge volume
    peak = np.max(np.abs(out))
    if peak > 0.99: out *= 0.99 / peak
    sf.write(output_path, out, sr, subtype='PCM_16')
    print(f"  Saved: {output_path}")
    return True

# =========================================================================
# CLI
# =========================================================================
if __name__ == '__main__':
    args = sys.argv[1:]
    method = 'chop'
    bpm = None
    replacement_start = None
    for a in args[:]:
        if a.startswith('--method='):
            method = a.split('=',1)[1]; args.remove(a)
        elif a.startswith('--bpm='):
            bpm = float(a.split('=',1)[1]); args.remove(a)
        elif a.startswith('--rep='):
            replacement_start = float(a.split('=',1)[1]); args.remove(a)
    audio_in, audio_out, t_start, t_end = args[0], args[1], float(args[2]), float(args[3])
    print(f"=== TAG ERASER ===")
    print(f"  Input:  {audio_in}")
    print(f"  Output: {audio_out}")
    print(f"  Tag region: {t_start}-{t_end}s | Method: {method}")
    if method == 'chop':
        remove_tag_chop(audio_in, audio_out, t_start, t_end, bpm=bpm, replacement_start=replacement_start)
    elif method == 'demucs':
        remove_tag_demucs(audio_in, audio_out, t_start, t_end)
    elif method == 'smudge':
        remove_tag_smudge(audio_in, audio_out, t_start, t_end)
    else:
        print(f"Unknown method: {method}. Use chop|demucs|smudge")
        sys.exit(1)
