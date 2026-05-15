# master-agent

Context-aware production mixing agent. Auto-detects song style from raw stems and applies an engineer-sourced chain.

## Prime Directive: ENHANCE, DO NOT GENERATE

This agent processes and mixes **human-recorded vocals only**.

It will **never**:
- Auto-generate ad-libs or harmonies via ElevenLabs/Suno/voice cloning
- Add backing vocals the artist did not record
- Insert AI-cloned vocals into a mix automatically

ElevenLabs and similar generation tools remain available as **standalone tools** in the toolbag (`gen_adlibs.py`, `place_adlibs.py`) but are **never invoked from `smart_mix.py`**. Backing vocals are only mixed in when explicitly provided via `--bv=<path>` to a stem the artist recorded or approved.

The agent's job is to enhance what the artist did. Period.

---

## What it does

1. **Analyze** music stem: tempo, key, spectral profile, onset density, sub-dominance
2. **Classify** style — drill (NY/UK), modern trap, melodic trap, emo rap, pop intimate, hyperpop, modern R&B
3. **Execute** the matching processing chain — vocal EQ, Auto-Tune, compression, de-essing, reverb, dynamic pocket carving, mastering
4. **Master** to genre-appropriate LUFS / dBTP targets

## Why "engineer-grounded"

Every parameter is sourced from primary engineering data — not hallucinated. See [RESEARCH.md](RESEARCH.md) for the full source notes including:

- **Max Lord** (Juice WRLD engineer) — Pensado's Place #491 transcript
- **Rob Kinelski** (Billie Eilish mix engineer) — Pensado's Place #412 transcript
- **Alex Tumay** (Young Thug engineer) — vocal mixing breakdown
- **Sean Costello** (Valhalla DSP creator) — reverb algorithm interview
- **Dan Worrall** (FabFilter) — EQ masterclass
- **ValhallaVintageVerb** official mode documentation
- **Antares Auto-Tune Pro** manual
- **ITU-R BS.1770-4** loudness standard

## Plugins used (all VST3, loaded via pedalboard)

- iZotope Ozone 9 Elements (master EQ + Imager + Maximizer)
- FabFilter Pro-Q 3 (surgical vocal EQ — both subtractive + additive passes)
- Antares Auto-Tune Pro (pitch correction, key/scale auto-set from detected song key)
- Antares Auto-Tune Vocal Compressor (vocal-tuned dynamics)
- Antares Vocal De-Esser (sibilance control)
- ValhallaVintageVerb (genre-specific mode: Smooth Plate / Dirty Plate / Plate / Chamber / Room)

## Usage

```bash
python smart_mix.py <vocal.wav> <music.wav> <output.wav> [style_override]
```

- `vocal.wav` — raw vocal stem
- `music.wav` — instrumental stem
- `output.wav` — destination for mastered mix
- `style_override` (optional) — force one of: `ny_drill`, `uk_drill`, `melodic_trap`, `emo_rap`, `modern_trap`, `pop_intimate`, `hyperpop`, `rnb_modern`

## Output spec per style

| Style | LUFS | Auto-Tune Retune | Reverb Mode |
|-------|------|------------------|-------------|
| ny_drill | -10.0 | 0 ms (hard) | Room |
| uk_drill | -10.5 | 10 ms | Plate |
| emo_rap | -10.5 | 15 ms | **Dirty Plate** (Max Lord RMX 16 emulation) |
| modern_trap | -10.5 | 20 ms | Plate (EMT 140 vibe) |
| melodic_trap | -11.0 | 25 ms | Chamber |
| pop_intimate | -14.0 | 30 ms | **Smooth Plate** (Rob Kinelski Valhalla) |
| hyperpop | -8.0 | 0 ms | Plate |
| rnb_modern | -12.0 | 30 ms | Smooth Room |

True-peak ceiling: -1 dBTP for all (ITU-R BS.1770-4 spec).

## Dependencies

```bash
pip install pedalboard librosa soundfile pyloudnorm scipy numpy
```

FFmpeg (with shared DLLs for torchcodec if you extend with Demucs).
