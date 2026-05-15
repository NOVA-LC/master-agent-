# Master Agent — Engineering Research Notes

Source data extracted from real engineer interviews via yt-dlp auto-caption scraping.
Plus official plugin documentation via WebFetch.

---

## Juice WRLD / Emo Rap (Max Lord — Pensado's Place #491)

**Vocal chain (hardware):**
- GML mic pre
- Tubetech tube preamp
- Bay Neve 1073 pre
- Bay 1176 compressor
- Shadow Hills mono opto compressor
- Reverb: **Lexicon RMX 16**

**Mic:** U67 (sometimes U87 — "swapped op-amps to differentiate")
**Saturation:** "necessary"
**Most annoying frequency to deal with:** "everything in the middle"
**Quote:** *"a really kind of dirty and not taken care of 1073 will have versus of a bay 1073"*

**Mapped to our VSTs:**
- Pro-Q 3 high-mid surgical cut (mids are problematic)
- ValhallaVintageVerb mode = **Dirty Plate** (emulates hardware warmth + grit, closest to RMX 16 character)
- Auto-Tune Pro retune ~15ms (Juice's signature melodic correction)

---

## Billie Eilish / Pop Intimate (Rob Kinelski — Pensado's Place #412)

**Plugins:**
- Fairchild 670 plugin (heavy compression)
- A "little bit of de-essing"
- UAD Neve 1073 (mic pre emulation)
- **FabFilter Pro-Q EQ — SUBTRACTIVE approach**
- Valhalla reverb (his cheap but effective pick)
- Waves Vocal Rider plugin (automation)
- Dangerous 2-Bus compressor (master bus)

**Workflow:**
- "Mix into the limiter / final chain from the start" (not at the end)
- Subtractive EQ first, broad sweep on the whole vocal
- Cuts around **2.5 kHz**
- Lots of automation rides on the vocals

**Quote:** *"I do a lot of subtractive if I'm doing EQ with her it's mostly subtractive because you take out a lot of 2.5"*

**Mapped to our VSTs:**
- Pro-Q 3 subtractive: -1.5 to -2 dB cut @ 2.5 kHz (the "Rob Kinelski cut")
- ValhallaVintageVerb mode = **Smooth Plate** (Valhalla docs: most transparent, "expensive sounding" vocals)
- Vocal lead very high (intimate/dry)
- Master LUFS more conservative (-14 standard, lets dynamics survive Spotify normalization)

---

## Young Thug / Modern Trap (Alex Tumay)

**Plugins:**
- **SSL bus compressor** — hard knee, ratios 4:1 → 10:1
- **FabFilter Pro-Q 3** — his go-to ("rubber band, makes it go into shapes")
- Fairchild 660 — "religious" use on vocals
- SPL Transient Designer instead of EQ on kicks
- **EMT 140 plate reverb** — "extremely bright, great density/fullness/thickness"
- Decapitator on 808 (parallel saturation)
- **Modulation on autotune'd vocal** (helps it blend — flanger or chorus subtle)
- TOA delays
- Stock Waves de-esser

**Key insight quote:** *"PR plate reverbs are great on vocals because they have a great density to them a great fullness and the thickness"*

**Quote on autotune blending:** *"using modulation like on an autotune vocal because it kind of just helps keep the ear distracted"*

**Mapped to our VSTs:**
- Pro-Q 3 for surgical EQ (his actual choice)
- ValhallaVintageVerb mode = **Bright Plate** (closest to EMT 140 character)
- Subtle modulation post-Auto-Tune (use Pro-Q dynamic band or saturation harmonic add)
- Master bus compression with SSL-style hard knee character

---

## ValhallaVintageVerb Modes (Sean Costello — Official Valhalla DSP docs)

| Mode | Character | Best For |
|------|-----------|----------|
| **Smooth Plate** | "Most transparent and naturalistic" | Pop intimate / "expensive" vocals |
| **Plate** | "Bright initial sound, high echo density, lush chorus" | Trap/Drill vocals (Alex Tumay's EMT 140 vibe) |
| **Dirty Plate** | "Warm, gritty artifacts, metallic sheen" | Juice WRLD / Emo rap (RMX 16 emulation) |
| **Chamber** | "Transparent, dense, less colored than Plate" | Melodic R&B |
| **Room** | "Medium diffusion, darker, chorused" | Drill (tight room) |
| **Cathedral** | "BIG yet clear, long open decays" | Rap ballad |
| **Hall1984** | Classic hall algorithm | Background pads |

**Quote from Sean Costello:** *"a well-tuned FDN would sound better than that"* — algorithm quality > preset choice. Both Smooth Plate and Smooth modes are recommended for "expensive sounding vocals."

---

## FabFilter Pro-Q 3 / Dan Worrall ("What Type of EQ")

**Key insights:**
- Pro-Q 3 is mathematically clean / free of nonlinearities → use it for SURGICAL work
- For saturation/character, use a "colorful" EQ (Volcano 3, Pultec emulation) — NOT Pro-Q 3
- Linear phase only when rolling your own crossover filters
- Don't double up bands on parallel — combined response distorts at crossover

**Mapped to our chain:**
- Pro-Q 3 used for CUTS (subtractive, clean) and broad BOOSTS (clean shelf)
- For warmth/saturation, rely on Antares Vocal Compressor's coloration

---

## Auto-Tune Pro Retune Speed (Antares Manual + SoundOracle)

| Style | Retune Speed | Humanize | Flex Tune |
|-------|--------------|----------|-----------|
| Drill hard tune | 0 ms | 0 | 0 |
| Modern trap | 10-20 ms | 20 | 20 |
| Melodic rap | 20-30 ms | 30 | 30 |
| Juice WRLD signature | 15 ms | 25 | 25 |
| Natural pop | 30+ ms | 40+ | 40+ |
| T-Pain hard | 0 ms | 0 | 0 |

**Quote (Antares manual):** *"Retune Speed = 0 = immediate changes from one pitch to another, suppresses vibrato"*
**Quote on Humanize:** *"Applies slower Retune Speed only during sustained portion of longer notes"* — keeps fast snap on transients but lets sustains breathe.

---

## Mastering Targets (Multiple Sources)

| Genre | LUFS Target | Peak Ceiling | Source |
|-------|-------------|--------------|--------|
| Spotify standard | -14 LUFS | -1 dBTP | ITU-R BS.1770-4 / Spotify spec |
| Apple Music | -16 LUFS | -1 dBTP | Apple Music guidelines |
| TikTok/YouTube | -13 to -14 LUFS | -1 dBTP | Platform docs |
| Commercial Hip-Hop reality | -7 to -9 LUFS | -1 dBTP | Genesis Mix Lab analysis |
| Drill production target | -10 to -11 LUFS | -1 dBTP | Oppro spec + Rys Up Audio guides |
| Hyperpop SLAM | -8 LUFS | -1 dBTP | Genre convention |
| Billie/intimate pop | -14 LUFS | -1 dBTP | Rob Kinelski approach + Spotify spec |

---

## Vocal Lead Above Music Bed

**Industry standard:** 3-6 dB above loudest instrument (RMS)
**Vocal momentary LUFS during peaks:** -10 to -8 LUFS
**Source:** thevocalmarket.com 2026 vocal mixing guide + horiamc.com LUFS 2026 guide

For Tyler's track: original mix had vocal at -21 dB RMS, music at -7.4 dB → vocal was **-14 dB BELOW** music. That's the root cause of "can't hear my voice" feedback.

---

## Sources

### YouTube Transcripts (yt-dlp auto-captions, full text extracted)
- [Pensado's Place #491 — Max Lord (Juice WRLD Engineer)](https://www.youtube.com/results?search_query=Pensado+Place+491+Max+Lord+Juice+WRLD)
- [Pensado's Place #412 — Rob Kinelski (Billie Eilish Mix Engineer)](https://www.youtube.com/results?search_query=Pensado+Place+412+Rob+Kinelski+Billie+Eilish)
- [The Magic Behind Valhalla Reverbs — Sean Costello](https://www.youtube.com/results?search_query=Sean+Costello+Valhalla+DSP+reverbs)
- [How Young Thug Gets Hits Mixed By Alex Tumay](https://www.youtube.com/results?search_query=Alex+Tumay+Young+Thug+mixing+vocals)
- [Dan Worrall — What Type of EQ Should I Use?](https://www.youtube.com/results?search_query=Dan+Worrall+EQ+FabFilter)

### Official Documentation (WebFetch)
- [ValhallaVintageVerb: The MODES — Valhalla DSP](https://valhalladsp.com/2023/02/10/valhallavintageverb-the-modes/)
- [Antares Auto-Tune Pro Manual](https://www.antarestech.com/documentation/auto-tune-pro/basic-auto-mode-controls)
- [FabFilter Pro-Q 3 Help](https://www.fabfilter.com/help/pro-q/using/dynamic-eq)

### Industry Articles
- [The Only LUFS Guide You Need in 2026 — Horia Stan](https://www.horiamc.com/blog/lufs-guide-2026-streaming-mastering)
- [Best Autotune Settings For Rappers — SoundOracle](https://soundoracle.net/blogs/soundoracle-net-blog/the-best-autotune-settings-for-rappers-in-2020)
- [Mix Vocals for Streaming — The Vocal Market](https://thevocalmarket.com/blogs/how-to/how-to-mix-vocals-for-streaming-lufs-loudness-2026)
- [Mastering for Spotify — Genesis Mix Lab](https://genesismixlab.com/ai-mastering/mastering-for-spotify/)
