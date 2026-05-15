"""
Physically download transcripts + article texts to knowledge_base/.
Engineer-grounded research input for smart_mix.py.

Sources:
- 10 YouTube videos (engineer interviews + plugin masterclasses)
- 10 articles (Sound On Sound, plugin manuals, ITU specs)
"""
import os, sys, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

KB = Path(__file__).parent / "knowledge_base"
KB.mkdir(exist_ok=True)

# =========================================================================
# 10 YOUTUBE VIDEO SOURCES
# =========================================================================
YT_VIDEOS = [
    # 1. Max Lord / Juice WRLD engineer
    ("01_max_lord_juice_wrld_pensados_491", "TSauoFu2X6Q"),
    # 2. Rob Kinelski / Billie Eilish mix engineer
    ("02_rob_kinelski_billie_pensados_412", "kIShzImfGqQ"),
    # 3. Sean Costello / Valhalla DSP
    ("03_sean_costello_valhalla_dsp", "L7Z8KFjdPN8"),
    # 4. Alex Tumay / Young Thug engineer
    ("04_alex_tumay_young_thug_mixing", "5fEs0LBgFCM"),
    # 5. FabFilter Pro-Q 3 / Dan Worrall
    ("05_dan_worrall_fabfilter_eq_masterclass", "p7zfUDl_J4w"),
    # 6. Juice WRLD $10K Vocal Chain (Archer Beats)
    ("06_juice_wrld_10k_vocal_chain", "EET4yoKCiKg"),
    # 7. Help Me Devvon - NY Drill Vocals
    ("07_devvon_ny_drill_mixing", "_PqGGd8LK08"),
    # 8. Waves Audio - Rob Kinelski groove
    ("08_rob_kinelski_waves_groove", "rZxK_yPdnxg"),
    # 9. Pensado's Place - Rob Kinelski 412 (alt source if 02 fails)
    ("09_rob_kinelski_pensados_412_alt", "kIShzImfGqQ"),
    # 10. Puremix Inside The Mix Rob Kinelski
    ("10_rob_kinelski_puremix_inside", "rZxK_yPdnxg"),
]

# =========================================================================
# 10 ARTICLE SOURCES
# =========================================================================
ARTICLES = [
    ("01_valhalla_vintage_verb_modes", "https://valhalladsp.com/2023/02/10/valhallavintageverb-the-modes/"),
    ("02_fabfilter_pro_q3_dynamic_eq", "https://www.fabfilter.com/help/pro-q/using/dynamic-eq"),
    ("03_antares_autotune_pro_manual", "https://www.antarestech.com/documentation/auto-tune-pro/basic-auto-mode-controls"),
    ("04_horiamc_lufs_2026_guide", "https://www.horiamc.com/blog/lufs-guide-2026-streaming-mastering"),
    ("05_vocal_market_streaming_lufs", "https://thevocalmarket.com/blogs/how-to/how-to-mix-vocals-for-streaming-lufs-loudness-2026"),
    ("06_genesis_mix_lab_spotify_mastering", "https://genesismixlab.com/ai-mastering/mastering-for-spotify/"),
    ("07_beatstorapon_rap_mastering_2026", "https://beatstorapon.com/blog/rap-mastering-settings-2025-professional-targets-presets-and-platform-delivery-for-rap-trap-rb/"),
    ("08_rys_up_audio_trap_vocals_2026", "https://rysupaudio.com/blogs/news/how-to-mix-trap-vocals"),
    ("09_musicguymixing_rap_vocal_eq", "https://www.musicguymixing.com/how-to-eq-rap-vocals/"),
    ("10_soundoracle_autotune_rap", "https://soundoracle.net/blogs/soundoracle-net-blog/the-best-autotune-settings-for-rappers-in-2020"),
]

def fetch_youtube_transcript(slug, video_id):
    """Fetch transcript via youtube-transcript-api; fall back gracefully."""
    out = KB / f"{slug}.txt"
    if out.exists():
        print(f"  [skip] {slug} (cached)")
        return True
    try:
        api = YouTubeTranscriptApi()
        # Newer API: fetch returns FetchedTranscript with .snippets
        fetched = api.fetch(video_id, languages=['en','en-US','en-GB'])
        if hasattr(fetched, 'snippets'):
            entries = fetched.snippets
            text = ' '.join(s.text for s in entries)
        else:
            entries = fetched
            text = ' '.join(e['text'] if isinstance(e, dict) else e.text for e in entries)
        text = re.sub(r'\s+', ' ', text).strip()
        out.write_text(text, encoding='utf-8')
        print(f"  [ok]   {slug} ({len(text)} chars)")
        return True
    except Exception as e:
        print(f"  [fail] {slug}: {type(e).__name__}: {str(e)[:80]}")
        # Fallback: yt-dlp auto-captions
        try:
            import subprocess
            result = subprocess.run(
                ['yt-dlp', '--write-auto-subs', '--skip-download', '--sub-langs', 'en.*',
                 '--sub-format', 'vtt', '-o', str(KB / f'{slug}.%(ext)s'),
                 f'https://www.youtube.com/watch?v={video_id}'],
                capture_output=True, text=True, timeout=60)
            # Find the VTT file
            for vtt in KB.glob(f'{slug}*.vtt'):
                lines = []
                with open(vtt, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('WEBVTT') or '-->' in line: continue
                        if line.startswith('Kind:') or line.startswith('Language:'): continue
                        line = re.sub(r'<[^>]+>', '', line)
                        if line: lines.append(line)
                seen = set(); deduped = []
                for l in lines:
                    if l not in seen: seen.add(l); deduped.append(l)
                text = ' '.join(deduped)
                out.write_text(text, encoding='utf-8')
                vtt.unlink()  # remove the VTT after extraction
                print(f"  [yt-dlp fallback ok] {slug} ({len(text)} chars)")
                return True
        except Exception as e2:
            print(f"  [yt-dlp fallback fail] {e2}")
        return False

def fetch_article(slug, url):
    """Fetch article HTML, parse to readable text."""
    out = KB / f"article_{slug}.txt"
    if out.exists():
        print(f"  [skip] {slug} (cached)")
        return True
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(r.text, 'html.parser')
        # Strip nav/scripts/styles
        for tag in soup(['script','style','nav','header','footer','aside','form']):
            tag.decompose()
        # Extract main content
        main = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|post|article|entry'))
        text = (main or soup).get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        out.write_text(f"URL: {url}\n\n{text}", encoding='utf-8')
        print(f"  [ok]   {slug} ({len(text)} chars)")
        return True
    except Exception as e:
        print(f"  [fail] {slug}: {type(e).__name__}: {str(e)[:80]}")
        return False

if __name__ == '__main__':
    print(f"Knowledge base: {KB}")
    print(f"\n=== YouTube transcripts ({len(YT_VIDEOS)}) ===")
    yt_ok = 0
    for slug, vid in YT_VIDEOS:
        if fetch_youtube_transcript(slug, vid): yt_ok += 1
    print(f"\n=== Articles ({len(ARTICLES)}) ===")
    art_ok = 0
    for slug, url in ARTICLES:
        if fetch_article(slug, url): art_ok += 1
        time.sleep(0.5)
    print(f"\nResult: {yt_ok}/{len(YT_VIDEOS)} videos + {art_ok}/{len(ARTICLES)} articles in {KB}")
