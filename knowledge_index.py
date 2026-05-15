"""
KnowledgeBase — indexed, queryable access to all transcripts/articles in knowledge_base/.

Every parameter decision in smart_mix.py can call .cite(topic) to attach a source.
Build the index once at startup; query as needed during processing.

The agent's decisions become AUDITABLE: each render produces a decisions.log
mapping every chain parameter to the transcript/article that taught it.
"""
import re, json, os
from pathlib import Path
from collections import defaultdict, Counter

KB_DIR = Path(__file__).parent / "knowledge_base"
INDEX_FILE = Path(__file__).parent / "knowledge_index.json"

# Map filename-prefix patterns to topic categories
CATEGORY_TAGS = {
    'bv_':        ['backing_vocals', 'ad_libs', 'doubles', 'harmonies'],
    'aura_':      ['atmosphere', 'reverb', 'pre_delay', 'delay_throws', 'depth', '3d_space'],
    'ai_':        ['ai_humanization', 'ai_vocals', 'eleven_labs', 'tape_saturation'],
    'tag_':       ['tag_removal', 'phase_cancellation', 'loop_chop', 'demucs_slice'],
    'video_':     ['general_mixing', 'engineer_interview'],
    'article_':   ['mixing_theory', 'mastering_targets', 'LUFS', 'platform_specs'],
}

# Map specific filename keywords to engineer/source labels
SOURCE_LABELS = {
    'max_lord':     'Max Lord (Juice WRLD engineer) — Pensado #491',
    'rob_kinelski': 'Rob Kinelski (Billie Eilish mix engineer) — Pensado #412',
    'sean_costello':'Sean Costello (Valhalla DSP creator)',
    'alex_tumay':   'Alex Tumay (Young Thug engineer)',
    'dan_worrall':  'Dan Worrall (FabFilter — EQ masterclass)',
    'devvon':       'Help Me Devvon (drill/rap vocal mixing)',
    'reid_stefan':  'Reid Stefan (vocal doubles & harmonies)',
    'pensado':      "Dave Pensado (Pensado's Place)",
    'travis':       'Travis Scott vocal delay throws (Tumay technique)',
    'jaycen':       'Jaycen Joshua (R&B/Trap BV phase + panning)',
    'finneas':      'Finneas/Kinelski (Billie Eilish vocal stacks)',
    'valhalla':     'Valhalla DSP official mode documentation',
    'autotune':     'Antares Auto-Tune Pro manual',
    'fabfilter':    'FabFilter Pro-Q 3 help docs',
    'horiamc':      'Horia Stan — LUFS Guide 2026',
    'vocal_market': 'The Vocal Market — Streaming LUFS Guide 2026',
    'genesis':      'Genesis Mix Lab — Mastering for Spotify',
    'beatstorapon': 'BeatsToRapOn — Rap Mastering Settings 2026',
    'rys_up':       'Rys Up Audio — Trap Vocals 2026',
    'musicguy':     'Music Guy Mixing — Rap Vocal EQ',
    'soundoracle':  'SoundOracle — Best Autotune Settings for Rappers',
}

class KnowledgeBase:
    def __init__(self, kb_dir=KB_DIR):
        self.dir = Path(kb_dir)
        self.documents = {}  # filename -> raw text
        self.tokens = {}     # filename -> list of (word, char_offset)
        self.index = defaultdict(list)  # keyword -> [(filename, offset, context)]
        self.metadata = {}   # filename -> dict(source, category, char_count)
        self._build()

    def _read_text(self, path):
        """Read .txt or .vtt as cleaned plaintext."""
        if path.suffix == '.txt':
            return path.read_text(encoding='utf-8', errors='ignore')
        elif path.suffix == '.vtt':
            lines = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('WEBVTT') or '-->' in line: continue
                    if line.startswith('Kind:') or line.startswith('Language:'): continue
                    line = re.sub(r'<[^>]+>', '', line)
                    if line: lines.append(line)
            seen = set(); out = []
            for l in lines:
                if l not in seen: seen.add(l); out.append(l)
            return ' '.join(out)
        return ''

    def _categorize(self, fname):
        cats = []
        for prefix, tags in CATEGORY_TAGS.items():
            if fname.startswith(prefix):
                cats.extend(tags)
                break
        return cats

    def _source_label(self, fname):
        fl = fname.lower()
        for key, label in SOURCE_LABELS.items():
            if key in fl:
                return label
        return fname.replace('.txt','').replace('.vtt','').replace('_',' ')

    def _build(self):
        if not self.dir.exists(): return
        files = [f for f in self.dir.glob('*') if f.suffix in ('.txt','.vtt') and '-orig' not in f.name and 'en-en' not in f.name]
        for f in files:
            text = self._read_text(f)
            if len(text) < 200: continue  # skip tiny/empty
            self.documents[f.name] = text
            # Tokenize: words >= 3 chars, lowercase
            tokens = [(m.group(0).lower(), m.start()) for m in re.finditer(r'\b[a-zA-Z][a-zA-Z0-9-]{2,}\b', text)]
            self.tokens[f.name] = tokens
            # Index every token
            for word, off in tokens:
                self.index[word].append((f.name, off))
            self.metadata[f.name] = {
                'source': self._source_label(f.name),
                'categories': self._categorize(f.name),
                'char_count': len(text),
            }

    def query(self, terms, k=3, context_chars=150):
        """Return top-k passages matching ANY of the terms (multi-term AND-boost).
           Returns list of dict: {source, passage, score, file}.
        """
        if isinstance(terms, str):
            terms = [t.lower() for t in terms.split() if t.strip()]
        else:
            terms = [t.lower() for t in terms]
        # Find candidate (file, offset) regions that match
        candidates = defaultdict(list)  # file -> list of offsets
        for t in terms:
            for entry in self.index.get(t, []):
                fn, off = entry
                candidates[fn].append((t, off))
        # Score: number of distinct terms matched per file
        scored = []
        for fn, hits in candidates.items():
            terms_matched = set(t for t, _ in hits)
            score = len(terms_matched) * 10 + len(hits)
            # Pick a representative offset (the densest cluster of hits)
            offsets = sorted([o for _, o in hits])
            if not offsets: continue
            # Find best cluster (sliding window of 500 chars)
            best_off = offsets[0]
            best_density = 1
            for i, o in enumerate(offsets):
                density = sum(1 for x in offsets if abs(x - o) < 500)
                if density > best_density:
                    best_density = density; best_off = o
            text = self.documents[fn]
            s = max(0, best_off - context_chars)
            e = min(len(text), best_off + context_chars * 2)
            passage = text[s:e].strip()
            scored.append({
                'source': self.metadata[fn]['source'],
                'file': fn,
                'score': score,
                'passage': passage,
                'matched_terms': list(terms_matched),
            })
        scored.sort(key=lambda x: -x['score'])
        return scored[:k]

    def cite(self, topic, brief=True):
        """Return a one-line citation string for a topic, or None if no good source."""
        hits = self.query(topic, k=1)
        if not hits: return None
        h = hits[0]
        if brief:
            return f"{h['source']}"
        return f"{h['source']}: \"{h['passage'][:150]}...\""

    def summary(self):
        """Print index summary."""
        by_cat = defaultdict(list)
        for fn, meta in self.metadata.items():
            for cat in (meta['categories'] or ['uncategorized']):
                by_cat[cat].append(meta['source'])
        out = f"KnowledgeBase: {len(self.documents)} docs, {sum(m['char_count'] for m in self.metadata.values()):,} chars\n"
        for cat, srcs in sorted(by_cat.items()):
            srcs_uniq = sorted(set(srcs))
            out += f"  [{cat}] {len(srcs_uniq)} sources\n"
        return out

    def export_index(self, path=INDEX_FILE):
        """Save lightweight index to JSON for inspection."""
        payload = {
            'doc_count': len(self.documents),
            'total_chars': sum(m['char_count'] for m in self.metadata.values()),
            'documents': {fn: meta for fn, meta in self.metadata.items()},
            'keyword_count': len(self.index),
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return path

# Singleton convenience
_kb = None
def get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb

if __name__ == '__main__':
    import sys
    kb = KnowledgeBase()
    print(kb.summary())
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        print(f"\nQuery: {query!r}\n")
        for h in kb.query(query, k=5):
            print(f"  [{h['source']}]  (score {h['score']}, matched: {h['matched_terms']})")
            print(f"    ...{h['passage'][:300]}...")
            print()
    kb.export_index()
    print(f"\nIndex exported: {INDEX_FILE}")
