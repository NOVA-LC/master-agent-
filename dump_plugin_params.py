"""
Dump every VST3 parameter (name, type, valid_values, default) to JSON.
Solves the "blind robot" problem — no guessing what params do.

Output: param_maps/<plugin>.json
"""
import json, sys, os
from pathlib import Path
from pedalboard import load_plugin

OUT = Path(__file__).parent / "param_maps"
OUT.mkdir(exist_ok=True)

PLUGINS = {
    'auto_tune_pro':     r"C:\Program Files\Common Files\VST3\Antares\Auto-Tune Pro.vst3",
    'auto_tune_vocal_eq':r"C:\Program Files\Common Files\VST3\Auto-Tune Vocal EQ.vst3",
    'fabfilter_pro_q3':  r"C:\Program Files\Common Files\VST3\FabFilter Pro-Q 3.vst3",
    'vocal_compressor':  r"C:\Program Files\Common Files\VST3\Auto-Tune Vocal Compressor.vst3",
    'vocal_de_esser':    r"C:\Program Files\Common Files\VST3\Vocal De-Esser.vst3",
    'valhalla_vintage':  r"C:\Program Files\Common Files\VST3\ValhallaVintageVerb.vst3",
    'ozone_9_elements':  r"C:\Program Files\Common Files\VST3\iZotope\Ozone 9 Elements.vst3",
}

for name, path in PLUGINS.items():
    try:
        p = load_plugin(path)
        params = {}
        for pname in p.parameters.keys():
            try:
                pr = p.parameters[pname]
                val = getattr(p, pname)
                entry = {'current': str(val)}
                if hasattr(pr, 'valid_values'):
                    vv = pr.valid_values
                    if isinstance(vv, list):
                        entry['type'] = 'enum' if any(isinstance(x,str) for x in vv) else 'discrete_numeric'
                        # Cap stored options at 30 for readability
                        entry['valid_values'] = [str(v) for v in (vv if len(vv) < 30 else vv[:5] + ['...'] + vv[-5:])]
                        entry['count'] = len(vv)
                if hasattr(pr, 'range'):
                    entry['range'] = str(pr.range)
                params[pname] = entry
            except Exception as e:
                params[pname] = {'error': str(e)[:80]}
        out_file = OUT / f"{name}.json"
        out_file.write_text(json.dumps(params, indent=2), encoding='utf-8')
        try:
            latency = getattr(p, 'latency_seconds', None)
            print(f"  {name}: {len(params)} params, latency={latency}s -> {out_file.name}")
        except:
            print(f"  {name}: {len(params)} params -> {out_file.name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
