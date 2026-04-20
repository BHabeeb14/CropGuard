import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
labels_path = ROOT / 'assets' / 'labels' / 'class_labels.txt'
advice_path = ROOT / 'assets' / 'advice' / 'advice.json'

labels = [l for l in labels_path.read_text(encoding='utf-8').splitlines() if l.strip()]
advice = json.loads(advice_path.read_text(encoding='utf-8'))

missing = [l for l in labels if l not in advice]
extra = [k for k in advice if not k.startswith('_') and k not in labels]

print(f"labels: {len(labels)}")
print(f"advice keys (non-meta): {sum(1 for k in advice if not k.startswith('_'))}")
print(f"missing: {missing}")
print(f"extra: {extra}")

sys.exit(1 if missing else 0)
