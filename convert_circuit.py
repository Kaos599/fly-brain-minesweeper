import json

import urllib.request

url = "https://raw.githubusercontent.com/cobanov/flyjump/main/public/data/connectome/graph.json"
dst = r"data/malecns_circuit.json"

print(f"Downloading from {url}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))

with open(dst, "w", encoding="utf-8") as out:
    json.dump(data, out, indent=2)

print(f"Successfully saved {dst}: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
