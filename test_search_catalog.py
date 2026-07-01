import json

with open("clean_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

print(f"Total items: {len(catalog)}")
verify_items = [item["name"] for item in catalog if "verify" in item["name"].lower()]
print(f"Verify items found ({len(verify_items)}):")
for name in verify_items[:10]:
    print(" -", name)
