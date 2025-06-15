import json

with open("data/annotations_auto_mine.json", encoding="utf-8") as f:
    data = json.load(f)

labelstudio_data = []

for item in data:
    text = item["text"]
    entry = {
        "data": {"text": text},
        "annotations": [{
            "result": []
        }]
    }
    for ent in item.get("entities", []):
        start = ent["start"]
        end = ent["end"]
        entry["annotations"][0]["result"].append({
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": text[start:end],
                "labels": ["PRODUCT"]
            }
        })
    labelstudio_data.append(entry)

with open("data/annotations_for_labelstudio_mine.json", "w", encoding="utf-8") as f:
    json.dump(labelstudio_data, f, ensure_ascii=False, indent=2)
