import json

# Пути к файлам
RAW_TEXTS = "data/raw_texts.jsonl"            # ваши исходные texts
LS_EXPORT = "data/annotations_from_label.json"    # экспорт из LS
OUTPUT    = "data/annotations.json"           # итоговый файл для обучения

# Загружаем тексты (по порядку)
texts = []
with open(RAW_TEXTS, encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        texts.append(entry["text"])

# Загружаем разметку LS
ls_data = json.load(open(LS_EXPORT, encoding="utf-8"))

annotations = []
for idx, item in enumerate(ls_data):
    text = texts[idx]
    ents = []
    # каждый annotation может содержать несколько result'ов
    for ann in item.get("annotations", []):
        for res in ann.get("result", []):
            v = res["value"]
            ents.append({
                "start": v["start"],
                "end":   v["end"],
                "label": v["labels"][0]
            })
    annotations.append({
        "text":     text,
        "entities": ents
    })

# Сохраняем в формате для train_ner.py
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"Готово {len(annotations)} записей в {OUTPUT}")
