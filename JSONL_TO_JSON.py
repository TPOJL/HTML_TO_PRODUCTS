import json
import re

def clean_text(text):
    # Удаляем нулевые байты и непечатаемые символы (кроме пробелов и переносов строк)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

jsonl_file = "data/raw_texts.jsonl"
label_studio_json = "data/raw_texts.json"

tasks = []
with open(jsonl_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            # Очищаем текст от запрещенных символов
            if "text" in data:
                data["text"] = clean_text(data["text"])
            tasks.append({"data": data})
        except json.JSONDecodeError:
            print(f"Skipping invalid line: {line}")

with open(label_studio_json, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"Файл {label_studio_json} успешно создан!")