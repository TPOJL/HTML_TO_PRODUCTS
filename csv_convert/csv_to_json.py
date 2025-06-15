# # scripts/csv_to_json.py
import json, csv

csv_file = 'data/annotations_corrected.csv'

temp = {}
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        i = int(row['doc_id'])
        if i not in temp:
            temp[i] = {'text': row['text'], 'entities': []}
        temp[i]['entities'].append({
            'start': int(row['start']),
            'end': int(row['end']),
            'label': row['label']
        })
# Сборка списка документов по порядку индексов
data = [temp[i] for i in sorted(temp)]
json.dump(data, open('data/annotations_from_label.json','w', encoding='utf-8'), ensure_ascii=False, indent=2)