# scripts/json_to_csv.py
import json, csv

data = json.load(open('data/annotations_auto.json', encoding='utf-8'))
with open('data/annotations_auto.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['doc_id','start','end','label','text'])
    for i, doc in enumerate(data):
        for ent in doc['entities']:
            writer.writerow([i, ent['start'], ent['end'], ent['label'], doc['text']])