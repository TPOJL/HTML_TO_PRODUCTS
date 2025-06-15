# auto_annotate.py
import json
from transformers import pipeline

def auto_annotate(input_file, output_file, model_name='dslim/bert-base-NER'):
    # Загружаем предобученную NER-модель
    ner = pipeline('ner', model=model_name, tokenizer=model_name, aggregation_strategy='simple')
    output = []
    # Читаем JSONL: в каждой строке JSON с полями 'url' и 'text'
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get('text', '')
            if not text:
                continue
            # Применяем модель к тексту
            annotations = ner(text)
            # Собираем результирующие сущности как PRODUCT
            ents = []
            for ann in annotations:
                ents.append({
                    'start': ann['start'],
                    'end': ann['end'],
                    'label': 'PRODUCT'
                })
            output.append({'text': text, 'entities': ents})
    # Сохраняем в правильном формате JSON
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw_texts.jsonl', help='JSONL входных данных')
    parser.add_argument('--output', default='data/annotations_auto.json', help='Файл разметки JSON')
    parser.add_argument('--model', default='dslim/bert-base-NER', help='HuggingFace NER модель')
    args = parser.parse_args()
    auto_annotate(args.input, args.output, args.model)