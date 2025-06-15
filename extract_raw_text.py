import argparse
import json
import requests
from bs4 import BeautifulSoup

def get_text(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', required=True, help='Файл с URL по одной строке')
    parser.add_argument('--output', required=True, help='Выходной JSONL файл')
    parser.add_argument('--limit', type=int, default=100, help='Сколько первых URL обработать')
    args = parser.parse_args()

    count = 0
    with open(args.urls, 'r', encoding='utf-8') as f, \
         open(args.output, 'w', encoding='utf-8') as out:
        for url in f:
            if count >= args.limit:
                break
            url = url.strip()
            if not url:
                continue
            text = get_text(url)
            if text:
                record = {'url': url, 'text': text}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1