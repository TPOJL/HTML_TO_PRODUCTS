import json
import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional


class ProductScraper:
    def __init__(self):
        self.product_selectors = [
            {'selector': '[itemprop="name"]'},
            {'selector': 'div.product-name'},
            {'selector': 'h2.product-item-title'},
            {'selector': 'meta[property="og:title"]', 'attr': 'content'},
            {'selector': 'a.product-link', 'attr': 'data-product-name'}
        ]
        self.forbidden_re = re.compile(r'[\$%\*@#&\^\+=<>\\/\[\]{}]')

    def extract_all_products(self, url: str) -> Optional[List[str]]:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            products = set()

            for sel in self.product_selectors:
                for el in soup.select(sel['selector']):
                    raw = el.get(sel.get('attr'), '') if sel.get('attr') else el.get_text(separator=' ', strip=True)
                    clean = self.clean_product_name(raw)
                    if clean:
                        products.add(clean)

            # отсекаем товары с запрещёнными символами
            products = [p for p in products if not self.forbidden_re.search(p)]
            return sorted(products, key=lambda x: -len(x))

        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    def clean_product_name(self, name: str) -> str:
        # нормализация переносов и пробелов
        name = re.sub(r'[\n\t\r]+', ' ', name)
        name = re.sub(r'\s{2,}', ' ', name).strip()[:200]

        # фильтрация спецсимволов
        if self.forbidden_re.search(name):
            return ''
        # фильтрация дубликатов слов (включая склеенные)
        if re.match(r'^(?P<ph>.+?)(?:\s+)?(?P=ph)$', name, re.IGNORECASE):
            return ''
        return name


class AnnotationBuilder:
    def __init__(self):
        self.scraper = ProductScraper()

    def extract_full_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        # оставляем весь текст, удаляем только script/style
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        # нормализуем пробелы, но сохраняем все строки
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def find_all_mentions(self, text: str, products: List[str]) -> List[Dict]:
        entities: List[Dict] = []
        lower = text.lower()

        for prod in sorted(products, key=lambda x: -len(x)):
            norm = prod.lower().strip()
            if not norm:
                continue
            start = 0
            while True:
                idx = lower.find(norm, start)
                if idx == -1:
                    break
                end = idx + len(norm)
                # границы слова или вхождение в любом виде
                before = text[idx-1] if idx>0 else ' '
                after = text[end] if end < len(text) else ' '
                if before.isalnum() or after.isalnum():
                    start = idx + 1
                    continue
                entities.append({'start': idx, 'end': end, 'label': 'PRODUCT'})
                start = idx + 1
        # сортировка и удаление вложенных/дубликатов
        out = []
        for ent in sorted(entities, key=lambda e:(e['start'], -(e['end']-e['start']))):
            if not any(o['start']<=ent['start']<o['end'] for o in out):
                out.append(ent)
        return out

    def process_urls(self, input_file: str, output_file: str, limit: int = None):
        with open(input_file, 'r', encoding='utf-8') as f:
            urls = [u.strip() for u in f if u.strip()]
        if limit:
            urls = urls[:limit]

        annotations = []
        for url in urls:
            print(f"Processing: {url}")
            products = self.scraper.extract_all_products(url)
            if not products:
                print("  No products found")
                continue

            try:
                resp = requests.get(url, timeout=15)
                text = self.extract_full_text(resp.text)
            except Exception as e:
                print(f"  Fetch failed: {e}")
                continue

            entities = self.find_all_mentions(text, products)
            print(f"  {len(entities)} annotations for {len(products)} products")
            annotations.append({
                'text': text,
                'entities': entities,
                'source_url': url,
                'products': products
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    AnnotationBuilder().process_urls(args.input, args.output, args.limit)
