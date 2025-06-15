import argparse
import requests
from bs4 import BeautifulSoup
from transformers import pipeline


def get_page_text(url):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Remove script/style tags
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(separator=' ')


def main(args):
    # Load NER pipeline
    ner = pipeline(
        'ner',
        model=args.model_dir,
        tokenizer=args.model_dir,
        aggregation_strategy='simple'
    )

    # Read URLs
    with open(args.urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Process each URL
    for url in urls:
        try:
            text = get_page_text(url)
            entities = ner(text)
            products = [ent['word'] for ent in entities if ent['entity_group'] == 'PRODUCT']
            print(f"URL: {url}\nProducts: {products}\n")
        except Exception as e:
            print(f"Failed to process {url}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--urls_file', required=True, help='File with one URL per line')
    args = parser.parse_args()
    main(args)