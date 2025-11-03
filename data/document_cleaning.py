import argparse
import re
import sys

letter_set= set(list("abcdefghijklmnopqrstuvwxyz,.;' "))

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    filtered = ''.join(ch for ch in text if ch in letter_set)
    filtered = re.sub(r' +', ' ', filtered).strip()
    return filtered


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args(argv)


    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f'Error reading input file: {e}', file=sys.stderr)
        sys.exit(2)


    cleaned = clean_text(content)


    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(cleaned + '\n')
    except Exception as e:
        print(f'Error writing output file: {e}', file=sys.stderr)
        sys.exit(3)


    print(f'Clean document written in: {args.output} (length: {len(cleaned)} characters)')




if __name__ == '__main__':
    main()