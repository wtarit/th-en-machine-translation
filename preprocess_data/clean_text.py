# This script is based on https://github.com/vistec-AI/thai2nmt/blob/master/scripts/clean_text.py

import os
import argparse
import glob
import unicodedata
import html
from pathlib import Path
from functools import partial

from tqdm.auto import tqdm

import pythainlp
import pandas as pd


def th_contain_escape_code(lang, text):
    charsets = [
        '\\x9e',
        '\\x95',
        # '\\x94',
        # '\\x93',
        '\\x90',
        # '\\x91',
    ]

    if lang == "th":
        for char in charsets:
            if char in repr(text):
                return True
    return False


def filter_thai_text_without_thai_chars(lang, text):
    if lang == "th":
        thai_char = [
            "ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช",
            "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด",
            "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ",
            "ฟ", "ภ", "ม", "ย", "ร", "ฤ", "ล", "ฦ", "ว", "ศ",
            "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
        ]
        for char in thai_char:
            if char in text:
                return False
        return True
    return False


def normalize_unicode(lang, text):

    if NORM_CODE is not 'NONE' and NORM_CODE in ['NFC', 'NFD', 'NFKC', 'NFKD']:
        text = unicodedata.normalize(NORM_CODE, text)
    if lang == "th":
        return text.replace(u'\x99', u'').replace(u'\x9c', u'')
    return text


def str_strip(lang, text):

    return str(text).strip()


def normalize_text(lang, text):
    text = text.replace('“', '"').replace(
        '”', '"').replace("‘", "'").replace("’", "'")
    return text


def html_unescape(lang, text):

    return html.unescape(text)


def normalize_thai_text(lang, text):
    """
        Remove redudant symbol of tones and vowels.
        and subsitute [“เ”, “เ”] to “แ”.
    """

    if lang == "th":
        return pythainlp.util.normalize(text)

    return text


def replace_escape_code(lang, text):
    mapping = {
        # "\\x9e": "",
        # "\\x95": "",
        "\x94": "\"",
        "\x93": "\"",
        # "\\x90": "",
        "\x91": "\'",
        "\x92": "\'",
        "\x96": "-",
    }
    for char in mapping:
        if char in text:
            text = text.replace(char, mapping[char])
    return text


def filter_blank_text(lang, text):
    if text == "":
        return True
    return False


CLEANING_RULES = [
    str_strip,
    html_unescape,
    normalize_unicode,
    normalize_text,
    normalize_thai_text,
    replace_escape_code,
]

FILTERING_RULES = [
    th_contain_escape_code,
    filter_thai_text_without_thai_chars,
    filter_blank_text,
]


def main(args):

    csv_file_paths = glob.glob(os.path.join(args.csv_dir, '*.csv'))

    for csv_file_path in csv_file_paths:

        df = pd.read_csv(csv_file_path, encoding='utf-8')
        csv_filename = Path(csv_file_path).stem

        print(f'Begin cleaning, filtering from sub-dataset: {csv_filename}')
        print(f'\nNumber of segment pairs (before): {df.shape[0]}')

        n_before = df.shape[0]

        for lang in ['th', 'en']:

            df[f'{lang}_text'] = df[f'{lang}_text'].apply(str)

            for rule in CLEANING_RULES:

                df[f'{lang}_text'] = df[f'{lang}_text'].apply(
                    lambda x: rule(lang, x))

            for rule in FILTERING_RULES:

                _rule = partial(rule, lang)

                df[f'{lang}_text_to_drop'] = df[f'{lang}_text'].apply(_rule)
                df = df.drop(df[df[f'{lang}_text_to_drop'] == True].index)

        n_after = df.shape[0]

        print(
            f'Number of segment pairs (after): {n_after} (filtered out {n_before - n_after})')
        print('\nDone cleaning and fitering.')

        if not os.path.exists(args.out_dir):
            print(f'\nCreate a directory at: `{args.out_dir}`')
            os.makedirs(args.out_dir, exist_ok=True)

        out_path = os.path.join(args.out_dir, f'{csv_filename}.csv')

        print(f'\nBegin writing file to: {out_path}\n')

        df = df.drop(columns=['en_text_to_drop', 'th_text_to_drop'])
        df = df[["en_text", "th_text"]]
        df.to_csv(out_path, index=False, encoding='utf-8')

        print('-'*30)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'csv_dir', help='Directory that stored the dataset in .csv format')
    parser.add_argument('--unicode_norm', default='none', type=str)
    parser.add_argument('--out_dir', help='Directory that stored cleaned dataset in .txt format',
                        default='./dataset/cleaned')

    args = parser.parse_args()

    NORM_CODE = args.unicode_norm.upper()

    print(f'Unicode Normalization Form specified is `{NORM_CODE}`')

    main(args)
