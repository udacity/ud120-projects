#!/usr/bin/env python3

import os
import logging
from urllib.request import urlopen
import tarfile

logging.basicConfig(format='%(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

LIBS = ['numpy', 'scipy', 'sklearn', 'nltk']  # Libs required for the course
URL = 'https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz'  # Download link
FILE_NAME = URL.rsplit('/', 1)[-1]  # Get file name, default download path is ./FILE_NAME
EXTRACT_TO = '.'  # Path to untar
MB = 1024 ** 2


def check_imports():
    logging.info('Checking env...')
    failed = []
    for i, l in enumerate(LIBS):
        i, n = i + 1, len(LIBS)
        step = f'[{i}/{n}]'

        def check_import():
            logging.info(f'{step} Checking {l}...')
            try:
                __import__(l)
            except ImportError:
                logging.warning(f'{step} You should install {l} before continuing!')
                failed.append(l)
            else:
                logging.info(f'{step} {l} is fine!')
            finally:
                logging.info(f'{step} Done!')

        check_import()
    logging.warning(f'You should install: {" ".join(failed)} before continuing!') if failed else None


def download():
    logging.info(f'Downloading the Enron dataset to {FILE_NAME} ...')
    chunk_size, curr_size = 4 * 1024, 0
    start = 'y'
    if os.path.exists(FILE_NAME):
        start = str(input(f'File {FILE_NAME} already exists! Download anyway? y/[n]: ')) or 'n'
    if start != 'y':
        return
    else:
        with urlopen(URL) as r, open(FILE_NAME, 'wb') as f:
            fsize = int(r.info()['Content-Length'])
            info_line = lambda dsize=0: f'{dsize // MB}MB/{fsize // MB}MB'

            print('Progress:', f'{info_line()}', end='')
            while True:
                chunk = r.read(chunk_size)
                if chunk:
                    f.write(chunk)
                    curr_size += chunk_size
                    if curr_size % MB == 0:
                        print('\b' * len(info_line(curr_size - MB)), info_line(curr_size), sep='', end='', flush=True)
                else:
                    print()
                    break
        logging.info('Download complete!')

    logging.info(f'Extracting Enron dataset to {EXTRACT_TO} ...')
    with tarfile.open(FILE_NAME, 'r:gz') as f: f.extractall(EXTRACT_TO)
    logging.info('You\'re ready to go!')


if __name__ == '__main__':
    [fn() for fn in [check_imports, download]]
