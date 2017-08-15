# -*- coding: utf-8 -*-
import os
import click
import logging
import shutil
#from dotenv import find_dotenv, load_dotenv
import data_helper as dh

def download_data():
    """ Downloads project data (saved in ../../data/external).
    """
    logger = logging.getLogger(__name__)
    logger.info('\nDownloading project data... \n(ProjectHome:/data/external/*)')

    to_download = {
        'lc_historical.csv.zip': '1d3d0cfdd641dd315a78c66803cb392e6a9c9b86',
        'LCDataDictionary.xlsx': '7878cb058dd17a735ac09d81984de6400de3cdf9'}
    baseurl='http://vnaidu.com/data-store/loan-status/'

    for filename, file_hash in to_download.items():
        url = baseurl + filename
        dlpath = os.path.join(project_dir, 'data/external', filename)
        if dh.download_file(url, dlpath, valid_hash=file_hash):
            logger.info('\nSuccess: ' + filename + ' -> ' + dlpath + ' (+)')
        elif os.path.isfile(dlpath):
            logger.info('\nFailed: ' + filename + ' ?-> ' + dlpath + ' (?)')
            os.remove(dlpath)
    return dh.validate_data_dirs(project_dir, hash_nfo).get('external')


@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if dh.validate_data_dirs(project_dir, hash_nfo).get('raw'):
        logger.info('(Success) External resources are valid.')
        return
    logger.info('Loading external resources...')

    to_copy = ['LCDataDictionary.xlsx']
    to_extract = ['lc_historical.csv.zip']
    if not dh.validate_data_dirs(project_dir, hash_nfo).get('external'):
        logger.info('\n-------- Invalid external resources:' +
                    '\n--------     downloading data...')
        if not download_data():
            logger.info('\n- Failed: Could not download external data (ERROR)')
            return
        else:
            logger.info('\n-- Success: Downloaded external project data.')
    for filename in to_extract:
        dh.extract_zip(filename, project_dir)
    for filename in to_copy:
        shutil.copy(os.path.join(project_dir, 'data/external', filename),
                    os.path.join(project_dir, 'data/raw', filename))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    hash_nfo = {
        'raw': {
            'lc_historical.csv': '2ab4770efe332d078ee935619ef3a6ea9bee85d5',
            'LCDataDictionary.xlsx': '7878cb058dd17a735ac09d81984de6400de3cdf9'},
        'external': {
            'lc_historical.csv.zip': '1d3d0cfdd641dd315a78c66803cb392e6a9c9b86',
            'LCDataDictionary.xlsx': '7878cb058dd17a735ac09d81984de6400de3cdf9'}}

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
