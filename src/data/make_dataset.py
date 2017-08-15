# -*- coding: utf-8 -*-
import os
import click
import logging
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import data_helper as dh

def clean_dataset(in_path, out_path):
    """ Function that processes raw data from (../raw) into cleaned
        data ready for feature engineering (saved in ../interim).
    """
    lc = pd.read_csv(in_path, index_col='id',
                     memory_map=True, low_memory=False)
    lc['loan_status'] = pd.Categorical(lc.loan_status,
                                    categories=['Fully Paid', 'Charged Off'])
    lc = lc.copy().dropna(axis=1, thresh=1)

    dt_features = ['earliest_cr_line', 'issue_d']
    lc[dt_features] = lc[dt_features].apply(
        lambda col: pd.to_datetime(col, format='%Y-%m-%d'), axis=0)

    cat_features =['purpose', 'home_ownership', 'addr_state']
    lc[cat_features] = lc[cat_features].apply(pd.Categorical, axis=0)

    lc.revol_util = (lc.revol_util
                     .str.extract('(\d+\.?\d?)', expand=False)
                     .astype('float'))

    lc.emp_length = (lc.emp_length
                     .str.extract('(< 1|10\+|\d+)', expand=False)
                     .replace('< 1', '0.5')
                     .replace('10+', '10.5')
                     .fillna('-1.5')
                     .astype('float'))
    lc.to_csv(out_path)

#dh.file_hash_sha1('../../data/interim/lc_historical.csv')

@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
#def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if not dh.validate_data_dirs(project_dir, hash_nfo).get('interim'):
        if dh.validate_data_dirs(project_dir, hash_nfo).get('raw'):
            logger.info('(Success) Making Dataset: Raw -> Clean')
            clean_dataset(raw_path, interim_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    os.system('python load_external_data.py')
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_path = os.path.join(project_dir, 'data/raw/lc_historical.csv')
    interim_path = os.path.join(project_dir, 'data/interim/lc_historical.csv')

    hash_nfo = {
        'raw': {
            'lc_historical.csv': '2ab4770efe332d078ee935619ef3a6ea9bee85d5',
            'LCDataDictionary.xlsx': '7878cb058dd17a735ac09d81984de6400de3cdf9'},
        'external': {
            'lc_historical.csv.zip': '1d3d0cfdd641dd315a78c66803cb392e6a9c9b86',
            'LCDataDictionary.xlsx': '7878cb058dd17a735ac09d81984de6400de3cdf9'},
        'interim': {
            'lc_historical.csv': 'abc43dd898cb75e440bfafe1f9c8a6be8afe0598'},
        'processed': {
            'lc_historical.csv': 'abc43dd898cb75e440bfafe1f9c8a6be8afe0598'}}

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
