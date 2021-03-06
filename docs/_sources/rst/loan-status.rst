
Classifying Risky P2P Loans
===========================

Abstract
--------

The prevalence of a global Peer-to-Peer (P2P) economy, coupled with the
recent deregulation of financial markets, has lead to the widespread
adoption of Artificial Intelligence driven by FinTech firms to manage
risk when speculating on unsecured P2P debt obligations. After
meticulously identifying ‘debt belonging to high-risk individuals’ by
leveraging an ensemble of Machine Learning algorithms, these firms are
able to find ideal trading opportunities.

While researching AI-driven portfolio management that favor
risk-minimization strategies by unmasking subtle interactions amongst
high dimensional features to identify prospective trades that exhibit
modest ,low-risk gains, I was impressed that the overall portfolio:
realized a modest return through a numerosity of individual gains;
achieved an impressive Sharpe ratio stemming from infrequent losses and
minimal portfolio volatility.

Project Overview
================

Objective
---------

Build a binary classification model that predicts the "Charged Off" or
"Fully Paid" Status of a loan by analyzing predominant characteristics
which differentiate the two classes in order to engineer new features
that may better enable our Machine Learning algorithms to reach efficacy
in minimizing portfolio risk while observing better-than-average
returns. Ultimately, the aim is to deploy this model to assist in
placing trades on loans immediately after they are issued by Lending
Club.

About P2P Lending
~~~~~~~~~~~~~~~~~

Peer-to-Peer (P2P) lending offers borrowers with bad credit to get the
necessary funds to meet emergency deadlines. It might seem careless to
lend even more money to people who have demonstrated an inability to
repay loans in the past. However, by implementing Machine Learning
algorithms to classify poor trade prospects, one can effectively
minimize portfolio risk.

There is a large social component to P2P lending, for sociological
factors (stigma of defaulting) often plays a greater role than financial
metrics in determining an applicant’s creditworthiness. For example the
“online friendships of borrowers act as signals of credit quality.” (
Lin et all, 2012)

The social benefit of providing finance for another individual has
wonderful implications, and, while it is nice to engage in philanthropic
activities, the motivating factor for underwriting speculating in p2p
lending markets is financial gain, especially since the underlying debt
is unsecured and investors are liable to defaults.

Project Setup
-------------

Import Libraries & Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from IPython.display import display
    from IPython.core.display import HTML
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import os
    if os.getcwd().split('/')[-1] == 'notebooks':
        os.chdir('../')

.. code:: ipython3

    import pandas as pd
    import numpy as np
    
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import pandas_profiling

.. code:: ipython3

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Imputer
    
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # written by Gilles Louppe and distributed under the BSD 3 clause
    from src.vn_datasci.blagging import BlaggingClassifier
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report
    
    # self-authored library that to facilatate ML classification and evaluation
    from src.vn_datasci.skhelper import LearningModel, eval_db

Notebook Config
~~~~~~~~~~~~~~~

.. code:: ipython3

    from IPython.display import display
    from IPython.core.display import HTML
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import os
    if os.getcwd().split('/')[-1] == 'notebooks':
        os.chdir('../')

.. code:: ipython3

    %matplotlib inline
    #%config figure_format='retina'
    plt.rcParams.update({'figure.figsize': (10, 7)})
    
    sns.set_context("notebook", font_scale=1.75, rc={"lines.linewidth": 1.25})
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    
    pd.options.display.width = 80
    pd.options.display.max_columns = 50
    pd.options.display.max_rows = 50

Data Preprocessing
==================

Load Dataset
------------

Data used for this project comes directly from Lending Club’s historical
loan records (the full record contains more than 100 columns).

.. code:: ipython3

    def load_dataset(path='data/raw/lc_historical.csv'):
        lc = pd.read_csv(path, index_col='id', memory_map=True, low_memory=False)
        lc.loan_status = pd.Categorical(lc.loan_status, categories=['Fully Paid', 'Charged Off'])
        return lc

.. code:: ipython3

    dataset = load_dataset()

Exploration
-----------

Summary
~~~~~~~

-  Target: loan-status
-  Number of features: 18
-  Number of observations: 138196
-  Feature datatypes:

   -  object: dti, bc\_util, fico\_range\_low, percent\_bc\_gt\_75,
      acc\_open\_past\_24mths, annual\_inc, recoveries, avg\_cur\_bal,
      loan\_amnt
   -  float64: revol\_util, earliest\_cr\_line, purpose, emp\_length,
      home\_ownership, addr\_state, issue\_d, loan\_status

-  Features with ALL missing or null values:

   -  inq\_last\_12m
   -  all\_util

-  Features with SOME missing or null values:

   -  avg\_cur\_bal (30%)
   -  bc\_util (21%)
   -  percent\_bc\_gt\_75 (21%)
   -  acc\_open\_past\_24mths (20%)
   -  emp\_length (0.18%)
   -  revol\_util (0.08%)

Missing Data
~~~~~~~~~~~~

Helper Functions
^^^^^^^^^^^^^^^^

.. code:: ipython3

    def calc_incomplete_stats(dataset):
        warnings.filterwarnings("ignore", 'This pattern has match groups')
        missing_data = pd.DataFrame(index=dataset.columns)
        missing_data['Null'] = dataset.isnull().sum()
        missing_data['NA_or_Missing'] = (
            dataset.apply(lambda col: (
                col.str.contains('(^$|n/a|^na$|^%$)', case=False).sum()))
            .fillna(0).astype(int))
        missing_data['Incomplete'] = (
            (missing_data.Null + missing_data.NA_or_Missing) / len(dataset))
        incomplete_stats = ((missing_data[(missing_data > 0).any(axis=1)])
                            .sort_values('Incomplete', ascending=False))
        return incomplete_stats
    
    def display_incomplete_stats(incomplete_stats):
        stats = incomplete_stats.copy()
        df_incomplete = (
            stats.style
            .set_caption('Missing')
            .background_gradient(cmap=sns.light_palette("orange", as_cmap=True),
                                 low=0, high=1, subset=['Null', 'NA_or_Missing'])
            .background_gradient(cmap=sns.light_palette("red", as_cmap=True),
                                 low=0, high=.6, subset=['Incomplete'])
            .format({'Null': '{:,}', 'NA_or_Missing': '{:,}', 'Incomplete': '{:.1%}'}))
        display(df_incomplete)
        
    def plot_incomplete_stats(incomplete_stats, ylim_range=(0, 100)):
        stats = incomplete_stats.copy()
        stats.Incomplete = stats.Incomplete * 100
        _ = sns.barplot(x=stats.index.tolist(), y=stats.Incomplete.tolist())
        for item in _.get_xticklabels():
            item.set_rotation(45)
        _.set(xlabel='Feature', ylabel='Incomplete (%)', 
              title='Features with Missing or Null Values',
              ylim=ylim_range)
        plt.show()
        
    def incomplete_data_report(dataset, display_stats=True, plot=True):
        incomplete_stats = calc_incomplete_stats(dataset)
        if display_stats:
            display_incomplete_stats(incomplete_stats)
        if plot:
            plot_incomplete_stats(incomplete_stats)
    
    
    incomplete_stats = load_dataset().pipe(calc_incomplete_stats)

.. code:: ipython3

    display(incomplete_stats)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Null</th>
          <th>NA_or_Missing</th>
          <th>Incomplete</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>all_util</th>
          <td>172745</td>
          <td>0</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>inq_last_12m</th>
          <td>172745</td>
          <td>0</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>avg_cur_bal</th>
          <td>51649</td>
          <td>0</td>
          <td>0.298990</td>
        </tr>
        <tr>
          <th>bc_util</th>
          <td>36407</td>
          <td>0</td>
          <td>0.210756</td>
        </tr>
        <tr>
          <th>percent_bc_gt_75</th>
          <td>36346</td>
          <td>0</td>
          <td>0.210403</td>
        </tr>
        <tr>
          <th>acc_open_past_24mths</th>
          <td>35121</td>
          <td>0</td>
          <td>0.203311</td>
        </tr>
        <tr>
          <th>emp_length</th>
          <td>0</td>
          <td>7507</td>
          <td>0.043457</td>
        </tr>
        <tr>
          <th>revol_util</th>
          <td>144</td>
          <td>0</td>
          <td>0.000834</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: ipython3

    plot_incomplete_stats(incomplete_stats)



.. image:: output_26_0.png


Data Munging
------------

Cleaning
~~~~~~~~

-  all\_util, inq\_last\_12m

   -  Drop features (all observations contain null/missing values)

-  revol\_util

   1. Remove the percent sign (%) from string
   2. Convert to a float

-  earliest\_cr\_line, issue\_d

   -  Convert to datetime data type.

-  emp\_length

   1. Strip leading and trailing whitespace
   2. Replace '< 1' with '0.5'
   3. Replace '10+' with '10.5'
   4. Fill null values with '-1.5'
   5. Convert to float

.. code:: ipython3

    def clean_data(lc):
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
        return lc

.. code:: ipython3

    dataset = load_dataset().pipe(clean_data)

Feature Engineering
-------------------

New Features
~~~~~~~~~~~~

-  loan\_amnt\_to\_inc

   -  the ratio of loan amount to annual income

-  earliest\_cr\_line\_age

   -  age of first credit line from when the loan was issued

-  avg\_cur\_bal\_to\_inc

   -  the ratio of avg current balance to annual income

-  avg\_cur\_bal\_to\_loan\_amnt

   -  the ratio of avg current balance to loan amount

-  acc\_open\_past\_24mths\_groups

   -  level of accounts opened in the last 2 yrs

.. code:: ipython3

    def add_features(lc):
        # ratio of loan amount to annual income
        group_labels = ['low', 'avg', 'high']
        lc['loan_amnt_to_inc'] = (
            pd.cut((lc.loan_amnt / lc.annual_inc), 3, labels=['low', 'avg', 'high'])
            .cat.set_categories(['low', 'avg', 'high'], ordered=True))
        
        # age of first credit line from when the loan was issued
        lc['earliest_cr_line_age'] = (lc.issue_d - lc.earliest_cr_line).astype(int)
        
        # the ratio of avg current balance to annual income
        lc['avg_cur_bal_to_inc'] = lc.avg_cur_bal / lc.annual_inc
        
        # the ratio of avg current balance to loan amount
        lc['avg_cur_bal_to_loan_amnt'] = lc.avg_cur_bal / lc.loan_amnt
        
        # grouping level of accounts opened in the last 2 yrs
        lc['acc_open_past_24mths_groups'] = (
            pd.qcut(lc.acc_open_past_24mths, 3, labels=['low', 'avg', 'high'])
            .cat.add_categories(['unknown']).fillna('unknown')
            .cat.set_categories(['low', 'avg', 'high', 'unknown'], ordered=True))
        
        return lc

.. code:: ipython3

    dataset = load_dataset().pipe(clean_data).pipe(add_features)

Drop Features
~~~~~~~~~~~~~

.. code:: ipython3

    def drop_features(lc):
        target_leaks = ['recoveries', 'issue_d']
        other_features = ['earliest_cr_line', 'acc_open_past_24mths', 'addr_state']
        to_drop = target_leaks + other_features
        return lc.drop(to_drop, axis=1)

.. code:: ipython3

    dataset = load_dataset().pipe(clean_data).pipe(add_features).pipe(drop_features)

Load & Prepare Function
-----------------------

.. code:: ipython3

    def load_and_preprocess_data():
        return (load_dataset()
                .pipe(clean_data)
                .pipe(add_features)
                .pipe(drop_features))

Exploratory Data Analysis (EDA)
===============================

Helper Functions
----------------

.. code:: ipython3

    def plot_factor_pct(dataset, feature):
        if feature not in dataset.columns:
            return
        y = dataset[feature]
        factor_counts = y.value_counts()
        x_vals = factor_counts.index.tolist()
        y_vals = ((factor_counts.values/factor_counts.values.sum())*100).round(2)
        sns.barplot(y=x_vals, x=y_vals);
    
    def plot_pct_charged_off(lc, feature):
        lc_counts = lc[feature].value_counts()
        charged_off = lc[lc.loan_status=='Charged Off']
        charged_off_counts = charged_off[feature].value_counts()
        charged_off_ratio = ((charged_off_counts / lc_counts * 100)
                             .round(2).sort_values(ascending=False))
    
        x_vals = charged_off_ratio.index.tolist()
        y_vals = charged_off_ratio
        sns.barplot(y=x_vals, x=y_vals);

Overview
--------

Missing Data
~~~~~~~~~~~~

.. code:: ipython3

    processed_dataset = load_and_preprocess_data()
    incomplete_stats = calc_incomplete_stats(processed_dataset)

.. code:: ipython3

    display(incomplete_stats)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Null</th>
          <th>NA_or_Missing</th>
          <th>Incomplete</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>avg_cur_bal</th>
          <td>51649</td>
          <td>0</td>
          <td>0.298990</td>
        </tr>
        <tr>
          <th>avg_cur_bal_to_inc</th>
          <td>51649</td>
          <td>0</td>
          <td>0.298990</td>
        </tr>
        <tr>
          <th>avg_cur_bal_to_loan_amnt</th>
          <td>51649</td>
          <td>0</td>
          <td>0.298990</td>
        </tr>
        <tr>
          <th>bc_util</th>
          <td>36407</td>
          <td>0</td>
          <td>0.210756</td>
        </tr>
        <tr>
          <th>percent_bc_gt_75</th>
          <td>36346</td>
          <td>0</td>
          <td>0.210403</td>
        </tr>
        <tr>
          <th>revol_util</th>
          <td>144</td>
          <td>0</td>
          <td>0.000834</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: ipython3

    plot_incomplete_stats(incomplete_stats)



.. image:: output_49_0.png


Factor Analysis
---------------

Target: loan\_status
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    processed_dataset.pipe(plot_factor_pct, 'loan_status')



.. image:: output_52_0.png


Summary Statistics
------------------

.. code:: ipython3

    HTML(processed_dataset.pipe(pandas_profiling.ProfileReport).html)




.. raw:: html

    <meta charset="UTF-8">
    
    <style>
    
            .variablerow {
                border: 1px solid #e1e1e8;
                border-top: hidden;
                padding-top: 2em;
                padding-bottom: 2em;
                padding-left: 1em;
                padding-right: 1em;
            }
    
            .headerrow {
                border: 1px solid #e1e1e8;
                background-color: #f5f5f5;
                padding: 2em;
            }
            .namecol {
                margin-top: -1em;
                overflow-x: auto;
            }
    
            .dl-horizontal dt {
                text-align: left;
                padding-right: 1em;
                white-space: normal;
            }
    
            .dl-horizontal dd {
                margin-left: 0;
            }
    
            .ignore {
                opacity: 0.4;
            }
    
            .container.pandas-profiling {
                max-width:975px;
            }
    
            .col-md-12 {
                padding-left: 2em;
            }
    
            .indent {
                margin-left: 1em;
            }
    
            /* Table example_values */
                table.example_values {
                    border: 0;
                }
    
                .example_values th {
                    border: 0;
                    padding: 0 ;
                    color: #555;
                    font-weight: 600;
                }
    
                .example_values tr, .example_values td{
                    border: 0;
                    padding: 0;
                    color: #555;
                }
    
            /* STATS */
                table.stats {
                    border: 0;
                }
    
                .stats th {
                    border: 0;
                    padding: 0 2em 0 0;
                    color: #555;
                    font-weight: 600;
                }
    
                .stats tr {
                    border: 0;
                }
    
                .stats tr:hover{
                    text-decoration: underline;
                }
    
                .stats td{
                    color: #555;
                    padding: 1px;
                    border: 0;
                }
    
    
            /* Sample table */
                table.sample {
                    border: 0;
                    margin-bottom: 2em;
                    margin-left:1em;
                }
                .sample tr {
                    border:0;
                }
                .sample td, .sample th{
                    padding: 0.5em;
                    white-space: nowrap;
                    border: none;
    
                }
    
                .sample thead {
                    border-top: 0;
                    border-bottom: 2px solid #ddd;
                }
    
                .sample td {
                    width:100%;
                }
    
    
            /* There is no good solution available to make the divs equal height and then center ... */
                .histogram {
                    margin-top: 3em;
                }
            /* Freq table */
    
                table.freq {
                    margin-bottom: 2em;
                    border: 0;
                }
                table.freq th, table.freq tr, table.freq td {
                    border: 0;
                    padding: 0;
                }
    
                .freq thead {
                    font-weight: 600;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
    
                }
    
                td.fillremaining{
                    width:auto;
                    max-width: none;
                }
    
                td.number, th.number {
                    text-align:right ;
                }
    
            /* Freq mini */
                .freq.mini td{
                    width: 50%;
                    padding: 1px;
                    font-size: 12px;
    
                }
                table.freq.mini {
                     width:100%;
                }
                .freq.mini th {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    max-width: 5em;
                    font-weight: 400;
                    text-align:right;
                    padding-right: 0.5em;
                }
    
                .missing {
                    color: #a94442;
                }
                .alert, .alert > th, .alert > td {
                    color: #a94442;
                }
    
    
            /* Bars in tables */
                .freq .bar{
                    float: left;
                    width: 0;
                    height: 100%;
                    line-height: 20px;
                    color: #fff;
                    text-align: center;
                    background-color: #337ab7;
                    border-radius: 3px;
                    margin-right: 4px;
                }
                .other .bar {
                    background-color: #999;
                }
                .missing .bar{
                    background-color: #a94442;
                }
                .tooltip-inner {
                    width: 100%;
                    white-space: nowrap;
                    text-align:left;
                }
    
                .extrapadding{
                    padding: 2em;
                }
    
    </style>
    
    <div class="container pandas-profiling">
        <div class="row headerrow highlight">
            <h1>Overview</h1>
        </div>
        <div class="row variablerow">
        <div class="col-md-6 namecol">
            <p class="h4">Dataset info</p>
            <table class="stats" style="margin-left: 1em;">
                <tbody>
                <tr>
                    <th>Number of variables</th>
                    <td>18 </td>
                </tr>
                <tr>
                    <th>Number of observations</th>
                    <td>172745 </td>
                </tr>
                <tr>
                    <th>Total Missing (%)</th>
                    <td>7.3% </td>
                </tr>
                <tr>
                    <th>Total size in memory</th>
                    <td>18.0 MiB </td>
                </tr>
                <tr>
                    <th>Average record size in memory</th>
                    <td>109.0 B </td>
                </tr>
                </tbody>
            </table>
        </div>
        <div class="col-md-6 namecol">
            <p class="h4">Variables types</p>
            <table class="stats" style="margin-left: 1em;">
                <tbody>
                <tr>
                    <th>Numeric</th>
                    <td>13 </td>
                </tr>
                <tr>
                    <th>Categorical</th>
                    <td>5 </td>
                </tr>
                <tr>
                    <th>Date</th>
                    <td>0 </td>
                </tr>
                <tr>
                    <th>Text (Unique)</th>
                    <td>0 </td>
                </tr>
                <tr>
                    <th>Rejected</th>
                    <td>0 </td>
                </tr>
                </tbody>
            </table>
        </div>
        <div class="col-md-12" style="padding-left: 1em;">
            <p class="h4">Warnings</p>
            <ul class="list-unstyled"><li><code>annual_inc</code> is highly skewed (γ1 = 35.012)</l><li><code>avg_cur_bal</code> has 51649 / 29.9% missing values <span class="label label-default">Missing</span></l><li><code>avg_cur_bal_to_inc</code> has 51649 / 29.9% missing values <span class="label label-default">Missing</span></l><li><code>avg_cur_bal_to_loan_amnt</code> has 51649 / 29.9% missing values <span class="label label-default">Missing</span></l><li><code>bc_util</code> has 36407 / 21.1% missing values <span class="label label-default">Missing</span></l><li><code>percent_bc_gt_75</code> has 21592 / 12.5% zeros</l><li><code>percent_bc_gt_75</code> has 36346 / 21.0% missing values <span class="label label-default">Missing</span></l> </ul>
        </div>
    </div>
        <div class="row headerrow highlight">
            <h1>Variables</h1>
        </div>
        <div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">acc_open_past_24mths_groups<br/>
                <small>Categorical</small>
            </p>
        </div><div class="col-md-3">
        <table class="stats ">
            <tr class="">
                <th>Distinct count</th>
                <td>4</td>
            </tr>
            <tr>
                <th>Unique (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (n)</th>
                <td>0</td>
            </tr>
        </table>
    </div>
    <div class="col-md-6 collapse in" id="minifreqtable-8076277048848854607">
        <table class="mini freq">
            <tr class="">
        <th>avg</th>
        <td>
            <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 34.4%">
                59367
            </div>
            
        </td>
    </tr><tr class="">
        <th>low</th>
        <td>
            <div class="bar" style="width:78%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 27.0%">
                46567
            </div>
            
        </td>
    </tr><tr class="">
        <th>unknown</th>
        <td>
            <div class="bar" style="width:59%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 20.3%">
                35121
            </div>
            
        </td>
    </tr>
        </table>
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#freqtable-8076277048848854607, #minifreqtable-8076277048848854607"
           aria-expanded="true" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="col-md-12 extrapadding collapse" id="freqtable-8076277048848854607">
        
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">avg</td>
            <td class="number">59367</td>
            <td class="number">34.4%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">low</td>
            <td class="number">46567</td>
            <td class="number">27.0%</td>
            <td>
                <div class="bar" style="width:78%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">unknown</td>
            <td class="number">35121</td>
            <td class="number">20.3%</td>
            <td>
                <div class="bar" style="width:59%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">high</td>
            <td class="number">31690</td>
            <td class="number">18.3%</td>
            <td>
                <div class="bar" style="width:53%">&nbsp;</div>
            </td>
    </tr>
    </table>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">annual_inc<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>14645</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>8.5%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>69396</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>4000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>7141800</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-2768471354743219373">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAArxJREFUeJzt2jFLqmEchvE7eamlsdZcGlzExRDDlpaO0FjgUiDhB2gLgqbOFwgJom8QgdWStkVByyECIVFokYamvkDJe4YDhcfjHYK%2BxuH6gYOvz6P/5eJ5BifCMAwF4J9i4x4A%2BM6CcQ/wt/RudeA9v37%2BGMEkACcIYBEIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYBAIYEyEYRiOewjgu%2BIEAQwCAQwCAQwCAQwCwdDd3NxocXFR29vbA%2B1bWVlRMpnseiUSCVUqlRFN%2BrVgbL%2BM/9Lx8bFOT08Vj8cH3lur1bret9ttFQoFLS0tDWu8gXGCYKimpqZsILVaTfl8XqlUSqurq7q4uOj7Xfv7%2B9ra2tLMzMyoxv0SJwiGanNzs%2B9nT09P2tnZ0eHhoRYWFvTw8KBSqaR4PK5UKtW19u7uTq1WS%2BVyedQjW5wgiMzJyYmWl5eVzWYVBIHS6bTy%2BbzOzs561pbLZZVKJU1OTo5h0k%2BcIIhMu93W9fW1rq6uPp6FYahcLte1rtFo6PHxUUdHR1GP2INAEJlYLKZCoaC9vT27rlqtKpfLaXp6OqLJ%2BuOKhcjMzc2p1Wp1PXt5eVGn0%2Bl6dnt7q0wmE%2BVofREIIrO2tqb7%2B3tVKhW9vb2p0WhofX2958rVbDY1Pz8/xkk/8W9eDFUymZQkvb%2B/S5KC4M8tvl6vS5IuLy91cHCg5%2Bdnzc7OamNjQ8Vi8WP/6%2Burstmszs/PlUgkIp6%2BF4EABlcswCAQwCAQwCAQwCAQwCAQwCAQwCAQwCAQwCAQwPgNAqacWLJI1FMAAAAASUVORK5CYII%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-2768471354743219373,#minihistogram-2768471354743219373"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-2768471354743219373">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-2768471354743219373"
                                                      aria-controls="quantiles-2768471354743219373" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-2768471354743219373" aria-controls="histogram-2768471354743219373"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-2768471354743219373" aria-controls="common-2768471354743219373"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-2768471354743219373" aria-controls="extreme-2768471354743219373"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-2768471354743219373">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>4000</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>25500</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>42000</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>60000</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>84450</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>141000</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>7141800</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>7137800</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>42450</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>55278</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.79657</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>3573.1</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>69396</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>29465</td>
                        </tr>
                        <tr class="alert">
                            <th>Skewness</th>
                            <td>35.012</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>11988000000</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>3055700000</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-2768471354743219373">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3XlU1XXi//EXiyvIajgupIYHFbi4jEuoUVia2ebXRFzKmhwqfxpi9pU0c5i0bFJHM1u0JnNqGlGpsdTUsawcxyZHpwQxxnE0XFIQL7KoyPL%2B/eHxfrvhgvBBlvt8nMOZw/t9P%2B/P%2B3XDMy8%2B93MvbsYYIwAAAFjGvbY3AAAA0NBQsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALCYZ21vwFXk5BRYvqa7u5sCArx06lSRysuN5evXRWQmc0NFZjI3ZLWZ%2B4YbWlzX813EFax6zN3dTW5ubnJ3d6vtrVw3ZHYNZHYNZHYdrpibggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgsVotWNu2bVO/fv00ZcoUp/GZM2fKZrM5fYWFhWn69OmSpMWLF6tr165O87169XIcb7fbNWXKFPXs2VO9e/fWs88%2Bq3Pnzjnm9%2B3bp1GjRikyMlLR0dFavny50/nXr1%2BvO%2B%2B8UzabTffcc4%2B2b99eg88CAABoaGqtYL311luaM2eO2rdvX2Fuzpw5SktLc3zt3r1bHTt21JAhQyRJ%2Bfn5GjFihNNj/vnPfzqOnzFjhnJzc7V582atW7dO%2B/bt0/z58yVJZ8%2BeVXx8vHr27KkdO3Zo8eLFev3117V582ZJUnp6upKSkjR58mTt3LlTDz/8sCZOnKjjx49fh2cFAAA0BLVWsJo0aaI1a9ZcsmD93IoVK9SuXTvdeuutki4UrBYtLv3BYbm5udq6daumT5%2Buli1bqlWrVkpMTFRqaqrOnz%2BvL774QiUlJZo6daq8vLzUvXt3xcXFKSUlRZKUmpqq6OhoDR06VE2bNlVsbKxCQ0O1du1a68IDAIAGrdY%2ByX3cuHGVelxeXp6WLVum999/3zGWn5%2BvY8eO6Z577tHx48cVGhqqGTNmKCIiQhkZGfL09FTnzp0djw8PD9eZM2d08OBBZWRkqEuXLvLw8HDMh4WFafXq1ZKkjIwMRUdHO%2B0hLCxM6enplc6WnZ2tnJwcpzFPz%2BYKCgqq9BqV4eHh7vS/roDMroHMroHMrsMVc9f5P5WzYsUK3XzzzQoNDXWMtWrVSj4%2BPnrxxRcVEBCg1157TY8%2B%2Bqg2btwou90ub29vubv/339EX19fSdKpU6dkt9sd31/k5%2BenvLw8lZeXy263y8/Pz2ne19dX%2B/fvr/SeU1JStGTJEqexiRMnKiEhodJrXAsfn2Y1sm5dRmbXQGbXQGbX4Uq563TBOn/%2BvFauXKnf//73TuO//e1vnb5/%2Bumn9cknn2jLli1q2rTpZddzc6v6R/Rfy7FxcXEaOHCg05inZ3PZ7UVVPv%2BleHi4y8enmfLzz6qsrNzStesqMpO5oSIzmRuy2szt7%2B91Xc93UZ0uWDt37pQxRn379r3i4zw8PNS6dWvl5OSoe/fuKigoUFlZmeNlQLvdLkkKDAxUYGCgsrKynI632%2B3y9/eXu7u7AgICHI//6XxAQECl9x0UFFTh5cCcnAKVltbMD1VZWXmNrV1Xkdk1kNk1kNl1uFLuOv1i6N/%2B9jf16dPH6eU%2BY4xeeukl7du3zzFWUlKiw4cPKzg4WGFhYSovL1dmZqZjfs%2BePWrRooU6dOggm82mzMxMlZaWOs1HRkZKkmw2m/bu3eu0j7S0NMc8AADA1dTpK1jff/%2B9evTo4TTm5uamY8eOafbs2Vq4cKG8vb21ePFiNW7cWHfccYeaN2%2Buu%2B66S3PnztXChQtVXFyshQsXKi4uTo0aNVJ0dLS8vLy0YMECTZo0SXv37tWqVau0aNEiSVJsbKxGjBihDRs2aODAgVq9erWysrI0bNiw2ngKrqrXsxtrewuV9mli/9reAgAA10WtFSybzSZJjitJW7ZskXThatFFOTk5FW44ly58TtaLL76oYcOGqaysTDabTe%2B%2B%2B66aN28u6cI9WsnJyRo0aJAaNWqke%2B%2B9V5MnT5YkNW7cWEuXLtWsWbMUFRWlwMBATZs2zfEREKGhoZo/f74WLFigpKQkhYSEaOnSpWrZsmXNPRkAAKBBcTPGmNrehCvIySmwfE1PT3cNmr/N8nVrihVXsDw93eXv7yW7vchlXscnM5kbKjK7RmapdnPfcMOlPzezptXpe7AAAADqIwoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGCxWi1Y27ZtU79%2B/TRlyhSn8a%2B//lqdO3eWzWZz%2BtqzZ48kyRijhQsXqn///urWrZseeeQRHT582HG83W7XlClT1LNnT/Xu3VvPPvuszp0755jft2%2BfRo0apcjISEVHR2v58uVO51%2B/fr3uvPNO2Ww23XPPPdq%2BfXsNPgsAAKChqbWC9dZbb2nOnDlq3759hbmCggJ16NBBaWlpTl%2BRkZGSpBUrVig1NVVvv/22tm/fruDgYE2aNEnGGEnSjBkzlJubq82bN2vdunXat2%2Bf5s%2BfL0k6e/as4uPj1bNnT%2B3YsUOLFy/W66%2B/rs2bN0uS0tPTlZSUpMmTJ2vnzp16%2BOGHNXHiRB0/fvw6PTMAAKC%2Bq7WC1aRJE61Zs%2BaSBev06dNq0aLFZY9dvXq1fv3rX6tr167y9vZWUlKSDhw4oG%2B//Va5ubnaunWrpk%2BfrpYtW6pVq1ZKTExUamqqzp8/ry%2B%2B%2BEIlJSWaOnWqvLy81L17d8XFxSklJUWSlJqaqujoaA0dOlRNmzZVbGysQkNDtXbt2hp7LgAAQMPiWVsnHjdu3GXn8vPzdfr0aT300EPat2%2BfWrVqpccee0z333%2B/iouLdeDAAUVERDge7%2B3trRtvvFHp6ekqLCyUp6enOnfu7JgPDw/XmTNndPDgQWVkZKhLly7y8PBwzIeFhWn16tWSpIyMDEVHRzvtJywsTOnp6ZXOlp2drZycHKcxT8/mCgoKqvQaleHhUb9uofP0rP5%2BL2aub9mrg8yugcyuwRUzS66Zu9YK1pV4e3urXbt2SkhIUNeuXfX555/r6aefVlBQkG666SYZY%2BTr6%2Bt0jK%2Bvr06dOiVfX195e3vL3d3daU6STp06JbvdXuFYPz8/5eXlqby8XHa7XX5%2BfhXW3r9/f6X3n5KSoiVLljiNTZw4UQkJCZVeoyHy9/eybC0fn2aWrVVfkNk1kNk1uGJmybVy18mCNXLkSI0cOdLx/dChQ7Vp0yatWbNG06ZNu%2Bxxbm5uV1z3avNWHRsXF6eBAwc6jXl6NpfdXlTl819KfftNwIr8Hh7u8vFppvz8syorK7dgV3UfmcncUJHZNTJLtZvbyl/ur0WdLFiX0q5dO6Wnp8vf31/u7u7Ky8tzmrfb7QoMDFRgYKAKCgpUVlbmeBnQbrdLkmM%2BKyurwrEX1w0ICHA8/qfzAQEBld5rUFBQhZcDc3IKVFrqOv%2BYLsXK/GVl5S73fJLZNZDZNbhiZsm1ctfJSyArV67Uhg0bnMYOHjyo4OBgNW7cWKGhodq7d69jLi8vT1lZWbLZbAoLC1N5ebkyMzMd83v27FGLFi3UoUMH2Ww2ZWZmqrS01Gn%2B4jsUbTab09qSnN7BCAAAcDV1smCVlpZqzpw5Sk9PV0lJidavX6%2BvvvpKo0ePliSNHj1ab7/9tr7//nsVFBRozpw5ioiIUGRkpPz9/XXXXXdp7ty5OnnypI4ePaqFCxcqLi5OjRo1UnR0tLy8vLRgwQIVFRXpm2%2B%2B0apVqzR27FhJUmxsrLZv364NGzbo3Llzeu%2B995SVlaVhw4bV5lMCAADqkVp7idBms0mS40rSli1bJF24WjR27Fjl5%2BcrISFBdrtdHTt21Guvvabw8HBJ0qhRo5STk6NHH31URUVF6tu3rxYvXuxY%2B7e//a2Sk5M1aNAgNWrUSPfee68mT54sSWrcuLGWLl2qWbNmKSoqSoGBgZo2bZpuvfVWSVJoaKjmz5%2BvBQsWKCkpSSEhIVq6dKlatmx53Z4bAABQv7mZi5/OiRqVk1Ng%2BZqenu4aNH%2Bb5evWlE8T%2B1d7DU9Pd/n7e8luL3KZ1/HJTOaGisyukVmq3dw33HD5z9WsSXXyJUIAAID6jIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFisVgvWtm3b1K9fP02ZMqXC3MaNG3XvvfeqR48eGjx4sFJSUhxzqamp6tKli2w2m9PXyZMnJUnFxcWaNWuW%2BvTpox49eighIUGnTp1yHH/kyBGNHz9e3bt3V1RUlObNm6fy8nLH/I4dO3TffffJZrNp0KBB%2Bvjjj2vwWQAAAA2NZ22d%2BK233tKaNWvUvn37CnN79uzRtGnTtGjRIt166636%2B9//rgkTJigkJES9evVSQUGB%2BvXrp3feeeeSa8%2BbN0%2B7d%2B9WamqqvLy89Mwzz2jGjBl68803ZYzRpEmT1KlTJ3355Zc6efKk4uPj1bJlS/3qV7/SiRMnNGHCBD311FOKjY3Vjh07lJiYqA4dOigyMrKmnxYAANAA1NoVrCZNmly2YOXl5emJJ57QwIED5eHhoVtuuUWdO3fWP//5T0nS6dOn1aJFi0uuW1paqo8%2B%2BkiJiYkKDg5WQECAkpKStHXrVp04cUJpaWnKzMzUzJkz5evrq5CQEMXHx2vlypWSpE8%2B%2BUTt27fXuHHj1KxZMw0cOFC333671qxZU3NPBgAAaFBq7QrWuHHjLjsXHR2t6Ohox/elpaXKzs5W69atJUn5%2BfnKysrSAw88oEOHDql9%2B/Z66qmnNGDAAGVlZamwsFDh4eGO40NCQtSsWTPt3btX2dnZatu2rfz8/Bzz4eHhOnTokAoLC5WRkeF0rCSFhYXp008/rXS27Oxs5eTkOI15ejZXUFBQpdeoDA%2BP%2BnULnadn9fd7MXN9y14dZHYNZHYNrphZcs3ctVawrsX8%2BfPl5eWlIUOGSJL8/f3Vpk0bJSQk6MYbb1RKSoqeeOIJrV27Vnl5eZIkX19fpzV8fHx06tQp2e32CnMXv7fb7bLb7erSpYvTvJ%2Bfn9M9XFeTkpKiJUuWOI1NnDhRCQkJlV6jIfL397JsLR%2BfZpatVV%2BQ2TWQ2TW4YmbJtXLX6YJljNH8%2BfO1bt06rVixQk2aNJEkTZo0yelxjzzyiNatW6ePP/7Y6crXz7m5uVV5L9dybFxcnAYOHOg05unZXHZ7UZXPfyn17TcBK/J7eLjLx6eZ8vPPqqys/OoHNABkJnNDRWbXyCzVbm4rf7m/FnW2YJWXl2v69OlKS0tTSkqK2rZte8XHt2vXTjk5OQoMDJR04T6u5s2bS7pQ1PLy8hQYGKiysjLHVa6L7Ha7JCkgIEABAQGXnA8ICKj03oOCgiq8HJiTU6DSUtf5x3QpVuYvKyt3ueeTzK6BzK7BFTNLrpW7zl4CefHFF3XgwAH9%2Bc9/rlCu3njjDf397393Gjt48KCCg4MVHBwsPz8/7d271zGXmZmp8%2BfPKyIiQjabTceOHXOUKunCuxY7deokLy8v2Ww2p2MvzvMOQgAAUFl1smDt2rVLa9eu1euvv17hfinpwk3uzz//vA4dOqTz589r%2BfLlysrK0vDhw%2BXh4aGRI0dq0aJFOnz4sHJzczV37lwNGTJELVu2VNeuXRUZGak5c%2BYoPz9fmZmZWrZsmcaOHStJuvfee3X06FG9%2B%2B67OnfunDZu3KivvvpKcXFx1/tpAAAA9VStvURos9kkXXiHoCRt2bJFkpSWlqbU1FQVFhbq9ttvdzqmd%2B/eeueddzRlyhSVlZXpwQcf1NmzZ9W5c2e9%2B%2B67atWqlSTpySefVFFRkYYPH66ysjLFxMQoOTnZsc4rr7yiWbNm6ZZbbpGXl5fGjBmjMWPGSJICAwO1dOlSzZ49WwsWLFCbNm20YMGCCje%2BAwAAXI6bMcbU9iZcQU5OgeVrenq6a9D8bZavW1M%2BTexf7TU8Pd3l7%2B8lu73IZV7HJzOZGyoyu0ZmqXZz33DDpT83s6bVyZcIAQAA6jMKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYrEoFq7Cw0Op9AAAANBhVKlgDBgzQ9OnTtXv3bqv3AwAAUO9VqWAlJycrJydH48aN09ChQ7V8%2BXKdOnXK6r0BAADUS1UqWMOGDdPbb7%2Btr776SqNHj9amTZt02223KTExUdu3b7d6jwAAAPVKtW5yDwgI0EMPPaSVK1dq7ty52r59u379619ryJAh2rhxo1V7BAAAqFc8q3Nwbm6uPvzwQ3344YfKysrSgAEDNHLkSOXk5Cg5OVlZWVl67LHHrNorAABAvVClgrVt2zatXr1an3/%2Bufz9/fXAAw9o5MiRatOmjeMxYWFhio%2BPp2ABAACXU6WC9dhjj6lfv35auHChBg4cKA8PjwqPiYyMVFBQULU3CAAAUN9UqWBt3rxZwcHBOn/%2BvKNcFRUVycvLy%2Blxn3zySfV3CAAAUM9U6SZ3Nzc33Xvvvfr8888dYykpKbr77rt1%2BPBhyzYHAABQH1WpYL3wwgu66aab1LNnT8fYfffdJ5vNphdeeMGyzQEAANRHVXqJcNeuXfriiy/UvHlzx1jLli313HPPKSYmxrLNAQAA1EdVuoJljFFpaWmF8bNnz6q8vLzamwIAAKjPqlSw%2Bvfvr2nTpikjI0P5%2BfnKy8vTrl27NGXKFA0YMKDS62zbtk39%2BvXTlClTKsytX79ed955p2w2m%2B655x6nT4g3xmjhwoXq37%2B/unXrpkceecTp3i%2B73a4pU6aoZ8%2Be6t27t5599lmdO3fOMb9v3z6NGjVKkZGRio6O1vLlyyt9bgAAgKupUsF67rnnVFxcrOHDh6tv376KiorSgw8%2BKA8PD82cObNSa7z11luaM2eO2rdvX2EuPT1dSUlJmjx5snbu3KmHH35YEydO1PHjxyVJK1asUGpqqt5%2B%2B21t375dwcHBmjRpkowxkqQZM2YoNzdXmzdv1rp167Rv3z7Nnz9f0oWrbPHx8erZs6d27NihxYsX6/XXX9fmzZsrdW4AAICrqVLBCgwM1PLly7V%2B/XotWbJEr776qtatW6d3331XLVu2rNQaTZo00Zo1ay5ZsFJTUxUdHa2hQ4eqadOmio2NVWhoqNauXStJWr16tX7961%2Bra9eu8vb2VlJSkg4cOKBvv/1Wubm52rp1q6ZPn66WLVuqVatWSkxMVGpqqs6fP68vvvhCJSUlmjp1qry8vNS9e3fFxcUpJSWlUucGAAC4mmr9qZyQkBAFBwc7vj9//rwkqXHjxlc9dty4cZedy8jIUHR0tNNYWFiY0tPTVVxcrAMHDigiIsIx5%2B3trRtvvFHp6ekqLCyUp6enOnfu7JgPDw/XmTNndPDgQWVkZKhLly5OH44aFham1atXX/XcAAAAlVGlgvXtt98qOTlZ//nPf1RWVlZhft%2B%2BfdXalN1ul5%2Bfn9OYr6%2Bv9u/fr7y8PBlj5OvrW2H%2B1KlT8vX1lbe3t9zd3Z3mJOnUqVOy2%2B0VjvXz81NeXp7Ky8uveO7Kys7OVk5OjtOYp2dzyz/Z3sOjWn%2Br%2B7rz9Kz%2Bfi9mrm/Zq4PMroHMrsEVM0uumbtKBSs5OVktWrTQs88%2Bq6ZNm1q9p8tyc3Or0Xmrjk1JSdGSJUucxiZOnKiEhIQqn78h8Pf3uvqDKsnHp5lla9UXZHYNZHYNrphZcq3cVSpYhw4d0j/%2B8Q81adLE6v1IkgICAmS3253G7Ha7AgIC5O/vL3d3d%2BXl5VWYDwwMVGBgoAoKClRWVuZ4GfDiWhfns7KyKhx7cd0rnbuy4uLiNHDgQKcxT8/mstuLKr1GZdS33wSsyO/h4S4fn2bKzz%2BrsjLX%2BEgQMpO5oSKza2SWaje3lb/cX4sqFaw2bdqopKSkxgqWzWbT3r17ncbS0tJ09913q3HjxgoNDdXevXvVu3dvSVJeXp6ysrJks9kUHBys8vJyZWZmKiwsTJK0Z88etWjRQh06dJDNZtPKlStVWloqT09Px3xkZORVz11ZQUFBFV4OzMkpUGmp6/xjuhQr85eVlbvc80lm10Bm1%2BCKmSXXyl2lSyBPP/205s6dq8LCQqv3I0mKjY3V9u3btWHDBp07d07vvfeesrKyNGzYMEnS6NGj9fbbb%2Bv7779XQUGB5syZo4iICEVGRsrf31933XWX5s6dq5MnT%2Bro0aNauHCh4uLi1KhRI0VHR8vLy0sLFixQUVGRvvnmG61atUpjx46t1LkBAACuxs1c/PCoazB8%2BHAdOXJEhYWF8vf3r3B/0t/%2B9rerrmGz2STJ8YnwF68mpaWlSZI2b96sBQsW6NixYwoJCdHMmTPVq1cvx/Gvvvqq/vznP6uoqEh9%2B/bV888/r1/84heSpIKCAiUnJ%2Bvzzz9Xo0aNdO%2B99yopKcnx7sb9%2B/dr1qxZ2rt3rwIDA/XYY49p9OjRjrWvdu6qyMkpqNbxl%2BLp6a5B87dZvm5N%2BTSxf7XX8PR0l7%2B/l%2Bz2Ipf5LYjMZG6oyOwamaXazX3DDS2u6/kuqlLBWrhwoRo1anTZ%2BUmTJlVrUw0RBYuCVVVkJnNDRWbXyCy5ZsGq0j1Yl/rTNgAAALigym9D%2B%2B677zR9%2BnQ9/PDDkqTy8nJ9%2Bumnlm0MAACgvqpSwfrss880ZswY2e127d69W5J0/PhxPffcc45PRAcAAHBVVSpYb775pubNm6c333zTcYN7mzZt9Morr%2Bjdd9%2B1cn8AAAD1TpUK1sGDBzV48GBJzp9wHhUVpaNHj1qzMwAAgHqqSgWrUaNGFT5JXbrwCe/X80/nAAAA1EVVKli33XabZs6cqQMHDki68Kdktm3bpsTERMXExFi6QQAAgPqmSgVr%2BvTpMsbo7rvvVnFxsfr166f4%2BHi1bt1azzzzjNV7BAAAqFeq9DlYPj4%2BWrp0qQ4cOKBDhw7Jzc1NHTt2VMeOHa3eHwAAQL1TpYJ1UUhIiEJCQqzaCwAAQINQpYI1YMCAy86VlZVpx44dVd4QAABAfVelghUXF%2Bf08Qzl5eU6cuSItm/frscff9yyzQEAANRHVSpYTz755CXH9%2BzZow8%2B%2BKBaGwIAAKjvqvy3CC8lMjJSaWlpVi4JAABQ71hasH744QedPn3ayiUBAADqnSq9RDhq1KgKY%2BfPn9d///tf3X777dXeFAAAQH1WpYLVoUMHp5vcJalJkyZ64IEH9MADD1iyMQAAgPqqSgXrpZdesnofAAAADUaVCtbq1avVqFGjSj122LBhVTkFAABAvVWlgvXCCy%2BouLhYxhincTc3N6cxNzc3ChYAAHA5VSpYb7zxht5//31NmDBBISEhKisr0/79%2B7Vs2TKNGzdOUVFRVu8TAACg3qhSwXrxxRf1hz/8QUFBQY6xHj166De/%2BY0effRRbdiwwbINAgAA1DdV%2BhysI0eOyMfHp8K4r6%2Bvjh07Vu1NAQAA1GdVKlgdO3bU3LlzZbfbHWOnT5/WggUL1LFjR8s2BwAAUB9V6SXCmTNnasKECVq1apW8vLzk5uamwsJCeXl56bXXXrN6jwAAAPVKlQpWz5499cUXX%2BjLL7/U8ePHZYxRq1atFB0dLW9vb6v3CAAAUK9UqWBJUrNmzTRo0CAdO3ZMwcHBVu4JAACgXqvSPVjnzp3Tb37zG3Xr1k133XWXJCk/P1%2BPPfaYCgoKLN0gAABAfVOlgrV48WJ9%2B%2B23mj9/vtzd/2%2BJkpIS/e53v7NscwAAAPVRlQrWli1btGjRIg0ZMsTxR599fHw0d%2B5cbd261dINAgAA1DdVKljZ2dnq0KFDhfHAwEAVFhZWd08AAAD1WpUK1i9%2B8Qvt3r27wvimTZvUunXram8KAACgPqvSuwgfeeQR/b//9/80YsQIlZWV6Z133lF6ero2b96sZ5991uo9AgAA1CtVKlijRo2Sn5%2Bfli9frubNm2vp0qXq2LGj5s%2BfryFDhlR7Uzt37tSjjz7qNGaMUUlJibZs2aI77rhDjRs3dpp/%2BeWXHe9oXLFihd59913l5uaqc%2BfOSk5OVnh4uCSpuLhYL7zwgjZu3KiSkhLdcsstSk5OVkBAgKQLfwboN7/5jXbt2qVmzZpp%2BPDhmjp1qtPN/AAAAFdSpYKVm5urIUOGWFKmLqV3795KS0tzGnv99df173//WwUFBWrUqFGF%2BYv%2B%2Bte/atGiRXrjjTcUGRmpd955R48//rg2b96s5s1oBOLIAAAeFElEQVSba968edq9e7dSU1Pl5eWlZ555RjNmzNCbb74pY4wmTZqkTp066csvv9TJkycVHx%2Bvli1b6le/%2BlWNZAUAAA3PNV%2BWKS8vV0xMjIwxNbGfSzp27JhWrFihadOm6fTp02rRosVlH7t69WqNGDFCN998s5o3b66JEydKkj7//HOVlpbqo48%2BUmJiooKDgxUQEKCkpCRt3bpVJ06cUFpamjIzMzVz5kz5%2BvoqJCRE8fHxWrly5fWKCgAAGoBrLlju7u7q16%2BfPv3005rYzyUtXLhQI0aMUJs2bZSfn6/y8nI9/vjj6t27twYPHqzly5c7Cl9GRobj5UBJcnNzU9euXZWenq6srCwVFhY6zYeEhKhZs2bau3evMjIy1LZtW/n5%2BTnmw8PDdejQId4dCQAAKq1KLxG2adNGL774opYtW6Ybb7xRjRo1cppfsGCBJZuTpB9%2B%2BEFbtmzRZ599Jklq3LixOnXqpLFjx%2BqVV17Rrl27lJCQIG9vb8XGxsputzsVJEny9fXVqVOnZLfbHd//lI%2BPj2P%2B53MXv7fb7ZX%2BO4vZ2dnKyclxGvP0bK6goKDKB68ED4/6dV%2BYp2f193sxc33LXh1kdg1kdg2umFlyzdxVKlj79%2B9Xx44dJclRWmrK%2B%2B%2B/r0GDBjluQo%2BJiVFMTIxjvn///oqLi1NqaqpiY2MdH3z6c5cbr%2Bz8tUhJSdGSJUucxiZOnKiEhATLzlEf%2Bft7WbaWj08zy9aqL8jsGsjsGlwxs%2BRaua%2BpYE2ZMkULFy7Ue%2B%2B95xh77bXXHPc51YRNmzYpOTn5io9p166dNm/eLEny9/dXXl6e07zdbldoaKgCAwMlSXl5eWrevLmkC%2B9OzMvLU2BgoMrKyi55rCRHwauMuLg4DRw40GnM07O57PaiSq9RGfXtNwEr8nt4uMvHp5ny88%2BqrKzcgl3VfWQmc0NFZtfILNVubit/ub8W11SwPv/88wpjy5Ytq7GCtX//fmVnZ6tPnz6OsY0bNyo3N1djx451jB08eFDBwcGSJJvNpvT0dA0bNkySVFZWpoyMDI0YMULBwcHy8/PT3r171aZNG0lSZmamzp8/r4iICOXk5OjYsWOy2%2B3y9/eXJO3Zs0edOnWSl1fl/wMFBQVVeDkwJ6dApaWu84/pUqzMX1ZW7nLPJ5ldA5ldgytmllwr9zVdArnUOwdr8t2E%2B/btU%2BvWrZ3ufWrSpIlefvllbd%2B%2BXaWlpfr73/%2BuNWvWOArXqFGjlJqaqq%2B//lpFRUX6/e9/r6ZNm2rgwIHy8PDQyJEjtWjRIh0%2BfFi5ubmaO3euhgwZopYtW6pr166KjIzUnDlzlJ%2Bfr8zMTC1btsypzAEAAFzNNV3ButR9Slbeu/RzOTk5FW5Yj4mJ0YwZM/T888/rxIkTateunZ577jndcccdkqTo6GhNmzZN06dPV25uriIiIrRs2TI1adJEkvTkk0%2BqqKhIw4cPV1lZmWJiYpxegnzllVc0a9Ys3XLLLfLy8tKYMWM0ZsyYGssIAAAaHjdzDZegunXrpu%2B%2B%2B%2B6qY6goJ6fA8jU9Pd01aP42y9etKZ8m9q/2Gp6e7vL395LdXuQyl5nJTOaGisyukVmq3dw33HD5z86sSfXrLmkAAIB64JpeIiwpKdHUqVOvOmbl52ABAADUN9dUsH75y18qOzv7qmMAAACu7JoK1k8//woAAACXxj1YAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABarswUrJiZGERERstlsjq/Zs2dLknbs2KH77rtPNptNgwYN0scff%2Bx07IoVKxQTE6PIyEjFxsZq7969jrni4mLNmjVLffr0UY8ePZSQkKBTp0455o8cOaLx48ere/fuioqK0rx581ReXn59QgMAgAahzhas/Px8/fGPf1RaWprj67nnntOJEyc0YcIEjRgxQt98842mT5%2BumTNnas%2BePZKkv/71r1q0aJHmzp2rr7/%2BWrfeeqsef/xxnTlzRpI0b9487d69W6mpqfrss8907tw5zZgxQ5JkjNGkSZPk7%2B%2BvL7/8Uu%2B//74%2B/fRTrVixotaeBwAAUP/UyYJVVlamoqIi%2Bfj4VJj75JNP1L59e40bN07NmjXTwIEDdfvtt2vNmjWSpNWrV2vEiBG6%2Beab1bx5c02cOFGS9Pnnn6u0tFQfffSREhMTFRwcrICAACUlJWnr1q06ceKE0tLSlJmZqZkzZ8rX11chISGKj4/XypUrr2t%2BAABQv9XJgpWfny9jjF599VUNGDBAAwYM0KxZs1RUVKSMjAyFh4c7PT4sLEzp6emSVGHezc1NXbt2VXp6urKyslRYWOg0HxISombNmmnv3r3KyMhQ27Zt5efn55gPDw/XoUOHVFhYWMOpAQBAQ%2BFZ2xu4lPPnz6tbt27q3bu3XnjhBWVnZ2vy5MlKTk6W3W5Xly5dnB7v5%2BfnuI/Kbrc7FSRJ8vX11alTp2S32x3f/5SPj49j/udzF7%2B32%2B3y9vau1P6zs7OVk5PjNObp2VxBQUGVOr6yPDzqZD%2B%2BLE/P6u/3Yub6lr06yOwayOwaXDGz5Jq562TBatWqlVatWuX43tvbW08//bSeeOIJ9erV65LHuLm5Of3v5eYv52rz1yIlJUVLlixxGps4caISEhIsO0d95O/vZdlaPj7NLFurviCzayCza3DFzJJr5a6TBetS2rVrp/Lycrm7uysvL89pzm63KyAgQJLk7%2B9/yfnQ0FAFBgZKkvLy8tS8eXNJF25sz8vLU2BgoMrKyi55rCTH%2BpURFxengQMHOo15ejaX3V5U6TUqo779JmBFfg8Pd/n4NFN%2B/lmVlbnGuzvJTOaGisyukVmq3dxW/nJ/Lepkwdq3b58%2B%2Bugjx7v7JOngwYNq3Lixbr31Vv3lL39xevyePXsUGRkpSbLZbEpPT9ewYcMkXbhhPiMjQyNGjFBwcLD8/Py0d%2B9etWnTRpKUmZmp8%2BfPKyIiQjk5OTp27Jjsdrv8/f0da3fq1EleXpX/DxQUFFTh5cCcnAKVlrrOP6ZLsTJ/WVm5yz2fZHYNZHYNrphZcq3cdfISSGBgoFavXq1ly5bp/PnzOnTokBYtWqTRo0fr/vvv19GjR/Xuu%2B/q3Llz2rhxo7766ivFxcVJkkaNGqXU1FR9/fXXKioq0u9//3s1bdpUAwcOlIeHh0aOHKlFixbp8OHDys3N1dy5czVkyBC1bNlSXbt2VWRkpObMmaP8/HxlZmZq2bJlGjt2bC0/IwAAoD6pk1ewgoKCtGzZMs2fP19vvPGG/P39NXToUCUkJKhx48ZaunSpZs%2BerQULFqhNmzZasGCB48b36OhoTZs2TdOnT1dubq4iIiK0bNkyNWnSRJL05JNPqqioSMOHD1dZWZliYmKUnJzsOPcrr7yiWbNm6ZZbbpGXl5fGjBmjMWPG1MbTAAAA6ik3Y4yp7U24gpycAsvX9PR016D52yxft6Z8mti/2mt4errL399LdnuRy1xmJjOZGyoyu0ZmqXZz33BDi%2Bt6vovq5EuEAAAA9RkFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwWJ0tWEeOHNGECRPUp08fRUVFadq0aTp9%2BrQkqby8XF26dFFERIRsNpvj6w9/%2BIPj%2BPXr1%2BvOO%2B%2BUzWbTPffco%2B3bt9dWFAAA4GLqbMGaMGGC/Pz8tHXrVq1du1YHDhzQyy%2B/LEkqKCiQMUafffaZ0tLSHF/jx4%2BXJKWnpyspKUmTJ0/Wzp079fDDD2vixIk6fvx4bUYCAAAuok4WrIKCAkVEROjpp5%2BWl5eXgoKCNHz4cP3zn/%2BUJOXn50uSfHx8Lnl8amqqoqOjNXToUDVt2lSxsbEKDQ3V2rVrr1sGAADguupkwWrRooXmzp2rwMBAx9ixY8fUunVrSdLp06fl5uammTNn6uabb1ZMTIwWLFigkpISSVJGRobCw8Od1gwLC1N6evr1CwEAAFyWZ21voDLS0tL03nvvaenSpY6xbt26KSYmRrNnz9aBAwc0adIkeXh4KDExUXa7XX5%2Bfk5r%2BPr6av/%2B/ddlv9nZ2crJyXEa8/RsrqCgIEvP4%2BFRJ/vxZXl6Vn%2B/FzPXt%2BzVQWbXQGbX4IqZJdfMXecL1q5duzRhwgRNnTpVUVFRkqSIiAilpKQ4HmOz2fTYY4/pzTffVGJi4mXXcnNzq/H9SlJKSoqWLFniNDZx4kQlJCRcl/PXVf7%2BXpat5ePTzLK16gsyuwYyuwZXzCy5Vu46XbC2bt2qp59%2BWrNmzdL9999/xce2a9dOp06dkjFGAQEBstvtTvN2u10BAQE1uV2HuLg4DRw40GnM07O57PYiS89T334TsCK/h4e7fHyaKT//rMrKyi3YVd1HZjI3VGR2jcxS7ea28pf7a1FnC9bu3buVlJSkxYsXq3///k5zO3bs0O7duzVx4kTH2MGDB9W2bVu5ubnJZrNp7969TsekpaXp7rvvvi57DwoKqvByYE5OgUpLXecf06VYmb%2BsrNzlnk8yuwYyuwZXzCy5Vu46eQmktLRUM2fO1OTJkyuUK0ny8/PTG2%2B8obVr16q0tFRpaWn6wx/%2BoLFjx0qSYmNjtX37dm3YsEHnzp3Te%2B%2B9p6ysLA0bNux6RwEAAC6oTl7B%2Bvbbb3XgwAG99NJLeumll5zmNm7cqK5du2rhwoV69dVX9Zvf/EZBQUF65JFH9NBDD0mSQkNDNX/%2BfC1YsEBJSUkKCQnR0qVL1bJly9qIAwAAXEydLFi9evVSZmbmFR8zaNAgDRo06LLzgwcP1uDBg63eGgAAwFXVyZcIAQAA6jMKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYl3DkyBGNHz9e3bt3V1RUlObNm6fy8vLa3hYAAKgnPGt7A3WNMUaTJk1Sp06d9OWXX%2BrkyZOKj49Xy5Yt9atf/aq2twcAAOoBrmD9TFpamjIzMzVz5kz5%2BvoqJCRE8fHxWrlyZW1vDQAA1BNcwfqZjIwMtW3bVn5%2Bfo6x8PBwHTp0SIWFhfL29r7qGtnZ2crJyXEa8/RsrqCgIEv36uFRv/qxp2f193sxc33LXh1kdg1kdg2umFlyzdwUrJ%2Bx2%2B3y9fV1Grv4vd1ur1TBSklJ0ZIlS5zGJk2apCeffNK6jepCkXv4F/sVFxdneXmrq7Kzs7VixdtkbuDITOaGyhUzS66Z23Wq5HUUFxenDz/80OkrLi7O8vPk5ORoyZIlFa6WNWRkdg1kdg1kdh2umJsrWD8TGBiovLw8pzG73S5JCggIqNQaQUFBLtPQAQBARVzB%2BhmbzaZjx445SpUk7dmzR506dZKXl1ct7gwAANQXFKyf6dq1qyIjIzVnzhzl5%2BcrMzNTy5Yt09ixY2t7awAAoJ7wSE5OTq7tTdQ1t9xyizZt2qTZs2drw4YNGj16tMaPH1/b27okLy8v9enTx6WurpHZNZDZNZDZdbhabjdjjKntTQAAADQkvEQIAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxClY9dOTIEY0fP17du3dXVFSU5s2bp/Ly8treliRp27Zt6tevn6ZMmVJhbv369brzzjtls9l0zz33aPv27Y45Y4wWLlyo/v37q1u3bnrkkUd0%2BPBhx7zdbteUKVPUs2dP9e7dW88%2B%2B6zOnTvnmN%2B3b59GjRqlyMhIRUdHa/ny5Zad%2B0qOHDmiCRMmqE%2BfPoqKitK0adN0%2BvTpGt9TTT4fV/P999/rkUceUa9evXTzzTdr8uTJys7OliTt2LFD9913n2w2mwYNGqSPP/7Y6dgVK1YoJiZGkZGRio2N1d69ex1zxcXFmjVrlvr06aMePXooISFBp06dcnqur/RzX51zX4sXX3xRnTt3tuS8dT1zTEyMIiIiZLPZHF%2BzZ89u8Llff/11DRgwQD169HD6t9cQM%2B/cudPpv6/NZlNERITjZ7whZr5uDOqV8vJyc//995upU6eavLw885///MfExMSYd955p7a3ZpYtW2YGDx5sRo0aZRITE53m0tLSTHh4uFm/fr05e/asWbVqlenWrZv58ccfjTHGLF%2B%2B3PTv399kZGSYgoICM3PmTHPfffeZ8vJyY4wxTzzxhHnooYdMTk6OOX78uPmf//kfM3v2bGOMMWfOnDH9%2B/c3v/vd70xhYaH517/%2BZXr16mU2bdpkybmv5J577jHPPPOMKSwsNCdOnDDDhw83M2bMqPE91eTzcSXFxcUmKirKLFmyxBQXF5vc3Fzz4IMPmgkTJpjjx4%2Bbbt26mRUrVpgzZ86Yzz77zNhsNvPdd98ZY4zZvHmz6d69u9mxY4cpKioyr776qunfv78pKioyxhgze/Zsc/fdd5usrCyTm5tr4uPjzeOPP26MufrPfXXPXVkZGRmmT58%2BJjQ01JLz1vXMPXv2NLt27aow3pBz/%2BlPfzIPPPCAOXLkiMnLyzPPPPOMef755xt05p977bXXzOTJk10qc02gYNUz3333nenSpYux2%2B2OsQ8%2B%2BMAMHjy4Fnd1wYoVK0x%2Bfr5JSkqqULCSk5PNhAkTnMZiY2PNm2%2B%2BaYwxZujQoWb58uWOuYKCAhMeHm52795tTp48aTp37mwyMjIc819%2B%2BaXp3r27KS4uNhs2bDB9%2BvQxpaWljvl58%2BaZRx99tNrnvpL8/HzzzDPPmJMnTzrG3n//fTN48OAa3VNNPx9XkpeXZ1atWmVKSkocY%2B%2B9954ZNGiQeeutt8x9993n9PjExETz3HPPGWOMiY%2BPN3PmzHHMlZeXm/79%2B5tPPvnElJSUmJ49e5q//vWvjvn//Oc/JjQ01Bw/fvyqP/fVOXdllZWVmdjYWPP66687ClZDzlxaWmo6d%2B5s9u/fX2GuIeceOHCg%2Bfbbb10q808dPXrU9OnTxxw9etRlMtcUXiKsZzIyMtS2bVv5%2Bfk5xsLDw3Xo0CEVFhbW4s6kcePGqUWLFpecy8jIUHh4uNNYWFiY0tPTVVxcrAMHDigiIsIx5%2B3trRtvvFHp6enKyMiQp6en08sy4eHhOnPmjA4ePKiMjAx16dJFHh4eFdau7rmvpEWLFpo7d64CAwMdY8eOHVPr1q1rdE81%2BXxcja%2Bvr2JjY%2BXp6SljjP773//qww8/1F133XXVdX8%2B7%2Bbmpq5duyo9PV1ZWVkqLCx0mg8JCVGzZs20d%2B/eq/7cV%2BfclbVy5Uo1bdpU9957r2OsIWfOz8%2BXMUavvvqqBgwYoAEDBmjWrFkqKipqsLlPnDih48eP64cfftDgwYPVt29fJSYmym63N9jMP7dw4UKNGDFCbdq0cZnMNYWCVc/Y7Xb5%2Bvo6jV383m6318aWKsVutzv9Q5Iu7PvUqVPKy8uTMeaSuU6dOiW73S5vb2%2B5u7s7zUlyzP/8WD8/P%2BXl5am8vLxa574WaWlpeu%2B99/T444/X6J5q8vmorKNHjyoiIkJDhw6VzWbT5MmTL3vei%2Bte6bwXf3Z/fryPj89lM/305746566MkydP6rXXXlNycrLTeEPOfP78eXXr1k29e/fWxo0b9cc//lH/%2Bte/lJyc3GBzHz9%2BXG5ubtqyZYtSUlL0l7/8RUePHtVzzz3XYDP/1A8//KAtW7Zo/PjxjnUbeuaaRMFCrXJzc6vR%2Beu19q5duzR%2B/HhNnTpVUVFRtbanmnw%2Bfqpt27ZKT0/Xxo0b9d///lf/%2B7//e9V1L7f%2B9chU1XNfNHfuXI0cOVI33XTTdTlvXcjcqlUrrVq1Sg8%2B%2BKC8vb1100036emnn9a6detUWlpaI%2Beu7dwlJSUqKSnR//7v/8rf31%2BtW7dWQkKCtmzZUmPnre3MP/X%2B%2B%2B9r0KBBCggIqNHz1qXMNYmCVc8EBgYqLy/PaezibwpX%2B0dRmwICAipcYbPb7QoICJC/v7/c3d0vmSswMFCBgYEqKChQWVmZ05wkx/yljr24bnXOXRlbt27VY489pmeffVYPP/ywY181taeafD6uhZubmzp06KBp06Zp3bp1atSo0SXPe3Fdf3//y85ffK5/Om%2BMUV5e3hUzSRd%2BtgICAqp87qvZsWOH0tPT9cQTT1SYq85563Lmy2nXrp3Ky8sv%2B/NZ33NfvCLi7e3tGGvbtq2MMSotLW2QmX9q06ZNGjJkiON7V/v5thoFq56x2Ww6duyY0/9B7tmzR506dZKXl1ct7uzKbDZbhbfQpqWlKTIyUo0bN1ZoaKjTfF5enrKysmSz2RQWFqby8nJlZmY65vfs2aMWLVqoQ4cOstlsyszMdPqtes%2BePYqMjKz2ua9m9%2B7dSkpK0uLFi3X//fc75a2pPdXk83E133zzje644w6ntS%2B%2Brbpfv34V1v35eX96f0RZWZkyMjIUGRmp4OBg%2Bfn5OR2fmZmp8%2BfPOz4m4Eo/95fKVNlzX83HH3%2Bs48ePKzo6Wn379tXw4cMlSX379lXnzp0bZGbpwkd9vPjii05jBw8eVOPGjXXrrbc2yNzt27eXt7e30/pHjx6Vp6enbrvttgaZ%2BaL9%2B/crOztbffr0cYxV57z1IXONq6Wb61ENI0eONE899ZQ5ffq0%2Bf77703//v3Nn/70p9relsOl3kWYmZlpbDab46MB/vjHP5qePXuanJwcY4wxf/7zn03//v3Nvn37TH5%2Bvpk6daqJjY11HD9lyhTz4IMPmpycHHPkyBFz9913m5dfftkYc%2BGjA2JiYsxLL71kCgsLzT/%2B8Q/TvXt388UXX1hy7sspKSkxd911l3n//fcrzNX0nmry%2BbiSgoIC069fP/PSSy%2BZM2fOmNzcXDN%2B/HgzZswYc/LkSdOzZ0%2BzfPlyc/bsWfPpp58am81m9u3bZ4z5v3c67tixwxQWFpqXX37Z3HbbbebcuXPGGGPmz59vhg4darKysszJkyfNuHHjzOTJkx3nvtLPfXXPfSV5eXnmxx9/dHz961//MqGhoebHH380R48ebZCZjTHmxIkTpnv37mbp0qWmuLjYHDx40Nx9993mhRdeaLD/rY0xZu7cueb%2B%2B%2B83P/74o8nOzjZxcXFm%2BvTpDTqzMcasXbvW3HbbbU5jDT1zTaNg1UM//vijiY%2BPN5GRkSYqKsq8%2Buqrtb0lY4wxERERJiIiwnTp0sV06dLF8f1FmzZtMoMHDzYRERHm/vvvNzt37nQ6fvHixSYqKspERkaa%2BPh4p89lys/PN0899ZTp3r276d27t3n%2B%2BedNcXGxY/7f//63GTVqlLHZbOa2224zH3zwgdPa1Tn35ezcudOEhoY6cv7068iRIzW6p5p%2BPq4kIyPDPPzww%2BaXv/yl6du3r0lISDDHjx93PCf33XefiYiIMIMHDzabN292OvaDDz4wt912m7HZbGb06NHm3//%2Bt2OuuLjY/Pa3vzW9evUyPXr0ME899ZTJz893zF/t5746574Whw8fdnxMQ0PP/M0335iRI0ea7t27m5iYGDNv3jzHz1lDzX1xb7179zZ9%2B/Y1zzzzjCkoKGjQmY0x5u233zbDhg2rMN6QM9c0N2OMqe2raAAAAA0J92ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABY7P8DKOzel7Eo6MEAAAAASUVORK5CYII%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-2768471354743219373">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">60000.0</td>
            <td class="number">6513</td>
            <td class="number">3.8%</td>
            <td>
                <div class="bar" style="width:6%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">50000.0</td>
            <td class="number">6008</td>
            <td class="number">3.5%</td>
            <td>
                <div class="bar" style="width:5%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">40000.0</td>
            <td class="number">5135</td>
            <td class="number">3.0%</td>
            <td>
                <div class="bar" style="width:5%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">45000.0</td>
            <td class="number">4651</td>
            <td class="number">2.7%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">65000.0</td>
            <td class="number">4538</td>
            <td class="number">2.6%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">70000.0</td>
            <td class="number">4224</td>
            <td class="number">2.4%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">75000.0</td>
            <td class="number">4035</td>
            <td class="number">2.3%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">55000.0</td>
            <td class="number">3972</td>
            <td class="number">2.3%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">80000.0</td>
            <td class="number">3971</td>
            <td class="number">2.3%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">30000.0</td>
            <td class="number">3512</td>
            <td class="number">2.0%</td>
            <td>
                <div class="bar" style="width:3%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (14635)</td>
            <td class="number">126186</td>
            <td class="number">73.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-2768471354743219373">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">4000.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:34%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4080.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:34%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4200.0</td>
            <td class="number">2</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:67%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4800.0</td>
            <td class="number">3</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4888.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:34%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">2039784.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">5000000.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">6000000.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">6100000.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">7141778.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">avg_cur_bal<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>36973</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>30.5%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (%)</th>
                        <td>29.9%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (n)</th>
                        <td>51649</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>12920</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>958080</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-6911166240170087198">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAAPtJREFUeJzt1bEJAlEQRVFXLGmLsCdje7IIexpzkQsbyN/gnHzgJZfZZmYuwE/X1QPgzG6rB3zbH6/DN%2B/n/Q9LwAeBJBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAjbzMzqEXBWPggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiED9obC49m6JdFAAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-6911166240170087198,#minihistogram-6911166240170087198"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-6911166240170087198">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-6911166240170087198"
                                                      aria-controls="quantiles-6911166240170087198" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-6911166240170087198" aria-controls="histogram-6911166240170087198"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-6911166240170087198" aria-controls="common-6911166240170087198"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-6911166240170087198" aria-controls="extreme-6911166240170087198"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-6911166240170087198">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>1041</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>2692</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>6441</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>18246</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>42192</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>958080</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>958080</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>15554</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>16414</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>1.2704</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>170.13</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>12920</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>11074</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>6.0427</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>1564600000</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>269430000</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-6911166240170087198">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3X18jHe%2B//F3klEkkWQSTVseqjYeSpKZonUTNylxWNVSRxFlV9tjQy3SoHVX1Cnb7C5KVbulPcVpt5Vq2tU7SktvjmV7Y1uS2JyuXy2hCJmRGxKSfH9/9GFOR6iIy0yG1/Px8Gjz/cx1XZ/rw%2BTxzlzXTIKMMUYAAACwTLC/GwAAALjaELAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwmM3fDVwrCgtLLN9ncHCQoqPDVFRUpupqY/n%2B8X%2BYte8wa99h1r7DrH3n3Flff30T//Thl6PCEsHBQQoKClJwcJC/W7nqMWvfYda%2Bw6x9h1n7Tn2ZNQELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACL%2BTVgff755%2BrWrZsmT55co7Zx40YNHDhQHTp0UL9%2B/ZSVleVVX7NmjXr37i2n06lhw4YpNzfXU6uoqNDcuXPVuXNndejQQenp6SoqKvLUCwoKNGbMGLVv315JSUlauHChqqurPfXt27dr0KBBcjgc6tu3r955550rcPYAAOBq5beA9eKLL2rBggVq2bJljdquXbs0bdo0TZ48WV999ZXmzJmj%2BfPn66uvvpIkbd68WUuXLlVmZqZ27NihO%2B%2B8U%2BPGjdPJkyclSQsXLtTOnTuVnZ2tjz/%2BWOXl5Zo1a5YkyRijiRMnym6369NPP9Wrr76qDRs2aM2aNZKkI0eOaPz48Ro6dKi%2B%2BOILzZw5U7Nnz9auXbt8NBkAABDo/BawGjZsqDfffPO8Acvtduvhhx9WSkqKQkJC1LNnT916662egLVu3ToNHTpUXbt2VWhoqCZMmCBJ2rJliyorK/X2228rIyNDLVq0UHR0tKZPn66tW7fqyJEj2r17t/Lz8zV79mxFRkYqLi5OaWlpWrt2rSTp3XffVcuWLTV69Gg1btxYKSkp6tOnj958803fDQcAAAQ0m78OPHr06AvWkpOTlZyc7Pm6srJSR48e1U033SRJysvL04ABAzz1oKAgtWvXTjk5OYqPj1dpaakSEhI89bi4ODVu3Fi5ubk6evSomjdvrqioKE89ISFB%2B/btU2lpqfLy8ry2laT4%2BHht2LDhss/5Srjj8Y3%2BbqHWNmR093cLAAD4hN8C1qVYtGiRwsLC1L9/f0mSy%2BXyCkiSFBkZqaKiIrlcLs/XPxUREeGpn1s7%2B7XL5ZLL5VLbtm296lFRUV73cF3M0aNHVVhY6LVms4UqNja21vuojZCQwHqPgs0WWP3%2B1NlZB9rMAxGz9h1m7TvM2nfqy6zrdcAyxmjRokV67733tGbNGjVs2FDSj69Ync%2BF1mtbt2rbrKwsLV%2B%2B3GttwoQJSk9Pr/PxrwZ2e5i/W7hsERGN/d3CNYNZ%2Bw6z9h1m7Tv%2BnnW9DVjV1dWaOXOmdu/eraysLDVv3txTs9vtcrvdXo93uVxq06aNYmJiJP14H1doaKikH4Oa2%2B1WTEyMqqqqzrutJEVHRys6Ovq89ejo6Fr3npqaqpSUFK81my1ULldZrfdRG/5O55fK6vP3pZCQYEVENFZx8SlVVVVffAPUGbP2HWbtO8zad86dtb9%2BuK%2B3Aeupp57S3r179frrr9e4pOdwOJSTk6PBgwdLkqqqqpSXl6ehQ4eqRYsWioqKUm5urpo1ayZJys/P1%2BnTp5WYmKjCwkIdOnRILpdLdrtd0o/vWmzdurXCwsLkcDj01ltveR1v165dcjqdte49Nja2xuXAwsISVVZe20%2Bqq%2BH8q6qqr4rzCATM2neYte8wa9/x96zr5UsgX3/9tdavX6/nn3%2B%2BRriSpBEjRig7O1s7duxQWVmZnn76aTVq1MjzrsPhw4dr6dKlOnDggI4fP67MzEz1799fTZs2Vbt27eR0OrVgwQIVFxcrPz9fK1eu1KhRoyRJAwcO1MGDB7V69WqVl5dr48aN%2Buyzz5SamurrMQAAgADlt1ewHA6HpB/fIShJH330kSRp9%2B7dys7OVmlpqfr06eO1TadOnfTyyy8rOTlZ06ZN08yZM3X8%2BHElJiZq5cqVnnu0Jk2apLKyMg0ZMkRVVVXq3bu35s2b59nPM888o7lz56pnz54KCwvTyJEjNXLkSElSTEyMVqxYofnz52vx4sVq1qyZFi9eXOPGdwAAgAsJMsYYfzdxLSgsLLF8nzZbsPou%2Btzy/V4pgfwxDTZbsOz2MLlcZby8f4Uxa99h1r7DrH3n3Flff30Tv/RRLy8RAgAABDICFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFjMrwHr888/V7du3TR58uQatffff1%2B//OUv5XA4dM8992jbtm2emjFGS5YsUffu3XXbbbfpwQcf1IEDBzx1l8ulyZMnq2PHjurUqZMef/xxlZeXe%2Bp79uzRiBEj5HQ6lZycrFWrVtX62AAAABfjt4D14osvasGCBWrZsmWNWk5OjqZPn65HHnlEX375pR544AFNmDBBhw8fliStWbNG2dnZeumll7Rt2za1aNFCEydOlDFGkjRr1iwdP35cmzZt0nvvvac9e/Zo0aJFkqRTp04pLS1NHTt21Pbt27Vs2TI9//zz2rRpU62ODQAAcDF%2BC1gNGzbUm2%2B%2Bed6AlZ2dreTkZA0YMECNGjXSsGHD1KZNG61fv16StG7dOv3mN79Ru3btFB4erunTp2vv3r365ptvdPz4cW3dulUzZ85U06ZNdcMNNygjI0PZ2dk6ffq0PvnkE505c0ZTp05VWFiY2rdvr9TUVGVlZdXq2AAAABdj89eBR48efcFaXl6ekpOTvdbi4%2BOVk5OjiooK7d27V4mJiZ5aeHi4br75ZuXk5Ki0tFQ2m0233nqrp56QkKCTJ0/q%2B%2B%2B/V15entq2bauQkBCvfa9bt%2B6ix66to0ePqrCw0GvNZgtVbGxsrfdRGyEhgXULnc0WWP3%2B1NlZB9rMAxGz9h1m7TvM2nfqy6z9FrB%2BjsvlUlRUlNdaZGSkvvvuO7ndbhljFBkZWaNeVFSkyMhIhYeHKzg42KsmSUVFRXK5XDW2jYqKktvtVnV19c8eu7aysrK0fPlyr7UJEyYoPT291vu4GtntYf5u4bJFRDT2dwvXDGbtO8zad5i17/h71vUyYF1IUFDQFa1btW1qaqpSUlK81my2ULlcZXU%2B/vn4O51fKqvP35dCQoIVEdFYxcWnVFVV7e92rmrM2neYte8wa985d9b%2B%2BuG%2BXgas6OhouVwurzWXy6Xo6GjZ7XYFBwfL7XbXqMfExCgmJkYlJSWqqqryXAY8u6%2Bz9f3799fY9ux%2Bf%2B7YtRUbG1vjcmBhYYkqK6/tJ9XVcP5VVdVXxXkEAmbtO8zad5i17/h71vXyJRCHw6Hc3Fyvtd27d8vpdOq6665TmzZtvOput1v79%2B%2BXw%2BFQfHy8qqurlZ%2Bf76nv2rVLTZo00S233CKHw6H8/HxVVlZ61Z1O50WPDQAAUBv1MmANGzZM27Zt0wcffKDy8nK98sor2r9/vwYPHixJuv/%2B%2B/XSSy/pH//4h0pKSrRgwQIlJibK6XTKbrfrrrvuUmZmpo4dO6aDBw9qyZIlSk1NVYMGDZScnKywsDAtXrxYZWVl%2BuKLL/TGG29o1KhRtTo2AADAxQSZsx8e5WMOh0OSPK8k2Ww/Xq3cvXu3JGnTpk1avHixDh06pLi4OM2ePVt33HGHZ/tnn31Wr7/%2BusrKytSlSxc9%2BeSTuvHGGyVJJSUlmjdvnrZs2aIGDRpo4MCBmj59uq677jpJ0nfffae5c%2BcqNzdXMTExGjt2rO6//37Pvi927LooLCy5rO3Px2YLVt9Fn1u%2B3ytlQ0Z3f7dQZzZbsOz2MLlcZby8f4Uxa99h1r7DrH3n3Flff30Tv/Tht4B1rSFgEbBQO8zad5i17zBr36kvAateXiIEAAAIZAQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALFZvA1Zubq5Gjx6t22%2B/Xd26ddO0adPkcrkkSdu3b9egQYPkcDjUt29fvfPOO17brlmzRr1795bT6dSwYcOUm5vrqVVUVGju3Lnq3LmzOnTooPT0dBUVFXnqBQUFGjNmjNq3b6%2BkpCQtXLhQ1dXVvjlpAABwVaiXAauqqkpjx45Vhw4dtH37dn3wwQc6duyY5s2bpyNHjmj8%2BPEaOnSovvjiC82cOVOzZ8/Wrl27JEmbN2/W0qVLlZmZqR07dujOO%2B/UuHHjdPLkSUnSwoULtXPnTmVnZ%2Bvjjz9WeXm5Zs2aJUkyxmjixImy2%2B369NNP9eqrr2rDhg1as2aN32YBAAACT70MWIWFhTp27JgGDhyo6667TlFRUerTp4/y8vL07rvvqmXLlho9erQaN26slJQU9enTR2%2B%2B%2BaYkad26dRo6dKi6du2q0NBQTZgwQZK0ZcsWVVZW6u2331ZGRoZatGih6OhoTZ8%2BXVu3btWRI0e0e/du5efna/bs2YqMjFRcXJzS0tK0du1af44DAAAEmHoZsG644QbFx8frjTfe0KlTp1RUVKTNmzerV69eysvLU0JCgtfj4%2BPjlZOTI0k16kFBQWrXrp1ycnK0f/9%2BlZaWetXj4uLUuHFj5ebmKi8vT82bN1dUVJSnnpCQoH379qm0tPQKnzUAALha2PzdwPkEBQVp2bJlevDBBz2X57p06aIpU6bot7/9rdq2bev1%2BKioKM99VC6XyysgSVJkZKSKioo893BFRkZ61SMiIjz1c2tnv3a5XAoPD69V/0ePHlVhYaHXms0WqtjY2FptX1shIfUyH1%2BQzRZY/f7U2VkH2swDEbP2HWbtO8zad%2BrLrOtlwDp9%2BrTGjRunAQMG6OGHH9apU6c0d%2B5cPfbYYxfcJigoyOu/F6pfbHsrZGVlafny5V5rEyZMUHp6umXHCER2e5i/W7hsERGN/d3CNYNZ%2Bw6z9h1m7Tv%2BnnW9DFh//etfVVBQoIyMDIWEhCgsLEyTJk3S4MGDdeedd8rtdns93uVyKTo6WpJkt9vPW2/Tpo1iYmIkSW63W6GhoZJ%2BvLHd7XYrJiZGVVVV591Wkmf/tZGamqqUlBSvNZstVC5XWa33URv%2BTueXyurz96WQkGBFRDRWcfEpVVXxrtIriVn7DrP2HWbtO%2BfO2l8/3NfLgGWMqfHRCGfOnJEkde3aVX/5y1%2B8art27ZLT6ZQkORwO5eTkaPDgwZJ%2BfEdiXl6ehg4dqhYtWigqKkq5ublq1qyZJCk/P1%2BnT59WYmKiCgsLdejQIblcLtntds%2B%2BW7durbCw2v8FxcbG1rgcWFhYosrKa/tJdTWcf1VV9VVxHoGAWfsOs/YdZu07/p51vXwJpH379goLC9Ozzz6r8vJynThxQi%2B%2B%2BKI6dOige%2B%2B9VwcPHtTq1atVXl6ujRs36rPPPlNqaqokacSIEcrOztaOHTtUVlamp59%2BWo0aNVJKSopCQkI0fPhwLV26VAcOHNDx48eVmZmp/v37q2nTpmrXrp2cTqcWLFig4uJi5efna%2BXKlRo1apSfJwIAAAJJvXwFy26368UXX9TChQvVo0cPNWjQQJ07d9bSpUsVExOjFStWaP78%2BVq8eLGaNWumxYsXe258T05O1rRp0zRz5kwdP35ciYmJWrlypRo2bChJmjRpksrKyjRkyBBVVVWpd%2B/emjdvnufYzzzzjObOnauePXsqLCxMI0eO1MiRI/0xBgAAEKCCjDHG301cCwoLSyzfp80WrL6LPrd8v1fKhozu/m6hzmy2YNntYXK5ynh5/wpj1r7DrH2HWfvOubO%2B/vomfumjXl4iBAAACGQELAAAAIvVKWDxqeYAAAAXVqeA1aNHD82cOVM7d%2B60uh8AAICAV6eANW/ePBUWFmr06NEaMGCAVq1a5flVNQAAANe6OgWswYMH66WXXtJnn32m%2B%2B%2B/Xx9%2B%2BKF69eqljIwMbdu2zeoeAQAAAspl3eQeHR2tX//611q7dq0yMzO1bds2/eY3v1H//v21ceNGq3oEAAAIKJf1QaPHjx/XW2%2B9pbfeekv79%2B9Xjx49NHz4cBUWFmrevHnav3%2B/xo4da1WvAAAAAaFOAevzzz/XunXrtGXLFtntdt13330aPny45/f7SVJ8fLzS0tIIWAAA4JpTp4A1duxYdevWTUuWLPH8jr9zOZ3OGr/wGAAA4FpQp4C1adMmtWjRQqdPn/aEq7KyMoWFhXk97t133738DgEAAAJMnW5yDwoK0sCBA7VlyxbPWlZWlu6%2B%2B24dOHDAsuYAAAACUZ0C1u9%2B9zv94he/UMeOHT1rgwYNksPh0O9%2B9zvLmgMAAAhEdbpE%2BPXXX%2BuTTz5RaGioZ61p06aaM2eOevfubVlzAAAAgahOr2AZY1RZWVlj/dSpU6qurr7spgAAAAJZnQJW9%2B7dNW3aNOXl5am4uFhut1tff/21Jk%2BerB49eljdIwAAQECp0yXCOXPm6NFHH9WQIUMUFBTkWe/SpYtmz55tWXMAAACBqE4BKyYmRqtWrdLevXu1b98%2BGWPUqlUrxcXFWd0fAABAwLmsX5UTFxenFi1aeL4%2Bffq0JOm66667vK4AAAACWJ0C1jfffKN58%2Bbpn//8p6qqqmrU9%2BzZc9mNAQAABKo6Bax58%2BapSZMmevzxx9WoUSOrewIAAAhodQpY%2B/bt09/%2B9jc1bNjQ6n4AAAACXp0%2BpqFZs2Y6c%2BaM1b0AAABcFeoUsB599FFlZmaqtLTU6n4AAAACXp0uES5fvlwFBQV6%2B%2B23ZbfbvT4LS5L%2B53/%2Bx5LmAAAAAlGdAlbPnj3VoEEDq3sBAAC4KtQpYE2ePNnqPgAAAK4adboHS5K%2B/fZbzZw5Uw888IAkqbq6Whs2bLCsMQAAgEBVp4D18ccfa%2BTIkXK5XNq5c6ck6fDhw5ozZ47WrVtnaYMAAACBpk4B64UXXtDChQv1wgsveG5wb9asmZ555hmtXr3ayv4AAAACTp0C1vfff69%2B/fpJktc7CJOSknTw4EFrOgMAAAhQdQpYDRo0kNvtrrG%2Bb98%2BfnUOAAC45tUpYPXq1UuzZ8/W3r17JUkul0uff/65MjIy1Lt3b0sbBAAACDR1ClgzZ86UMUZ33323Kioq1K1bN6Wlpemmm27SjBkzrO4RAAAgoNTpc7AiIiK0YsUK7d27V/v27VNQUJBatWqlVq1aWd0fAABAwKlTwDorLi5OcXFxVvUCAABwVahTwOrRo8cFa1VVVdq%2BfXudGwIAAAh0dQpYqampXh/PUF1drYKCAm3btk3jxo2zrDkAAIBAVKeANWnSpPOu79q1S6%2B99tplNQQAABDo6vy7CM/H6XRq9%2B7dVu4SAAAg4FgasP71r3/pxIkTlu3v%2BeefV48ePdShQwc9%2BOCDOnDggCRp%2B/btGjRokBwOh/r27at33nnHa7s1a9aod%2B/ecjqdGjZsmHJzcz21iooKzZ07V507d1aHDh2Unp6uoqIiT72goEBjxoxR%2B/btlZSUpIULF6q6utqycwIAAFe/Ol0iHDFiRI2106dP6//9v/%2BnPn36XHZTkvTaa69py5YtysrKUnh4uH7/%2B99r9erVGjt2rMaPH68pU6Zo2LBh2r59uzIyMnTLLbfI6XRq8%2BbNWrp0qf70pz/J6XTq5Zdf1rhx47Rp0yaFhoZq4cKF2rlzp7KzsxUWFqYZM2Zo1qxZeuGFF2SM0cSJE9W6dWt9%2BumnOnbsmNLS0tS0aVM99NBDlpwXAAC4%2BtUpYN1yyy1eN7lLUsOGDXXffffpvvvus6Sx//qv/9LTTz%2Bt5s2bS5IyMzMlSS%2B99JJatmyp0aNHS5JSUlLUp08fvfnmm3I6nVq3bp2GDh2qrl27SpImTJigtWvXasuWLerfv7/efvtt/eEPf1CLFi0kSdOnT9eAAQN05MgRHTlyRPn5%2BVq9erUiIyMVGRmptLQ0rV69moAFAABqrU4B6/e//73VfXg5cuSIDh8%2BrH/961967LHHdOLECSUlJemJJ55QXl6eEhISvB4fHx%2BvDRs2SJLy8vI0YMAATy0oKEjt2rVTTk6O4uPjVVpa6rV9XFycGjdurNzcXB09elTNmzdXVFSUp56QkKB9%2B/aptLRU4eHhter/6NGjKiws9Fqz2UIVGxt7ybP4OSEhll7hveJstsDq96fOzjrQZh6ImLXvMGvfYda%2BU19mXaeAtW7dOjVo0KBWjx08ePAl7//w4cMKCgrSRx99pKysLJWXlys9PV1z5sxRWVmZ2rZt6/X4qKgoz31ULpfLKyBJUmRkpIqKiuRyuTxf/1RERISnfm7t7Ncul6vWASsrK0vLly/3WpswYYLS09Nrtf3Vym4P83cLly0iorG/W7hmMGvfYda%2Bw6x9x9%2BzrlPA%2Bt3vfqeKigoZY7zWg4KCvNaCgoLqFLDOnDmjM2fO6LHHHpPdbpckpaenKy0tTUlJSefd5uwly3MvXZ5bv5CL1S9FamqqUlJSvNZstlC5XGWWHUPyfzq/VFafvy%2BFhAQrIqKxiotPqaqKNz1cSczad5i17zBr3zl31v764b5OAetPf/qTXn31VY0fP15xcXGqqqrSd999p5UrV2r06NEXDEG1dfYVqJ%2B%2BYtS8eXMZY1RZWSm32%2B31eJfLpejoaEmS3W4/b71NmzaKiYmRJLndboWGhkqSjDFyu92KiYlRVVXVebeV5Nl/bcTGxta4HFhYWKLKymv7SXU1nH9VVfVVcR6BgFn7DrP2HWbtO/6edZ1eAnnqqaf0xBNPKDExUY0bN1Z4eLg6dOigJ554QvPnz7/splq2bKnw8HCvj1c4ePCgbDabevXq5bUu/fgBp06nU5LkcDiUk5PjqVVVVSkvL09Op1MtWrRQVFSU1/b5%2Bfk6ffq0EhMT5XA4dOjQIU%2BoOrvv1q1bKyws8C9vAQAA36hTwCooKFBERESN9cjISB06dOiym2rQoIGGDRumRYsW6fDhwyosLNRzzz2ne%2B%2B9V4MHD9bBgwe1evVqlZeXa%2BPGjfrss8%2BUmpoq6cePkMjOztaOHTtUVlamp59%2BWo0aNVJKSopCQkI0fPhwLV26VAcOHNDx48eVmZmp/v37q2nTpmrXrp2cTqcWLFig4uJi5efna%2BXKlRo1atRlnxMAALh21ClgtWrVSpmZmV6v9Jw4cUKLFy9Wq1atLGlsypQp6tixowYNGqSBAweqVatWmjVrlmJiYrRixQq9/fbb6tSpk5YsWaLFixd7bnxPTk7WtGnTNHPmTCUlJenvf/%2B7Vq5cqYYNG0r68df8dOnSRUOGDFHfvn3VtGlTr1fdnnnmGZWUlKhnz5566KGHNGLECI0cOdKScwIAANeGIHPuneq1sHPnTo0fP17FxcUKCwtTUFCQSktLFRYWpueee05dunS5Er0GtMLCEsv3abMFq%2B%2Bizy3f75WyIaO7v1uoM5stWHZ7mFyuMu6fuMKYte8wa99h1r5z7qyvv76Jf/qoy0YdO3bUJ598ok8//VSHDx%2BWMUY33HCDkpOTa/1RBgAAAFerOgUsSWrcuLH69u2rQ4cOeT4VHQAAAHW8B6u8vFxPPPGEbrvtNt11112SpOLiYo0dO1YlJdZfCgMAAAgkdQpYy5Yt0zfffKNFixYpOPj/dnHmzBn94Q9/sKw5AACAQFSngPXRRx9p6dKl6t%2B/v%2BcT0CMiIpSZmamtW7da2iAAAECgqVPAOnr0qG655ZYa6zExMSotLb3cngAAAAJanQLWjTfeqJ07d9ZY//DDD3XTTTdddlMAAACBrE7vInzwwQf129/%2BVkOHDlVVVZVefvll5eTkaNOmTXr88cet7hEAACCg1ClgjRgxQlFRUVq1apVCQ0O1YsUKtWrVSosWLVL//v2t7hEAACCg1ClgHT9%2BXP379ydMAQAAnMcl34NVXV2t3r17qw6/YQcAAOCacMkBKzg4WN26ddOGDRuuRD8AAAABr06XCJs1a6annnpKK1eu1M0336wGDRp41RcvXmxJcwAAAIGoTgHru%2B%2B%2BU6tWrSRJLpfL0oYAAAAC3SUFrMmTJ2vJkiV65ZVXPGvPPfecJkyYYHljAAAAgeqS7sHasmVLjbWVK1da1gwAAMDV4JIC1vneOci7CQEAALxdUsA6%2B4udL7YGAABwLavT7yIEAADAhRHsxplGAAAZZklEQVSwAAAALHZJ7yI8c%2BaMpk6detE1PgcLAABcyy4pYN1%2B%2B%2B06evToRdcAAACuZZcUsH76%2BVcAAAA4P%2B7BAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACwWEAHrqaee0q233ur5evv27Ro0aJAcDof69u2rd955x%2Bvxa9asUe/eveV0OjVs2DDl5uZ6ahUVFZo7d646d%2B6sDh06KD09XUVFRZ56QUGBxowZo/bt2yspKUkLFy5UdXX1lT9JAABw1aj3AWvPnj1av3695%2BsjR45o/PjxGjp0qL744gvNnDlTs2fP1q5duyRJmzdv1tKlS5WZmakdO3bozjvv1Lhx43Ty5ElJ0sKFC7Vz505lZ2fr448/Vnl5uWbNmiVJMsZo4sSJstvt%2BvTTT/Xqq69qw4YNWrNmje9PHAAABKx6HbCqq6v1xBNP6MEHH/Ssvfvuu2rZsqVGjx6txo0bKyUlRX369NGbb74pSVq3bp2GDh2qrl27KjQ0VBMmTJAkbdmyRZWVlXr77beVkZGhFi1aKDo6WtOnT9fWrVt15MgR7d69W/n5%2BZo9e7YiIyMVFxentLQ0rV271h%2BnDwAAAlS9Dlhr165Vo0aNNHDgQM9aXl6eEhISvB4XHx%2BvnJyc89aDgoLUrl075eTkaP/%2B/SotLfWqx8XFqXHjxsrNzVVeXp6aN2%2BuqKgoTz0hIUH79u1TaWnplTpNAABwlbH5u4ELOXbsmJ577jm98sorXusul0tt27b1WouKivLcR%2BVyubwCkiRFRkaqqKhILpfL8/VPRUREeOrn1s5%2B7XK5FB4eXqvejx49qsLCQq81my1UsbGxtdq%2BtkJC6nU%2BrsFmC6x%2Bf%2BrsrANt5oGIWfsOs/YdZu079WXW9TZgZWZmavjw4frFL36hgoKCiz4%2BKCjI678Xql9seytkZWVp%2BfLlXmsTJkxQenq6ZccIRHZ7mL9buGwREY393cI1g1n7DrP2HWbtO/6edb0MWNu3b1dOTo6eeuqpGrXo6Gi53W6vNZfLpejoaEmS3W4/b71NmzaKiYmRJLndboWGhkr68cZ2t9utmJgYVVVVnXfbs8etrdTUVKWkpHit2WyhcrnKar2P2vB3Or9UVp%2B/L4WEBCsiorGKi0%2Bpqop3lV5JzNp3mLXvMGvfOXfW/vrhvl4GrHfeeUeHDx9WcnKypB9DkCR16dJFY8aM0Xvvvef1%2BF27dsnpdEqSHA6HcnJyNHjwYElSVVWV8vLyNHToULVo0UJRUVHKzc1Vs2bNJEn5%2Bfk6ffq0EhMTVVhYqEOHDsnlcslut3v23bp1a4WF1f4vKDY2tsblwMLCElVWXttPqqvh/Kuqqq%2BK8wgEzNp3mLXvMGvf8fes6%2BVLIDNmzNCHH36o9evXa/369Vq5cqUkaf369brnnnt08OBBrV69WuXl5dq4caM%2B%2B%2BwzpaamSpJGjBih7Oxs7dixQ2VlZXr66afVqFEjpaSkKCQkRMOHD9fSpUt14MABHT9%2BXJmZmerfv7%2BaNm2qdu3ayel0asGCBSouLlZ%2Bfr5WrlypUaNG%2BXMcAAAgwNTLV7AiIyO9bjavrKyUJN14442SpBUrVmj%2B/PlavHixmjVrpsWLF3tufE9OTta0adM0c%2BZMHT9%2BXImJiVq5cqUaNmwoSZo0aZLKyso0ZMgQVVVVqXfv3po3b57nWM8884zmzp2rnj17KiwsTCNHjtTIkSN9dOYAAOBqEGTOXn/DFVVYWGL5Pm22YPVd9Lnl%2B71SNmR093cLdWazBctuD5PLVcbL%2B1cYs/YdZu07zNp3zp319dc38Usf9fISIQAAQCAjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDF6m3AKigo0Pjx49W5c2clJSVp2rRpOnHihCRpz549GjFihJxOp5KTk7Vq1Sqvbd9//3398pe/lMPh0D333KNt27Z5asYYLVmyRN27d9dtt92mBx98UAcOHPDUXS6XJk%2BerI4dO6pTp056/PHHVV5e7puTBgAAV4V6G7DGjx%2BvqKgobd26VevXr9fevXv1xz/%2BUadOnVJaWpo6duyo7du3a9myZXr%2B%2Bee1adMmSVJOTo6mT5%2BuRx55RF9%2B%2BaUeeOABTZgwQYcPH5YkrVmzRtnZ2XrppZe0bds2tWjRQhMnTpQxRpI0a9YsHT9%2BXJs2bdJ7772nPXv2aNGiRX6bAwAACDz1MmCVlJQoMTFRjz76qMLCwhQbG6shQ4boq6%2B%2B0ieffKIzZ85o6tSpCgsLU/v27ZWamqqsrCxJUnZ2tpKTkzVgwAA1atRIw4YNU5s2bbR%2B/XpJ0rp16/Sb3/xG7dq1U3h4uKZPn669e/fqm2%2B%2B0fHjx7V161bNnDlTTZs21Q033KCMjAxlZ2fr9OnT/hwJAAAIIDZ/N3A%2BTZo0UWZmptfaoUOHdNNNNykvL09t27ZVSEiIpxYfH69169ZJkvLy8pScnOy1bXx8vHJyclRRUaG9e/cqMTHRUwsPD9fNN9%2BsnJwclZaWymaz6dZbb/XUExISdPLkSX3//fde6z/n6NGjKiws9Fqz2UIVGxtbuwHUUkhIvczHF2SzBVa/P3V21oE280DErH2HWfsOs/ad%2BjLrehmwzrV792698sorWrFihd5//31FRkZ61aOiouR2u1VdXS2Xy6WoqCivemRkpL777ju53W4ZY2psHxkZqaKiIkVGRio8PFzBwcFeNUkqKiqqdb9ZWVlavny519qECROUnp5e631cjez2MH%2B3cNkiIhr7u4VrBrP2HWbtO8zad/w963ofsL7%2B%2BmuNHz9eU6dOVVJSkt5///067ScoKOiK1n8qNTVVKSkpXms2W6hcrrJa76M2/J3OL5XV5%2B9LISHBiohorOLiU6qqqvZ3O1c1Zu07zNp3mLXvnDtrf/1wX68D1tatW/Xoo49q7ty5uvfeeyVJMTEx2r9/v9fjXC6X7Ha7goODFR0dLZfLVaMeHR3teYzb7a5Rj4mJUUxMjEpKSlRVVeW5BHl2XzExMbXuOzY2tsblwMLCElVWXttPqqvh/Kuqqq%2BK8wgEzNp3mLXvMGvf8fes6%2B1LIDt37tT06dO1bNkyT7iSJIfDofz8fFVWVnrWdu3aJafT6ann5uZ67Wv37t1yOp267rrr1KZNG6%2B62%2B3W/v375XA4FB8fr%2BrqauXn53vtu0mTJrrllluu0JkCAICrTb0MWJWVlZo9e7YeeeQRde/e3auWnJyssLAwLV68WGVlZfriiy/0xhtvaNSoUZKkYcOGadu2bfrggw9UXl6uV155Rfv379fgwYMlSffff79eeukl/eMf/1BJSYkWLFigxMREOZ1O2e123XXXXcrMzNSxY8d08OBBLVmyRKmpqWrQoIHP5wAAAAJTkDn7AVD1yFdffaVRo0bpuuuuq1HbuHGjTp48qblz5yo3N1cxMTEaO3as7r//fs9jNm3apMWLF%2BvQoUOKi4vT7Nmzdccdd3jqzz77rF5//XWVlZWpS5cuevLJJ3XjjTdK%2BvEjIubNm6ctW7aoQYMGGjhwoKZPn37eXi5FYWHJZW1/PjZbsPou%2Btzy/V4pGzK6X/xB9ZTNFiy7PUwuVxkv719hzNp3mLXvMGvfOXfW11/fxC991MuAdTUiYBGwUDvM2neYte8wa9%2BpLwGrXl4iBAAACGQELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAes8CgoKNGbMGLVv315JSUlauHChqqur/d0WAAAIEDZ/N1DfGGM0ceJEtW7dWp9%2B%2BqmOHTumtLQ0NW3aVA899JC/2wMAAAGAV7DOsXv3buXn52v27NmKjIxUXFyc0tLStHbtWn%2B3BgAAAgSvYJ0jLy9PzZs3V1RUlGctISFB%2B/btU2lpqcLDwy%2B6j6NHj6qwsNBrzWYLVWxsrKW9hoQEVj6%2Ba%2Bk2f7dwSTY/2tPz/2dnHWgzD0TM2neYte8wa9%2BpL7MmYJ3D5XIpMjLSa%2B3s1y6Xq1YBKysrS8uXL/damzhxoiZNmmRdo/oxyD1w43dKTU21PLzB29GjR7VmzUvM2geYte8wa99h1r5TX2ZNlL4CUlNT9dZbb3n9SU1Ntfw4hYWFWr58eY1Xy2A9Zu07zNp3mLXvMGvfqS%2Bz5hWsc8TExMjtdnutuVwuSVJ0dHSt9hEbG8tPKAAAXMN4BescDodDhw4d8oQqSdq1a5dat26tsLAwP3YGAAACBQHrHO3atZPT6dSCBQtUXFys/Px8rVy5UqNGjfJ3awAAIECEzJs3b56/m6hvevbsqQ8//FDz58/XBx98oPvvv19jxozxd1vnFRYWps6dO/Pqmg8wa99h1r7DrH2HWftOfZh1kDHG%2BO3oAAAAVyEuEQIAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyAFYAKCgo0ZswYtW/fXklJSVq4cKGqq6v93Va9UVBQoPHjx6tz585KSkrStGnTdOLECUnSnj17NGLECDmdTiUnJ2vVqlVe277//vv65S9/KYfDoXvuuUfbtm3z1IwxWrJkibp3767bbrtNDz74oA4cOOCpu1wuTZ48WR07dlSnTp30%2BOOPq7y83FO/2LED3VNPPaVbb73V8/X27ds1aNAgORwO9e3bV%2B%2B8847X49esWaPevXvL6XRq2LBhys3N9dQqKio0d%2B5cde7cWR06dFB6erqKioo89Ys9By527ED2/PPPq0ePHurQoYPXv0Hmba3c3FyNHj1at99%2Bu7p166Zp06bJ5XJJYtaX6/PPP1e3bt00efLkGjV/fg%2B%2BnGOfl0FAqa6uNvfee6%2BZOnWqcbvd5p///Kfp3bu3efnll/3dWr1xzz33mBkzZpjS0lJz5MgRM2TIEDNr1ixz8uRJ0717d/OHP/zBlJaWmr///e/mjjvuMB9%2B%2BKExxpjdu3ebhIQE8/7775tTp06ZN954w9x2223mhx9%2BMMYYs2rVKtO9e3eTl5dnSkpKzOzZs82gQYNMdXW1McaYhx9%2B2Pz61782hYWF5vDhw%2Bbf//3fzfz5840x5qLHDnR5eXmmc%2BfOpk2bNsYYYw4fPmxuu%2B02s2bNGnPy5Enz8ccfG4fDYb799ltjjDGbNm0y7du3N9u3bzdlZWXm2WefNd27dzdlZWXGGGPmz59v7r77brN//35z/Phxk5aWZsaNG2eMufhz4GLHDmR//vOfzX333WcKCgqM2%2B02M2bMME8%2B%2BSTztlhlZaXp1q2befrpp01FRYVxuVzmoYceMunp6cz6Mq1cudL069fPjBgxwmRkZHjV/Pk9%2BHKPfT4ErADz7bffmrZt2xqXy%2BVZe%2B2110y/fv382FX9UVxcbGbMmGGOHTvmWXv11VdNv379zAcffGA6d%2B5sKisrPbWFCxea//iP/zDGGDNv3jwzfvx4r/0NGzbMvPDCC8YYYwYMGGBWrVrlqZWUlJiEhASzc%2BdOc%2BzYMXPrrbeavLw8T/3TTz817du3NxUVFRc9diCrqqoyw4YNM88//7wnYL344otm0KBBXo/LyMgwc%2BbMMcYYk5aWZhYsWOCpVVdXm%2B7du5t3333XnDlzxnTs2NFs3rzZU//nP/9p2rRpYw4fPnzR58DFjh3IUlJSzDfffFNjnXlb64cffjBt2rQx3333nWft1VdfNf/2b//GrC/TmjVrTHFxsZk%2BfXqNgOXP78GXc%2BwL4RJhgMnLy1Pz5s0VFRXlWUtISNC%2BfftUWlrqx87qhyZNmigzM1MxMTGetUOHDummm25SXl6e2rZtq5CQEE8tPj5eOTk5kn6cbUJCgtf%2BztYrKiq0d%2B9eJSYmemrh4eG6%2BeablZOTo7y8PNlsNq9LZAkJCTp58qS%2B//77ix47kK1du1aNGjXSwIEDPWs/N8vz1YOCgtSuXTvl5ORo//79Ki0t9arHxcWpcePGys3Nvehz4GLHDlRHjhzR4cOH9a9//Uv9%2BvVTly5dlJGRIZfLxbwtdsMNNyg%2BPl5vvPGGTp06paKiIm3evFm9evVi1pdp9OjRatKkyXlr/vwefDnHvhACVoBxuVyKjIz0Wjv79dn7A/B/du/erVdeeUXjxo077%2ByioqLkdrtVXV0tl8vl9Y1N%2BnG2RUVFcrvdMsacd/ZFRUVyuVwKDw9XcHCwV02Sp/5zxw5Ux44d03PPPad58%2BZ5rV/ofM/ea/Jzsz777/jc7SMiIi44y58%2BBy527EB1%2BPBhBQUF6aOPPlJWVpb%2B8pe/6ODBg5ozZw7ztlhQUJCWLVumjz/%2B2HMvVHV1taZMmcKsryB/fg%2B%2BnGNfCAELV62vv/5aY8aM0dSpU5WUlFTn/QQFBV3ReiDLzMzU8OHD9Ytf/KJWjz87iwvN5ErOMtD/Hs6cOaMzZ87osccek91u10033aT09HR99NFHF9yGedfN6dOnNW7cOA0YMEA7d%2B7Utm3bFB4erscee%2ByC2zDrK8ef34MvZ98ErAATExMjt9vttXb2p6Lo6Gh/tFQvbd26VWPHjtXjjz%2BuBx54QNKFZ2e32xUcHKzo6OgarwK6XC5FR0d7HnO%2B7WNiYhQTE6OSkhJVVVV51c4e92LHDkTbt29XTk6OHn744Rq16Ojo857v2X%2Bjdrv9gvWzl3d/WjfGyO12/%2Bwszx73YscOVGd/ug4PD/esNW/eXMYYVVZWMm8L/fWvf1VBQYEyMjIUFhampk2batKkSdq8ebMaNGjArK8Qf34PvpxjX0hgfme/hjkcDh06dMjrH8KuXbvUunVrhYWF%2BbGz%2BmPnzp2aPn26li1bpnvvvdez7nA4lJ%2Bfr8rKSs/arl275HQ6PfWfvp1a%2BvESo9Pp1HXXXac2bdp41d1ut/bv3y%2BHw6H4%2BHhVV1crPz/fa99NmjTRLbfcctFjB6J33nlHhw8fVnJysrp06aIhQ4ZIkrp06aJbb721xizPnfVP712oqqpSXl6enE6nWrRooaioKK/t8/Pzdfr0aSUmJl70OXC%2Bv8dAn7UktWzZUuHh4V7ndvDgQdlsNvXq1Yt5W8gYU%2BPS/ZkzZyRJXbt2ZdZXiD%2B/B1/OsS/oZ272Rz01fPhwM2XKFHPixAnzj3/8w3Tv3t38%2Bc9/9ndb9cKZM2fMXXfdZV599dUatYqKCtO7d2/z%2B9//3pSWlpq//e1vpn379uaTTz4xxhiTn59vHA6H5226//3f/206duxoCgsLjTHGvP7666Z79%2B5mz549pri42EydOtUMGzbMs//JkyebX/3qV6awsNAUFBSYu%2B%2B%2B2/zxj3%2Bs1bEDkdvtNj/88IPnz9///nfTpk0b88MPP5iDBw%2Bajh07mlWrVplTp06ZDRs2GIfDYfbs2WOM%2Bb9392zfvt2UlpaaP/7xj6ZXr16mvLzcGGPMokWLzIABA8z%2B/fvNsWPHzOjRo80jjzziOfbPPQeOHTv2s8cOZJmZmebee%2B81P/zwgzl69KhJTU01M2fOvOg5M%2B9LU1RUZDp37myWLFliTp06Zdxut5k4caJJTU1l1hY537sI/fk9%2BHKPfT4ErAD0ww8/mLS0NON0Ok1SUpJ59tln/d1SvfHll1%2BaNm3amMTExBp/CgoKzP/%2B7/%2BaESNGGIfDYXr16mVee%2B01r%2B0//PBD069fP5OYmGjuvfde8%2BWXX3rVly1bZpKSkozT6TRpaWmez0gx5sePiJgyZYpp37696dSpk3nyySdNRUWFp36xYwe6AwcOeD6mwZgf/y4GDRpkEhMTTb9%2B/cymTZu8Hv/aa6%2BZXr16GYfDYe6//37zv//7v55aRUWF%2Bc///E9zxx13mA4dOpgpU6aY4uJiT/1iz4GLHTtQnZ1Lp06dTJcuXcyMGTNMSUmJMYZ5W%2B3bb781v/rVr8ztt99uunbtatLT0z3Pd2Zdd2e/H7dt29a0bdvW8/VZ/vwefDnHPp8gY4yp02t5AAAAOC/uwQIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALDY/wfDhx%2B4x7OTQwAAAABJRU5ErkJggg%3D%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-6911166240170087198">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">1250.0</td>
            <td class="number">28</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2352.0</td>
            <td class="number">27</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2120.0</td>
            <td class="number">27</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2589.0</td>
            <td class="number">27</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1583.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1336.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2587.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1971.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1724.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (36962)</td>
            <td class="number">120831</td>
            <td class="number">69.9%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="missing">
            <td class="fillremaining">(Missing)</td>
            <td class="number">51649</td>
            <td class="number">29.9%</td>
            <td>
                <div class="bar" style="width:43%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-6911166240170087198">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4.0</td>
            <td class="number">2</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:8%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">383983.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">477255.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">502002.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">800008.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">958084.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">avg_cur_bal_to_inc<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>103048</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>85.1%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (%)</th>
                        <td>29.9%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (n)</th>
                        <td>51649</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>0.18212</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>6.3872</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-6590192124298511576">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAAQlJREFUeJzt1cEJAlEQBUEVQzIIc/JsTgaxOY13kQbB5YtU3QfepZnjzMwBeOu0egD8svPqAa8ut8fHN9v9usMS8EEgCQSCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCefWAb7jcHh/fbPfrDkv4Nz4IhOPMzOoR8Kt8EAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAhPx4oOkVsT5s0AAAAASUVORK5CYII%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-6590192124298511576,#minihistogram-6590192124298511576"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-6590192124298511576">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-6590192124298511576"
                                                      aria-controls="quantiles-6590192124298511576" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-6590192124298511576" aria-controls="histogram-6590192124298511576"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-6590192124298511576" aria-controls="common-6590192124298511576"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-6590192124298511576" aria-controls="extreme-6590192124298511576"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-6590192124298511576">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>0.020754</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>0.050874</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>0.11024</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>0.2552</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>0.55037</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>6.3872</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>6.3872</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>0.20432</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>0.19319</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>1.0608</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>21.742</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>0.18212</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>0.13916</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>2.8321</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>22054</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>0.037321</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-6590192124298511576">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3X90VPWd//FXkuFXJiSZBKPAUsSwCCEzglWQH0YTSou0qIcC4ccu0lKgFAywIjQQIlsoaQtRVOTUaCtUTyVCdKMoiMoPWQqlyLr5RbMWpfxaSEgmQgIhZDLfP1zm62WwhPCByYTn4xyOZ%2B5n7r3veZ14zitzb2ZCvF6vVwAAADAmNNADAAAAtDQULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgmC3QA9wsysvPGD9maGiIYmLsqqysUUOD1/jxgw15WJGHPzKxIg8r8vDXEjK55Zb2ATkv72AFsdDQEIWEhCg0NCTQozQL5GFFHv7IxIo8rMjDH5k0HQULAADAMAoWAACAYQEtWDt37tTAgQM1Z84cv7XNmzdrxIgR6tu3r7773e8qNzfXsr527VolJyfL5XJp9OjRKi4u9q2dP39emZmZ6tevn/r27au0tDRVVlb61o8eParJkyerT58%2BGjBggJYvX66Ghgbf%2Bu7du/Xwww/L6XRq6NChevvtt6/DqwcAAC1VwArWSy%2B9pKVLl6pr165%2BawUFBZo3b57mzJmjffv2adGiRVqyZIn27dsnSfrggw%2B0cuVKZWVlac%2BePXrggQc0bdo0nT17VpK0fPly7d%2B/X3l5efroo49UW1urBQsWSJK8Xq9mzpwph8OhHTt26LXXXtOmTZu0du1aSdLJkyc1ffp0jRo1Snv37lV6eroyMjJUUFBwg5IBAADBLmAFq02bNtqwYcNlC1ZVVZV%2B%2BtOfKiUlRWFhYbr//vt15513%2BgrW%2BvXrNWrUKN13330KDw/XjBkzJElbt25VfX293nrrLc2ePVtdunRRTEyM5s%2Bfr23btunkyZMqLCxUaWmpMjIyFBUVpfj4eE2ZMkXr1q2TJL3zzjvq2rWrJk6cqHbt2iklJUVDhgzRhg0bblw4AAAgqAXsYxomTpz4jWtJSUlKSkryPa6vr1dZWZk6duwoSSopKdHw4cN96yEhIerVq5eKioqUkJCg6upq9e7d27ceHx%2Bvdu3aqbi4WGVlZercubOio6N9671799ahQ4dUXV2tkpISy76SlJCQoE2bNjX6tZWVlam8vNyyzWYLV1xcXKOP0RhhYaGW/97syMOKPPyRiRV5WJGHPzJpuqD4HKwVK1bIbrdr2LBhkiS3220pSJIUFRWlyspKud1u3%2BOvi4yM9K1funbxsdvtltvtVs%2BePS3r0dHRlnu4riQ3N1erVq2ybJsxY4bS0tIafYyrERnZ7rocN1iRhxV5%2BCMTK/KwIg9/ZHL1mnXB8nq9WrFihTZu3Ki1a9eqTZs2kr56x%2Bpyvml7Y9dN7ZuamqqUlBTLNpstXG53TZPPfzlhYaGKjGyn06fPyeNpuPIOLRx5WJGHPzKxIg8r8vDXEjJxOOwBOW%2BzLVgNDQ1KT09XYWGhcnNz1blzZ9%2Baw%2BFQVVWV5flut1s9evRQbGyspK/u4woPD5f0VVGrqqpSbGysPB7PZfeVpJiYGMXExFx2PSYmptGzx8XF%2BV0OLC8/o/r66/PD6fE0XLdjByPysCIPf2RiRR5W5OGPTK5es72oumzZMh08eFCvv/66pVxJktPpVFFRke%2Bxx%2BNRSUmJXC6XunTpoujoaMvHNpSWlqqurk6JiYlyOp06fvy4r1RJX/3VYvfu3WW32%2BV0Oi37Xlx3uVzX6ZUCAICWplkWrE8%2B%2BUT5%2BflavXq13/1SkjR27Fjl5eVpz549qqmp0dNPP622bdv6/upwzJgxWrlypY4cOaKKigplZWVp2LBh6tChg3r16iWXy6WlS5fq9OnTKi0tVU5OjiZMmCBJGjFihI4dO6Y1a9aotrZWmzdv1scff6zU1NQbHQMAAAhSAbtE6HQ6JX31F4KS9OGHH0qSCgsLlZeXp%2Brqag0ZMsSyz7333qvf//73SkpK0rx585Senq6KigolJiYqJyfHd4/W448/rpqaGo0cOVIej0fJyclavHix7zjPPvusMjMzdf/998tut2v8%2BPEaP368JCk2NlYvvviilixZouzsbHXq1EnZ2dl%2BN74DAAB8kxCv1xucX48dZMrLzxg/ps0WKofDLre7hmvjIo9LkYc/MrEiDyvy8NcSMrnllvYBOW%2BzvckdjXPPws2BHqHRNs0eFOgRAAC4IZrlPVgAAADBjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABgW0IK1c%2BdODRw4UHPmzPFbe/fdd/W9731PTqdTP/jBD7Rr1y7fmtfr1TPPPKNBgwbprrvu0qRJk3TkyBHfutvt1pw5c3T33Xfr3nvv1cKFC1VbW%2BtbP3DggMaOHSuXy6WkpCS98sorjT43AADAlQSsYL300ktaunSpunbt6rdWVFSk%2BfPna9asWfrLX/6ixx57TDNmzNCJEyckSWvXrlVeXp5efvll7dq1S126dNHMmTPl9XolSQsWLFBFRYW2bNmijRs36sCBA1qxYoUk6dy5c5oyZYruvvtu7d69W88995xWr16tLVu2NOrcAAAAVxKwgtWmTRtt2LDhsgUrLy9PSUlJGj58uNq2bavRo0erR48eys/PlyStX79eP/nJT9SrVy9FRERo/vz5OnjwoD799FNVVFRo27ZtSk9PV4cOHXTrrbdq9uzZysvLU11dnbZv364LFy7oiSeekN1uV58%2BfZSamqrc3NxGnRsAAOBKbIE68cSJE79xraSkRElJSZZtCQkJKioq0vnz53Xw4EElJib61iIiIvStb31LRUVFqq6uls1m05133ulb7927t86ePasvvvhCJSUl6tmzp8LCwizHXr9%2B/RXP3VhlZWUqLy%2B3bLPZwhUXF9foYzRGWFhw3UJns13feS/mEWy5XC/k4Y9MrMjDijz8kUnTBaxg/SNut1vR0dGWbVFRUfrss89UVVUlr9erqKgov/XKykpFRUUpIiJCoaGhljVJqqyslNvt9ts3OjpaVVVVamho%2BIfnbqzc3FytWrXKsm3GjBlKS0tr9DFaIofDfkPOExnZ7oacJ1iQhz8ysSIPK/LwRyZXr1kWrG8SEhJyXddN7ZuamqqUlBTLNpstXG53TZPPfznB9huF6dd/qbCwUEVGttPp0%2Bfk8TRc13MFA/LwRyZW5GFFHv5aQiY36pf7SzXLghUTEyO3223Z5na7FRMTI4fDodDQUFVVVfmtx8bGKjY2VmfOnJHH4/FdBrx4rIvrhw8f9tv34nH/0bkbKy4uzu9yYHn5GdXXB%2BcPpyk36vV7PA03fdZfRx7%2ByMSKPKzIwx%2BZXL1m%2BRaI0%2BlUcXGxZVthYaFcLpdat26tHj16WNarqqp0%2BPBhOZ1OJSQkqKGhQaWlpb71goICtW/fXrfffrucTqdKS0tVX19vWXe5XFc8NwAAQGM0y4I1evRo7dq1S%2B%2B9955qa2v16quv6vDhw3r00UclSePGjdPLL7%2Bsv/71rzpz5oyWLl2qxMREuVwuORwOPfTQQ8rKytKpU6d07NgxPfPMM0pNTVWrVq2UlJQku92u7Oxs1dTUaO/evXrjjTc0YcKERp0bAADgSkK8Fz886gZzOp2S5HsnyWb76mplYWGhJGnLli3Kzs7W8ePHFR8fr4yMDN1zzz2%2B/Z9//nm9/vrrqqmpUf/%2B/fWLX/xCt912myTpzJkzWrx4sbZu3apWrVppxIgRmj9/vlq3bi1J%2Buyzz5SZmani4mLFxsZq6tSpGjdunO/YVzp3U5SXn7mm/S/HZgvV0BU7jR/3etk0e9B1Pb7NFiqHwy63u4a3skUel0MmVuRhRR7%2BWkImt9zSPiDnDVjButlQsChYNxp5%2BCMTK/KwIg9/LSGTQBWsZnmJEAAAIJhRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhjXbglVcXKyJEyfq29/%2BtgYOHKh58%2BbJ7XZLknbv3q2HH35YTqdTQ4cO1dtvv23Zd%2B3atUpOTpbL5dLo0aNVXFzsWzt//rwyMzPVr18/9e3bV2lpaaqsrPStHz16VJMnT1afPn00YMAALV%2B%2BXA0NDTfmRQMAgBahWRYsj8ejqVOnqm/fvtq9e7fee%2B89nTp1SosXL9bJkyc1ffp0jRo1Snv37lV6eroyMjJUUFAgSfrggw%2B0cuVKZWVlac%2BePXrggQc0bdo0nT17VpK0fPly7d%2B/X3l5efroo49UW1urBQsWSJK8Xq9mzpwph8OhHTt26LXXXtOmTZu0du3agGUBAACCT7MsWOXl5Tp16pRGjBih1q1bKzo6WkOGDFFJSYneeecdde3aVRMnTlS7du2UkpKiIUOGaMOGDZKk9evXa9SoUbrvvvsUHh6uGTNmSJK2bt2q%2Bvp6vfXWW5o9e7a6dOmimJgYzZ8/X9u2bdPJkydVWFio0tJSZWRkKCoqSvHx8ZoyZYrWrVsXyDgAAECQsQV6gMu59dZblZCQoDfeeENz5szRuXPn9MEHH%2BjBBx9USUmJevfubXl%2BQkKCNm3aJEkqKSnR8OHDfWshISHq1auXioqKlJCQoOrqasv%2B8fHxateunYqLi1VWVqbOnTsrOjrat967d28dOnRI1dXVioiIaNT8ZWVlKi8vt2yz2cIVFxd31Vn8I2FhzbIffyOb7frOezGPYMvleiEPf2RiRR5W5OGPTJquWRaskJAQPffcc5o0aZLv8lz//v31b//2b/rZz36mnj17Wp4fHR3tu4/K7XZbCpIkRUVFqbKy0ncPV1RUlGU9MjLSt37p2sXHbre70QUrNzdXq1atsmybMWOG0tLSGrV/S%2BVw2G/IeSIj292Q8wQL8vBHJlbkYUUe/sjk6jXLglVXV6dp06Zp%2BPDh%2BulPf6pz584pMzNTTz755DfuExISYvnvN61faX8TUlNTlZKSYtlms4XL7a4xdg4p%2BH6jMP36LxUWFqrIyHY6ffqcPB7%2BMIE8/JGJFXlYkYe/lpDJjfrl/lLNsmD96U9/0tGjRzV79myFhYXJbrfr8ccf16OPPqoHHnhAVVVVlue73W7FxMRIkhwOx2XXe/ToodjYWElSVVWVwsPDJX11Y3tVVZViY2Pl8Xguu68k3/EbIy4uzu9yYHn5GdXXB%2BcPpyk36vV7PA03fdZfRx7%2ByMSKPKzIwx%2BZXL1m%2BRaI1%2Bv1%2B2iECxcuSJLuu%2B8%2By8cuSFJBQYFcLpckyel0qqioyLfm8XhUUlIil8ulLl26KDo62rJ/aWmp6urqlJiYKKfTqePHj/tK1cVjd%2B/eXXZ7YBowAAAIPs2yYPXp00d2u13PP/%2B8amtr9eWXX%2Bqll15S37599cgjj%2BjYsWNas2aNamtrtXnzZn388cdKTU2VJI0dO1Z5eXnas2ePampq9PTTT6tt27ZKSUlRWFiYxowZo5UrV%2BrIkSOqqKhQVlaWhg0bpg4dOqhXr15yuVxaunSpTp8%2BrdLSUuXk5GjChAkBTgQAAASTZnmJ0OFw6KWXXtLy5cs1ePBgtWrVSv369dPKlSsVGxurF198UUuWLFF2drY6deqk7Oxs343vSUlJmjdvntLT01VRUaHExETl5OSoTZs2kqTHH39cNTU1GjlypDwej5KTk7V48WLfuZ999lllZmbq/vvvl91u1/jx4zV%2B/PhAxAAAAIJUiNfr9QZ6iJtBefkZ48e02UI1dMVO48e9XjbNHnRdj2%2BzhcrhsMvtruFeAZHH5ZCJFXlYkYe/lpDJLbe0D8h5m%2BUlQgAAgGBGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGNalgVVdXm54DAACgxWhSwRo8eLDS09O1f/9%2B0/MAAAAEvSYVrMWLF6u8vFwTJ07U8OHD9corr6iystL0bAAAAEGpSQXr0Ucf1csvv6yPP/5Y48aN0/vvv68HH3xQs2fP1q5du0zPCAAAEFSu6Sb3mJgY/eu//qvWrVunrKws7dq1Sz/5yU80bNgwbd682dSMAAAAQcV2LTtXVFTozTff1JtvvqnDhw9r8ODBGjNmjMrLy7V48WIdPnxYU6dONTUrAABAUGhSwdq5c6fWr1%2BvrVu3yuFw6Ic//KHGjBmjTp06%2BZ6TkJCgKVOmULAAAMBNp0kFa%2BrUqRo4cKCeeeYZpaSkKCwszO85LpdLcXFx1zwgAABAsGlSwdqyZYu6dOmiuro6X7mqqamR3W63PO%2Bdd9659gkBAACCTJNucg8JCdGIESO0detW37bc3Fx9//vf15EjR4wNBwAAEIyaVLB%2B%2Bctf6o477tDdd9/t2/bwww/L6XTql7/8pbHhAAAAglGTLhF%2B8skn2r59u8LDw33bOnTooEWLFik5OdnYcAAAAMGoSe9geb1e1dfX%2B20/d%2B6cGhoarnkoAACAYNakgjVo0CDNmzdPJSUlOn36tKqqqvTJJ59ozpw5Gjx4sOkZAQAAgkqTLhEuWrRIc%2BfO1ciRIxUSEuLb3r9/f2VkZBgbDgAAIBg1qWDFxsbqlVde0cGDB3Xo0CF5vV5169ZN8fHxpucDAAAIOtf0VTnx8fHq0qWL73FdXZ0kqXXr1tc2FQAAQBBrUsH69NNPtXjxYv3tb3%2BTx%2BPxWz9w4MA1DwYAABCsmlSwFi9erPbt22vhwoVq27at6ZkAAACCWpMK1qFDh/TnP/9Zbdq0MT0PAABA0GvSxzR06tRJFy5cMD0LAABAi9CkgjV37lxlZWWpurra9DwAAABBr0mXCFetWqWjR4/qrbfeksPhsHwWliT953/%2Bp5HhAAAAglGTCtb999%2BvVq1amZ4FAACgRWhSwZozZ47pOQAAAFqMJt2DJUn//d//rfT0dD322GOSpIaGBm3atMnYYAAAAMGqSQXro48%2B0vjx4%2BV2u7V//35J0okTJ7Ro0SKtX7/e6IAAAADBpkkF67e//a2WL1%2Bu3/72t74b3Dt16qRnn31Wa9asMTkfAABA0GlSwfriiy/03e9%2BV5Isf0E4YMAAHTt2zMxkAAAAQapJBatVq1aqqqry237o0CG%2BOgcAANz0mlSwHnzwQWVkZOjgwYOSJLfbrZ07d2r27NlKTk42OiAAAECwaVLBSk9Pl9fr1fe//32dP39eAwcO1JQpU9SxY0f9/Oc/Nz0jAABAUGnS52BFRkbqxRdf1MGDB3Xo0CGFhISoW7du6tatm%2Bn5AAAAgk6TCtZF8fHxio%2BPNzULAABAi9CkS4SDBw/%2Bxn8DBgwwNtzq1as1ePBg9e3bV5MmTdKRI0ckSbt379bDDz8sp9OpoUOH6u2337bst3btWiUnJ8vlcmn06NEqLi72rZ0/f16ZmZnq16%2Bf%2Bvbtq7S0NFVWVvrWjx49qsmTJ6tPnz4aMGCAli9froaGBmOvCQAAtHxNKlipqakaO3as79%2BYMWM0cOBAeb1eTZs2zchgf/zjH7V161bl5uZq%2B/bt6tixo9asWaOTJ09q%2BvTpGjVqlPbu3av09HRlZGSooKBAkvTBBx9o5cqVysrK0p49e/TAAw9o2rRpOnv2rCRp%2BfLl2r9/v/Ly8vTRRx%2BptrZWCxYskCR5vV7NnDlTDodDO3bs0GuvvaZNmzZp7dq1Rl4TAAC4OTTpEuHjjz9%2B2e0FBQX64x//eE0DXfS73/1OTz/9tDp37ixJysrKkiS9/PLL6tq1qyZOnChJSklJ0ZAhQ7Rhwwa5XC6tX79eo0aN0n333SdJmjFjhtatW6etW7dq2LBheuutt/TrX/9aXbp0kSTNnz9fw4cP18mTJ3Xy5EmVlpZqzZo1ioqKUlRUlKZMmaI1a9boRz/6kZHXBQAAWr5rugfrUi6XS%2Bnp6dd8nJMnT%2BrEiRP6%2B9//rieffFJffvmlBgwYoKeeekolJSXq3bu35fkJCQm%2B70EsKSnR8OHDfWshISHq1auXioqKlJCQoOrqasv%2B8fHxateunYqLi1VWVqbOnTsrOjrat967d28dOnRI1dXVioiIaNT8ZWVlKi8vt2yz2cIVFxd31Vn8I2FhTf4qyYCw2a7vvBfzCLZcrhfy8EcmVuRhRR7%2ByKTpjBasv//97/ryyy%2Bv%2BTgnTpxQSEiIPvzwQ%2BXm5qq2tlZpaWlatGiRampq1LNnT8vzo6OjffdRud1uS0GSpKioKFVWVsrtdvsef11kZKRv/dK1i4/dbnejC1Zubq5WrVpl2TZjxgylpaU1av%2BWyuGw35DzREa2uyHnCRbk4Y9MrMjDijz8kcnVa1LBGjt2rN%2B2uro6ff755xoyZMg1D3XhwgVduHBBTz75pBwOhyQpLS1NU6ZM%2Bcab6C9%2BZc/Xv7rncuvf5ErrVyM1NVUpKSmWbTZbuNzuGmPnkILvNwrTr/9SYWGhioxsp9Onz8nj4Q8TyMMfmViRhxV5%2BGsJmdyoX%2B4v1aSCdfvtt/sVkjZt2uiHP/yhfvjDH17zUBffgfr6O0adO3eW1%2BtVfX2939f0uN1uxcTESJIcDsdl13v06KHY2FhJUlVVlcLDwyV9dWN7VVWVYmNj5fF4LruvJN/xGyMuLs7vcmB5%2BRnV1wfnD6cpN%2Br1ezwNN33WX0ce/sjEijysyMMfmVy9JhWsX/3qV6bnsOjatasiIiJUXFyswYMHS5KOHTsmm82mBx98UPn5%2BZbnFxQUyOVySZKcTqeKior06KOPSpI8Ho9KSko0atQodenSRdHR0SouLlanTp0kSaWlpaqrq1NiYqLKy8t1/Phxud1u3ztnBQUF6t69u%2Bz2wDRgAAAQfJpUsNavX69WrVo16rkXi87VaNWqlUaPHq0VK1aoe/fuCgsL0wsvvKBHHnlEjz76qFavXq01a9Zo7Nix2r59uz7%2B%2BGO98cYbkr66fDlr1ix95zvfkdPp1OrVq9W2bVulpKQoLCxMY8aM0cqVK9WzZ0%2BFh4crKytLw4YNU4cOHdShQwe5XC4tXbpUTz31lP73f/9XOTk5%2BtnPfnbVrwEAANy8Qrxer/dqd%2BrTp4/Onz%2BvS3cNCQmxbAsJCdGBAweaNFhdXZ1%2B9atfaePGjQoNDVVycrIWLlyoiIgI7du3T0uWLNHnn3%2BuTp06ae7cuRo6dKhv39dff105OTmqqKhQYmKi/v3f/13//M//bDnuO%2B%2B8I4/Ho%2BTkZC1evFjt27eX9NUN9pmZmfrzn/8su92u8ePHa%2BbMmU16DV9XXn7mmo9xKZstVENX7DR%2B3Otl0%2BxB1/X4NluoHA673O4a3soWeVwOmViRhxV5%2BGsJmdxyS/uAnLdJBWv37t167bXXNH36dMXHx8vj8eizzz5TTk6OJk6caPTT3FsKChYF60YjD39kYkUeVuThryVkEqiC1aRLhMuWLdPvfvc7y43cffv21VNPPaUf//jHeu%2B994wNCAAAEGya9Hf%2BR48eVWRkpN/2qKgoHT9%2B/JqHAgAACGZNKljdunVTVlaW7yMMJOnLL79Udna2unXrZmw4AACAYNSkS4QZGRmaPn263njjDdntdoWEhKi6ulp2u10vvPCC6RkBAACCSpMK1t13363t27drx44dOnHihLxer2699VYlJSU1%2ButkAAAAWqomfxdhu3btNHToUB0/flxdunQxORMAAEBQa9I9WLW1tXrqqad011136aGHHpIknT59WlOnTtWZM%2BY/jgAAACCYNKlgPffcc/r000%2B1YsUKhYb%2B/0NcuHBBv/71r40NBwAAEIyaVLA%2B/PBDrVy5UsOGDfN96XNkZKSysrK0bds2owMCAAAEmyYVrLKyMt1%2B%2B%2B1%2B22NjY1VdXX2tMwEAAAS1JhWs2267Tfv37/fb/v7776tjx47XPBQAAEAwa9JfEU6aNEk/%2B9nPNGrUKHk8Hv3%2B979XUVGRtmzZooULF5qeEQAAIKg0qWCNHTtW0dHReuWVVxQeHq4XX3xR3bp104oVKzRs2DDTMwIAAASVJhWsiooKDRs2jDIFAABwGVd9D1ZDQ4OSk5Pl9XqvxzwAAABB76oLVmhoqAYOHKhNmzZdj3kAAACCXpMuEXbq1EnLli1TTk6OvvWtb6lVq1aW9ezsbCPDAQAABKMmFazPPvtM3bp1kyS53W6jAwEAAAS7qypYc%2BbM0TPPPKNXX33Vt%2B2FF17QjBkzjA8GAAAQrK7qHqytW7f6bcvJyTE2DAAAQEtwVQXrcn85yF8TAgAAWF1Vwbr4xc5X2gYAAHAza9J3EQIAAOCbUbAAAAAMu6q/Irxw4YKlQWwjAAAVE0lEQVSeeOKJK27jc7AAAMDN7KoK1re//W2VlZVdcRsAAMDN7KoK1tc//woAAACXxz1YAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMCwoChYy5Yt05133ul7vHv3bj388MNyOp0aOnSo3n77bcvz165dq%2BTkZLlcLo0ePVrFxcW%2BtfPnzyszM1P9%2BvVT3759lZaWpsrKSt/60aNHNXnyZPXp00cDBgzQ8uXL1dDQcP1fJAAAaDGafcE6cOCA8vPzfY9Pnjyp6dOna9SoUdq7d6/S09OVkZGhgoICSdIHH3yglStXKisrS3v27NEDDzygadOm6ezZs5Kk5cuXa//%2B/crLy9NHH32k2tpaLViwQJLk9Xo1c%2BZMORwO7dixQ6%2B99po2bdqktWvX3vgXDgAAglazLlgNDQ166qmnNGnSJN%2B2d955R127dtXEiRPVrl07paSkaMiQIdqwYYMkaf369Ro1apTuu%2B8%2BhYeHa8aMGZKkrVu3qr6%2BXm%2B99ZZmz56tLl26KCYmRvPnz9e2bdt08uRJFRYWqrS0VBkZGYqKilJ8fLymTJmidevWBeLlAwCAIGUL9AD/yLp169S2bVuNGDFCK1eulCSVlJSod%2B/eluclJCRo06ZNvvXhw4f71kJCQtSrVy8VFRUpISFB1dXVlv3j4%2BPVrl07FRcXq6ysTJ07d1Z0dLRvvXfv3jp06JCqq6sVERHRqLnLyspUXl5u2WazhSsuLu7qAriCsLBm3Y/92GzXd96LeQRbLtcLefgjEyvysCIPf2TSdM22YJ06dUovvPCCXn31Vct2t9utnj17WrZFR0f77qNyu92WgiRJUVFRqqyslNvt9j3%2BusjISN/6pWsXH7vd7kYXrNzcXK1atcqybcaMGUpLS2vU/i2Vw2G/IeeJjGx3Q84TLMjDH5lYkYcVefgjk6vXbAtWVlaWxowZozvuuENHjx694vNDQkIs//2m9Svtb0JqaqpSUlIs22y2cLndNcbOIQXfbxSmX/%2BlwsJCFRnZTqdPn5PHwx8mkIc/MrEiDyvy8NcSMrlRv9xfqlkWrN27d6uoqEjLli3zW4uJiVFVVZVlm9vtVkxMjCTJ4XBcdr1Hjx6KjY2VJFVVVSk8PFzSVze2V1VVKTY2Vh6P57L7XjxvY8XFxfldDiwvP6P6%2BuD84TTlRr1%2Bj6fhps/668jDH5lYkYcVefgjk6vXLN8Cefvtt3XixAklJSWpf//%2BGjlypCSpf//%2BuvPOOy0fuyBJBQUFcrlckiSn06mioiLfmsfjUUlJiVwul7p06aLo6GjL/qWlpaqrq1NiYqKcTqeOHz/uK1UXj929e3fZ7YFpwAAAIPg0y4L185//XO%2B//77y8/OVn5%2BvnJwcSVJ%2Bfr5%2B8IMf6NixY1qzZo1qa2u1efNmffzxx0pNTZUkjR07Vnl5edqzZ49qamr09NNPq23btkpJSVFYWJjGjBmjlStX6siRI6qoqFBWVpaGDRumDh06qFevXnK5XFq6dKlOnz6t0tJS5eTkaMKECYGMAwAABJlmeYkwKirKcrN5fX29JOm2226TJL344otasmSJsrOz1alTJ2VnZ/tufE9KStK8efOUnp6uiooKJSYmKicnR23atJEkPf7446qpqdHIkSPl8XiUnJysxYsX%2B8717LPPKjMzU/fff7/sdrvGjx%2Bv8ePH36BXDgAAWoIQr9frDfQQN4Py8jPGj2mzhWroip3Gj3u9bJo96Loe32YLlcNhl9tdw70CIo/LIRMr8rAiD38tIZNbbmkfkPM2y0uEAAAAwYyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwrNkWrKNHj2r69Onq16%2BfBgwYoHnz5unLL7%2BUJB04cEBjx46Vy%2BVSUlKSXnnlFcu%2B7777rr73ve/J6XTqBz/4gXbt2uVb83q9euaZZzRo0CDdddddmjRpko4cOeJbd7vdmjNnju6%2B%2B27de%2B%2B9WrhwoWpra2/MiwYAAC1Csy1Y06dPV3R0tLZt26b8/HwdPHhQv/nNb3Tu3DlNmTJFd999t3bv3q3nnntOq1ev1pYtWyRJRUVFmj9/vmbNmqW//OUveuyxxzRjxgydOHFCkrR27Vrl5eXp5Zdf1q5du9SlSxfNnDlTXq9XkrRgwQJVVFRoy5Yt2rhxow4cOKAVK1YELAcAABB8mmXBOnPmjBITEzV37lzZ7XbFxcVp5MiR2rdvn7Zv364LFy7oiSeekN1uV58%2BfZSamqrc3FxJUl5enpKSkjR8%2BHC1bdtWo0ePVo8ePZSfny9JWr9%2BvX7yk5%2BoV69eioiI0Pz583Xw4EF9%2Bumnqqio0LZt25Senq4OHTro1ltv1ezZs5WXl6e6urpARgIAAIKILdADXE779u2VlZVl2Xb8%2BHF17NhRJSUl6tmzp8LCwnxrCQkJWr9%2BvSSppKRESUlJln0TEhJUVFSk8%2BfP6%2BDBg0pMTPStRURE6Fvf%2BpaKiopUXV0tm82mO%2B%2B807feu3dvnT17Vl988YVl%2Bz9SVlam8vJyyzabLVxxcXGNC6CRwsKaZT/%2BRjbb9Z33Yh7Blsv1Qh7%2ByMSKPKzIwx%2BZNF2zLFiXKiws1KuvvqoXX3xR7777rqKioizr0dHRqqqqUkNDg9xut6Kjoy3rUVFR%2Buyzz1RVVSWv1%2Bu3f1RUlCorKxUVFaWIiAiFhoZa1iSpsrKy0fPm5uZq1apVlm0zZsxQWlpao4/REjkc9htynsjIdjfkPMGCPPyRiRV5WJGHPzK5es2%2BYH3yySeaPn26nnjiCQ0YMEDvvvtuk44TEhJyXde/LjU1VSkpKZZtNlu43O6aRh%2BjMYLtNwrTr/9SYWGhioxsp9Onz8njabiu5woG5OGPTKzIw4o8/LWETG7UL/eXatYFa9u2bZo7d64yMzP1yCOPSJJiY2N1%2BPBhy/PcbrccDodCQ0MVExMjt9vttx4TE%2BN7TlVVld96bGysYmNjdebMGXk8Ht8lyIvHio2NbfTccXFxfpcDy8vPqL4%2BOH84TblRr9/jabjps/468vBHJlbkYUUe/sjk6jXbt0D279%2Bv%2BfPn67nnnvOVK0lyOp0qLS1VfX29b1tBQYFcLpdvvbi42HKswsJCuVwutW7dWj169LCsV1VV6fDhw3I6nUpISFBDQ4NKS0stx27fvr1uv/326/RKAQBAS9MsC1Z9fb0yMjI0a9YsDRo0yLKWlJQku92u7Oxs1dTUaO/evXrjjTc0YcIESdLo0aO1a9cuvffee6qtrdWrr76qw4cP69FHH5UkjRs3Ti%2B//LL%2B%2Bte/6syZM1q6dKkSExPlcrnkcDj00EMPKSsrS6dOndKxY8f0zDPPKDU1Va1atbrhOQAAgOAU4r34AVDNyL59%2BzRhwgS1bt3ab23z5s06e/asMjMzVVxcrNjYWE2dOlXjxo3zPWfLli3Kzs7W8ePHFR8fr4yMDN1zzz2%2B9eeff16vv/66ampq1L9/f/3iF7/QbbfdJumrj4hYvHixtm7dqlatWmnEiBGaP3/%2BZWe5GuXlZ65p/8ux2UI1dMVO48e9XjbNHnTlJ10Dmy1UDoddbncNb2WLPC6HTKzIw4o8/LWETG65pX1AztssC1ZLRMGiYN1o5OGPTKzIw4o8/LWETAJVsJrlJUIAAIBgRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDBboAfAzeOhlbsCPcJV2TR7UKBHAAAEKd7BAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEUrMs4evSoJk%2BerD59%2BmjAgAFavny5GhoaAj0WAAAIEvwV4SW8Xq9mzpyp7t27a8eOHTp16pSmTJmiDh066Ec/%2BlGgxwMAAEGAd7AuUVhYqNLSUmVkZCgqKkrx8fGaMmWK1q1bF%2BjRAABAkOAdrEuUlJSoc%2BfOio6O9m3r3bu3Dh06pOrqakVERFzxGGVlZSovL7dss9nCFRcXZ3TWsDD68fUUbJ/b9cHc%2By2PL/588HPy/5GJFXlYkYc/Mmk6CtYl3G63oqKiLNsuPna73Y0qWLm5uVq1apVl28yZM/X444%2BbG1RfFbnHbvtMqampxstbMCorK1Nubi55/J%2BysjKtXfsyeXwNmViRhxV5%2BCOTpqOSXgepqal68803Lf9SU1ONn6e8vFyrVq3ye7fsZkUeVuThj0ysyMOKPPyRSdPxDtYlYmNjVVVVZdnmdrslSTExMY06RlxcHE0fAICbGO9gXcLpdOr48eO%2BUiVJBQUF6t69u%2Bx2ewAnAwAAwYKCdYlevXrJ5XJp6dKlOn36tEpLS5WTk6MJEyYEejQAABAkwhYvXrw40EM0N/fff7/ef/99LVmyRO%2B9957GjRunyZMnB3qsy7Lb7erXrx/vrv0f8rAiD39kYkUeVuThj0yaJsTr9XoDPQQAAEBLwiVCAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWEHo6NGjmjx5svr06aMBAwZo%2BfLlamhoCPRYAbVz504NHDhQc%2BbMCfQozcLRo0c1ffp09evXTwMGDNC8efP05ZdfBnqsgPnrX/%2BqSZMm6Z577tF9992nWbNmqaysLNBjNRvLli3TnXfeGegxAio5OVmJiYlyOp2%2Bf0uWLAn0WAG3evVqDR48WH379tWkSZN05MiRQI8UNChYQcbr9WrmzJlyOBzasWOHXnvtNW3atElr164N9GgB89JLL2np0qXq2rVroEdpNqZPn67o6Ght27ZN%2Bfn5OnjwoH7zm98EeqyAqKur049//GPde%2B%2B9%2BtOf/qT33ntPlZWV4mtYv3LgwAHl5%2BcHeoyAO336tP7whz%2BosLDQ92/RokWBHiug/vjHP2rr1q3Kzc3V9u3b1bFjR61ZsybQYwUNClaQKSwsVGlpqTIyMhQVFaX4%2BHhNmTJF69atC/RoAdOmTRtt2LCBgvV/zpw5o8TERM2dO1d2u11xcXEaOXKk9u3bF%2BjRAuLcuXOaM2eOpk2bptatWysmJkbf%2B9739Le//S3QowVcQ0ODnnrqKU2aNCnQowSUx%2BNRTU2NIiMjAz1Ks/K73/1OixYtUufOnRUVFaWsrKybvnReDQpWkCkpKVHnzp0VHR3t29a7d28dOnRI1dXVAZwscCZOnKj27dsHeoxmo3379srKylJsbKxv2/Hjx9WxY8cAThU4UVFRGj16tGw2m7xerz7//HO9%2BeabeuihhwI9WsCtW7dObdu21YgRIwI9SkCdPn1aXq9Xzz//vAYPHqzBgwcrMzNTNTU1gR4tYE6ePKkTJ07o73//u7773e%2Bqf//%2Bmj17ttxud6BHCxoUrCDjdrsVFRVl2XbxMT/4uJzCwkK9%2BuqrmjZtWqBHCahjx44pMTFRw4cPl9Pp1KxZswI9UkCdOnVKL7zwApdK9dVl5Lvuukv33nuvNm/erD/84Q/6r//6r5s6mxMnTigkJEQffvihcnNz9R//8R86duwY72BdBQoW0IJ98sknmjx5sp544gkNGDAg0OMEVOfOnVVUVKTNmzfr888/15NPPhnokQIqKytLY8aM0R133BHoUQLu1ltv1RtvvKF/%2BZd/UUREhO644w7NnTtXGzduVF1dXaDHC4gLFy7owoULevLJJ%2BVwONSxY0elpaXpww8/1Pnz5wM9XlCgYAWZ2NhYVVVVWbZdfOcqJiYmECOhmdq2bZumTp2qhQsX6rHHHgv0OM1CSEiIbr/9ds2bN08bN25UZWVloEcKiN27d6uoqEg//elPAz1Ks/VP//RPamhoUEVFRaBHCYiLt6FERET4tnXu3Fler/emzeRqUbCCjNPp1PHjxy2XAwsKCtS9e3fZ7fYATobmZP/%2B/Zo/f76ee%2B45PfLII4EeJ6D27t2r73znO6qvr/dtu/ixJmFhYYEaK6DefvttnThxQklJSerfv79GjhwpSerfv7/efffdAE934x04cEDLli2zbPviiy/UunVrxcXFBWiqwOratasiIiJUXFzs23bs2DHZbLabNpOrRcEKMr169ZLL5dLSpUt1%2BvRplZaWKicnRxMmTAj0aGgm6uvrlZGRoVmzZmnQoEGBHifgEhISdO7cOWVnZ%2BvcuXOqrKzU888/r3vuucfvfsabxc9//nO9//77ys/PV35%2BvnJyciRJ%2Bfn5SklJCfB0N15sbKzWr1%2BvnJwc1dXV6dChQ1q5cqXGjRt305bwVq1aafTo0VqxYoVOnDih8vJyvfDCC3rkkUdks9kCPV5QCPF6vd5AD4Grc%2BLECWVmZurPf/6z7Ha7xo8fr5kzZwZ6rIBxOp2S5HuH4uL//IWFhQGbKZD27dunCRMmqHXr1n5rmzdvVufOnQMwVWAdOHBAv/71r1VUVCSbzab%2B/ftrwYIFuvXWWwM9WrNw9OhRDRkyRKWlpYEeJWD%2B8pe/aMWKFfqf//kfORwODR8%2BXGlpaZf9/%2BhmUVdXp1/96lfauHGjQkNDlZycrIULF1ouG%2BKbUbAAAAAM4xIhAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABj2/wCMnZmZ3xudmgAAAABJRU5ErkJggg%3D%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-6590192124298511576">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.041</td>
            <td class="number">21</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.045</td>
            <td class="number">20</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.044</td>
            <td class="number">19</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.075</td>
            <td class="number">19</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.05</td>
            <td class="number">18</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.054</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.042</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.034</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.06</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (103037)</td>
            <td class="number">120905</td>
            <td class="number">70.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="missing">
            <td class="fillremaining">(Missing)</td>
            <td class="number">51649</td>
            <td class="number">29.9%</td>
            <td>
                <div class="bar" style="width:43%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-6590192124298511576">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2.77777777778e-05</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4.44444444444e-05</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4.7619047619e-05</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.000120481927711</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">3.19985833333</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.36695</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.5543</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.68520833333</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">6.38722666667</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">avg_cur_bal_to_loan_amnt<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>95901</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>79.2%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (%)</th>
                        <td>29.9%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (n)</th>
                        <td>51649</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>1.374</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>172.55</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram393930976851884880">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAAPtJREFUeJzt1bEJAlEQRVFXLGmLsCdje7IIexpzkQsbyN/gnHzgJZfZZmYuwE/X1QPgzG6rB3zbH6/DN%2B/n/Q9LwAeBJBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAjbzMzqEXBWPggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiED9obC49m6JdFAAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives393930976851884880,#minihistogram393930976851884880"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives393930976851884880">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles393930976851884880"
                                                      aria-controls="quantiles393930976851884880" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram393930976851884880" aria-controls="histogram393930976851884880"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common393930976851884880" aria-controls="common393930976851884880"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme393930976851884880" aria-controls="extreme393930976851884880"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles393930976851884880">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>0.10397</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>0.2542</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>0.62682</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>1.5551</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>4.7668</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>172.55</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>172.55</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>1.3009</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>2.574</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>1.8734</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>337.31</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>1.374</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>1.2728</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>11.39</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>166380</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>6.6256</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram393930976851884880">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xl01PW9//FXFpYsJJkEo4CIGA5LkhmWKhiBKOFiKQXkIBCWW8QiIAURlILBiLRQ0hYiiKAlLpCrrUSIXNxAVFCQQhWpzYa5lisNy4UEMkNIJIQk8/uDw/w6BCSGT0gm83yc09OTz2fmO%2B/XTGhf%2BX4nEx%2Bn0%2BkUAAAAjPFt6AEAAACaGgoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADDMv6EH8BZFRWeNH9PX10fh4UEqLi5TdbXT%2BPEbM7KT3duyS96dn%2Bxkr2v2m25qZXiq2uEMlgfz9fWRj4%2BPfH19GnqUG47sZPdG3pyf7GT3NBQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIY1aMHavXu37rnnHs2ZM6fG3rZt2zRs2DD17NlT999/vzIyMtz209PTNWDAANlsNo0ePVq5ubmuvfPnz2vhwoXq3bu3evbsqVmzZqm4uNi1f/ToUU2ePFk9evRQXFycli1bpurqatf%2B3r17NXz4cFmtVg0aNEjvvPNOPaQHAABNVYMVrJdffllLlixRhw4dauxlZWVp3rx5mjNnjvbv369nnnlGixcv1v79%2ByVJH330kVauXKmUlBTt27dP9957r6ZNm6bvv/9ekrRs2TIdOHBAmZmZ%2BuSTT1ReXq4FCxZIkpxOp2bOnCmLxaLPPvtMb7zxhrZu3ar09HRJ0smTJzV9%2BnSNGjVKX3zxhZKSkpScnKysrKwb9MwAAABP12AFq0WLFtq0adMVC5bD4dCjjz6qhIQE%2Bfn5qX///urSpYurYG3cuFGjRo3S3XffrcDAQM2YMUOStGPHDlVWVmrz5s2aPXu22rdvr/DwcM2fP187d%2B7UyZMnlZ2drfz8fCUnJys0NFRRUVGaMmWKNmzYIEl699131aFDB02cOFEBAQFKSEjQwIEDtWnTphv35AAAAI/m31APPHHixKvuxcfHKz4%2B3vV1ZWWlCgsL1aZNG0lSXl6ehgwZ4tr38fFRt27dlJOTo%2BjoaJWWliomJsa1HxUVpYCAAOXm5qqwsFDt2rVTWFiYaz8mJkaHDx9WaWmp8vLy3O4rSdHR0dq6det1Z64Pdz69raFHqLWts/s29AgAANwQDVawfozly5crKChIgwcPliTZ7Xa3giRJoaGhKi4ult1ud33970JCQlz7l%2B9d%2Btput8tut6tr165u%2B2FhYW7v4bqWwsJCFRUVua35%2BwcqMjKy1seoDT8/z/odBX9/c/Neyu5pz4EJZPfO7JJ35yc72T1Noy5YTqdTy5cv13vvvaf09HS1aNFC0sUzVldytfXa7pu6b0ZGhlavXu22NmPGDM2aNavOj98UWCxBxo8ZEhJg/Jieguzey5vzk907eWL2RluwqqurlZSUpOzsbGVkZKhdu3auPYvFIofD4XZ7u92uzp07KyIiQtLF93EFBgZKuljUHA6HIiIiVFVVdcX7SlJ4eLjCw8OvuB8eHl7r2RMTE5WQkOC25u8fKLu9rNbHqA1Pa/Qm8/v5%2BSokJEAlJedUVVV97Ts0IWT3zuySd%2BcnO9nrmr0%2BfrivjUZbsJYuXapDhw7pzTffrHFJz2q1KicnRyNGjJAkVVVVKS8vT6NGjVL79u0VFham3NxctW3bVpKUn5%2BviooKxcbGqqioSMePH5fdbpfFYpF08bcWO3XqpKCgIFmtVr399ttuj5eVlSWbzVbr2SMjI2tcDiwqOqvKSu/6h3G5%2BshfVVXttc8r2b0zu%2BTd%2BclOdk/RKE%2BBfPXVV9qyZYtefPHFGuVKksaOHavMzEzt27dPZWVleu6559SyZUvXbx2OGTNGK1eu1JEjR3T69GmlpKRo8ODBat26tbp16yabzaYlS5aopKRE%2Bfn5SktL04QJEyRJw4YN07Fjx7R%2B/XqVl5dr27Zt2rVrlxITE2/00wAAADxUg53Bslqtki7%2BhqAkffzxx5Kk7OxsZWZmqrS0VAMHDnS7z1133aXXXntN8fHxmjdvnpKSknT69GnFxsYqLS3N9R6txx57TGVlZRo5cqSqqqo0YMAALVq0yHWc559/XgsXLlT//v0VFBSk8ePHa/z48ZKkiIgIrV27VosXL1Zqaqratm2r1NTUGm98BwAAuBofp9PpbOghvEFR0Vnjx/T399Wg5buNH7e%2BmPyYBn9/X1ksQbLbyzzutPH1Irt3Zpe8Oz/ZyV7X7Dfd1MrwVLXTKC8RAgAAeDIKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgWIMWrN27d%2Buee%2B7RnDlzauy9//77%2BulPfyqr1aqhQ4dqz549rj2n06kVK1aob9%2B%2B6t69uyZNmqQjR4649u12u%2BbMmaNevXrprrvu0tNPP63y8nLX/sGDBzV27FjZbDbFx8dr3bp1tX5sAACAa2mwgvXyyy9ryZIl6tChQ429nJwczZ8/X48//ri%2B/PJLPfTQQ5oxY4ZOnDghSUpPT1dmZqZeeeUV7dmzR%2B3bt9fMmTPldDolSQsWLNDp06e1fft2vffeezp48KCWL18uSTp37pymTJmiXr16ae/evVq1apVefPFFbd%2B%2BvVaPDQAAcC0NVrBatGihTZs2XbFgZWZmKj4%2BXkOGDFHLli01evRode7cWVu2bJEkbdy4UY888oi6deum4OBgzZ8/X4cOHdLXX3%2Bt06dPa%2BfOnUpKSlLr1q118803a/bs2crMzFRFRYU%2B/fRTXbhwQU8%2B%2BaSCgoLUo0cPJSYmKiMjo1aPDQAAcC3%2BDfXAEydOvOpeXl6e4uPj3daio6OVk5Oj8%2BfP69ChQ4qNjXXtBQcH67bbblNOTo5KS0vl7%2B%2BvLl26uPZjYmL0/fff67vvvlNeXp66du0qPz8/t2Nv3Ljxmo9dW4WFhSoqKnJb8/cPVGRkZK2PURt%2Bfp71Fjp/f3PzXsruac%2BBCWT3zuySd%2BcnO9k9TYMVrB9it9sVFhbmthYaGqpvv/1WDodDTqdToaGhNfaLi4sVGhqq4OBg%2Bfr6uu1JUnFxsex2e437hoWFyeFwqLq6%2Bgcfu7YyMjK0evVqt7UZM2Zo1qxZtT5GU2SxBBk/ZkhIgPFjegqyey9vzk927%2BSJ2RtlwboaHx%2Bfet03dd/ExEQlJCS4rfn7B8puL6vz41%2BJpzV6k/n9/HwVEhKgkpJzqqqqNnZcT0B278wueXd%2BspO9rtnr44f72miUBSs8PFx2u91tzW63Kzw8XBaLRb6%2BvnI4HDX2IyIiFBERobNnz6qqqsp1GfDSsS7tFxQU1LjvpeP%2B0GPXVmRkZI3LgUVFZ1VZ6V3/MC5XH/mrqqq99nklu3dml7w7P9nJ7ika5SkQq9Wq3Nxct7Xs7GzZbDY1b95cnTt3dtt3OBwqKCiQ1WpVdHS0qqurlZ%2Bf79rPyspSq1atdPvtt8tqtSo/P1%2BVlZVu%2Bzab7ZqPDQAAUBuNsmCNHj1ae/bs0QcffKDy8nK9/vrrKigo0IgRIyRJ48aN0yuvvKJvvvlGZ8%2Be1ZIlSxQbGyubzSaLxaKf/exnSklJ0alTp3Ts2DGtWLFCiYmJatasmeLj4xUUFKTU1FSVlZXpiy%2B%2B0FtvvaUJEybU6rEBAACupcEuEVqtVklynUn6%2BOOPJV08W9S5c2ctX75cqampmj9/vqKiorR27Vq1bt1akjR27FgVFRXpl7/8pcrKytSnTx%2BtWrXKdezf/OY3WrRokQYNGqRmzZpp2LBhevzxxyVJzZs319q1a7Vw4ULFxcUpIiJC8%2BbN07333itJ13xsAACAa/FxXvp0TtSroqKzxo/p7%2B%2BrQct3Gz9ufdk6u6%2BxY/n7%2B8piCZLdXuZx1%2BWvF9m9M7vk3fnJTva6Zr/pplaGp6qdRnmJEAAAwJNRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYY22YOXm5mrixIn6yU9%2BonvuuUfz5s2T3W6XJO3du1fDhw%2BX1WrVoEGD9M4777jdNz09XQMGDJDNZtPo0aOVm5vr2jt//rwWLlyo3r17q2fPnpo1a5aKi4td%2B0ePHtXkyZPVo0cPxcXFadmyZaqurr4xoQEAQJPQKAtWVVWVpk6dqp49e2rv3r364IMPdOrUKS1atEgnT57U9OnTNWrUKH3xxRdKSkpScnKysrKyJEkfffSRVq5cqZSUFO3bt0/33nuvpk2bpu%2B//16StGzZMh04cECZmZn65JNPVF5ergULFkiSnE6nZs6cKYvFos8%2B%2B0xvvPGGtm7dqvT09AZ7LgAAgOdplAWrqKhIp06d0rBhw9S8eXOFhYVp4MCBysvL07vvvqsOHTpo4sSJCggIUEJCggYOHKhNmzZJkjZu3KhRo0bp7rvvVmBgoGbMmCFJ2rFjhyorK7V582bNnj1b7du3V3h4uObPn6%2BdO3fq5MmTys7OVn5%2BvpKTkxUaGqqoqChNmTJFGzZsaMinAwAAeJhGWbBuvvlmRUdH66233tK5c%2BdUXFysjz76SPfdd5/y8vIUExPjdvvo6Gjl5ORIUo19Hx8fdevWTTk5OSooKFBpaanbflRUlAICApSbm6u8vDy1a9dOYWFhrv2YmBgdPnxYpaWl9ZwaAAA0Ff4NPcCV%2BPj4aNWqVZo0aZLr8lyfPn30xBNP6Fe/%2BpW6du3qdvuwsDDX%2B6jsdrtbQZKk0NBQFRcXu97DFRoa6rYfEhLi2r9879LXdrtdwcHBtZq/sLBQRUVFbmv%2B/oGKjIys1f1ry8%2BvUfbjq/L3Nzfvpeye9hyYQHbvzC55d36yk93TNMqCVVFRoWnTpmnIkCF69NFHde7cOS1cuFC//vWvr3ofHx8ft/%2B%2B2v617m9CRkaGVq9e7bY2Y8YMzZo1y9hjeCKLJcj4MUNCAowf01OQ3Xt5c36yeydPzN4oC9Zf//pXHT16VLNnz5afn5%2BCgoL02GOPacSIEbr33nvlcDjcbm%2B32xUeHi5JslgsV9zv3LmzIiIiJEkOh0OBgYGSLr6x3eFwKCIiQlVVVVe8ryTX8WsjMTFRCQkJbmv%2B/oGy28tqfYza8LRGbzK/n5%2BvQkICVFJyTlVV3vVbnmT3zuySd%2BcnO9nrmr0%2BfrivjUZZsJxOZ42PRrhw4YIk6e6779Z///d/u%2B1lZWXJZrNJkqxWq3JycjRixAhJF38jMS8vT6NGjVL79u0VFham3NxctW3bVpKUn5%2BviooKxcbGqqioSMePH5fdbpfFYnEdu1OnTgoKqv0LFBkZWeNyYFHRWVVWetc/jMvVR/6qqmqvfV7J7p3ZJe/OT3aye4pGeQqkR48eCgoK0gsvvKDy8nKdOXNGL7/8snr27KkHHnhAx44d0/r161VeXq5t27Zp165dSkxMlCSNHTtWmZmZ2rdvn8rKyvTcc8%2BpZcuWSkhIkJ%2Bfn8aMGaOVK1fqyJEjOn36tFJSUjR48GC1bt1a3bp1k81m05IlS1RSUqL8/HylpaVpwoQJDfyMAAAAT9Ioz2BZLBa9/PLLWrZsmfr166dmzZqpd%2B/eWrlypSIiIrR27VotXrxYqampatu2rVJTU11vfI%2BPj9e8efOUlJSk06dPKzY2VmlpaWrRooUk6bHHHlNZWZlGjhypqqoqDRgwQIsWLXI99vPPP6%2BFCxeqf//%2BCgoK0vjx4zV%2B/PiGeBoAAICH8nE6nc6GHsIbFBWdNX5Mf39fDVq%2B2/hx68vW2X2NHcvf31cWS5Ds9jKPO218vcjundkl785PdrLXNftNN7UyPFXtNMpLhAAAAJ6MggUAAGBYnQoWn2oOAABwdXUqWP369VNSUpIOHDhgeh4AAACPV6eCtWjRIhUVFWnixIkaMmSI1q1b5/pTNQAAAN6uTgVrxIgReuWVV7Rr1y6NGzdOH374oe677z7Nnj1be/bsMT0jAACAR7muN7mHh4frF7/4hTZs2KCUlBTt2bNHjzzyiAYPHqxt27aZmhEAAMCjXNcHjZ4%2BfVpvv/223n77bRUUFKhfv34aM2aMioqKtGjRIhUUFGjq1KmmZgUAAPAIdSpYu3fv1saNG7Vjxw5ZLBY9%2BOCDGjNmjOvv%2B0lSdHS0pkyZQsECAABep04Fa%2BrUqbrnnnu0YsUK19/4u5zNZqvxB48BAAC8QZ0K1vbt29W%2BfXtVVFS4ylVZWZmCgoLcbvfuu%2B9e/4QAAAAepk5vcvfx8dGwYcO0Y8cO11pGRoZ%2B/vOf68iRI8aGAwAA8ER1Kli/%2B93vdMcdd6hXr16uteHDh8tqtep3v/udseEAAAA8UZ0uEX711Vf69NNPFRgY6Fpr3bq1nnnmGQ0YMMDYcAAAAJ6oTmewnE6nKisra6yfO3dO1dXV1z0UAACAJ6tTwerbt6/mzZunvLw8lZSUyOFw6KuvvtKcOXPUr18/0zMCAAB4lDpdInzmmWc0d%2B5cjRw5Uj4%2BPq71Pn36KDk52dhwAAAAnqhOBSsiIkLr1q3ToUOHdPjwYTmdTnXs2FFRUVGm5wMAAPA41/WncqKiotS%2BfXvX1xUVFZKk5s2bX99UAAAAHqxOBevrr7/WokWL9M9//lNVVVU19g8ePHjdgwEAAHiqOhWsRYsWqVWrVnr66afVsmVL0zMBAAB4tDoVrMOHD%2Btvf/ubWrRoYXoeAAAAj1enj2lo27atLly4YHoWAACAJqFOBWvu3LlKSUlRaWmp6XkAAAA8Xp0uEa5evVpHjx7V5s2bZbFY3D4LS5I%2B//xzI8MBAAB4ojoVrP79%2B6tZs2amZwEAAGgS6lSw5syZY3oOAACAJqNO78GSpH/84x9KSkrSQw89JEmqrq7W1q1bjQ0GAADgqepUsD755BONHz9edrtdBw4ckCSdOHFCzzzzjDZu3Gh0QAAAAE9Tp4L1pz/9ScuWLdOf/vQn1xvc27Ztq%2Beff17r1683OR8AAIDHqVPB%2Bu6773T//fdLkttvEMbFxenYsWNmJgMAAPBQdSpYzZo1k8PhqLF%2B%2BPBh/nQOAADwenUqWPfdd5%2BSk5N16NAhSZLdbtfu3bs1e/ZsDRgwwOiAAAAAnqZOBSspKUlOp1M///nPdf78ed1zzz2aMmWK2rRpo6eeesr0jAAAAB6lTp%2BDFRISorVr1%2BrQoUM6fPiwfHx81LFjR3Xs2NH0fAAAAB6nTgXrkqioKEVFRZmaBQAAoEmoU8Hq16/fVfeqqqq0d%2B/eOg8EAADg6epUsBITE90%2BnqG6ulpHjx7Vnj17NG3aNGPDAQAAeKI6FazHHnvsiutZWVn6y1/%2Bcl0DAQAAeLo6/y3CK7HZbMrOzjZ5SAAAAI9jtGD961//0pkzZ4wd78UXX1S/fv3Us2dPTZo0SUeOHJEk7d27V8OHD5fVatWgQYP0zjvvuN0vPT1dAwYMkM1m0%2BjRo5Wbm%2BvaO3/%2BvBYuXKjevXurZ8%2BemjVrloqLi137R48e1eTJk9WjRw/FxcVp2bJlqq6uNpYJAAA0fXW6RDh27NgaaxUVFfrf//1fDRw48LqHkqS//OUv2rFjhzIyMhQcHKzf//73Wr9%2BvaZOnarp06friSee0OjRo7V3717Nnj1bt99%2Bu2w2mz766COtXLlSL730kmw2m1577TVNmzZN27dvV2BgoJYtW6YDBw4oMzNTQUFBeuqpp7RgwQL96U9/ktPp1MyZM9WpUyd99tlnOnXqlKZMmaLWrVvr4YcfNpILAAA0fXUqWLfffrvbm9wlqUWLFnrwwQf14IMPGhns1Vdf1XPPPad27dpJklJSUiRJr7zyijp06KCJEydKkhISEjRw4EBt2rRJNptNGzdu1KhRo3T33XdLkmbMmKENGzZox44dGjx4sDZv3qw//OEPat%2B%2BvSRp/vz5GjJkiE6ePKmTJ08qPz9f69evV2hoqEJDQzVlyhStX7%2BeggUAAGqtTgXr97//vek53Jw8eVInTpzQv/71L/3617/WmTNnFBcXp2effVZ5eXmKiYlxu310dLS2bt0qScrLy9OQIUNcez4%2BPurWrZtycnIUHR2t0tJSt/tHRUUpICBAubm5KiwsVLt27RQWFubaj4mJ0eHDh1VaWqrg4OBazV9YWKiioiK3NX//QEVGRv7o5%2BKH%2BPkZvcJb7/z9zc17KbunPQcmkN07s0venZ/sZPc0dSpYGzduVLNmzWp12xEjRvzo4584cUI%2BPj76%2BOOPlZGRofLycs2aNUvPPPOMysrK1LVrV7fbh4WFud5HZbfb3QqSJIWGhqq4uFh2u9319b8LCQlx7V%2B%2Bd%2Blru91e64KVkZGh1atXu63NmDFDs2bNqtX9myqLJcj4MUNCAowf01OQ3Xt5c36yeydPzF6ngvW73/1O58%2Bfl9PpdFv38fFxW/Px8alTwbpw4YIuXLigX//617JYLJKkWbNmacqUKYqLi7vifS5dsrz80uXl%2B1dzrf0fIzExUQkJCW5r/v6BstvLjD2G5HmN3mR%2BPz9fhYQEqKTknKqqvOuXEMjundkl785PdrLXNXt9/HBfG3UqWC%2B99JLeeOMNTZ8%2BXVFRUaqqqtK3336rtLQ0TZw48aolqLYunYH69zNG7dq1k9PpVGVlpRwOh9vt7Xa7wsPDJUkWi%2BWK%2B507d1ZERIQkyeFwKDAwUJLkdDrlcDgUERGhqqqqK95Xkuv4tREZGVnjcmBR0VlVVnrXP4zL1Uf%2Bqqpqr31eye6d2SXvzk92snuKOp0CWbp0qZ599lnFxsYqICBAwcHB6tmzp5599lktXrz4uofq0KGDgoOD3T5e4dixY/L399d9993nti5d/IBTm80mSbJarcrJyXHtVVVVKS8vTzabTe3bt1dYWJjb/fPz81VRUaHY2FhZrVYdP37cVaouHbtTp04KCmqYBgwAADxPnQrW0aNHFRISUmM9NDRUx48fv%2B6hmjVrptGjR2v58uU6ceKEioqKtGbNGj3wwAMaMWKEjh07pvXr16u8vFzbtm3Trl27lJiYKOniR0hkZmZq3759Kisr03PPPaeWLVsqISFBfn5%2BGjNmjFauXKkjR47o9OnTSklJ0eDBg9W6dWt169ZNNptNS5YsUUlJifLz85WWlqYJEyZcdyYAAOA96lSwOnbsqJSUFLczPWfOnFFqaqo6duxoZLAnnnhCvXr10vDhwzVs2DB17NhRCxYsUEREhNauXavNmzfrrrvu0ooVK5Samup643t8fLzmzZunpKQkxcXF6e9//7vS0tLUokULSRf/zE%2BfPn00cuRIDRo0SK1bt3Y76/b888/r7Nmz6t%2B/vx5%2B%2BGGNHTtW48ePN5IJAAB4Bx/n5e9Ur4UDBw5o%2BvTpKikpUVBQkHx8fFRaWqqgoCCtWbNGffr0qY9ZPVpR0Vnjx/T399Wg5buNH7e%2BbJ3d19ix/P19ZbEEyW4v87jr8teL7N6ZXfLu/GQne12z33RTK8NT1U6d3uTeq1cvffrpp/rss8904sQJOZ1O3XzzzYqPj6/1RxkAAAA0VXUqWJIUEBCgQYMG6fjx465PRQcAAEAd34NVXl6uZ599Vt27d9fPfvYzSVJJSYmmTp2qs2fNXwoDAADwJHUqWKtWrdLXX3%2Bt5cuXy9f3/x/iwoUL%2BsMf/mBsOAAAAE9Up4L18ccfa%2BXKlRo8eLDrE9BDQkKUkpKinTt3Gh0QAADA09SpYBUWFur222%2BvsR4REaHS0tLrnQkAAMCj1alg3XLLLTpw4ECN9Q8//FBt2rS57qEAAAA8WZ1%2Bi3DSpEn61a9%2BpVGjRqmqqkqvvfaacnJytH37dj399NOmZwQAAPAodSpYY8eOVVhYmNatW6fAwECtXbtWHTt21PLlyzV48GDTMwIAAHiUOhWs06dPa/DgwZQpAACAK/jR78Gqrq7WgAEDVIe/sAMAAOAVfnTB8vX11T333KOtW7fWxzwAAAAer06XCNu2baulS5cqLS1Nt912m5o1a%2Ba2n5qaamQ4AAAAT1SngvXtt9%2BqY8eOkiS73W50IAAAAE/3owrWnDlztGLFCr3%2B%2BuuutTVr1mjGjBnGBwMAAPBUP%2Bo9WDt27KixlpaWZmwYAACApuBHFawr/eYgv00IAADg7kcVrEt/2PlaawAAAN6sTn%2BLEAAAAFdHwQIAADDsR/0W4YULF/Tkk09ec43PwQIAAN7sRxWsn/zkJyosLLzmGgAAgDf7UQXr3z//CgAAAFdJ6U0LAAAZ8ElEQVTGe7AAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZ5RMFaunSpunTp4vp67969Gj58uKxWqwYNGqR33nnH7fbp6ekaMGCAbDabRo8erdzcXNfe%2BfPntXDhQvXu3Vs9e/bUrFmzVFxc7No/evSoJk%2BerB49eiguLk7Lli1TdXV1/YcEAABNRqMvWAcPHtSWLVtcX588eVLTp0/XqFGj9MUXXygpKUnJycnKysqSJH300UdauXKlUlJStG/fPt17772aNm2avv/%2Be0nSsmXLdODAAWVmZuqTTz5ReXm5FixYIElyOp2aOXOmLBaLPvvsM73xxhvaunWr0tPTb3xwAADgsRp1waqurtazzz6rSZMmudbeffdddejQQRMnTlRAQIASEhI0cOBAbdq0SZK0ceNGjRo1SnfffbcCAwM1Y8YMSdKOHTtUWVmpzZs3a/bs2Wrfvr3Cw8M1f/587dy5UydPnlR2drby8/OVnJys0NBQRUVFacqUKdqwYUNDxAcAAB6qUResDRs2qGXLlho2bJhrLS8vTzExMW63i46OVk5OzhX3fXx81K1bN%2BXk5KigoEClpaVu%2B1FRUQoICFBubq7y8vLUrl07hYWFufZjYmJ0%2BPBhlZaW1ldMAADQxPg39ABXc%2BrUKa1Zs0avv/6627rdblfXrl3d1sLCwlzvo7Lb7W4FSZJCQ0NVXFwsu93u%2BvrfhYSEuPYv37v0td1uV3BwcK1mLywsVFFRkduav3%2BgIiMja3X/2vLza9T9uAZ/f3PzXsruac%2BBCWT3zuySd%2BcnO9k9TaMtWCkpKRozZozuuOMOHT169Jq39/Hxcfvvq%2B1f6/4mZGRkaPXq1W5rM2bM0KxZs4w9hieyWIKMHzMkJMD4MT0F2b2XN%2Bcnu3fyxOyNsmDt3btXOTk5Wrp0aY298PBwORwOtzW73a7w8HBJksViueJ%2B586dFRERIUlyOBwKDAyUdPGN7Q6HQxEREaqqqrrifS89bm0lJiYqISHBbc3fP1B2e1mtj1EbntboTeb38/NVSEiASkrOqarKu37Lk%2BzemV3y7vxkJ3tds9fHD/e10SgL1jvvvKMTJ04oPj5e0sUSJEl9%2BvTR5MmT9d5777ndPisrSzabTZJktVqVk5OjESNGSJKqqqqUl5enUaNGqX379goLC1Nubq7atm0rScrPz1dFRYViY2NVVFSk48ePy263y2KxuI7dqVMnBQXV/gWKjIyscTmwqOisKiu96x/G5eojf1VVtdc%2Br2T3zuySd%2BcnO9k9RaM8BfLUU0/pww8/1JYtW7RlyxalpaVJkrZs2aKhQ4fq2LFjWr9%2BvcrLy7Vt2zbt2rVLiYmJkqSxY8cqMzNT%2B/btU1lZmZ577jm1bNlSCQkJ8vPz05gxY7Ry5UodOXJEp0%2BfVkpKigYPHqzWrVurW7dustlsWrJkiUpKSpSfn6%2B0tDRNmDChIZ8OAADgYRrlGazQ0FC3N5tXVlZKkm655RZJ0tq1a7V48WKlpqaqbdu2Sk1Ndb3xPT4%2BXvPmzVNSUpJOnz6t2NhYpaWlqUWLFpKkxx57TGVlZRo5cqSqqqo0YMAALVq0yPVYzz//vBYuXKj%2B/fsrKChI48eP1/jx429QcgAA0BT4OC9df0O9Kio6a/yY/v6%2BGrR8t/Hj1pets/saO5a/v68sliDZ7WUed9r4epHdO7NL3p2f7GSva/abbmpleKraaZSXCAEAADwZBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMKzRFqyjR49q%2BvTp6t27t%2BLi4jRv3jydOXNGknTw4EGNHTtWNptN8fHxWrdundt933//ff30pz%2BV1WrV0KFDtWfPHtee0%2BnUihUr1LdvX3Xv3l2TJk3SkSNHXPt2u11z5sxRr169dNddd%2Bnpp59WeXn5jQkNAACahEZbsKZPn66wsDDt3LlTW7Zs0aFDh/THP/5R586d05QpU9SrVy/t3btXq1at0osvvqjt27dLknJycjR//nw9/vjj%2BvLLL/XQQw9pxowZOnHihCQpPT1dmZmZeuWVV7Rnzx61b99eM2fOlNPplCQtWLBAp0%2Bf1vbt2/Xee%2B/p4MGDWr58eYM9DwAAwPM0yoJ19uxZxcbGau7cuQoKClJkZKRGjhyp/fv369NPP9WFCxf05JNPKigoSD169FBiYqIyMjIkSZmZmYqPj9eQIUPUsmVLjR49Wp07d9aWLVskSRs3btQjjzyibt26KTg4WPPnz9ehQ4f09ddf6/Tp09q5c6eSkpLUunVr3XzzzZo9e7YyMzNVUVHRkE8JAADwIP4NPcCVtGrVSikpKW5rx48fV5s2bZSXl6euXbvKz8/PtRcdHa2NGzdKkvLy8hQfH%2B923%2BjoaOXk5Oj8%2BfM6dOiQYmNjXXvBwcG67bbblJOTo9LSUvn7%2B6tLly6u/ZiYGH3//ff67rvv3NZ/SGFhoYqKitzW/P0DFRkZWbsnoJb8/BplP74qf39z817K7mnPgQlk987sknfnJzvZPU2jLFiXy87O1uuvv661a9fq/fffV2hoqNt%2BWFiYHA6HqqurZbfbFRYW5rYfGhqqb7/9Vg6HQ06ns8b9Q0NDVVxcrNDQUAUHB8vX19dtT5KKi4trPW9GRoZWr17ttjZjxgzNmjWr1sdoiiyWIOPHDAkJMH5MT0F27%2BXN%2BcnunTwxe6MvWF999ZWmT5%2BuJ598UnFxcXr//ffrdBwfH5963f93iYmJSkhIcFvz9w%2BU3V5W62PUhqc1epP5/fx8FRISoJKSc6qqqjZ2XE9Adu/MLnl3frKTva7Z6%2BOH%2B9po1AVr586dmjt3rhYuXKgHHnhAkhQREaGCggK329ntdlksFvn6%2Bio8PFx2u73Gfnh4uOs2Doejxn5ERIQiIiJ09uxZVVVVuS5BXjpWREREreeOjIyscTmwqOisKiu96x/G5eojf1VVtdc%2Br2T3zuySd%2BcnO9k9RaM9BXLgwAHNnz9fq1atcpUrSbJarcrPz1dlZaVrLSsrSzabzbWfm5vrdqzs7GzZbDY1b95cnTt3dtt3OBwqKCiQ1WpVdHS0qqurlZ%2Bf73bsVq1a6fbbb6%2BnpAAAoKlplAWrsrJSycnJevzxx9W3b1%2B3vfj4eAUFBSk1NVVlZWX64osv9NZbb2nChAmSpNGjR2vPnj364IMPVF5ertdff10FBQUaMWKEJGncuHF65ZVX9M033%2Bjs2bNasmSJYmNjZbPZZLFY9LOf/UwpKSk6deqUjh07phUrVigxMVHNmjW74c8DAADwTD7OSx8A1Yjs379fEyZMUPPmzWvsbdu2Td9//70WLlyo3NxcRUREaOrUqRo3bpzrNtu3b1dqaqqOHz%2BuqKgoJScn684773Ttv/DCC3rzzTdVVlamPn366Le//a1uueUWSRc/ImLRokXasWOHmjVrpmHDhmn%2B/PlXnOXHKCo6e133vxJ/f18NWr7b%2BHHry9bZfa99o1ry9/eVxRIku73M404bXy%2Bye2d2ybvzk53sdc1%2B002tDE9VO42yYDVFFCwKlilk987sknfnJzvZPa1gNcpLhAAAAJ6MggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBuoKjR49q8uTJ6tGjh%2BLi4rRs2TJVV1c39FgAAMBD%2BDf0AI2N0%2BnUzJkz1alTJ3322Wc6deqUpkyZotatW%2Bvhhx9u6PEAAIAH4AzWZbKzs5Wfn6/k5GSFhoYqKipKU6ZM0YYNGxp6NAAA4CE4g3WZvLw8tWvXTmFhYa61mJgYHT58WKWlpQoODr7mMQoLC1VUVOS25u8fqMjISKOz%2Bvl5Vj/%2B2co9DT3Cj/LR3P4NPcIVXXrdPe31N8Gbs0venZ/sZPc0FKzL2O12hYaGuq1d%2Btput9eqYGVkZGj16tVuazNnztRjjz1mblBdLHIP3fKtEhMTjZe3xq6wsFAZGRlemz09/RWye1l2ybvzk53snpbd8yqhB0hMTNTbb7/t9p/ExETjj1NUVKTVq1fXOFvmDchOdm/kzfnJTnZPwxmsy0RERMjhcLit2e12SVJ4eHitjhEZGelxTRsAAJjDGazLWK1WHT9%2B3FWqJCkrK0udOnVSUFBQA04GAAA8BQXrMt26dZPNZtOSJUtUUlKi/Px8paWlacKECQ09GgAA8BB%2BixYtWtTQQzQ2/fv314cffqjFixfrgw8%2B0Lhx4zR58uSGHuuKgoKC1Lt3b688u0Z2snsjb85PdrJ7Eh%2Bn0%2Bls6CEAAACaEi4RAgAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwfJAR48e1eTJk9WjRw/FxcVp2bJlqq6ubuix6s3Ro0c1ffp09e7dW3FxcZo3b57OnDmj6upqde3aVbGxsbJara7/vPrqqw09sjEDBgyokW/x4sWSpL1792r48OGyWq0aNGiQ3nnnnQae1pwvv/zSLbPValVsbKy6dOmiI0eOqEuXLjX2t27d2tBjX5fdu3frnnvu0Zw5c2rsvf/%2B%2B/rpT38qq9WqoUOHas%2BePa49p9OpFStWqG/fvurevbsmTZqkI0eO3MjRr9sPZd%2B2bZuGDRumnj176v7771dGRoZrLzMzU127dq3xvXDq1KkbOf51u1r%2Bffv2XfF7PSsrS1LTfu2Tk5Nr5I6OjlZSUpIkadWqVerWrZvb/p133tkQEa7OCY9SXV3tfOCBB5xPPvmk0%2BFwOP/5z386BwwY4HzttdcaerR6M3ToUOdTTz3lLC0tdZ48edI5cuRI54IFC5wOh8PZuXNn54kTJxp6xHrTq1cv51dffVVj/cSJE87u3bs709PTnd9//73zk08%2BcVqtVuc//vGPBpjyxlizZo3z8ccfd%2Bbm5jpjYmIaehyj0tLSnPfff79z7NixztmzZ7vtZWdnO2NiYpzvv/%2B%2B89y5c8633nrL2b17d%2Bf//d//OZ1Op3PdunXOvn37OvPy8pxnz551JicnO4cPH%2B6srq5uiCg/2g9l/8c//uG0Wq3OTz75xFlZWenctWuXMyYmxvnll186nc6L2R9%2B%2BOGGGNuYH8q/fft25/3333/V%2Bzbl1/5yFRUVziFDhjg//fRTp9PpdC5evNiZnJx8I8asM85geZjs7Gzl5%2BcrOTlZoaGhioqK0pQpU7Rhw4aGHq1enD17VrGxsZo7d66CgoIUGRmpkSNHav/%2B/SopKZEkhYSENPCU9aOqqkplZWVXzPfuu%2B%2BqQ4cOmjhxogICApSQkKCBAwdq06ZNDTBp/Tt%2B/LjS09NdZy9btWrV0CMZ1aJFC23atEkdOnSosZeZman4%2BHgNGTJELVu21OjRo9W5c2dt2bJFkrRx40Y98sgj6tatm4KDgzV//nwdOnRIX3/99Y2OUSc/lN3hcOjRRx9VQkKC/Pz81L9/f3Xp0kX79%2B%2BXpCbxvfBD%2Ba%2BVrym/9pdLT0/XrbfeqnvvvVeSVFJS0uhfewqWh8nLy1O7du0UFhbmWouJidHhw4dVWlragJPVj1atWiklJUURERGutePHj6tNmzY6c%2BaMfHx8lJycrLvvvlsDBgxQamqqLly40IATm1NSUiKn06kXXnhB/fr1U79%2B/bRw4UKVlZUpLy9PMTExbrePjo5WTk5OA01bv1asWKFRo0apbdu2KikpUXV1taZNm6a77rpL999/v9atWyenB//d%2BokTJ171/yx%2B6LU%2Bf/68Dh06pNjYWNdecHCwbrvtNo/5Xvih7PHx8frVr37l%2BrqyslKFhYVq06aNpIv/RgoKCvTggw/qJz/5iUaOHKnPP//8hsxtyg/lLykp0ZkzZ/SLX/xCd955p37%2B85%2B7inVTf%2B3/ncPhUFpamp588knXWklJibKysjR06FDdeeedGj9%2BfKPLTcHyMHa7XaGhoW5rl7622%2B0NMdINlZ2drddff13Tpk2TJHXv3l0DBgzQjh07tGrVKr3zzjtas2ZNA09pRkVFhbp376677rpL27Zt03/913/p73//uxYtWnTF74OwsDAVFxc30LT151//%2Bpc%2B/vhjTZ48WZLUvHlzderUSRMmTNDu3bv17LPPavXq1U327J3dbnf7gUq6%2BG%2B%2BuLhYDodDTqfziv%2Bb0BS/F5YvX66goCANHjxYkmSxWNS2bVstXbpUn3/%2BuYYPH65HH31Uhw4dauBJzQgODtatt96qJ554Qp9//rlmzJihpKQk7d2716te%2B/T0dN19993q3Lmza%2B3mm29W27Zt9eKLL%2BrTTz9V9%2B7d9ctf/rJRZadgwWN89dVXmjx5sp588knFxcUpNjZWGRkZGjp0qAIDA2W1WjV16lRlZmY29KhG3HzzzXrrrbf0n//5nwoODtYdd9yhuXPn6r333lNlZeUV7%2BPj43ODp6x/b7zxhgYNGqTw8HBJF9/4/%2Bc//1nx8fFq2bKl%2Bvbtq8TExCbzutfWtV7rpvS94HQ6tWzZMr333ntas2aNWrRoIUmaOXOm1qxZoy5duiggIECTJk1S165dm8wvfIwZM0br1q1Tz5491bJlSw0ZMkSDBg265g8TTem1r6io0IYNGzRu3Di39d/85jf64x//qNtuu03BwcGaO3eumjdvro8//riBJq2JguVhIiIi5HA43NYunbm69H9ATdHOnTs1depUPf3003rooYeuertbb71VxcXFHn256Ifceuutqq6ulq%2Bv7xW/D5ri98CHH37oOmNxNbfeeqvH/eZYbYWHh9c4O33ptbZYLFf9Xvj3y%2BqerLq6Wk899ZR27typjIwMRUVF/eDtb731VhUVFd2g6W68S9/r3vDaSxd/o9jpdKpPnz4/eDs/Pz%2B1adOmUb32FCwPY7Vadfz4cbf/wc3KylKnTp0UFBTUgJPVnwMHDmj%2B/PlatWqVHnjgAdf63r17a1wO/O6779SuXbsm8RPcwYMHtXTpUre17777Ts2bN9e9996r3Nxct72srCzZbLYbOWK9%2B/bbb1VYWKjevXu71rZt26Y///nPbrf77rvv1L59%2Bxs93g1htVprvNbZ2dmy2Wxq3ry5Onfu7LbvcDhUUFAgq9V6o0etF0uXLtWhQ4f05ptvql27dm57L730kv7617%2B6rTWl74UNGzbogw8%2BcFu7lM8bXntJ%2Bvzzz9W7d2/5%2Bv7/uuJ0OvX73/9eBw8edK1duHBBR44caVSvPQXLw3Tr1k02m01LlixRSUmJ8vPzlZaWpgkTJjT0aPWisrJSycnJevzxx9W3b1%2B3vbCwML300kvasmWLKisrlZ2drVdffbXJPBcRERHauHGj0tLSVFFRocOHD2vlypUaN26cHnjgAR07dkzr169XeXm5tm3bpl27dikxMbGhxzbq4MGDatOmjYKDg11rLVq00B//%2BEft2bNHlZWV%2Butf/6pNmzY1mdf9cqNHj9aePXv0wQcfqLy8XK%2B//roKCgo0YsQISdK4ceP0yiuv6JtvvtHZs2e1ZMkSxcbGNomy/dVXX2nLli168cUXa7zXSLr4Ruff/va3Onz4sCoqKrRu3ToVFBRo5MiRDTCteZWVlVqyZIlycnJ04cIFvf/%2B%2B9q1a5frcllTfu0v%2Beabb9SpUye3NR8fHx0/flyLFy/WyZMnVVZWpuXLl6t58%2Bb6j//4jwaatCYfZ1O9ltKEnThxQgsXLtTf/vY3BQUFafz48Zo5c2ZDj1Uv9u/frwkTJqh58%2BY19rZt26a8vDy98MILKigoUGRkpBITE/Xwww%2B7/bTjyb788kstX75c//M//yOLxaIhQ4Zo1qxZat68ufbv36/Fixfrf//3f9W2bVvNnTtXgwYNauiRjXr11Vf13nvvafPmzW7rGRkZeu2113Ty5Endeuut%2BuUvf%2BnR/6d66YzDpffW%2Bfv7S7p4pkqStm/frtTUVB0/flxRUVFKTk52%2B1DFF154QW%2B%2B%2BabKysrUp08f/fa3v9Utt9xyg1PUzQ9lX7BggTZv3uxau%2BSuu%2B7Sa6%2B9poqKCi1fvlwffPCBzp07py5dumj%2B/Pnq3r37jQ1xHX4ov9Pp1EsvvaRNmzbJbrerY8eOevzxx10fVSA13df%2BkqFDh2rMmDGaOHGi231LSkq0dOlSffbZZ6qqqpLVatXTTz%2BtO%2B644wZNf20ULAAAAMOaxo/5AAAAjQgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACG/T8xe7j7QwvbywAAAABJRU5ErkJggg%3D%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common393930976851884880">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.28</td>
            <td class="number">22</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.16</td>
            <td class="number">21</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.18</td>
            <td class="number">18</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.196</td>
            <td class="number">18</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.32</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.15</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.3</td>
            <td class="number">16</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.14</td>
            <td class="number">16</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.384</td>
            <td class="number">16</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (95890)</td>
            <td class="number">120909</td>
            <td class="number">70.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="missing">
            <td class="fillremaining">(Missing)</td>
            <td class="number">51649</td>
            <td class="number">29.9%</td>
            <td>
                <div class="bar" style="width:43%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme393930976851884880">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.000129032258065</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.000182648401826</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.000347826086957</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.000444444444444</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">88.445</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">90.194</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">101.27</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">108.851</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">172.548</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">bc_util<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>1178</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.9%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (%)</th>
                        <td>21.1%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (n)</th>
                        <td>36407</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>66.058</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>339.6</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.7%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-7293805005243820192">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAARhJREFUeJzt28EJAkEQAEEVQzIIc/J9ORmEOa0JSOMJuotU/Q/m08w9Zo9jjHEAXjrNHgBWdp49wCyX2333N4/t%2BoVJWJkNAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBD%2B4sntJ89n4R02CASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQFju3N3pOiuxQSAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgLHeLtbK9d2KP7fqlSfiV4xhjzB4CVuUXC4JAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBMITf0QRY9yxT/MAAAAASUVORK5CYII%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-7293805005243820192,#minihistogram-7293805005243820192"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-7293805005243820192">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-7293805005243820192"
                                                      aria-controls="quantiles-7293805005243820192" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-7293805005243820192" aria-controls="histogram-7293805005243820192"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-7293805005243820192" aria-controls="common-7293805005243820192"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-7293805005243820192" aria-controls="extreme-7293805005243820192"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-7293805005243820192">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>14.4</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>48.2</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>71.3</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>88.5</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>98.4</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>339.6</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>339.6</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>40.3</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>26.359</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.39902</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-0.37079</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>66.058</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>21.95</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>-0.65476</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>9006300</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>694.79</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-7293805005243820192">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xl8lOW9//93kmHJQlaMCqZow0EgmQE5rALBxIMCKvKgCUFocUHkcIAAYkEwYFrQ2EIEFWvJsQLVViJEixuICi7lgBu12TBHEWQ7QEwmhIQ1k/v3B1/mx8gW7RVmhnk9Hw8e88h13fd1X5/Le8Z37ntmEmRZliUAAAAYE%2BztCQAAAFxuCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDCbtycQKCoqDhsfMzg4SLGx4aqqqlNDg2V8fF9H/YFdv8QaUD/1B3L9UuPW4IorWl3iWZ3CFSw/FhwcpKCgIAUHB3l7Kl5B/YFdv8QaUD/1B3L9km%2BvAQELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAyzeXsCgK8avHiTt6fwo6yd2tfbUwAA/D9cwQIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY5pMB67PPPpPdbvf4l5ycrOuvv16StHnzZg0dOlR2u10DBw7U66%2B/7rH/ihUrlJqaKofDoYyMDJWWlrr7jh8/rrlz56pnz5664YYblJWVpaqqKnf/nj17NHbsWHXt2lV9%2BvTRggUL1NDQcGkKBwAAlwWfDFg9evRQcXGxx7//%2Bq//0uDBg3XgwAFNmDBB6enp%2BvTTTzVr1ixlZ2erqKhIkvTuu%2B9q8eLFys3N1ZYtWzRgwACNHz9eR44ckSQtWLBAW7duVWFhod5//30dO3ZMs2fPliRZlqVJkyYpJiZGH374oV566SWtXbtWK1as8NpaAAAA/%2BOTAeuH9u3bpxUrVmjGjBl644031K5dO40ZM0ahoaFKS0vTzTffrNWrV0uSVq1apfT0dPXu3VthYWGaOHGiJGnDhg2qr6/Xa6%2B9pqlTpyohIUGxsbGaOXOmNm7cqAMHDqi4uFjl5eXKzs5WVFSUEhMTNW7cOK1cudKb5QMAAD/jF3%2BLcNGiRUpPT1ebNm1UVlampKQkj/7OnTtr7dq1kqSysjINGTLE3RcUFKROnTqppKREnTt3Vm1trcf%2BiYmJCg0NVWlpqQ4ePKi2bdsqOjra3Z%2BUlKSdO3eqtrZWERERjZrvwYMHVVFR4dFms4UpPj7%2BR9d%2BISEhwR6PgSbQ6/8hmy3w1iHQzwHqp/4zHwORL6%2BBzwes7777Tu%2B9957ef/99SZLT6VTHjh09tomOjna/j8rpdHoEJEmKiopSVVWVnE6n%2B%2BczRUZGuvt/2Hf6Z6fT2eiAVVBQoCVLlni0TZw4UVlZWY3a/8eKjAxtknH9RaDXf1pMTLi3p%2BA1gX4OUD/1BzpfXAOfD1gvvfSSBg4cqNjY2AtuFxQU5PF4vv6L7W9CZmam0tLSPNpstjA5nXXGjiGdSuyRkaGqqTkqlyvw3ogf6PX/kOnzyx8E%2BjlA/dQfyPVLjVsDb/3y6fMB65133lFOTo7759jYWFVXV3ts43Q63QEsJibmnP0dOnRQXFycJKm6ulphYWGSTr2xvbq6WnFxcXK5XOfc9/RxGys%2BPv6s24EVFYdVX980TwCXq6HJxvYHgV7/aYG8BoF%2BDlA/9Qdy/ZJvroHv3bQ8w9dff62DBw%2BqZ8%2Be7ja73e7xtQuSVFRUJIfD4e4vKSlx97lcLpWVlcnhcCghIUHR0dEe%2B5eXl%2BvEiRNKTk6W3W7Xvn373KHq9Njt27dXeHjg3n4BAAA/jk8HrG3btunqq6/2eO/THXfcob1792r58uU6duyY1q1bp48%2B%2BkiZmZmSpJEjR6qwsFBbtmxRXV2dnnzySbVs2VJpaWkKCQnRiBEjtHjxYu3evVuVlZXKzc3VoEGD1Lp1a3Xq1EkOh0Pz589XTU2NysvLlZ%2Bfr9GjR3trCQAAgB/y6VuEFRUVZ71hPS4uTkuXLtW8efOUl5enNm3aKC8vz/3G95SUFM2YMUOzZs1SZWWlkpOTlZ%2BfrxYtWkiSJk%2BerLq6Og0fPlwul0upqaketyCfeuopzZ07V/3791d4eLhGjRqlUaNGXbKaAQCA/wuyLMvy9iQCQUXFYeNj2mzBiokJl9NZ53P3ni%2BFpq5/8OJNxsdsSmun9vX2FC45ngPUT/2BW7/UuDW44opWl3hWp/j0LUIAAAB/RMACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw3w6YP3hD39Qv379dMMNN%2Biee%2B7R7t27JUmbN2/W0KFDZbfbNXDgQL3%2B%2Buse%2B61YsUKpqalyOBzKyMhQaWmpu%2B/48eOaO3euevbsqRtuuEFZWVmqqqpy9%2B/Zs0djx45V165d1adPHy1YsEANDQ2XpmAAAHBZ8NmA9de//lUbNmxQQUGBPvjgA1199dVavny5Dhw4oAkTJig9PV2ffvqpZs2apezsbBUVFUmS3n33XS1evFi5ubnasmWLBgwYoPHjx%2BvIkSOSpAULFmjr1q0qLCzU%2B%2B%2B/r2PHjmn27NmSJMuyNGnSJMXExOjDDz/USy%2B9pLVr12rFihVeWwcAAOB/fDZg/elPf9KcOXPUtm1bRUVFKTc3V3PmzNEbb7yhdu3aacyYMQoNDVVaWppuvvlmrV69WpK0atUqpaenq3fv3goLC9PEiRMlSRs2bFB9fb1ee%2B01TZ06VQkJCYqNjdXMmTO1ceNGHThwQMXFxSovL1d2draioqKUmJiocePGaeXKld5cCgAA4Gd8MmAdOHBA%2B/fv13fffadbbrlFvXr10tSpU%2BV0OlVWVqakpCSP7Tt37qySkhJJOqs/KChInTp1UklJiXbt2qXa2lqP/sTERIWGhqq0tFRlZWVq27atoqOj3f1JSUnauXOnamtrm7hqAABwubB5ewLnsn//fgUFBem9995TQUGBjh07pqysLM2ZM0d1dXXq2LGjx/bR0dHu91E5nU6PgCRJUVFRqqqqktPpdP98psjISHf/D/tO/%2Bx0OhUREdGo%2BR88eFAVFRUebTZbmOLj4xu1f2OFhAR7PAaaQK//h2y2wFuHQD8HqJ/6z3wMRL68Bj4ZsE6ePKmTJ0/q17/%2BtWJiYiRJWVlZGjdunPr06XPOfYKCgjwez9d/Phfr/zEKCgq0ZMkSj7aJEycqKyvL2DHOFBkZ2iTj%2BotAr/%2B0mJhwb0/BawL9HKB%2B6g90vrgGPhmwTl%2BBOvOKUdu2bWVZlurr61VdXe2xvdPpVGxsrCQpJibmnP0dOnRQXFycJKm6ulphYWGSTr2xvbq6WnFxcXK5XOfcV5J7/MbIzMxUWlqaR5vNFians67RYzRGSEiwIiNDVVNzVC5X4H3SMdDr/yHT55c/CPRzgPqpP5Drlxq3Bt765dMnA1a7du0UERGh0tJS9evXT5K0d%2B9e2Ww23XTTTVqzZo3H9kVFRXI4HJIku92ukpISDRs2TJLkcrlUVlam9PR0JSQkKDo6WqWlpWrTpo0kqby8XCdOnFBycrIqKiq0b98%2BOZ1O95WzoqIitW/fXuHhjf8PFB8ff9btwIqKw6qvb5ongMvV0GRj%2B4NAr/%2B0QF6DQD8HqJ/6A7l%2ByTfXwPduWkpq1qyZMjIytHDhQu3fv18VFRV69tlndeedd2rYsGHau3evli9frmPHjmndunX66KOPlJmZKUkaOXKkCgsLtWXLFtXV1enJJ59Uy5YtlZaWppCQEI0YMUKLFy/W7t27VVlZqdzcXA0aNEitW7dWp06d5HA4NH/%2BfNXU1Ki8vFz5%2BfkaPXq0l1cEAAD4E5%2B8giVJDz74oJ544gkNHTpUwcHBSk1N1ezZsxUREaGlS5dq3rx5ysvLU5s2bZSXl%2Bd%2B43tKSopmzJihWbNmqbKyUsnJycrPz1eLFi0kSZMnT1ZdXZ2GDx8ul8ul1NRU5eTkuI/71FNPae7cuerfv7/Cw8M1atQojRo1yhtLAAAA/FSQZVmWtycRCCoqDhsf02YLVkxMuJzOOp%2B7NHopNHX9gxdvMj5mU1o7ta%2B3p3DJ8RygfuoP3Pqlxq3BFVe0usSzOsUnbxECAAD4MwIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMMxnA1ZqaqqSk5Nlt9vd/%2BbNmydJ2rx5s4YOHSq73a6BAwfq9ddf99h3xYoVSk1NlcPhUEZGhkpLS919x48f19y5c9WzZ0/dcMMNysrKUlVVlbt/z549Gjt2rLp27ao%2BffpowYIFamhouDRFAwCAy4LPBqyamhr9%2Bc9/VnFxsfvfnDlzdODAAU2YMEHp6en69NNPNWvWLGVnZ6uoqEiS9O6772rx4sXKzc3Vli1bNGDAAI0fP15HjhyRJC1YsEBbt25VYWGh3n//fR07dkyzZ8%2BWJFmWpUmTJikmJkYffvihXnrpJa1du1YrVqzw2joAAAD/45MBy%2BVyqa6uTpGRkWf1vfHGG2rXrp3GjBmj0NBQpaWl6eabb9bq1aslSatWrVJ6erp69%2B6tsLAwTZw4UZK0YcMG1dfX67XXXtPUqVOVkJCg2NhYzZw5Uxs3btSBAwdUXFys8vJyZWdnKyoqSomJiRo3bpxWrlx5SesHAAD%2BzScDVk1NjSzL0jPPPKN%2B/fqpX79%2Bmjt3rurq6lRWVqakpCSP7Tt37qySkhJJOqs/KChInTp1UklJiXbt2qXa2lqP/sTERIWGhqq0tFRlZWVq27atoqOj3f1JSUnauXOnamtrm7hqAABwubB5ewLncuLECXXp0kU9evTQY489poMHD2rKlCnKycmR0%2BlUx44dPbaPjo52v4/K6XR6BCRJioqKUlVVlZxOp/vnM0VGRrr7f9h3%2Bmen06mIiIhGzf/gwYOqqKjwaLPZwhQfH9%2Bo/RsrJCTY4zHQBHr9P2SzBd46BPo5QP3Uf%2BZjIPLlNfDJgHXllVfqlVdecf8cERGhhx56SP/5n/%2Bp7t27n3OfoKAgj8fz9Z/Pxfp/jIKCAi1ZssSjbeLEicrKyjJ2jDNFRoY2ybj%2BItDrPy0mJtzbU/CaQD8HqJ/6A50vroFPBqxzueaaa9TQ0KDg4GBVV1d79DmdTsXGxkqSYmJiztnfoUMHxcXFSZKqq6sVFhYm6dQb26urqxUXFyeXy3XOfSW5x2%2BMzMxMpaWlebTZbGFyOusaPUZjhIQEKzIyVDU1R%2BVyBd4nHQO9/h8yfX75g0A/B6if%2BgO5fqlxa%2BCtXz59MmBt27ZNr732mvvTfZK0Y8cONW/eXAMGDNDf/vY3j%2B2LiorkcDgkSXa7XSUlJRo2bJikU2%2BYLysrU3p6uhISEhQdHa3S0lK1adNGklReXq4TJ04oOTlZFRUV2rdvn5xOp2JiYtxjt2/fXuHhjf8PFB8ff9btwIqKw6qvb5ongMvV0GRj%2B4NAr/%2B0QF6DQD8HqJ/6A7l%2ByTfXwPduWkqKi4vTqlWrlJ%2BfrxMnTmjnzp1avHix7rrrLt15553au3evli9frmPHjmndunX66KOPlJmZKUkaOXKkCgsLtWXLFtXV1enJJ59Uy5YtlZaWppCQEI0YMUKLFy/W7t27VVlZqdzcXA0aNEitW7dWp06d5HA4NH/%2BfNXU1Ki8vFz5%2BfkaPXq0l1cEAAD4E5%2B8ghUfH6/8/HwtXLhQzz33nGJiYjRkyBBlZWWpefPmWrp0qebNm6e8vDy1adNGeXl57je%2Bp6SkaMaMGZo1a5YqKyuVnJys/Px8tWjRQpI0efJk1dXVafjw4XK5XEpNTVVOTo772E899ZTmzp2r/v37Kzw8XKNGjdKoUaO8sQwAAMBPBVmWZXl7EoGgouKw8TFttmDFxITL6azzuUujl0JT1z948SbjYzaltVP7ensKlxzPAeqn/sCtX2rcGlxxRatLPKtTfPIWIQAAgD8jYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGGQ9YtbW1pocEAADwK8YDVr9%2B/TRr1ixt3brV9NAAAAB%2BwXjAysnJUUVFhcaMGaMhQ4Zo2bJlqqqqMn0YAAAAn2U8YA0bNkzPP/%2B8PvroI91111165513dNNNN2nq1KnatMm//rYbAADAT9Fkb3KPjY3Vr371K61cuVK5ubnatGmT7r//fg0aNEjr1q1rqsMCAAB4na2pBq6srNSrr76qV199Vbt27VK/fv00YsQIVVRUKCcnR7t27dIDDzzQVIcHAADwGuMB6%2BOPP9aqVau0YcMGxcTE6Be/%2BIVGjBihNm3auLfp3Lmzxo0bR8ACAACXJeMB64EHHtCNN96oRYsWKS0tTSEhIWdt43A4FB8fb/rQAAAAPsF4wFq/fr0SEhJ04sQJd7iqq6tTeHi4x3ZvvPGG6UMDAAD4BONvcg8KCtIdd9yhDRs2uNsKCgp02223affu3aYPBwAA4HOMB6zHHntMP//5z9WtWzd329ChQ2W32/XYY4%2BZPhwAAIDPMX6L8IsvvtAHH3ygsLAwd1vr1q01Z84cpaammj4cAACAzzF%2BBcuyLNXX15/VfvToUTU0NJg%2BHAAAgM8xHrD69u2rGTNmqKysTDU1NaqurtYXX3yhadOmqV%2B/fqYPBwAA4HOM3yKcM2eOHnroIQ0fPlxBQUHu9l69eik7O9v04QAAAHyO8YAVFxenZcuWafv27dq5c6csy9J1112nxMRE04eCnxm8mL9FCQAIDE32p3ISExOVkJDg/vnEiROSpObNmzfVIQEAAHyC8YD15ZdfKicnR998841cLtdZ/du2bTN9SAAAAJ9iPGDl5OSoVatWeuSRR9SyZUvTwwMAAPg84wFr586d%2BuSTT9SiRQvTQwMAAPgF41/T0KZNG508edL0sAAAAH7DeMB66KGHlJubq9raWtNDAwAA%2BAXjtwiXLFmiPXv26LXXXlNMTIzHd2FJ0t///nfThwQAAPApxgNW//791axZM6NjPv7441qxYoXKy8slSZs3b1Zubq527Nihq666SpMnT9bQoUPd269YsULLly9XZWWlrr/%2BeuXk5CgpKUmSdPz4cT322GNat26dTp48qf79%2BysnJ0exsbGSpD179ujRRx/VF198odDQUA0fPlzTp09XcLDxi30AAOAyZTxgTZs2zeh427Zt05o1a9w/HzhwQBMmTNCDDz6ojIwMbd68WVOnTtW1114rh8Ohd999V4sXL9Zzzz0nh8OhF154QePHj9f69esVFhamBQsWaOvWrSosLFR4eLgefvhhzZ49W3/84x9lWZYmTZqk9u3b68MPP9T333%2BvcePGqXXr1rr33nuN1gUAAC5fTXJZ5p///KdmzZqlu%2B%2B%2BW5LU0NCgtWvX/uhxGhoa9Oijj%2Bqee%2B5xt73xxhtq166dxowZo9DQUKWlpenmm2/W6tWrJUmrVq1Senq6evfurbCwME2cOFGStGHDBtXX1%2Bu1117T1KlTlZCQoNjYWM2cOVMbN27UgQMHVFxcrPLycmVnZysqKkqJiYkaN26cVq5c%2Ba8vCgAACBjGr2C9//77ysrKUv/%2B/bV161ZJ0v79%2BzVnzhzV1tYqIyOj0WOtXLlSLVu21B133KHFixdLksrKyty3%2B07r3LmzO8CVlZVpyJAh7r6goCB16tRJJSUl6ty5s2praz32T0xMVGhoqEpLS3Xw4EG1bdtW0dHR7v6kpCTt3LlTtbW1ioiIaNS8Dx48qIqKCo82my1M8fHxja69MUJCgj0eEdhstsA7DwL9OUD91H/mYyDy5TUwHrD%2B%2BMc/asGCBRoyZIgcDoekU1/d8NRTT%2Bnxxx9vdMD6/vvv9eyzz%2BrFF1/0aHc6nerYsaNHW3R0tKqqqtz9ZwYkSYqKilJVVZWcTqf75zNFRka6%2B3/Yd/pnp9PZ6IBVUFCgJUuWeLRNnDhRWVlZjdr/x4qMDG2SceFfYmLCvT0Frwn05wD1U3%2Bg88U1MB6wduzYoVtuuUWSPD5B2KdPH%2B3du7fR4%2BTm5mrEiBH6%2Bc9/rj179lx0%2B9PH%2BuGnFn/Yf7H9TcjMzFRaWppHm80WJqezztgxpFOJPTIyVDU1R%2BVyNRgdG/7H9PnlDwL9OUD91B/I9UuNWwNv/fJpPGA1a9ZM1dXVat26tUf7zp07G/2nczZv3qySkhI9/vjjZ/XFxsaqurrao83pdLo/BRgTE3PO/g4dOiguLk6SVF1drbCwMEmSZVmqrq5WXFycXC7XOfc9fdzGio%2BPP%2Bt2YEXFYdXXN80TwOVqaLKx4T8C%2BRwI9OcA9VN/INcv%2BeYaGL9pedNNNyk7O1vbt2%2BXdCqgfPzxx5o6dapSU1MbNcbrr7%2Bu/fv3KyUlRb169dLw4cMlSb169dL111%2Bv0tJSj%2B2LiorctyPtdrtKSkrcfS6XS2VlZXI4HEpISFB0dLTH/uXl5Tpx4oSSk5Nlt9u1b98%2Bd6g6PXb79u0VHh64t18AAMCPYzxgzZo1S5Zl6bbbbtPx48d14403aty4cbr66qv18MMPN2qMhx9%2BWO%2B8847WrFmjNWvWKD8/X5K0Zs0a3X777dq7d6%2BWL1%2BuY8eOad26dfroo4%2BUmZkpSRo5cqQKCwu1ZcsW1dXV6cknn1TLli2VlpamkJAQjRgxQosXL9bu3btVWVmp3NxcDRo0SK1bt1anTp3kcDg0f/581dTUqLy8XPn5%2BRo9erTpZQIAAJcx47cIIyMjtXTpUm3fvl07d%2B5UUFCQrrvuOl133XWNHiMqKsrjzeb19fWSpKuuukqStHTpUs2bN095eXlq06aN8vLy3G98T0lJ0YwZMzRr1ixVVlYqOTlZ%2Bfn57j8%2BPXnyZNXV1Wn48OFyuVxKTU1VTk6O%2B1hPPfWU5s6dq/79%2Bys8PFyjRo3SqFGj/tVlAQAAASTIsizL25MIBBUVh42PabMFKyYmXE5nnc/dez6XwYs3eXsKl7W1U/t6ewqXnL89B0yjfuoP5Pqlxq3BFVe0usSzOsX4Fax%2B/fqdt8/lcmnz5s2mDwkAAOBTjAeszMxMj688aGho0J49e7Rp0yaNHz/e9OEAAAB8jvGANXny5HO2FxUV6a9//avpwwEAAPicS/bd8g6HQ8XFxZfqcAAAAF5zyQLWd999p0OHDl2qwwEAAHiN8VuEI0eOPKvtxIkT%2Bvbbb3XzzTebPhwAAIDPMR6wrr322rP%2Brl%2BLFi30i1/8Qr/4xS9MHw4AAMDnGA9YTzzxhOkhAQAA/IrxgLVq1So1a9asUdsOGzbM9OEBAAC8znjAeuyxx3T8%2BHH98Avig4KCPNqCgoIIWAAA4LJkPGA999xzeumllzRhwgQlJibK5XLp66%2B/Vn5%2BvsaMGaM%2BffqYPiQAAIBPMR6wHn/8cf3pT39SfHy8u%2B2GG27Qo48%2Bqvvuu09vv/226UMCAAD4FOPfg7Vnzx5FRkae1R4VFaV9%2B/aZPhwAAIDPMR6wrrvuOuXm5srpdLrbDh06pLy8PF133XWmDwcAAOBzjN8izM7O1oQJE/TKK68oPDxcQUFBqq2tVXh4uJ599lnThwMAAPA5xgNWt27d9MEHH%2BjDDz/U/v37ZVmWrrzySqWkpCgiIsL04QAAAHyO8YAlSaGhoRo4cKD27dunhISEpjgEAACAzzL%2BHqxjx47p0UcfVZcuXTR48GBJUk1NjR544AEdPnzY9OEAAAB8jvGA9fTTT%2BvLL7/UwoULFRz8/w9/8uRJ/e53vzN9OAAAAJ9jPGC99957Wrx4sQYNGuT%2Bo8%2BRkZHKzc3Vxo0bTR8OAADA5xgPWAcPHtS11157VntcXJxqa2tNHw4AAMDnGA9YV111lbZu3XpW%2BzvvvKOrr77a9OEAAAB8jvFPEd5zzz36r//6L6Wnp8vlcumFF15QSUmJ1q9fr0ceecT04QAAAHyO8YA1cuRIRUdHa9myZQoLC9PSpUt13XXXaeHChRo0aJDpwwEAAPgc4wGrsrJSgwYNIkwBAICAZfQ9WA0NDUpNTZVlWSaHBQAA8CtGA1ZwcLBuvPFGrV271uSwAAAAfsX4LcI2bdro8ccfV35%2Bvn72s5%2BpWbNmHv15eXmmDwkAAOBTjAesr7/%2BWtddd50kyel0mh4eAADA5xkLWNOmTdOiRYv04osvutueffZZTZw40dQhAAAA/IKx92Bt2LDhrLb8/PyfPN5XX32le%2B65R927d1fv3r01ZcoUHTx4UJK0efNmDR06VHa7XQMHDtTrr7/use%2BKFSuUmpoqh8OhjIwMlZaWuvuOHz%2BuuXPnqmfPnrrhhhuUlZWlqqoqd/%2BePXs0duxYde3aVX369NGCBQvU0NDwk%2BsAAACBx1jAOtcnB3/qpwlPnDih%2B%2B67Tz169ND//M8EJAKZAAAeNklEQVT/6O2331ZVVZVycnJ04MABTZgwQenp6fr00081a9YsZWdnq6ioSJL07rvvavHixcrNzdWWLVs0YMAAjR8/XkeOHJEkLViwQFu3blVhYaHef/99HTt2TLNnz3bPd9KkSYqJidGHH36ol156SWvXrtWKFSt%2B4qoAAIBAZCxgnf7Dzhdra4yjR49q2rRpGj9%2BvJo3b67Y2Fjdeuut%2Buabb/TGG2%2BoXbt2GjNmjEJDQ5WWlqabb75Zq1evliStWrVK6enp6t27t8LCwty3KDds2KD6%2Bnq99tprmjp1qhISEhQbG6uZM2dq48aNOnDggIqLi1VeXq7s7GxFRUUpMTFR48aN08qVK3/6wgAAgIBj/G8RmhAVFaWMjAzZbDZZlqVvv/1Wr776qgYPHqyysjIlJSV5bN%2B5c2eVlJRI0ln9QUFB6tSpk0pKSrRr1y7V1tZ69CcmJio0NFSlpaUqKytT27ZtFR0d7e5PSkrSzp07%2BUPVAACg0Yx/itCkvXv36pZbbpHL5VJmZqamTJmisWPHqmPHjh7bRUdHu99H5XQ6PQKSdCqwVVVVuT/VGBUV5dEfGRnp7v9h3%2BmfnU6nIiIiGjXvgwcPqqKiwqPNZgtTfHx8o/ZvrJCQYI9HBDabLfDOg0B/DlA/9Z/5GIh8eQ2MBayTJ09q%2BvTpF237Md%2BD1bZtW5WUlOi7777TnDlz9Otf//q8256%2BHXm%2B25IXu135U29nnktBQYGWLFni0TZx4kRlZWUZO8aZIiNDm2Rc%2BJeYmHBvT8FrAv05QP3UH%2Bh8cQ2MBax///d/d3/K70JtP1ZQUJCuvfZazZgxQ%2Bnp6RowYICqq6s9tnE6nYqNjZUkxcTEnLO/Q4cOiouLkyRVV1crLCxM0qk3tldXVysuLk4ul%2Buc%2B0pyj98YmZmZSktL82iz2cLkdNY1eozGCAkJVmRkqGpqjsrl4pOOgc70%2BeUPAv05QP3UH8j1S41bA2/98mksYJ35/Vf/qk8//VSzZ8/WunXrZLOdmuLpr0q48cYb9eqrr3psX1RUJIfDIUmy2%2B0qKSnRsGHDJEkul0tlZWVKT09XQkKCoqOjVVpaqjZt2kiSysvLdeLECSUnJ6uiokL79u2T0%2BlUTEyMe%2Bz27dsrPLzx/4Hi4%2BPPuh1YUXFY9fVN8wRwuRqabGz4j0A%2BBwL9OUD91B/I9Uu%2BuQa%2Bd9NSp960fvToUeXl5eno0aOqqqrSM888o%2B7du%2BuOO%2B7Q3r17tXz5ch07dkzr1q3TRx99pMzMTEnSyJEjVVhYqC1btqiurk5PPvmkWrZsqbS0NIWEhGjEiBFavHixdu/ercrKSuXm5mrQoEFq3bq1OnXqJIfDofnz56umpkbl5eXKz8/X6NGjvbwiAADAn/hkwIqIiNDzzz%2Bvbdu2qX///hoyZIjCw8P15JNPKi4uTkuXLtVrr72mHj16aNGiRcrLy3O/8T0lJUUzZszQrFmz1KdPH/3jH/9Qfn6%2BWrRoIUmaPHmyevXqpeHDh2vgwIFq3bq15s2b5z72U089pcOHD6t///669957NXLkSI0aNcor6wAAAPxTkPVTvw0UP0pFxWHjY9pswYqJCZfTWedzl0bPZfDiTd6ewmVt7dS%2B3p7CJedvzwHTqJ/6A7l%2BqXFrcMUVrS7xrE7xyStYAAAA/oyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAzz2YC1Z88eTZgwQT179lSfPn00Y8YMHTp0SJK0bds2jRw5Ug6HQykpKVq2bJnHvm%2B99ZZuvfVW2e123X777dq0aZO7z7IsLVq0SH379lWXLl10zz33aPfu3e5%2Bp9OpadOmqVu3burRo4ceeeQRHTt27NIUDQAALgs%2BG7AmTJig6Ohobdy4UWvWrNH27dv1%2B9//XkePHtW4cePUrVs3bd68WU8//bT%2B8Ic/aP369ZKkkpISzZw5U1OmTNFnn32mu%2B%2B%2BWxMnTtT%2B/fslSStWrFBhYaGef/55bdq0SQkJCZo0aZIsy5IkzZ49W5WVlVq/fr3efPNNbdu2TQsXLvTaOgAAAP/jkwHr8OHDSk5O1kMPPaTw8HDFx8dr%2BPDh%2Bvzzz/XBBx/o5MmTmj59usLDw9W1a1dlZmaqoKBAklRYWKiUlBQNGTJELVu2VEZGhjp06KA1a9ZIklatWqX7779fnTp1UkREhGbOnKnt27fryy%2B/VGVlpTZu3KhZs2apdevWuvLKKzV16lQVFhbqxIkT3lwSAADgR3wyYLVq1Uq5ubmKi4tzt%2B3bt09XX321ysrK1LFjR4WEhLj7OnfurJKSEklSWVmZkpKSPMY73X/8%2BHFt375dycnJ7r6IiAj97Gc/U0lJicrKymSz2XT99de7%2B5OSknTkyBHt2LGjqcoFAACXGZu3J9AYxcXFevHFF7V06VK99dZbioqK8uiPjo5WdXW1Ghoa5HQ6FR0d7dEfFRWlr7/%2BWtXV1bIs66z9o6KiVFVVpaioKEVERCg4ONijT5KqqqoaPd%2BDBw%2BqoqLCo81mC1N8fHyjx2iMkJBgj0cENpst8M6DQH8OUD/1n/kYiHx5DXw%2BYH3xxReaMGGCpk%2Bfrj59%2Buitt976SeMEBQU1af%2BZCgoKtGTJEo%2B2iRMnKisrq9Fj/BiRkaFNMi78S0xMuLen4DWB/hygfuoPdL64Bj4dsDZu3KiHHnpIc%2BfO1Z133ilJiouL065duzy2czqdiomJUXBwsGJjY%2BV0Os/qj42NdW9TXV19Vn9cXJzi4uJ0%2BPBhuVwu9y3I02OdebvyYjIzM5WWlubRZrOFyemsa/QYjRESEqzIyFDV1ByVy9VgdGz4H9Pnlz8I9OcA9VN/INcvNW4NvPXLp88GrK1bt2rmzJl6%2Bumn1bdvX3e73W7XypUrVV9fL5vt1PSLiorkcDjc/aWlpR5jFRcX67bbblPz5s3VoUMHlZaWqkePHpKk6upq7dq1S3a7XQkJCWpoaFB5ebk6d%2B7sHrtVq1a69tprGz33%2BPj4s24HVlQcVn29%2BSdA90fWGR8T/qkpzi9/4XI1UD/1e3saXhPo9Uu%2BuQa%2Bd9NSUn19vbKzszVlyhSPcCVJKSkpCg8PV15enurq6vTpp5/qlVde0ejRoyVJGRkZ2rRpk95%2B%2B20dO3ZML774onbt2qVhw4ZJku666y49//zz%2Buqrr3T48GHNnz9fycnJcjgciomJ0eDBg5Wbm6vvv/9ee/fu1aJFi5SZmalmzZpd8nUAAAD%2BKcg6/QVQPuTzzz/X6NGj1bx587P61q1bpyNHjmju3LkqLS1VXFycHnjgAd11113ubdavX6%2B8vDzt27dPiYmJys7OVvfu3d39zzzzjF5%2B%2BWXV1dWpV69e%2Bu1vf6urrrpK0qmviMjJydGGDRvUrFkz3XHHHZo5c%2BY55/JjVFQc/pf2PxebLVgDF35sfFz4p7VT%2B158o8uMzRasmJhwOZ11Pvfb66VA/dQfyPVLjVuDK65odYlndYpPBqzLEQELTY2AFXj/g6F%2B6g/k%2BiXfDlg%2BeYsQAADAnxGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGE%2BHbA%2B/vhj3XjjjZo2bdpZfW%2B99ZZuvfVW2e123X777dq0aZO7z7IsLVq0SH379lWXLl10zz33aPfu3e5%2Bp9OpadOmqVu3burRo4ceeeQRHTt2zN2/bds2jRw5Ug6HQykpKVq2bFnTFgoAAC4rPhuw/vu//1vz589Xu3btzuorKSnRzJkzNWXKFH322We6%2B%2B67NXHiRO3fv1%2BStGLFChUWFur555/Xpk2blJCQoEmTJsmyLEnS7NmzVVlZqfXr1%2BvNN9/Utm3btHDhQknS0aNHNW7cOHXr1k2bN2/W008/rT/84Q9av379pSseAAD4NZ8NWC1atNDq1avPGbAKCwuVkpKiIUOGqGXLlsrIyFCHDh20Zs0aSdKqVat0//33q1OnToqIiNDMmTO1fft2ffnll6qsrNTGjRs1a9YstW7dWldeeaWmTp2qwsJCnThxQh988IFOnjyp6dOnKzw8XF27dlVmZqYKCgou9RIAAAA/ZfP2BM5nzJgx5%2B0rKytTSkqKR1vnzp1VUlKi48ePa/v27UpOTnb3RURE6Gc/%2B5lKSkpUW1srm82m66%2B/3t2flJSkI0eOaMeOHSorK1PHjh0VEhLiMfaqVasaPfeDBw%2BqoqLCo81mC1N8fHyjx2iMkBCfzcfwApst8M6H08%2BBQH0uUD/1n/kYiHx5DXw2YF2I0%2BlUdHS0R1tUVJS%2B/vprVVdXy7IsRUVFndVfVVWlqKgoRUREKDg42KNPkqqqquR0Os/aNzo6WtXV1WpoaPDY73wKCgq0ZMkSj7aJEycqKyvrR9UJ/BgxMeHenoLXREaGensKXkX91B/ofHEN/DJgnU9QUFCT9jdWZmam0tLSPNpstjA5nXVGxj/NFxM7vMf0%2BeUPQkKCFRkZqpqao3K5Grw9nUuO%2Bqk/kOuXGrcG3vrl0y8DVmxsrJxOp0eb0%2BlUbGysYmJiFBwcrOrq6rP64%2BLiFBcXp8OHD8vlcrlvA54e63T/rl27ztr39LiNER8ff9btwIqKw6qvD8wnAC6NQD6/XK4G6qd%2Bb0/DawK9fsk318AvL4HY7XaVlpZ6tBUXF8vhcKh58%2Bbq0KGDR391dbV27dolu92uzp07q6GhQeXl5e7%2BoqIitWrVStdee63sdrvKy8tVX1/v0e9wOJq%2BMAAAcFnwy4CVkZGhTZs26e2339axY8f04osvateuXRo2bJgk6a677tLzzz%2Bvr776SocPH9b8%2BfOVnJwsh8OhmJgYDR48WLm5ufr%2B%2B%2B%2B1d%2B9eLVq0SJmZmWrWrJlSUlIUHh6uvLw81dXV6dNPP9Urr7yi0aNHe7lqAADgL3z2FqHdbpck95Wk9957T9KpK1UdOnTQwoULlZeXp5kzZyoxMVFLly5V69atJUkjR45URUWF7rvvPtXV1alXr156%2Bumn3WP/5je/UU5OjgYOHKhmzZrpjjvu0JQpUyRJzZs319KlSzV37lz16dNHcXFxmjFjhgYMGHApywcAAH4syDr97ZtoUhUVh42PabMFa%2BDCj42PC/%2B0dmpfb0/hkrPZghUTEy6ns87n3n9xKVA/9Qdy/VLj1uCKK1pd4lmd4pe3CAEAAHwZAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYZvP2BHzRnj179Oijj%2BqLL75QaGiohg8frunTpys4mDwK3zV48SZvT6HR1k7t6%2B0pAECTImD9gGVZmjRpktq3b68PP/xQ33//vcaNG6fWrVvr3nvv9fb0AACAH%2BCSzA8UFxervLxc2dnZioqKUmJiosaNG6eVK1d6e2oAAMBPcAXrB8rKytS2bVtFR0e725KSkrRz507V1tYqIiLiomMcPHhQFRUVHm02W5ji4%2BONzjUkhHwM/2SzmTl3Tz8HAvW5QP3Uf%2BZjIPLlNSBg/YDT6VRUVJRH2%2BmfnU5nowJWQUGBlixZ4tE2adIkTZ482dxEdSrI3X3V18rMzDQe3vzBwYMHVVBQQP0BWr90ag1WrHg%2BYNeA%2Bqk/kOuXfHsNfC/yXQYyMzP16quvevzLzMw0fpyKigotWbLkrKtlgYL6A7t%2BiTWgfuoP5Pol314DrmD9QFxcnKqrqz3anE6nJCk2NrZRY8THx/tckgYAAJcOV7B%2BwG63a9%2B%2Bfe5QJUlFRUVq3769wsPDvTgzAADgLwhYP9CpUyc5HA7Nnz9fNTU1Ki8vV35%2BvkaPHu3tqQEAAD8RkpOTk%2BPtSfia/v3765133tG8efP09ttv66677tLYsWO9Pa1zCg8PV8%2BePQP26hr1B3b9EmtA/dQfyPVLvrsGQZZlWd6eBAAAwOWEW4QAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGw/NCePXs0duxYde3aVX369NGCBQvU0NDg7Wk1qdTUVCUnJ8tut7v/zZs3T5K0efNmDR06VHa7XQMHDtTrr7/u5dma8fHHH%2BvGG2/UtGnTzup76623dOutt8put%2Bv222/Xpk2b3H2WZWnRokXq27evunTponvuuUe7d%2B%2B%2BlFM34nz1b9myRddff73HuWC321VUVCTp8ql/z549mjBhgnr27Kk%2BffpoxowZOnTokCRp27ZtGjlypBwOh1JSUrRs2TKPfS90fviL89Xf0NCgjh07nvV68Kc//cm97%2BVQ/1dffaV77rlH3bt3V%2B/evTVlyhQdPHhQ0sVf81asWKHU1FQ5HA5lZGSotLTUGyX8y863Brt37z7na8DatWvd%2B/rEGljwKw0NDdadd95pTZ8%2B3aqurra%2B%2BeYbKzU11XrhhRe8PbUm1a1bN%2BuLL744q33//v1Wly5drBUrVlhHjhyx3n//fctut1v//Oc/vTBLc/Lz861bbrnFGjlypDV16lSPvuLiYispKcl66623rKNHj1qvvPKK1aVLF%2Bv//u//LMuyrGXLlll9%2B/a1ysrKrMOHD1vZ2dnW0KFDrYaGBm%2BU8pNcqP7169dbt9xyy3n3vRzqtyzLuv32262HH37Yqq2ttQ4cOGANHz7cmj17tnXkyBGrb9%2B%2B1u9%2B9zurtrbW%2Bsc//mF1797deueddyzLuvj54S/OV391dbXVoUMHa//%2B/efc73Ko//jx41afPn2sJUuWWMePH7cqKyutX/7yl9aECRMu%2Bpq3fv16q2vXrtbmzZuturo665lnnrH69u1r1dXVebmqH%2BdCa1BaWmolJSWdd19fWQOuYPmZ4uJilZeXKzs7W1FRUUpMTNS4ceO0cuVKb0%2BtybhcLtXV1SkyMvKsvjfeeEPt2rXTmDFjFBoaqrS0NN18881avXq1F2ZqTosWLbR69Wq1a9furL7CwkKlpKRoyJAhatmypTIyMtShQwetWbNGkrRq1Srdf//96tSpkyIiIjRz5kxt375dX3755aUu4ye7UP2HDh1Sq1atzrvv5VD/4cOHlZycrIceekjh4eGKj4/X8OHD9fnnn%2BuDDz7QyZMnNX36dIWHh6tr167KzMxUQUGBpIufH/7gQvXX1NRI0jlfD6TLo/6jR49q2rRpGj9%2BvJo3b67Y2Fjdeuut%2Buabby76mrdq1Sqlp6erd%2B/eCgsL08SJEyVJGzZs8GZJP9qF1qAxrwG%2BsAYELD9TVlamtm3bKjo62t2WlJSknTt3qra21oszazo1NTWyLEvPPPOM%2BvXrp379%2Bmnu3Lmqq6tTWVmZkpKSPLbv3LmzSkpKvDRbM8aMGXPeF5AL1Xz8%2BHFt375dycnJ7r6IiAj97Gc/86s1uVD9NTU1OnTokH71q1%2Bpe/fuuu2229z/87xc6m/VqpVyc3MVFxfnbtu3b5%2BuvvpqlZWVqWPHjgoJCXH3nXnOXw7PiQvVf%2BjQIQUFBSk7O1u9e/dWamqq8vLydPLkSUmXR/1RUVHKyMiQzWaTZVn69ttv9eqrr2rw4MEXre%2BH/UFBQerUqZNf1S9deA1qamrU0NCg8ePHq0ePHrrlllu0bNkyWZYlyXfWgIDlZ5xOp6KiojzaTv/sdDq9MaUmd%2BLECXXp0kU9evTQunXr9Oc//1n/%2BMc/lJOTc871iI6OVlVVlZdm2/ScTqdHwJZOnQNVVVWqrq6WZVnnPEculzWJiIjQNddcowcffFB///vfNXHiRM2aNUubN2%2B%2BbOsvLi7Wiy%2B%2BqPHjx5/3nK%2BurlZDQ8MFzw9/dWb9ktSlSxelpqZqw4YNevrpp/X666/r2WeflXTh54e/2bt3r5KTkzVkyBDZ7XZNmTLloq95l1P90rnXoHnz5mrfvr1Gjx6tjz/%2BWI8%2B%2BqiWLFnivornK2tAwILPu/LKK/XKK6/ol7/8pSIiIvTzn/9cDz30kN58803V19efc5%2BgoKBLPEvvu1jNl8uajBgxQsuWLdMNN9ygli1basiQIRo4cOBFbwv7a/1ffPGFxo4dq%2BnTp6tPnz4/eZzLpf7k5GQVFBTo9ttvV1hYmOx2ux544AEVFhZecBx/rL9t27YqKSnRunXr9O233%2BrXv/71ebc9Xd/56vTH%2BqVzr0Fqaqr%2B8pe/KCUlRS1btlTfvn2VmZnpPgd8ZQ0IWH4mLi5O1dXVHm2nr1zFxsZ6Y0pecc0116ihoUHBwcHnXI/LeS1iY2PPulp5uuaYmJjzrsmZt1suN9dcc42%2B//77y67%2BjRs36oEHHtAjjzyiu%2B%2B%2BW9L5XwNO136h88PfnKv%2Bc7nmmmtUVVUly7Iuq/qlU6Hg2muv1YwZM/Tmm2%2BqWbNmF3zNi4mJuexeE3%2B4Bue6EnX6NUDynTUgYPkZu92uffv2ebyAFBUVqX379goPD/fizJrOtm3b9Pjjj3u07dixQ82bN9eAAQPO%2BvhtUVGRHA7HpZziJWW328%2Bqubi4WA6HQ82bN1eHDh08%2Bqurq7Vr1y7Z7fZLPdUmsXLlSr399tsebTt27FBCQsJlVf/WrVs1c%2BZMPf3007rzzjvd7Xa7XeXl5R5Xb8885y90fviT89W/efNm9%2B3A03bs2KG2bdsqKCjosqj/008/1X/8x394/Dc%2B/VU8N9544wVf8%2Bx2u8d7jVwul8rKyvyqfunCa/DJJ5/oL3/5i8f2p18DJB9ag0v6mUUYMWLECOvBBx%2B0Dh06ZH311VdW3759rb/85S/enlaTOXDggNW1a1dr6dKl1vHjx60dO3ZYt912m/XYY49Z33//vdWtWzdr2bJl1tGjR621a9dadrvd2rZtm7enbcTMmTPP%2BpqC8vJyy263uz%2BG/uc//9nq1q2bVVFRYVmWZb388stW3759rW3btlk1NTXW9OnTrYyMDG9M/192rvpffPFFq0%2BfPlZxcbF14sQJ680337SSkpKskpISy7Iuj/pPnjxpDR482HrppZfO6jt%2B/LiVmppqPfHEE1Ztba31ySefWF27drU%2B%2BOADy7Iufn74gwvVX1ZWZiUlJVl/%2B9vfrJMnT1pFRUVWv379rOXLl1uWdXnUf/jwYevGG2%2B0nnjiCevIkSNWZWWlNXbsWGvUqFEXfc378MMP3V9RUFtba/3%2B97%2B3brrpJuvYsWNerurHudAabNiwwXI4HNbf//536%2BTJk9amTZusrl27Wu%2B%2B%2B65lWb6zBkGW9f/edg%2B/sX//fs2dO1effPKJwsPDNWrUKE2aNMnb02pSn332mRYuXKj//d//VUxMjIYMGaKsrCw1b95cn3/%2BuebNm6dvv/1Wbdq00UMPPaSBAwd6e8r/ktNXW07/9maz2SSd%2Bk1cktavX6%2B8vDzt27dPiYmJys7OVvfu3d37P/PMM3r55ZdVV1enXr166be//a2uuuqqS1zFT3eh%2Bi3L0nPPPafVq1fL6XTquuuu05QpUzRgwAD3/v5e/%2Beff67Ro0erefPmZ/WtW7dOR44c0dy5c1VaWqq4uDg98MADuuuuu9zbXOz88HUXq7%2BsrEzPPPOMdu3apfj4eGVmZuree%2B9VcPCpmzL%2BXr906sr97373O5WUlMhms6lXr16aPXu2rrzyyou%2B5r388svKz89XZWWlkpOT9Zvf/Eb/9m//5sVqfpoLrUFBQYFeeOEFHThwQNdcc43uu%2B8%2BDR8%2B3L2vL6wBAQsAAMAw3oMFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIb9f8Uz5VWGdTgiAAAAAElFTkSuQmCC"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-7293805005243820192">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">1153</td>
            <td class="number">0.7%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">98.2</td>
            <td class="number">350</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97.4</td>
            <td class="number">349</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97.9</td>
            <td class="number">347</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">98.6</td>
            <td class="number">340</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">96.8</td>
            <td class="number">337</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">98.3</td>
            <td class="number">332</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97.5</td>
            <td class="number">331</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97.3</td>
            <td class="number">329</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97.7</td>
            <td class="number">325</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (1167)</td>
            <td class="number">132145</td>
            <td class="number">76.5%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="missing">
            <td class="fillremaining">(Missing)</td>
            <td class="number">36407</td>
            <td class="number">21.1%</td>
            <td>
                <div class="bar" style="width:28%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-7293805005243820192">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">1153</td>
            <td class="number">0.7%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.1</td>
            <td class="number">81</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:7%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.2</td>
            <td class="number">67</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:6%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.3</td>
            <td class="number">61</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:6%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.4</td>
            <td class="number">53</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:5%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">165.7</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">173.2</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">182.5</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">187.9</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">339.6</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">dti<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>3499</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>2.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>16.08</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>34.99</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.1%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-8805093359342398371">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAATdJREFUeJzt29FtwjAUQFFAHalDdKd%2BsxNDdCezALpKkFBc55x/JH/k6tlxuI4xxgV46Xb0AmBmX0cvYGXfv4/dv/m7/3xgJbxLIDu888Dzv9liQRAIBIFAcAaZjIP9XEwQCAKBIBAIpz2DuNNgCxMEgkAgnHaLtZK920WvhbczQSAIBIJAIAgEgkAgCASCQCAIBIJAIAgEwhKfmvgyl08xQSAsMUHYx//etzNBIAgEgkAgCASCQCAIBIJAILgHYZOz3p2YIBAEAmG6LZYPD5mJCQJBIBAEAmG6MwjrWOHVsAkCQSAQBAJBIBAEAkEgEK5jjHH0ImBWJggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiEJ%2BqjID0nIHb9AAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-8805093359342398371,#minihistogram-8805093359342398371"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-8805093359342398371">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-8805093359342398371"
                                                      aria-controls="quantiles-8805093359342398371" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-8805093359342398371" aria-controls="histogram-8805093359342398371"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-8805093359342398371" aria-controls="common-8805093359342398371"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-8805093359342398371" aria-controls="extreme-8805093359342398371"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-8805093359342398371">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>4.08</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>10.34</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>15.74</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>21.52</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>29.28</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>34.99</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>34.99</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>11.18</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>7.6032</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.47285</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-0.60719</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>16.08</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>6.2649</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>0.17839</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>2777700</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>57.809</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-8805093359342398371">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xtc1XWex/E3cCCVOxiWSmm2XoCD5mqmODTaWmlaPixFx8nsoq2ro7iyMio5TFpMJaurdnMsc9zZRKV5WN5iJ3Uy1i7qONzMMdLFy5oEB0G8IHD2jx6emSNqWF88t9fz8eDhg%2B/3/L6/z%2Bccfoe3v9/hHD%2B73W4XAAAAjPF3dQEAAADehoAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAyzuLoAX1FeXmN8TX9/P0VFBauyslaNjXbj67srX%2B1b8t3e6du3%2BpZ8t3f6Nt/3zTeHGl2vuTiD5cH8/f3k5%2Bcnf38/V5dyQ/lq35Lv9k7fvtW35Lu907f39E3AAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDLK4uAIDvGbok39UlXJetqUmuLgGAh%2BEMFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAY74MFeAlPe28pAPBmnMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwi6sLAAB3N3RJvqtLuC5bU5NcXQLg8ziDBQAAYBgBCwAAwDACFgAAgGFuG7COHTumKVOm6O6771b//v01e/ZsnT59Wo2NjerevbsSEhJktVodX2%2B99ZZj282bN%2BuBBx6Q1WrV8OHDlZ//t9dP2O12LV68WElJSerZs6cmTpyoo0ePOuZtNptmzpyp3r17q2/fvpo3b57Onz9/Q3sHAACezW0D1pQpUxQREaEdO3Zo48aNKi0t1csvv6yamhrZ7XZ99NFHKiwsdHw9/fTTkqSioiKlp6drxowZ%2BuKLL/TEE09o6tSpOnnypCRp9erVys3N1cqVK5Wfn6/Y2FhNmzZNdrtdkjR37lxVVFQoLy9PmzZt0oEDB7Ro0SKX3Q8AAMDzuGXAqqmpUUJCgtLS0hQcHKyYmBiNGjVKe/bsUXV1tSQpLCzsitvm5uYqOTlZw4YNU6tWrTR69Gh17dpVGzdulCStX79ezzzzjHr06KGQkBClp6ertLRU%2B/fvV0VFhXbs2KE5c%2Baobdu2ateunVJTU5Wbm6u6urob1j8AAPBsbvk2DaGhocrKynIaO3HihG699VadPn1afn5%2BysjIUH5%2Bvlq3bq3hw4dr%2BvTpCgwMVElJiZKTk522jYuLU1FRkS5cuKDS0lIlJCQ45kJCQnTbbbepqKhIZ86ckcViUbdu3Rzz8fHxOnv2rA4fPuw0fi2nTp1SeXm505jF0kYxMTHXe1dcU0CAv9O/vsJX%2B5Z8u3c0n8Xi%2BT8fvvqzTt/e07dbBqzLFRYWas2aNXrzzTclST179tSgQYO0YMEClZaWatq0aQoICFBqaqpsNpsiIiKctg8PD9ehQ4dUVVUlu92u8PDwJvOVlZUKDw9XSEiI/P39neYkqbKystn15uTkaPny5U5jU6dO1fTp06%2Br7%2BYKC2vdIuu6O1/tW/Lt3vH9IiODXV2CMb76s07fns/tA9bevXs1ZcoUzZo1S/3795f0XYC5xGq1avLkyXrjjTeUmpp61XX8/PyuuZ8fO//3UlJSNHjwYKcxi6WNbLbaZq/RHAEB/goLa63q6nNqaGg0urY789W%2BJd/uHc1n%2BrnGFXz1Z52%2Bzfftqv9wuHXA2rFjh9LS0jR//nw98sgjV71dx44dVVlZKbvdrqioKNlsNqd5m82mqKgoRUZGyt/fX1VVVU3mo6OjFR0drZqaGjU0NCggIMAxJ0nR0dHNrjsmJqbJ5cDy8hrV17fMwdLQ0Nhia7szX%2B1b8u3e8f286WfDV3/W6dvzue3Fzn379ik9PV1Lly51Cle7d%2B/Wq6%2B%2B6nTbw4cPq0OHDvLz85PValVxcbHTfGFhoRITExUUFKSuXbs6zVdVVamsrExWq1VxcXFqbGzUwYMHHfMFBQUKDQ1Vp06dWqZRAADgddwyYNXX1ysjI0MzZsxQUpLzZ2pFRETo9ddf18aNG1VfX6/CwkK99dZbGj9%2BvCRp9OjRys/P15YtW3T%2B/HmtWbNGZWVlGjlypCRp3LhxWrlypb788kvV1NRo4cKFSkhIUGJioiIjIzV06FBlZWXp22%2B/1fHjx7V48WKlpKQoMDDwht8PAADAM/nZL70BlBvZs2ePxo8fr6CgoCZz27ZtU0lJiZYtW6aysjLFxMQoJSVFTz75pOPF6Xl5ecrOztaJEyfUpUsXZWRkqE%2BfPo41li1bpnfffVe1tbXq16%2Bfnn/%2Bed1yyy2SvnuLiMzMTG3fvl2BgYEaMWKE0tPTr1jL9Sgvr/lR21%2BJxeKvyMhg2Wy1XnNKtTl8tW/p2r172gcSo%2BV4w4c9%2B%2BpxTt/m%2B7755lCj6zWXWwYsb0TAMsdX%2B5YIWGgeApbnom/vCVhueYkQAADAkxGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDLK4uAHBXQ5fku7oEAICH4gwWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGWVxdAADArKFL8l1dQrNtTU1ydQlAi%2BAMFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABjmtgHr2LFjmjJliu6%2B%2B271799fs2fP1unTpyVJBw4c0NixY5WYmKjk5GStWrXKadvNmzfrgQcekNVq1fDhw5Wf/7e/qLHb7Vq8eLGSkpLUs2dPTZw4UUePHnXM22w2zZw5U71791bfvn01b948nT9//sY0DQAAvILbBqwpU6YoIiJCO3bs0MaNG1VaWqqXX35Z586d06RJk9S7d2/t3r1bS5cu1Wuvvaa8vDxJUlFRkdLT0zVjxgx98cUXeuKJJzR16lSdPHlSkrR69Wrl5uZq5cqVys/PV2xsrKZNmya73S5Jmjt3rioqKpSXl6dNmzbpwIEDWrRokcvuBwAA4HncMmDV1NQoISFBaWlpCg4OVkxMjEaNGqU9e/Zo586dunjxombNmqXg4GD16tVLKSkpysnJkSTl5uYqOTlZw4YNU6tWrTR69Gh17dpVGzdulCStX79ezzzzjHr06KGQkBClp6ertLRU%2B/fvV0VFhXbs2KE5c%2Baobdu2ateunVJTU5Wbm6u6ujpX3iUAAMCDuGXACg0NVVZWlqKjox1jJ06c0K233qqSkhJ1795dAQEBjrm4uDgVFRVJkkpKShQfH%2B%2B03qX5CxcuqLS0VAkJCY65kJAQ3XbbbSoqKlJJSYksFou6devmmI%2BPj9fZs2d1%2BPDhlmoXAAB4GY94J/fCwkKtWbNGb775pjZv3qzw8HCn%2BYiICFVVVamxsVE2m00RERFO8%2BHh4Tp06JCqqqpkt9ubbB8eHq7KykqFh4crJCRE/v7%2BTnOSVFlZ2ex6T506pfLycqcxi6WNYmJimr1GcwQE%2BDv96yt8tW/AG1ksVz6OffU4p2/v6dvtA9bevXs1ZcoUzZo1S/3799fmzZt/0Dp%2Bfn4tOv/3cnJytHz5cqexqVOnavr06c1e43qEhbVukXXdna/2DXiTyMjga8776nFO357PrQPWjh07lJaWpvnz5%2BuRRx6RJEVHR6usrMzpdjabTZGRkfL391dUVJRsNluT%2BaioKMdtqqqqmsxHR0crOjpaNTU1amhocFyCvLTW31%2Bu/D4pKSkaPHiw05jF0kY2W22z12iOgAB/hYW1VnX1OTU0NBpd2535at%2BAN7ra86KvHuf0bb7v7wvxLcVtA9a%2BffuUnp6upUuXKinpbx8GarVatXbtWtXX18ti%2Ba78goICJSYmOuaLi4ud1iosLNRDDz2koKAgde3aVcXFxerbt68kqaqqSmVlZbJarYqNjVVjY6MOHjyouLg4x9qhoaHq1KlTs2uPiYlpcjmwvLxG9fUtc7A0NDS22NruzFf7BrzJ9x3Dvnqc07fnc8uLnfX19crIyNCMGTOcwpUkJScnKzg4WNnZ2aqtrdXnn3%2BudevWafz48ZKk0aNHKz8/X1u2bNH58%2Be1Zs0alZWVaeTIkZKkcePGaeXKlfryyy9VU1OjhQsXKiEhQYmJiYqMjNTQoUOVlZWlb7/9VsePH9fixYuVkpKiwMDAG34/AAAAz%2BRnv/QGUG5kz549Gj9%2BvIKCgprMbdu2TWfPntX8%2BfNVXFys6OhoTZ48WePGjXPcJi8vT9nZ2Tpx4oS6dOmijIwM9enTxzG/bNkyvfvuu6qtrVW/fv30/PPP65ZbbpH03VtEZGZmavv27QoMDNSIESOUnp5%2BxVquR3l5zY/a/kosFn9FRgbLZqv1msTfHDeq76FL8r//RgB%2BlK2pSVcc5/mNvk25%2BeZQo%2Bs1l1sGLG9EwDKHgAV4DwKWM/r2noDllpcIAQAAPBkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYZjxgnTlzxvSSAAAAHsV4wBo4cKDmzJmjffv2mV4aAADAIxgPWJmZmSovL9eECRM0bNgwrVq1SpWVlaZ3AwAA4LaMB6yRI0dq5cqV%2BvjjjzVu3Dh9%2BOGH%2BulPf6rU1FTl5%2Beb3h0AAIDbabEXuUdFRenxxx/X2rVrlZWVpfz8fD3zzDN68MEHtW3btpbaLQAAgMtZWmrhiooKvffee3rvvfdUVlamgQMHasyYMSovL1dmZqbKyso0efLklto9AACAyxgPWLt27dL69eu1fft2RUZG6tFHH9WYMWPUvn17x23i4uI0adIkAhYAAPBKxgPW5MmTNWDAAC1evFiDBw9WQEBAk9skJiYqJibG9K4BAADcgvGAlZeXp9jYWNXV1TnCVW1trYKDg51u98EHH5jeNQAAgFsw/iJ3Pz8/jRgxQtu3b3eM5eTk6KGHHtLRo0dN7w4AAMDtGA9YL7zwgu644w717t3bMfbwww/LarXqhRdeML07AAAAt2P8EuHevXu1c%2BdOtWnTxjHWtm1bPffccxo0aJDp3QEAALgd42ew7Ha76uvrm4yfO3dOjY2NpncHAADgdowHrKSkJM2ePVslJSWqrq5WVVWV9u7dq5kzZ2rgwIGmdwcAAOB2jF8ifO6555SWlqZRo0bJz8/PMd6vXz9lZGSY3h0AAIDbMR6woqOjtWrVKpWWlurIkSOy2%2B3q3LmzunTpYnpXAAAAbqnFPiqnS5cuio2NdXxfV1cnSQoKCmqpXQIAALgF4wFr//79yszM1FdffaWGhoYm8wcOHDC9SwAAALdiPGBlZmYqNDRU8%2BbNU6tWrUwvDwAA4PaMB6wjR47os88%2B00033WR6aQAAAI9g/G0a2rdvr4sXL5peFgAAwGMYD1hpaWnKysrSmTNnTC8NAADgEYxfIly%2BfLmOHTumP/zhD4qMjHR6LyxJ%2BuSTT0zvEgAAwK0YD1g/%2BclPFBgYaHpZAAAAj2E8YM2cOdP0kgAAAB7F%2BGuwJOkvf/mL5syZoyeeeEKS1NjYqK1bt7bErgAAANyO8YD10Ucf6Wc/%2B5lsNpv27dsnSTp58qSee%2B45rV%2B/3vTuAAAA3I7xgPXGG2/olVde0RtvvOF4gXv79u31H//xH3rnnXdM7w4AAMDtGA9Yhw8f1v333y9JTn9B2L9/fx0/fvy61tq1a5cGDBjQ5HVdn376qbp16yar1er0VVBQIEmy2%2B1avHixkpKS1LNnT02cOFFHjx51bG%2Bz2TRz5kz17t1bffv21bx583T%2B/HnH/IEDBzR27FglJiYqOTlZq1atuu77AQAA%2BC7jASswMFBVVVVNxo8cOXJdH53z29/%2BVgsXLtTtt9/eZK6mpkadOnVSYWGh01diYqIkafXq1crNzdXKlSuVn5%2Bv2NhYTZs2TXa7XZI0d%2B5cVVRUKC8vT5s2bdKBAwe0aNEiSdK5c%2Bc0adIk9e7dW7t379bSpUv12muvKS8v74fcHQAAwAcZD1g//elPlZGRodLSUknfnS3atWuXUlNTNWjQoGavc9NNN2nDhg1XDFinT59WaGjoVbddv369nnnmGfXo0UMhISFKT09XaWmp9u/fr4qKCu3YsUNz5sxR27Zt1a5dO6Wmpio3N1d1dXXauXOnLl68qFmzZik4OFi9evVSSkqKcnJyrv/OAAAAPsl4wJozZ47sdrseeughXbhwQQMGDNCkSZN066236pe//GWz15kwYcJVQ1R1dbVOnz6txx9/XH369NFDDz2kjRs3SpIuXLig0tJSJSQkOG4fEhKi2267TUVFRSopKZHFYlG3bt0c8/Hx8Tp79qwOHz6skpISde/eXQEBAY75uLg4FRUVXe9dAQAAfJTx98EKCwvTm2%2B%2BqdLSUh05ckR%2Bfn7q3LmzOnfubGwfISEh6tixo6ZPn64ePXpo%2B/btSktLU0xMjO644w7Z7XaFh4c7bRMeHq7KykqFh4crJCRE/v7%2BTnOSVFlZKZvN1mTbiIgIVVVVqbGx0Wm7qzl16pTKy8udxiyWNoqJifmhLV9RQIC/07%2B%2Bwlf7BryRxXLl49hXj3P69p6%2BjQesS7p06aIuXbq0yNpjxozRmDFjHN8PGzZMH374oTZs2KDZs2dfdbvLP7bneuebKycnR8uXL3camzp1qqZPn25k/cuFhbVukXXdna/2DXiTyMjga8776nFO357PeMAaOHDgVecaGhq0e/du07uUJHXs2FFFRUWKjIyUv79/kxfa22w2RUdHKzo6WjU1NWpoaHBcBrTZbJLkmC8rK2uy7aV1myMlJUWDBw92GrNY2shmq/2h7V1RQIC/wsJaq7r6nBoaGo2u7c58tW/AG13tedFXj3P6Nt/394X4lmI8YKWkpDidCWpsbNSxY8eUn5%2BvZ5991sg%2B1q5dq7CwMA0bNswxdvjwYcXGxiooKEhdu3ZVcXGx%2BvbtK0mqqqpSWVmZrFarYmNj1djYqIMHDyouLk6SVFBQoNDQUHXq1ElWq1Vr165VfX29LBaLY/7SXyg2R0xMTJPLgeXlNaqvb5mDpaGhscXWdme%2B2jfgTb7vGPbV45y%2BPZ/xgPWLX/ziiuMFBQX6r//6LyP7qK%2Bv18KFC3XbbbepW7duysvL08cff%2Bz4S79x48Zp%2BfLluueee9ShQwctXLhQCQkJjpA0dOhQZWVlafHixbpw4YIWL16slJQUBQYGKjk5WcHBwcrOzta0adNUXFysdevWacmSJUZq92VDl%2BS7ugQAAG6IFnsN1uUSExM1Z86cZt/earVK%2Bi5MSdIf//hHSVJhYaHGjx%2Bv6upqTZ8%2BXTabTZ07d9arr76q%2BPh4SdLYsWNVXl6up556SrW1terXr5%2BWLl3qWPvXv/61MjMzNWTIEAUGBmrEiBGaMWOGJCkoKEhvvvmm5s%2Bfr/79%2Bys6OlqzZ8/Wvffea%2BR%2BAAAA3s/PfundN1vY//7v/2r8%2BPH65JNPbsTu3E55eY3xNS0Wf0VGBstmq/WIU6qcwQJwua2pSVcc97TnN1Po23zfN9989ffNbEnGz2CNHTu2yVhdXZ2%2B/vpr3XfffaZ3BwAA4HaMB6xOnTo1ebuDm266SY8%2B%2BqgeffRR07sDAABwO8YD1m9%2B8xvTSwIAAHgU4wFr/fr1CgwMbNZtR44caXr3AAAALmc8YL3wwgu6cOGCLn/tvJ%2Bfn9OYn58fAQsAAHgl4wHr9ddf13/%2B539qypQp6tKlixoaGnTo0CGtWLFCEyZMUP/%2B/U3vEgAAwK0YD1gvvvii3nrrLad3Mr/rrrv0q1/9Sk899ZS2bNliepcAAABuxfjHVh87dkxhYWFNxsPDw3XixAnTuwMAAHA7xgNW586dlZWV5fgAZUk6ffq0srOz1blzZ9O7AwAAcDvGLxFmZGRoypQpWrdunYKDg%2BXn56czZ84oODhYr776qundAQAAuB3jAat3797auXOn/vSnP%2BnkyZOy2%2B1q166dkpOTFRISYnp3AAAAbqdFPuy5devWGjJkiE6cOKHY2NiW2AUAAIDbMh6wzp8/r6ysLOXm5kqSioqKVF1drbS0NGVnZys01DUfuggAcD%2Be9iHwV/twauByxl/kvnTpUu3fv1%2BLFi2Sv//flr948aJeeukl07sDAABwO8YD1h//%2BEctWbJEDz74oONDn8PCwpSVlaUdO3aY3h0AAIDbMR6wTp06pU6dOjUZj46O1pkzZ0zvDgAAwO0YD1i33HKL9u3b12T8ww8/1K233mp6dwAAAG7H%2BIvcJ06cqH/5l3/RY489poaGBr399tsqKipSXl6e5s2bZ3p3AAAAbsd4wBo7dqwiIiK0atUqtWnTRm%2B%2B%2BaY6d%2B6sRYsW6cEHHzS9OwAAALdjPGBVVFTowQcfJEwBAACfZfQ1WI2NjRo0aJDsdrvJZQEAADyK0YDl7%2B%2BvAQMGaOvWrSaXBQAA8CjGLxG2b99eL774olasWKHbbrtNgYGBTvPZ2dmmdwkAAOBWjAesQ4cOqXPnzpIkm81menkAAAC3ZyxgzZw5U4sXL9aaNWscY6%2B%2B%2BqqmTp1qahcAAAAewdhrsLZv395kbMWKFaaWBwAA8BjGAtaV/nKQvyYEAAC%2ByFjAuvTBzt83BgAA4O2MfxYhAACAryNgAQAAGGbsrwgvXryoWbNmfe8Y74MFAAC8nbGA9Y//%2BI86derU944BAAB4O2MB6%2B/f/woAAMCX8RosAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIa5dcDatWuXBgwYoJkzZzaZ27x5sx544AFZrVYNHz5c%2Bfn5jjm73a7FixcrKSlJPXv21MSJE3X06FHHvM1m08yZM9W7d2/17dtX8%2BbN0/nz5x3zBw4c0NixY5WYmKjk5GStWrWqZRsFAABexW0D1m9/%2B1stXLhQt99%2Be5O5oqIipaena8aMGfriiy/0xBNPaOrUqTp58qQkafXq1crNzdXKlSuVn5%2Bv2NhYTZs2TXa7XZI0d%2B5cVVRUKC8vT5s2bdKBAwe0aNEiSdK5c%2Bc0adIk9e7dW7t379bSpUv12muvKS8v78Y1DwAAPJrbBqybbrpJGzZsuGLAys3NVXJysoYNG6ZWrVpp9OjR6tq1qzZu3ChJWr9%2BvZ555hn16NFDISEhSk9PV2lpqfbv36%2BKigrt2LFDc%2BbMUdu2bdWuXTulpqYqNzdXdXV12rlzpy5evKhZs2YpODhYvXr1UkpKinJycm70XQAAADyUxdUFXM2ECROuOldSUqLk5GSnsbi4OBUVFenChQsqLS1VQkKCYy4kJES33XabioqKdObMGVksFnXr1s0xHx8fr7Nnz%2Brw4cMqKSlR9%2B7dFRAQ4LT2%2BvXrm137qVOnVF5e7jRmsbRRTExMs9dojoAAf6d/AQAty2Jp2edbX31e98a%2B3TZgXYvNZlNERITTWHh4uA4dOqSqqirZ7XaFh4c3ma%2BsrFR4eLhCQkLk7%2B/vNCdJlZWVstlsTbaNiIhQVVWVGhsbnba7mpycHC1fvtxpbOrUqZo%2Bffp19dlcYWGtW2RdAICzyMjgG7IfX31e96a%2BPTJgXY2fn1%2BLzjdXSkqKBg8e7DRmsbSRzVZrZP1LAgL8FRbWWtXV59TQ0Gh0bQBAU6afxy/nq8/rLdn3jQrFl/PIgBUVFSWbzeY0ZrPZFBUVpcjISPn7%2B6uqqqrJfHR0tKKjo1VTU6OGhgbHZcBLa12aLysra7LtpXWbIyYmpsnlwPLyGtXXt8zB0tDQ2GJrAwD%2B5kY91/rq87o39e2RFzutVquKi4udxgoLC5WYmKigoCB17drVab6qqkplZWWyWq2Ki4tTY2OjDh486JgvKChQaGioOnXqJKvVqoMHD6q%2Bvt5pPjExseUbAwAAXsEjA9bo0aOVn5%2BvLVu26Pz581qzZo3Kyso0cuRISdK4ceO0cuVKffnll6qpqdHChQuVkJCgxMRERUZGaujQocrKytK3336r48ePa/HixUpJSVFgYKCSk5MVHBys7Oxs1dbW6vPPP9e6des0fvx4F3cNAAA8hZ/90ptDuRmr1SpJjjNJFst3VzMLCwslSXl5ecrOztaJEyfUpUsXZWRkqE%2BfPo7tly1bpnfffVe1tbXq16%2Bfnn/%2Bed1yyy2SpJqaGmVmZmr79u0KDAzUiBEjlJ6erqCgIEnSoUOHNH/%2BfBUXFys6OlqTJ0/WuHHjflQ/5eU1P2r7K7FY/BUZGSybrdYjTqkOXZL//TcCADe2NTWpRdf3tOd1U1qy75tvDjW6XnO5bcDyNgQsAhYAz0fAahneGLA88hIhAACAOyNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAF69ZhMAAARfUlEQVQAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhFlcXAACApxi6JN/VJVyXralJri7BZ3EGCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMs7i6APw4feZtc3UJAADgMpzBAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGeWzAGjRokBISEmS1Wh1fCxYskCTt3r1bDz/8sKxWq4YMGaL333/fadvVq1dr0KBBSkxM1OjRo1VcXOyYu3DhgubPn6%2B7775bd911l6ZPn67Kysob2hsAAPBsHhuwqqur9bvf/U6FhYWOr%2Beee07ffPONpkyZoscee0yff/655syZo4yMDBUUFEiS/vu//1tLlixRVlaWPv30U91777169tlndfbsWUnSK6%2B8on379ik3N1cfffSRzp8/r7lz57qyVQAA4GE8MmA1NDSotrZWYWFhTeY%2B%2BOAD3X777ZowYYJat26twYMH67777tOGDRskSevXr9djjz2me%2B65R23atNHUqVMlSdu3b1d9fb3%2B8Ic/KDU1VbGxsYqKilJ6erp27Nihb7755ob2CAAAPJfF1QX8ENXV1bLb7Vq2bJn27t0rSRo8eLDS09NVUlKi%2BPh4p9vHxcVp69atkqSSkhINGzbMMefn56cePXqoqKhIcXFxOnPmjNP2Xbp0UevWrVVcXKx27do1q75Tp06pvLzcacxiaaOYmJgf1O/VBAR4ZD4GANwgFotn/J649PvMm36veWTAqqurU8%2BePdW3b1%2B98MILOnXqlGbMmKHMzEzZbDZ1797d6fYRERGO11HZbDZFREQ4zYeHh6uyslI2m83x/d8LCwu7rtdh5eTkaPny5U5jU6dO1fTp05u9BgAAP1ZkZLCrS7guYWGtXV2CMR4ZsNq1a6d169Y5vg8JCVFaWpr%2B%2BZ//WX369LniNn5%2Bfk7/Xm3%2Bar5v/u%2BlpKRo8ODBTmMWSxvZbLXNXqM5vCnpAwDMM/17p6UEBPgrLKy1qqvPqaGh0ejargqZHhmwrqRjx45qbGyUv7%2B/qqqqnOZsNpuioqIkSZGRkVec79q1q6KjoyVJVVVVatOmjSTJbrerqqrKMdccMTExTS4HlpfXqL7e7A8NAADX4mm/dxoaGj2u5qvxyFMgBw4c0Isvvug0dvjwYQUFBenee%2B91etsFSSooKFBiYqIkyWq1qqioyDHX0NCgkpISJSYmKjY2VhEREU7bHzx4UHV1dUpISGjBjgAAgDfxyIAVHR2t9evXa8WKFaqrq9ORI0e0ZMkSjRs3To888oiOHz%2Bud955R%2BfPn9e2bdv08ccfKyUlRZI0duxY5ebm6tNPP1Vtba3%2B/d//Xa1atdLgwYMVEBCgMWPGaMmSJTp69KgqKiqUlZWlBx98UG3btnVx1wAAwFP42e12u6uL%2BCG%2B%2BOILLVq0SH/9618VGRmpYcOGafr06QoKCtKePXu0YMECff3112rfvr3S0tI0ZMgQx7bvvvuuVqxYoYqKCiUkJOjXv/61/uEf/kHSdy%2Bg/81vfqMPPvhADQ0NGjRokDIzMxUaGvqj6i0vr/lR21%2BJxeKvIYt2GV8XAOAdtqYmubqEZrFY/BUZGSybrdb4JcKbb/5xv79/KI8NWJ6GgAUAuNEIWK4LWB55iRAAAMCdec1fEQIAAGdDl%2BS7uoRm2/PCg64uwSjOYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAuoJjx47p6aefVq9evdS/f3%2B98soramxsdHVZAADAQ1hcXYC7sdvtmjZtmu6880796U9/0rfffqtJkyapbdu2evLJJ11dHgAA8ACcwbpMYWGhDh48qIyMDIWHh6tLly6aNGmS1q5d6%2BrSAACAh%2BAM1mVKSkrUoUMHRUREOMbi4%2BN15MgRnTlzRiEhId%2B7xqlTp1ReXu40ZrG0UUxMjNFaAwLIxwAA7%2BFNv9cIWJex2WwKDw93Grv0vc1ma1bAysnJ0fLly53Gpk2bpl/84hfmCtV3Qe6JWw4pJSXFeHhzZ6dOnVJOTo7P9S35bu/07Vt9S77buy/3vWzZMq/q23uiohtJSUnRe%2B%2B95/SVkpJifD/l5eVavnx5k7Nl3s5X%2B5Z8t3f69q2%2BJd/tnb69p2/OYF0mOjpaVVVVTmM2m02SFBUV1aw1YmJivCaBAwCA68cZrMtYrVadOHHCEaokqaCgQHfeeaeCg4NdWBkAAPAUBKzL9OjRQ4mJiVq4cKGqq6t18OBBrVixQuPHj3d1aQAAwEMEZGZmZrq6CHfzk5/8RB9%2B%2BKEWLFigLVu2aNy4cXr66addXdYVBQcH6%2B677/a5s2u%2B2rfku73Tt2/1Lflu7/TtHX372e12u6uLAAAA8CZcIgQAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIDlgY4dO6ann35avXr1Uv/%2B/fXKK6%2BosbHR1WXdEIMGDVJCQoKsVqvja8GCBa4uq0Xs2rVLAwYM0MyZM5vMbd68WQ888ICsVquGDx%2Bu/Px8F1TYMq7W96effqpu3bo5PfZWq1UFBQUuqtSsY8eOacqUKbr77rvVv39/zZ49W6dPn5YkHThwQGPHjlViYqKSk5O1atUqF1dr1tV6b2xsVPfu3Zsc82%2B99ZarSzbiyy%2B/1MSJE9WnTx/dc889mjFjhk6dOiVJ2r17tx5%2B%2BGFZrVYNGTJE77//vourNedqfR89evSKx/jWrVtdXfIPY4dHaWxstD/yyCP2WbNm2auqquxfffWVfdCgQfa3337b1aXdEL1797bv3bvX1WW0uBUrVtjvv/9%2B%2B9ixY%2B2pqalOc4WFhfb4%2BHj75s2b7efOnbOvW7fO3rNnT/v//d//uahac67Vd15env3%2B%2B%2B93UWUtb/jw4fZf/vKX9jNnzti/%2BeYb%2B6hRo%2Bxz5861nz171p6UlGR/6aWX7GfOnLH/%2Bc9/tvfp08f%2B4YcfurpkY67We1VVlb1r1672kydPurpE4y5cuGDv37%2B/ffny5fYLFy7YKyoq7D//%2Bc/tU6ZMsZ88edLes2dP%2B%2BrVq%2B1nz561f/TRR3ar1Wr/y1/%2B4uqyf7Rr9V1cXGyPj493dYnGcAbLwxQWFurgwYPKyMhQeHi4unTpokmTJmnt2rWuLq3FNTQ0qLa2VmFhYa4upcXddNNN2rBhg26//fYmc7m5uUpOTtawYcPUqlUrjR49Wl27dtXGjRtdUKlZ1%2Br79OnTCg0NdUFVLa%2BmpkYJCQlKS0tTcHCwYmJiNGrUKO3Zs0c7d%2B7UxYsXNWvWLAUHB6tXr15KSUlRTk6Oq8s24lq9V1dXS5JXHvPnzp3TzJkz9eyzzyooKEhRUVF64IEH9NVXX%2BmDDz7Q7bffrgkTJqh169YaPHiw7rvvPm3YsMHVZf9o1%2Brb245xApaHKSkpUYcOHRQREeEYi4%2BP15EjR3TmzBkXVtbyqqurZbfbtWzZMg0cOFADBw7U/PnzVVtb6%2BrSjJswYcJVn2hKSkoUHx/vNBYXF6eioqIbUVqLulbf1dXVOn36tB5//HH16dNHDz30kFeESkkKDQ1VVlaWoqOjHWMnTpzQrbfeqpKSEnXv3l0BAQGOOW95vKVr93769Gn5%2BfkpIyND99xzjwYNGqTs7GxdvHjRhRWbER4ertGjR8tischut%2Bvrr7/We%2B%2B9p6FDh3r1MX6tvqurq9XY2Khnn31Wffv21f33369Vq1bJbre7uuwfhIDlYWw2m8LDw53GLn1vs9lcUdINU1dXp549e6pv377atm2bfve73%2BnPf/6zMjMzXV3aDWWz2ZwCtvTdz0BlZaWLKroxQkJC1LFjR/3rv/6rPvnkE02dOlVz5szR7t27XV2acYWFhVqzZo2effbZKx7zERERqqqq8srXXv5975LUs2dPDRo0SNu3b9fSpUv1/vvv69VXX3VxleYcP35cCQkJGjZsmKxWq2bMmHHVx9ybjvEr9R0UFKQ777xT48eP165du/SrX/1Ky5cv99gzdwQseIx27dpp3bp1%2BvnPf66QkBDdcccdSktL06ZNm1RXV%2Bfq8lzOz8/P1SW0qDFjxmjVqlW666671KpVKw0bNkxDhgzx2Cffq9m7d6%2BefvppzZo1S/3793d1OTfU5b0nJCQoJydHw4cPV5s2bWS1WjV58mTl5ua6ulRjOnTooKKiIm3btk1ff/21/u3f/u2qt/WmY/xKfQ8aNEi///3vlZycrFatWikpKUkpKSke%2B3gTsDxMdHS0qqqqnMYunbmKiopyRUku1bFjRzU2NqqiosLVpdwwUVFRTc5W2mw2n338v/32W1eXYcyOHTs0efJkzZs3T0888YSkqx/zkZGR8vf3nqfwK/V%2BJR07dlRlZaXHXja6Ej8/P3Xq1EmzZ8/Wpk2bFBgYeMXH3NuO8cv7vtIZOk8%2Bxr3n6PQRVqtVJ06ccPoFW1BQoDvvvFPBwcEurKzlHThwQC%2B%2B%2BKLT2OHDhxUUFKSYmBgXVXXjWa1WFRcXO40VFhYqMTHRRRXdGGvXrtWWLVucxg4fPqzY2FgXVWTWvn37lJ6erqVLl%2BqRRx5xjFutVh08eFD19fWOsYKCAq96vK/W%2B%2B7du5tcDjx8%2BLA6dOjg8WdzPv/8c/3TP/2T0%2BN66ZLvgAEDmhzj3vKYX6vvzz77TL///e%2Bdbu/JxzgBy8P06NFDiYmJWrhwoaqrq3Xw4EGtWLFC48ePd3VpLS46Olrr16/XihUrVFdXpyNHjmjJkiUaN26c0wuAvd3o0aOVn5%2BvLVu26Pz581qzZo3Kyso0cuRIV5fWourr67Vw4UIVFRXp4sWL2rx5sz7%2B%2BGONGzfO1aX9aPX19crIyNCMGTOUlJTkNJecnKzg4GBlZ2ertrZWn3/%2BudatW%2Bc1x/y1eo%2BIiNDrr7%2BujRs3qr6%2BXoWFhXrrrbe8ove4uDidO3dO2dnZOnfunCorK7Vs2TL16dNHI0aM0PHjx/XOO%2B/o/Pnz2rZtmz7%2B%2BGOlpKS4uuwf7Vp9t2rVSi%2B//LLy8/NVX1%2Bv//mf/9GGDRs89vH2s3vTeVYfcfLkSc2fP1%2BfffaZgoOD9bOf/UzTpk1zdVk3xBdffKFFixbpr3/9qyIjIzVs2DBNnz5dQUFBri7NKKvVKkmO/%2BVZLBZJ352pkqS8vDxlZ2frxIkT6tKlizIyMtSnTx/XFGvQtfq22%2B16/fXXtWHDBtlsNnXu3FkzZszQvffe67J6TdmzZ4/Gjx9/xZ/jbdu26ezZs5o/f76Ki4sVHR2tyZMne0WwlL6/95KSEi1btkxlZWWKiYlRSkqKnnzySa%2B4PHrgwAG99NJLKioqksViUb9%2B/TR37ly1a9dOe/bs0YIFC/T111%2Brffv2SktL05AhQ1xdshHX6jsnJ0dvv/22vvnmG3Xs2FFPPfWURo0a5eqSfxACFgAAgGGe/18AAAAAN0PAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBh/w9BmFO9xCEOYQAAAABJRU5ErkJggg%3D%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-8805093359342398371">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">220</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">14.4</td>
            <td class="number">186</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">16.8</td>
            <td class="number">153</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">12.0</td>
            <td class="number">152</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">18.0</td>
            <td class="number">143</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">19.2</td>
            <td class="number">139</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">20.4</td>
            <td class="number">138</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">15.6</td>
            <td class="number">137</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">13.2</td>
            <td class="number">137</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">21.6</td>
            <td class="number">130</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (3489)</td>
            <td class="number">171210</td>
            <td class="number">99.1%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-8805093359342398371">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">220</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.01</td>
            <td class="number">6</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:3%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.02</td>
            <td class="number">6</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:3%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.03</td>
            <td class="number">3</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:2%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.04</td>
            <td class="number">4</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:2%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">34.95</td>
            <td class="number">11</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:84%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34.96</td>
            <td class="number">9</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:69%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34.97</td>
            <td class="number">7</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:54%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34.98</td>
            <td class="number">13</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34.99</td>
            <td class="number">9</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:69%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">earliest_cr_line_age<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>2173</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>1.3%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>4.7171e+17</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>94608000000000000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>2064268800000000000</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram5272898482068229639">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAA8VJREFUeJzt3M0rtHscx/EPBgsWKLFiqdRk5KHGSmzOiLIYsqEorBSK6EiKs1IWlI3/QMrDgoOshNRJIkTZSFEmsvIwuM7iLiW3L6MZ5tb7VTZz9a3ftXj3%2B5m5umIcx3EE4Ldiv3sBQDRzffcCwqHw739Dnvnvn78isBL8NOwggIFAAAOBAAYCAQwEAhgIBDAQCGAgEMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDD/ipQ2fwYse8BHsIICBQABD1B2xPnP0ASKFHQQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGAgEMBAIYCAQwBB1z2JFs1CfE%2BPx%2BD8fOwhgIBDAQCCAgUAAA4EABgIBDHzNG0G8OeXPxw4CGNhBogy7TnQhkB%2BAX/gjJ8ZxHOe7FwFEK/4HAQwEAhgIBDAQCGAgEIRsdXVVJSUl6ujoCHk2EAioqalJOTk5uru7e3Ftb29PDQ0NKigoUElJibq7u3V1dRWuZX8KgSAkExMTGhoaUnZ2dsizh4eH8vv9Sk1NfXXt8fFRLS0tys/P18bGhubn5xUIBDQwMBCGVX8egSAkiYmJmpqaejOQxcVF%2BXw%2B5eXlqbKyUnNzc8/XLi8vNTIyotra2ldzFxcXCgQCqqqqUkJCglJSUlReXq79/f2I3ctH8EMhQtLQ0PDmtePjY/X09Gh8fFxFRUXa3t5Wc3OzsrOzlZeXJ6/XK0na3Nx8NZuRkaHc3FxNTk6qo6NDNzc3Wl5eVmlpaaRu5UPYQRA2k5OTKisrk9frlcvlUmFhoXw%2Bn2ZmZt6djYmJ0ejoqFZWVuTxeOT1evX09KTOzs4vWPnbCARhc3JyooWFBbnd7ue/ubk5nZ2dvTt7f3%2Bv1tZWVVRUaGtrS2tra0pOTlZXV9cXrPxtHLEQNrGxsaqrq1N/f3/Is%2Bvr6zo9PVV7e7vi4uKUlJSktrY2VVdX6/LyUmlpaRFY8fvYQRA2WVlZOjo6evHZ%2Bfm5Hh8f3511HEdPT08vPgsGg5J%2BHb%2B%2BC4EgbPx%2Bv7a2tjQ9Pa1gMKiDgwPV1NRoaWnp3VmPx6OkpCSNjY3p9vZW19fXmpiYUH5%2B/m%2B/Fv4qPM2LkLjdbknSw8ODJMnl%2BnVK393dlSQtLCxodHRUp6enSk9PV319vRobGyVJfX19mp2dleM4CgaDSkhIkCQNDg6qurpaOzs7Gh4e1sHBgeLj41VcXKze3l5lZmZ%2B9W0%2BIxDAwBELMBAIYCAQwEAggIFAAAOBAAYCAQwEAhgIBDAQCGD4HxsgAm7EfBs8AAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives5272898482068229639,#minihistogram5272898482068229639"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives5272898482068229639">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles5272898482068229639"
                                                      aria-controls="quantiles5272898482068229639" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram5272898482068229639" aria-controls="histogram5272898482068229639"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common5272898482068229639" aria-controls="common5272898482068229639"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme5272898482068229639" aria-controls="extreme5272898482068229639"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles5272898482068229639">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>94608000000000000</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>1.7876e+17</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>3.2072e+17</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>4.2863e+17</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>5.813e+17</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>9.0729e+17</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>2064268800000000000</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>1969660800000000000</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>2.6058e+17</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>2.2491e+17</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.4768</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>1.7574</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>4.7171e+17</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>1.7187e+17</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>1.1264</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>6909064824910512128</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>5.0585e+34</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram5272898482068229639">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzs3XtUVXXi//8XFxUEgQNGpZE1%2BDG5ndTUJFODxlJL82MoXmasycwcDHVqZDQiGi/MfJSkcmrkU6N8avXRlFxlpVlpN0c/NrkaBIwpR/NCCsFBBEXksH9/%2BPN8PWoJtZVzds/HWizXeb/32Xu/zgVe7r05%2BBiGYQgAAACm8W3rHQAAALAaChYAAIDJKFgAAAAmo2ABAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJiMggUAAGAyChYAAIDJKFgAAAAmo2ABAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJiMggUAAGAyChYAAIDJKFgAAAAmo2ABAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJiMggUAAGAyChYAAIDJKFgAAAAm82/rHfi5qKqqU3h4kKqr69XcbLT17pjK19fHstkka%2Bcjm/eycj4rZ5Osnc8Ts11xRac22S5HsC4TX18f%2Bfj4yNfXp613xXRWziZZOx/ZvJeV81k5m2TtfFbO1loULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJP5t/UO4OdjeN7Wtt6FVtkwa2Bb7wIAwEtxBAsAAMBkHlmwPvvsMyUkJLh9xcfH64YbbpAkbdu2TaNGjVJCQoKGDh2qN9980%2B3%2BBQUFSkpKkt1u19ixY1VSUuKaO3nypLKystS/f3/17t1b6enpqq6uds0fPHhQU6ZMUa9evZSYmKjFixerubn58gQHAACW4JEFq1%2B/ftq1a5fb129/%2B1sNHz5cR44c0fTp05WSkqIdO3Zo7ty5yszMVFFRkSTpvffeU15ennJycrR9%2B3YNGTJE06ZN0/HjxyVJixcv1s6dO1VYWKgPPvhADQ0NmjdvniTJMAzNmDFDNptNH330kV555RVt2LBBBQUFbfZYAAAA7%2BORBetc5eXlKigo0Jw5c7R%2B/Xp169ZNkydPVmBgoJKTk3X77bdr7dq1kqQ1a9YoJSVFAwYMUMeOHZWWliZJ2rx5s5qamrRu3TrNmjVLUVFRCg8PV0ZGhrZs2aIjR45o165dKisrU2ZmpkJDQxUdHa2pU6dq1apVbRkfAAB4Ga8oWEuXLlVKSoq6dOmi0tJSxcXFuc3HxsaquLhYks6b9/HxUUxMjIqLi7V//37V1dW5zUdHRyswMFAlJSUqLS1V165dFRYW5pqPi4vTvn37VFdXd4lTAgAAq/D43yL85ptv9P777%2BuDDz6QJDkcDvXs2dNtmbCwMNd1VA6Hw60gSVJoaKiqq6vlcDhct88WEhLimj937sxth8Oh4ODgFu1zRUWFKisr3cYCAoJlswXJz88rOm2rnMlktWz%2B/u65rJZPIps3s3I%2BK2eTrJ3Pytlay%2BML1iuvvKKhQ4cqPDz8B5fz8fFx%2B/f75i92fzOsXr1ay5YtcxtLS0tTenq6QkICTduOp7FaNpstyO221fKdjWzey8r5rJxNsnY%2BK2drKY8vWO%2B%2B%2B66ys7Ndt8PDw1VTU%2BO2jMPhcBUwm812wfkePXooIiJCklRTU6OOHTtKOn1he01NjSIiIuR0Oi943zPbbanU1FQlJye7jQUEnD76VVt7Qk6ntX4r0c/PVyEhgZbL5nDUS7JuPols3szK%2BaycTbJ2Pk/Mdu5/li8Xjy5YX331lSoqKtS/f3/XWEJCgl5//XW35YqKimS3213zxcXFGj16tCTJ6XSqtLRUKSkpioqKUlhYmEpKStSlSxdJUllZmRobGxUfH6/KykqVl5fL4XDIZrO51t29e3cFBbX8CYqMjFRkZKTb2Jkf1k5ns5qaPONFZzarZTs3i9XynY1s3svK%2BaycTbJ2PitnaymPPkm6e/duXX311W7XPo0cOVKHDh3SypUr1dDQoI0bN%2Brjjz9WamqqJGn8%2BPEqLCzU9u3bVV9fr6effloBAQFKTk6Wn5%2Bfxo0bp7y8PB04cEBVVVXKycnRsGHD1LlzZ8XExMhut2vBggWqra1VWVmZ8vPzNWnSpLZ6CAAAgBfy6CNYlZWV512wHhERoeXLl2v%2B/PnKzc1Vly5dlJub67rwffDgwZozZ47mzp2rqqoqxcfHKz8/Xx06dJAkPfLII6qvr9eYMWPkdDqVlJTkdgrymWeeUVZWlgYNGqSgoCBNnDhREydOvGyZAQCA9/MxDMNo6534OXA46mWzBcnhqLfcYVN/f98WZfPWv0XY0nzeiGzey8r5rJxNsnY%2BT8x2xRWd2mS7Hn2KEAAAwBtRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZB5dsJ5//nndeuut6t27t%2B6//34dOHBAkrRt2zaNGjVKCQkJGjp0qN588023%2BxUUFCgpKUl2u11jx45VSUmJa%2B7kyZPKyspS//791bt3b6Wnp6u6uto1f/DgQU2ZMkW9evVSYmKiFi9erObm5ssTGAAAWILHFqxXX31Vmzdv1urVq/Xhhx/q6quv1sqVK3XkyBFNnz5dKSkp2rFjh%2BbOnavMzEwVFRVJkt577z3l5eUpJydH27dv15AhQzRt2jQdP35ckrR48WLt3LlThYWF%2BuCDD9TQ0KB58%2BZJkgzD0IwZM2Sz2fTRRx/plVde0YYNG1RQUNBmjwMAAPA%2BHluwXnrpJT3xxBPq2rWrQkNDlZOToyeeeELr169Xt27dNHnyZAUGBio5OVm333671q5dK0las2aNUlJSNGDAAHXs2FFpaWmSpM2bN6upqUnr1q3TrFmzFBUVpfDwcGVkZGjLli06cuSIdu3apbKyMmVmZio0NFTR0dGaOnWqVq1a1ZYPBQAA8DL%2Bbb0DF3LkyBEdPnxY33zzjX7/%2B9/r6NGjSkxM1JNPPqnS0lLFxcW5LR8bG6sNGzZIkkpLSzVixAjXnI%2BPj2JiYlRcXKzY2FjV1dW53T86OlqBgYEqKSlRRUWFunbtqrCwMNd8XFyc9u3bp7q6OgUHB7do/ysqKlRZWek2FhAQLJstSH5%2BHttpf7QzmayWzd/fPZfV8klk82ZWzmflbJK181k5W2t5ZME6fPiwfHx89P7772v16tVqaGhQenq6nnjiCdXX16tnz55uy4eFhbmuo3I4HG4FSZJCQ0NVXV0th8Phun22kJAQ1/y5c2duOxyOFhes1atXa9myZW5jaWlpSk9PV0hIYIvW4Y2sls1mC3K7bbV8ZyOb97JyPitnk6ydz8rZWsojC9apU6d06tQp/f73v5fNZpMkpaena%2BrUqUpMTLzgfXx8fNz%2B/b7573Ox%2BdZITU1VcnKy21hAwOlyVlt7Qk6ntS6a9/PzVUhIoOWyORz1kqybTyKbN7NyPitnk6ydzxOznfuf5cvFIwvWmSNQZx8x6tq1qwzDUFNTk2pqatyWdzgcCg8PlyTZbLYLzvfo0UMRERGSpJqaGnXs2FHS6Qvba2pqFBERIafTecH7SnKtvyUiIyMVGRl5znpO/7B2OpvV1OQZLzqzWS3buVmslu9sZPNeVs5n5WyStfNZOVtLeeRJ0m7duik4ONjt4xUOHTokf39/3XbbbW7jklRUVCS73S5JSkhIUHFxsWvO6XSqtLRUdrtdUVFRCgsLc7t/WVmZGhsbFR8fr4SEBJWXl7tK1Zl1d%2B/eXUFBbdOAAQCA9/HIgtWuXTuNHTtWS5Ys0eHDh1VZWam//OUvuueeezR69GgdOnRIK1euVENDgzZu3KiPP/5YqampkqTx48ersLBQ27dvV319vZ5%2B%2BmkFBAQoOTlZfn5%2BGjdunPLy8nTgwAFVVVUpJydHw4YNU%2BfOnRUTEyO73a4FCxaotrZWZWVlys/P16RJk9r4EQEAAN7EI08RStLvfvc7/elPf9KoUaPk6%2BurpKQkzZs3T8HBwVq%2BfLnmz5%2Bv3NxcdenSRbm5ua4L3wcPHqw5c%2BZo7ty5qqqqUnx8vPLz89WhQwdJ0iOPPKL6%2BnqNGTNGTqdTSUlJys7Odm33mWeeUVZWlgYNGqSgoCBNnDhREydObIuHAAAAeCkfwzCMtt6JnwOHo142W5AcjnrLnZf29/dtUbbheVsv4179dBtmDZTU8nzeiGzey8r5rJxNsnY%2BT8x2xRWd2mS7HnmKEAAAwJtRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZB5bsJKSkhQfH6%2BEhATX1/z58yVJ27Zt06hRo5SQkKChQ4fqzTffdLtvQUGBkpKSZLfbNXbsWJWUlLjmTp48qaysLPXv31%2B9e/dWenq6qqurXfMHDx7UlClT1KtXLyUmJmrx4sVqbm6%2BPKEBAIAleGzBqq2t1f/8z/9o165drq8nnnhCR44c0fTp05WSkqIdO3Zo7ty5yszMVFFRkSTpvffeU15ennJycrR9%2B3YNGTJE06ZN0/HjxyVJixcv1s6dO1VYWKgPPvhADQ0NmjdvniTJMAzNmDFDNptNH330kV555RVt2LBBBQUFbfY4AAAA7%2BORBcvpdKq%2Bvl4hISHnza1fv17dunXT5MmTFRgYqOTkZN1%2B%2B%2B1au3atJGnNmjVKSUnRgAED1LFjR6WlpUmSNm/erKamJq1bt06zZs1SVFSUwsPDlZGRoS1btujIkSPatWuXysrKlJmZqdDQUEVHR2vq1KlatWrVZc0PAAC8m39b78CF1NbWyjAMPffcc/r8888lScnJycrIyFBpaani4uLclo%2BNjdWGDRskSaWlpRoxYoRrzsfHRzExMSouLlZsbKzq6urc7h8dHa3AwECVlJSooqJCXbt2VVhYmGs%2BLi5O%2B/btU11dnYKDg1u0/xUVFaqsrHQbCwgIls0WJD8/j%2By0P8mZTFbL5u/vnstq%2BSSyeTMr57NyNsna%2BaycrbU8smA1NjbqxhtvVL9%2B/bRw4UJVVFRo5syZys7OlsPhUM%2BePd2WDwsLc11H5XA43AqSJIWGhqq6uloOh8N1%2B2whISGu%2BXPnztx2OBwtLlirV6/WsmXL3MbS0tKUnp6ukJDAFq3DG1ktm80W5HbbavnORjbvZeV8Vs4mWTuflbO1lEcWrCuvvFKvvfaa63ZwcLAee%2BwxPfzww%2Brbt%2B8F7%2BPj4%2BP27/fNf5%2BLzbdGamqqkpOT3cYCAk6Xs9raE3I6rXXRvJ%2Bfr0JCAi2XzeGol2TdfBLZvJmV81k5m2TtfJ6Y7dz/LF8uHlmwLuSaa65Rc3OzfH19VVNT4zbncDgUHh4uSbLZbBec79GjhyIiIiRJNTU16tixo6TTF7bX1NQoIiJCTqfzgveV5Fp/S0RGRioyMvKc9Zz%2BYe10NqupyTNedGazWrZzs1gt39nI5r2snM/K2SRr57NytpbyyJOku3fv1qJFi9zG9u7dq/bt22vIkCFuH7sgSUVFRbLb7ZKkhIQEFRcXu%2BacTqdKS0tlt9sVFRWlsLAwt/uXlZWpsbHR9ZEQ5eXlrlJ1Zt3du3dXUFDbNGAAAOB9PLJgRUREaM2aNcrPz1djY6P27dunvLw8TZgwQffcc48OHTqklStXqqGhQRs3btTHH3%2Bs1NRUSdL48eNVWFio7du3q76%2BXk8//bQCAgKUnJwsPz8/jRs3Tnl5eTpw4ICqqqqUk5OjYcOGqXPnzoqJiZHdbteCBQtUW1ursrIy5efna9KkSW38iAAAAG/ikacIIyMjlZ%2BfryVLluiFF16QzWbTiBEjlJ6ervbt22v58uWaP3%2B%2BcnNz1aVLF%2BXm5roufB88eLDmzJmjuXPnqqqqSvHx8crPz1eHDh0kSY888ojq6%2Bs1ZswYOZ1OJSUlKTs727XtZ555RllZWRo0aJCCgoI0ceJETZw4sS0eBgAA4KV8DMMw2nonfg4cjnrZbEFyOOotd17a39%2B3RdmG5229jHv1022YNVBSy/N5I7J5Lyvns3I2ydr5PDHbFVd0apPteuQpQgAAAG9GwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJOZXrDq6urMXiUAAIBXMb1g3XrrrZo7d6527txp9qoBAAC8gukFKzs7W5WVlZo8ebJGjBihFStWqLq62uzNAAAAeCzTC9bo0aP14osv6uOPP9aECRP07rvv6rbbbtOsWbO0detWszcHAADgcS7ZRe7h4eH69a9/rVWrViknJ0dbt27Vgw8%2BqGHDhmnjxo2XarMAAABtzv9Srbiqqkqvv/66Xn/9de3fv1%2B33nqrxo0bp8rKSmVnZ2v//v166KGHLtXmAQAA2ozpBeuTTz7RmjVrtHnzZtlsNt17770aN26cunTp4lomNjZWU6dOpWABAABLMr1gPfTQQ7rlllu0dOlSJScny8/P77xl7Ha7IiMjzd40AACARzC9YG3atElRUVFqbGx0lav6%2BnoFBQW5Lbd%2B/XqzNw0AAOARTL/I3cfHRyNHjtTmzZtdY6tXr9Zdd92lAwcOmL05AAAAj2N6wVq4cKF%2B8YtfqE%2BfPq6xUaNGKSEhQQsXLjR7cwAAAB7H9FOEn3/%2BuT788EN17NjRNda5c2c98cQTSkpKMntzAAAAHsf0I1iGYaipqem88RMnTqi5udnszQEAAHgc0wvWwIEDNWfOHJWWlqq2tlY1NTX6/PPPNXv2bN16661mbw4AAMDjmH6K8IknntBjjz2mMWPGyMfHxzV%2B8803KzMz0%2BzNAQAAeBzTC1ZERIRWrFihPXv2aN%2B%2BfTIMQ9dff72io6PN3hQAAIBHumR/Kic6OlpRUVGu242NjZKk9u3bX6pNAgAAeATTC9YXX3yh7Oxsff3113I6nefN79692%2BxNAgAAeBTTC1Z2drY6deqkxx9/XAEBAWavHgAAwOOZXrD27dun//u//1OHDh1MW%2BeiRYtUUFCgsrIySdK2bduUk5OjvXv36qqrrtIjjzyiUaNGuZYvKCjQypUrVVVVpRtuuEHZ2dmKi4uTJJ08eVILFy7Uxo0bderUKQ0aNEjZ2dkKDw%2BXJB08eFBPPvmkPv/8cwUGBmrMmDF69NFH5etr%2Bi9cAgAAizK9NXTp0kWnTp0ybX27d%2B/WG2%2B84bp95MgRTZ8%2BXSkpKdqxY4fmzp2rzMxMFRUVSZLee%2B895eXlKScnR9u3b9eQIUM0bdo0HT9%2BXJK0ePFi7dy5U4WFhfrggw/U0NCgefPmSTr9GV4zZsyQzWbTRx99pFdeeUUbNmxQQUGBaXkAAID1mV6wHnvsMeXk5Kiuru4nr6u5uVlPPvmk7r//ftfY%2BvXr1a1bN02ePFmBgYFKTk7W7bffrrVr10qS1qxZo5SUFA0YMEAdO3ZUWlqaJGnz5s1qamrSunXrNGvWLEVFRSk8PFwZGRnasmWLjhw5ol27dqmsrEyZmZkKDQ1VdHS0pk6dqlWrVv3kLAAA4OfD9FOEy5Yt08GDB7Vu3TrZbDa3z8KSpE8//bTF61q1apUCAgI0cuRI5eXlSZJKS0tdp/vOiI2N1YYNG1zzI0aMcM35%2BPgoJiZGxcXFio2NVV1dndv9o6OjFRgYqJKSElVUVKhr164KCwtzzcfFxWnfvn2qq6tTcHBwyx8IAADws2V6wRo0aJDatWv3k9fz3Xff6S9/%2BYtefvllt3GHw6GePXu6jYWFham6uto1f3ZBkqTQ0FBVV1fL4XC4bp8tJCTENX/u3JnbDoejxQWroqJClZWVbmMBAcGy2YLk52e9a7nOZLJaNn9/91xWyyeRzZtZOZ%2BVs0nWzmflbK1lesGaPXu2KevJycnRuHHj9Itf/EIHDx686PJnjpSde8Ts3PmL3d8Mq1ev1rJly9zG0tLSlJ6erpCQQNO242msls1mC3K7bbV8ZyOb97JyPitnk6ydz8rZWuqSfNDoP//5T61atUrl5eUqKChQc3Oz3n33XQ0fPrxF99%2B2bZuKi4u1aNGi8%2BbCw8NVU1PjNuZwOFy/BWiz2S4436NHD0VEREiSampq1LFjR0mnL2yvqalRRESEnE7nBe97ZrstlZqaquTkZLexgIDTR79qa0/I6bTWH7328/NVSEig5bI5HPWSrJtPIps3s3I%2BK2eTrJ3PE7Od%2B5/ly8X0gvXBBx8oPT1dgwYN0s6dOyVJhw8f1hNPPKG6ujqNHTv2out48803dfjwYQ0ePFjS6RIknf57hlOmTNFbb73ltnxRUZHsdrskKSEhQcXFxRo9erQkyel0qrS0VCkpKYqKilJYWJhKSkrUpUsXSVJZWZkaGxsVHx%2BvyspKlZeXy%2BFwyGazudbdvXt3BQW1/AmKjIxUZGSk29iZH9ZOZ7OamjzjRWc2q2U7N4vV8p2NbN7LyvmsnE2ydj4rZ2sp0wvWX//6Vy1evFgjRoxwlZ4uXbromWee0aJFi1pUsP7whz9o5syZrtuHDx9Wamqq3njjDTU3N2v58uVauXKlxo8frw8//FAff/yxXnvtNUnS%2BPHjNXPmTP3yl79UQkKCnn/%2BeQUEBCg5OVl%2Bfn4aN26c8vLy1LNnT3Xs2FE5OTkaNmyYOnfurM6dO8tut2vBggV68skn9e233yo/P1%2B//e1vzX6Y4AWG521t611olQ2zBrb1LgAA/n%2BmF6y9e/fqjjvukOR%2BXVNiYqIOHTrUonWEhoa6XWze1NQkSbrqqqskScuXL9f8%2BfOVm5urLl26KDc313Xh%2B%2BDBgzVnzhzNnTtXVVVVio%2BPV35%2BvuuDTx955BHV19drzJgxcjqdSkpKUnZ2tmtbzzzzjLKysjRo0CAFBQVp4sSJmjhx4o9/QAAAwM%2BO6QWrXbt2qqmpUefOnd3G9%2B3b96P/dM4111zj%2BhR3Serbt6/bh4%2Bea8KECZowYcIF59q3b6%2BsrCxlZWVdcP6qq65Sfn7%2Bj9pPAAAA6RJ80Ohtt92mzMxM7dmzR9Lpi8Q/%2BeQTzZo1S0lJSWZvDgAAwOOYXrDmzp0rwzB011136eTJk7rllls0depUXX311frDH/5g9uYAAAA8jumnCENCQrR8%2BXLt2bNH%2B/btk4%2BPj66//npdf/31Zm8KAADAI12Sz8GSTv8Jmujo6Eu1egAAAI9lesG69dZbv3fO6XRq27ZtZm8SAADAo5hesFJTU90%2BnqG5uVkHDx7U1q1bNW3aNLM3BwAA4HFML1iPPPLIBceLior06quvmr05AAAAj3PZ/ty13W7Xrl27LtfmAAAA2sxlK1jffPONjh49erk2BwAA0GZMP0U4fvz488YaGxv173//W7fffrvZmwMAAPA4phes6667zu0id0nq0KGD7r33Xt17771mbw4AAMDjmF6w/vSnP5m9SgAAAK9iesFas2aN2rVr16JlR48ebfbmAQAA2pzpBWvhwoU6efKkDMNwG/fx8XEb8/HxoWABAABLMr1gvfDCC3rllVc0ffp0RUdHy%2Bl06quvvlJ%2Bfr4mT56sxMREszcJAADgUUwvWIsWLdJLL72kyMhI11jv3r315JNP6oEHHtA777xj9iYBAAA8iukF6%2BDBgwoJCTlvPDQ0VOXl5WZv7mdveN7Wtt4FAABwDtM/aPT6669XTk6OHA6Ha%2Bzo0aPKzc3V9ddfb/bmAAAAPI7pR7AyMzM1ffp0vfbaawoKCpKPj4/q6uoUFBSkv/zlL2ZvDgAAwOOYXrD69OmjDz/8UB999JEOHz4swzB05ZVXavDgwQoODjZ7cwAAAB7H9IIlSYGBgRo6dKjKy8sVFRV1KTYBAADgsUy/BquhoUFPPvmkbrzxRg0fPlySVFtbq4ceekjHjh0ze3MAAAAex/SC9eyzz%2BqLL77QkiVL5Ov7/1Z/6tQp/fnPfzZ7cwAAAB7H9IL1/vvvKy8vT8OGDXP90eeQkBDl5ORoy5YtZm8OAADA45hesCoqKnTdddedNx4REaG6ujqzNwcAAOBxTC9YV111lXbu3Hne%2BLvvvqurr77a7M0BAAB4HNN/i/D%2B%2B%2B/Xb3/7W6WkpMjpdOpvf/ubiouLtWnTJj3%2B%2BONmbw4AAMDjmF6wxo8fr7CwMK1YsUIdO3bU8uXLdf3112vJkiUaNmyY2ZsDAADwOKYXrKqqKg0bNowyBQAAfrZMvQarublZSUlJMgzDzNUCAAB4FVMLlq%2Bvr2655RZt2LDBzNUCAAB4FdNPEXbp0kWLFi1Sfn6%2Brr32WrVr185tPjc31%2BxNAgAAeBTTP6bhq6%2B%2B0vXXX69OnTrJ4XCooqLC7aulvvzyS91///3q27evBgwYoJkzZ7ruv23bNo0aNUoJCQkaOnSo3nzzTbf7FhQUKCkpSXa7XWPHjlVJSYlr7uTJk8rKylL//v3Vu3dvpaenq7q62jV/8OBBTZkyRb169VJiYqIWL16s5ubmn/ioAACAnxPTCtbs2bMlSS%2B//LLra8CAAW63X3755Ratq7GxUQ888ID69eunv//973rnnXdUXV2sjPQSAAAgAElEQVSt7OxsHTlyRNOnT1dKSop27NihuXPnKjMzU0VFRZKk9957T3l5ecrJydH27ds1ZMgQTZs2TcePH5ckLV68WDt37lRhYaE%2B%2BOADNTQ0aN68eZIkwzA0Y8YM2Ww2ffTRR3rllVe0YcMGFRQUmPUwAQCAnwHTCtbmzZvPG8vPz/9R6zpx4oRmz56tadOmqX379goPD9edd96pr7/%2BWuvXr1e3bt00efJkBQYGKjk5WbfffrvWrl0rSVqzZo1SUlI0YMAAdezYUWlpaa79a2pq0rp16zRr1ixFRUUpPDxcGRkZ2rJli44cOaJdu3aprKxMmZmZCg0NVXR0tKZOnapVq1b9%2BAcGAAD87Jh2DdaFfnPwx/42YWhoqMaOHetax969e/X6669r%2BPDhKi0tVVxcnNvysbGxrgvrS0tLNWLECNecj4%2BPYmJiVFxcrNjYWNXV1bndPzo6WoGBgSopKVFFRYW6du2qsLAw13xcXJz27dunuro6BQcHt2j/KyoqVFlZ6TYWEBAsmy1Ifn6mn5UFJEn%2B/q1/bZ15PVrxdWnlbJK181k5m2TtfFbO1lqmFawzf9j5YmOtcejQId1xxx1yOp1KTU3VzJkzNWXKFPXs2dNtubCwMNd1VA6Hw60gSacLW3V1tRwOh%2Bv22UJCQlzz586due1wOFpcsFavXq1ly5a5jaWlpSk9PV0hIYEtWgfQWjZb0I%2B%2Br5Vfl1bOJlk7n5WzSdbOZ%2BVsLWX6bxGaqWvXriouLtY333yjJ554Qr///e%2B/d9kzZe77St3Fyt5PLYNnS01NVXJysttYQMDpclZbe0JOJxfNw3wOR32r7%2BPn56uQkEBLvi6tnE2ydj4rZ5Osnc8Ts/2U/3z%2BFB5dsKTTxee6667TnDlzlJKSoiFDhqimpsZtGYfDofDwcEmSzWa74HyPHj0UEREhSaqpqVHHjh0lnT4FWVNTo4iICDmdzgveV5Jr/S0RGRmpyMjIc9Zz%2Boef09mspibPeNHBWn7K68rKr0srZ5Osnc/K2SRr57NytpYyrWCdOnVKjz766EXHWvI5WDt27NC8efO0ceNG%2Bfuf3sUzH5Vwyy236PXXX3dbvqioSHa7XZKUkJCg4uJijR49WpLkdDpVWlqqlJQURUVFKSwsTCUlJerSpYskqaysTI2NjYqPj1dlZaXKy8vlcDhks9lc6%2B7evbuCgtqmAQMAAO9j2lVoN91003mfeXWhsZaIjY3ViRMnlJubqxMnTqi6ulrPPfec%2Bvbtq5EjR%2BrQoUNauXKlGhoatHHjRn388cdKTU2VdPqPTRcWFmr79u2qr6/X008/rYCAACUnJ8vPz0/jxo1TXl6eDhw4oKqqKuXk5GjYsGHq3LmzYmJiZLfbtWDBAtXW1qqsrEz5%2BfmaNGmSWQ8TAAD4GTDtCFZLP%2BOqJYKDg/Xiiy/qz3/%2BswYNGiR/f3/dfPPNWrhwoSIiIrR8%2BXLNnz9fubm56tKli3Jzc10Xvg8ePFhz5szR3LlzVVVVpfj4eOXn56tDhw6SpEceeUT19fUaM2aMnE6nkpKSlJ2d7dr2M888o6ysLA0aNEhBQUGaOHGiJk6caFo2AABgfT4Gf5n5snA46mWzBcnhqDf1vPTwvK2mrQvebcOsga2%2Bj7%2B/7yV5XXoCK2eTrJ3Pytkka%2BfzxGxXXNGpTbbLB1UAAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJiMggUAAGAyChYAAIDJKFgAAAAmo2ABAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJiMggUAAGAyChYAAIDJKFgAAAAmo2ABAACYjIIFAABgMgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMk8tmAdPHhQ06dPV//%2B/ZWYmKg5c%2Bbo6NGjkqTdu3dr/PjxstvtGjx4sFasWOF237ffflt33nmnEhISdPfdd2vr1q2uOcMwtHTpUg0cOFA33nij7r//fh04cMA173A4NHv2bPXp00f9%2BvXT448/roaGhssTGgAAWILHFqzp06crLCxMW7Zs0RtvvKE9e/bov/7rv3TixAlNnTpVffr00bZt2/Tss8/q%2Beef16ZNmyRJxcXFysjI0MyZM/XZZ5/pvvvuU1pamg4fPixJKigoUGFhoV588UVt3bpVUVFRmjFjhgzDkCTNmzdPVVVV2rRpk9566y3t3r1bS5YsabPHAQAAeB%2BPLFjHjh1TfHy8HnvsMQUFBSkyMlJjxozRP/7xD3344Yc6deqUHn30UQUFBalXr15KTU3V6tWrJUmFhYUaPHiwRowYoYCAAI0dO1Y9evTQG2%2B8IUlas2aNHnzwQcXExCg4OFgZGRnas2ePvvjiC1VVVWnLli2aO3euOnfurCuvvFKzZs1SYWGhGhsb2/IhAQAAXsQjC1anTp2Uk5OjiIgI11h5ebmuvvpqlZaWqmfPnvLz83PNxcbGqri4WJJUWlqquLg4t/WdmT958qT27Nmj%2BPh411xwcLCuvfZaFRcXq7S0VP7%2B/rrhhhtc83FxcTp%2B/Lj27t17qeICAACL8W/rHWiJXbt26eWXX9by5cv19ttvKzQ01G0%2BLCxMNTU1am5ulsPhUFhYmNt8aGiovvrqK9XU1MgwjPPuHxoaqurqaoWGhio4OFi%2Bvr5uc5JUXV3d4v2tqKhQZWWl21hAQLBstiD5%2BXlkp4UF%2BPu3/rV15vVoxdellbNJ1s5n5WyStfNZOVtreXzB%2BvzzzzV9%2BnQ9%2BuijSkxM1Ntvv/2j1uPj43NJ58%2B2evVqLVu2zG0sLS1N6enpCgkJbPF6gNaw2YJ%2B9H2t/Lq0cjbJ2vmsnE2ydj4rZ2spjy5YW7Zs0WOPPaasrCzdc889kqSIiAjt37/fbTmHwyGbzSZfX1%2BFh4fL4XCcNx8eHu5apqam5rz5iIgIRURE6NixY3I6na5TkGfWdfbpyotJTU1VcnKy21hAQLAkqbb2hJzO5havC2gph6O%2B1ffx8/NVSEigJV%2BXVs4mWTuflbNJ1s7nidl%2Byn8%2BfwqPLVg7d%2B5URkaGnn32WQ0cONA1npCQoFWrVqmpqUn%2B/qd3v6ioSHa73TVfUlLitq5du3bprrvuUvv27dWjRw%2BVlJSoX79%2BkqSamhrt379fCQkJioqKUnNzs8rKyhQbG%2Btad6dOnXTddde1eN8jIyMVGRnpNnbmh5/T2aymJs940cFafsrrysqvSytnk6ydz8rZJGvns3K2lvLIk6RNTU3KzMzUzJkz3cqVJA0ePFhBQUHKzc1VfX29duzYoddee02TJk2SJI0dO1Zbt27VO%2B%2B8o4aGBr388svav3%2B/Ro8eLUmaMGGCXnzxRX355Zc6duyYFixYoPj4eNntdtlsNg0fPlw5OTn67rvvdOjQIS1dulSpqalq167dZX8cAACAd/IxznwAlAf5xz/%2BoUmTJql9%2B/bnzW3cuFHHjx9XVlaWSkpKFBERoYceekgTJkxwLbNp0ybl5uaqvLxc0dHRyszMVN%2B%2BfV3zzz33nP73f/9X9fX1uvnmm/XHP/5RV111laTTHxGRnZ2tzZs3q127dho5cqQyMjIuuC%2Bt4XDUy2YLksNRb2qrH5639eIL4Wdhw6yBF1/oHP7%2BvpfkdekJrJxNsnY%2BK2eTrJ3PE7NdcUWnNtmuRxYsK6Jg4VKjYLmzcjbJ2vmsnE2ydj5PzNZWBcsjTxECAAB4MwoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMk89pPcAbSON31kx4/5SAkA8CYcwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADCZRxesTz75RLfccotmz5593tzbb7%2BtO%2B%2B8UwkJCbr77ru1detW15xhGFq6dKkGDhyoG2%2B8Uffff78OHDjgmnc4HJo9e7b69Omjfv366fHHH1dDQ4Nrfvfu3Ro/frzsdrsGDx6sFStWXNqgAADAUjy2YP33f/%2B3FixYoG7dup03V1xcrIyMDM2cOVOfffaZ7rvvPqWlpenw4cOSpIKCAhUWFurFF1/U1q1bFRUVpRkzZsgwDEnSvHnzVFVVpU2bNumtt97S7t27tWTJEknSiRMnNHXqVPXp00fbtm3Ts88%2Bq%2Beff16bNm26fOEBAIBX89iC1aFDB61du/aCBauwsFCDBw/WiBEjFBAQoLFjx6pHjx564403JElr1qzRgw8%2BqJiYGAUHBysjI0N79uzRF198oaqqKm3ZskVz585V586ddeWVV2rWrFkqLCxUY2OjPvzwQ506dUqPPvqogoKC1KtXL6Wmpmr16tWX%2ByEAAABeyr%2Btd%2BD7TJ48%2BXvnSktLNXjwYLex2NhYFRcX6%2BTJk9qzZ4/i4%2BNdc8HBwbr22mtVXFysuro6%2Bfv764YbbnDNx8XF6fjx49q7d69KS0vVs2dP%2Bfn5ua17zZo1Ld73iooKVVZWuo0FBATLZguSn5/HdlrgsvH3v/TvgzPvNau%2B56ycz8rZJGvns3K21vLYgvVDHA6HwsLC3MZCQ0P11VdfqaamRoZhKDQ09Lz56upqhYaGKjg4WL6%2Bvm5zklRdXS2Hw3HefcPCwlRTU6Pm5ma3%2B32f1atXa9myZW5jaWlpSk9PV0hIYKuyAlZkswVdtm1Z/T1n5XxWziZZO5%2BVs7WUVxas7%2BPj43NJ51sqNTVVycnJbmMBAcGSpNraE3I6m03ZDuCtHI76S74NPz9fhYQEWvY9Z%2BV8Vs4mWTufJ2a7nP%2BhO5tXFqzw8HA5HA63MYfDofDwcNlsNvn6%2Bqqmpua8%2BYiICEVEROjYsWNyOp2u04Bn1nVmfv/%2B/efd98x6WyIyMlKRkZHnrOP0DxSns1lNTZ7xogPayuV8D1j9PWflfFbOJlk7n5WztZRXniRNSEhQSUmJ29iuXbtkt9vVvn179ejRw22%2BpqZG%2B/fvV0JCgmJjY9Xc3KyysjLXfFFRkTp16qTrrrtOCQkJKisrU1NTk9u83W6/9MEAAIAleGXBGjt2rLZu3ap33nlHDQ0Nevnll7V//36NHj1akjRhwgS9%2BOKL%2BvLLL3Xs2DEtWLBA8fHxstvtstlsGj58uHJycvTdd9/p0KFDWrp0qVJTU9WuXTsNHjxYQUFBys3NVX19vXbs2KHXXntNkyZNauPUAADAW3jsKcKEhARJch1Jev/99yWdPlLVo0cPLVmyRLm5ucrIyFB0dLSWL1%2Buzp07S5LGjx%2BvyspKPfDAA6qvr9fNN9%2BsZ5991rXup556StnZ2Ro6dKjatWunkSNHaubMmZKk9u3ba/ny5crKylJiYqIiIiI0Z84cDRky5HLGBwAAXszHOPPpm7ikHI562WxBcjjqTT0vPTxv68UXAjzMhlkDL/k2/P19L8l7zlNYOZ%2BVs0nWzueJ2a64olObbNcrTxECAAB4MgoWAACAyShYAAAAJqNgAQAAmIyCBQAAYDIKFgAAgMkoWAAAACajYAEAAJjMYz/JHYB1edsH5F6OD0YFYC0cwQIAADAZBQsAAMBkFCwAAACTUbAAAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwQIAADAZBQsAAMBkFCwAAACTUbAAAABM5t/WOwAAnm543ta23oVW2TBrYFvvAvCzxxEsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwbqAgwcPasqUKerVq5cSExO1ePFiNTc3t/VuAQAAL8HnYJ3DMAzNmDFD3bt310cffaTvvvtOU6dOVefOnfWb3/ymrXcPAC7Kmz63i8/sglVxBOscu3btUllZmTIzMxUaGqro6GhNnTpVq1atautdAwAAXoIjWOcoLS1V165dFRYW5hqLi4vTvn37VFdXp%2BDg4Iuuo6KiQpWVlW5jAQHBstmC5OdHpwWAM7zpaJskvffYoJ%2B8jjM/B6z488DK2VqLgnUOh8Oh0NBQt7Eztx0OR4sK1urVq7Vs2TK3sd/85jcKCgpSamqqIiMjTdvffywcZtq6fqyKigqtXr3a9Gyewsr5yOa9rJzPytmk0/kKCl60ZD4rZ2stKuYlkJqaqtdff93ta%2BDAgVq2bNl5R7asoLKy0rLZJGvnI5v3snI%2BK2eTrJ3PytlaiyNY54iIiFBNTY3bmMPhkCSFh4e3aB2RkZHnNfeSkhJzdhAAAHg8jmCdIyEhQeXl5a5SJUlFRUXq3r27goKC2nDPAACAt6BgnSMmJkZ2u10LFixQbW2tysrKlJ%2Bfr0mTJrX1rgEAAC/hl52dnd3WO%2BFpBg0apHfffVfz58/XO%2B%2B8owkTJmjKlCk/eb1BQUHq37%2B/JY%2BEWTmbZO18ZPNeVs5n5WyStfNZOVtr%2BBiGYbT1TgAAAFgJpwgBAABMRsECAAAwGQULAADAZBQsAAAAk1GwAAAATEbBAgAAMBkFCwAAwGQULAAAAJNRsAAAAExGwfoJDh48qClTpqhXr15KTEzU4sWL1dzcfMFlCwoKlJSUJLvdrrFjx6qkpMQ1d/LkSWVlZal///7q3bu30tPTVV1dfbliXFBrsr366qu644471Lt3b40cOVLvv/%2B%2Ba%2B7ZZ59VTEyMEhISXF99%2B/a9XDG%2BV0vzFRYWqmfPnm77n5CQoO%2B%2B%2B06Sdz93DzzwwHm5YmJitGzZMklSRkaGYmNj3eZHjRp1ueOc55NPPtEtt9yi2bNn/%2BByhmFo6dKlGjhwoG688Ubdf//9OnDggGve4XBo9uzZ6tOnj/r166fHH39cDQ0Nl3r3f1Brsi1btky33XabevfurbFjx%2Bof//iHa97bn7uLfd/w5ufuzjvvPO9917NnT61bt06S9Ktf/UpxcXFu8w8//PDliPC9Dh48qOnTp6t///5KTEzUnDlzdPTo0Qsu%2B/bbb7sy3n333dq6datr7mLvScsx8KM0Nzcb99xzj/Hoo48aNTU1xtdff20kJSUZf/vb385bdtOmTUavXr2Mbdu2GfX19cZzzz1nDBw40KivrzcMwzDmz59v3HXXXcb%2B/fuNqqoqY%2BrUqca0adMudySX1mR79913jZtuusnYuXOncerUKWPt2rVGXFyc8c033xiGcTpbZmbm5Y7wg1qTb8WKFcZvfvOb712XNz9356qpqTEGDhxofPnll4ZhGMbDDz9svPDCC5d6l1slPz/fuOOOO4zx48cbs2bN%2BsFlV6xYYQwcONAoLS01jh07ZmRmZhqjRo0ympubDcM4ne/Xv/61UVlZaRw%2BfNj4z//8T2P%2B/PmXI8YFtSbbSy%2B9ZCQlJRlfffWV0djYaDz33HNGv379jGPHjhmG4f3P3cW%2Bb3jzc3eub775xkhMTDQqKysNwzCMkSNHGm%2B%2B%2Beal2M0f7e677zb%2B8Ic/GHV1dcaRI0eMMWPGGPPmzTtvuV27dhlxcXHG22%2B/bZw4ccJ47bXXjBtvvNH49ttvDcO4%2BHvSajiC9SPt2rVLZWVlyszMVGhoqKKjozV16lStWrXqvGXXrFmjlJQUDRgwQB07dlRaWpokafPmzWpqatK6des0a9YsRUVFKTw8XBkZGdqyZYuOHDlyuWNJal22hoYGPfroo%2Brdu7f8/f117733Kjg4WP/85z8lSbW1terUqdPljvCDWpPv6NGj37v/3v7cnSsvL0933HGHbrjhBkme%2Bdx16NBBa9euVbdu3S667Jo1a/Tggw8qJiZGwcHBysjI0J49e/TFF1%2BoqqpKW7Zs0dy5c9W5c2ddeeWVmjVrlgoLC9XY2HgZkpyvNdn8/Pw0Z84cde/eXe3atdMDDzygo0eP6l//%2Bpck73/ufmj/vf25O9eCBQs0ZcoUde7cWdLp7CEhIWbv4o927NgxxcfH67HHHlNQUJAiIyM1ZswYtyOmZxQWFmrw4MEaMWKEAgICNHbsWPXo0UNvvPGGpB9%2BT1oRBetHKi0tVdeuXRUWFuYai4uL0759%2B1RXV3fesnFxca7bPj4%2BiomJUXFxsfbv36%2B6ujq3%2BejoaAUGBrqdRrycWpNt1KhRmjBhgut2bW2t6urqdPXVV7tuFxUV6e6771bfvn01ceJEFRcXX54g36M1%2BWpra7V//37de%2B%2B9uummmzRmzBh9%2BumnkuT1z93Z/v3vf2v9%2BvWaMWOGa6y2tlZbtmzR0KFD1b9/fz344IPav3//Jd3/i5k8eXKLisPJkye1Z88excfHu8aCg4N17bXXqri4WKWlpfL393eVSen043T8%2BHHt3bv3kuz7xbQ0myTdd999GjZsmOv2t99%2BK0lu7ztvfe6kH/6%2B4e3P3dm2bdumf/3rX/r1r3/tGjt69Khee%2B013XbbbRowYIBmzZqlqqoqM3e3VTp16qScnBxFRES4xsrLy12vtbOd%2B7NOkmJjY1VcXHzR96QVUbB%2BJIfDodDQULexM7cdDsd5y579A%2B/MstXV1a5lz11XSEhIm13L05psZzMMQ5mZmerdu7duuukmSdKVV16pLl266Pnnn9eHH36oG2%2B8UQ888ECbXqfUmnw2m01dunTRokWL9Omnn2rUqFF6%2BOGHtWfPHks9d3/96181duxYhYeHu8aioqJ03XXX6ZVXXtG7776rkJAQPfjgg212lKA1ampqZBjGBR%2BLM%2B%2B74OBg%2Bfr6us1JavNr6FqrsbFRjz/%2BuEaPHu36oefNz530w983rPTcLVu2TFOnTlX79u0lnf4e2qNHD9c1WW%2B88Yaqq6uVnp7exnv6/%2BzatUsvv/yypk2bdt7cD/2su9h70or823oHfg58fHxaNd7SeU9y6tQp/eEPf9DXX3%2BtgoIC174/9dRTbss99thjWr9%2Bvd5//32NGzeuLXa1Vc4%2BoiNJ999/v9566y29%2BeabGjx48Pfez5ueu6qqKm3YsEFvv/222/jzzz/vdvuPf/yjbr75Zn322WcaOHDg5dxFU1npfVdXV6e0tDT5%2B/srKyvLNe7tz90Pfd8ICAj43vt503O3e/dulZaWavny5a4xHx8frV692m25rKws3XXXXdq3b5%2Buu%2B66y7yX7j7//HNNnz5djz76qBITE1t8Pyu951qDI1g/UkREhGpqatzGzhwhOPsogHT6KMiFlg0PD3cddj173jAM1dTUuB2SvZxak006fR3WtGnTVF5erldffVVXXHHF967bz89PV199tSorK83d6VZobb5zXXPNNaqsrLTEcydJH3zwgf7jP/5D11577Q%2BuOzg4WKGhoW363LWUzWaTr6/vBR%2BLiIgIRURE6NixY3I6nW5zktrsuWut6upq/epXv1JISIheeuklBQUFfe%2By3vTcXcjZ3zes8NxJ0saNG3XrrbcqODj4B5e75pprJMn1m8ttZcuWLXrooYf0%2BOOP67777rvgMuHh4Rc8gxMeHn7R96QVUbB%2BpISEBJWXl7u9mIqKitS9e/fzvtElJCS4nWN2Op0qLS2V3W5XVFSUwsLC3K7ZKSsrU2Njo9u56supNdkMw9Ds2bPVvn17rVy50u3wsGEY%2BtOf/qTdu3e7xk6dOqUDBw4oKirq0gf5Hq3J98ILL%2Bjvf/%2B729jevXsVFRXl9c/dGZ9%2B%2Bqluvvlmt7G6ujplZ2e7XazvcDjkcDja9Llrqfbt26tHjx5uz01NTY3279%2BvhIQExcbGqrm5WWVlZa75oqIiderUqc2PErTEyZMnNW3aNNntdj3zzDPq0KGDa87bn7uLfd/w9ufujAu978rLy5WVleX2kRNnritry%2Bdu586dysjI0LPPPqt77rnne5dLSEg47/rTXbt2yW63X/Q9aUUUrB8pJiZGdrtdCxYsUG1trcrKypSfn69JkyZJkoYNG%2Bb6LYvx48ersLBQ27dvV319vZ5%2B%2BmkFBAQoOTlZfn5%2BGjdunPLy8nTgwAFVVVUpJydHw4YNc/1WiSdnW79%2BvcrKyrR06VK3b/LS6cO%2B5eXlmj9/vo4cOaL6%2BnotWbJE7du31y9/%2BcvLnuuM1uSrra3VH//4R%2B3bt0%2BNjY1asWKF9u/frzFjxnj9c3fGl19%2Bqe7du7uNBQcHq6ioSAsXLtTRo0d19OhRPfXUU4qJiVHv3r0vW57WOHLkiIYNG%2Bb6XJ0JEyboxRdf1Jdffqljx45pwYIFio%2BPl91ul81m0/Dhw5WTk6PvvvtOhw4d0tKlS5Wamqp27dq1cZLznZvtb3/7m3x8fJSdne12LZLk/c/dxb5vePtzJ50ukWVlZee97zp37qwPP/xQixcv1okTJ3TkyBEtWrRIv/zlL3XllVde7l2XdPq3pTMzMzVz5swLnl6%2B77779M4770iSxo4dq61bt%2Bqdd95RQ0ODXn75Ze3fv1%2BjR4%2BW9MPvSUtqq8%2BHsIJvv/3WmDp1qmG3243ExETjueeec8316NHD%2BOijj1y3X331VeO2224zEhISjAkTJhj/%2Bte/XHMnT540nnrqKaNv375G7969jd/97ndGbW3tZc1yrpZmmzx5shETE2PEx8e7fT3%2B%2BOOGYRjG0aNHjYyMDGPAgAFGv379jAceeMDYs2dPm2Q6W0vznTx50li4cKExcOBAo0%2BfPsaECROML774wrWsNz93Z/Tq1ct4//33z1vPoUOHjLS0NKNfv35GYmKikZ6ebhw%2BfPiS7/8POfP66tmzp9GzZ0/XbcMwjAMHDhg9evQwvv76a9fyzz77/7V39yCtQ2EYx5%2BC6OKgk2O3CqXa%2BtEhxUEtCBXFRRyLCNW5BcXBQcTZoaCLu0NxkA5KEScRkYK4%2BDW4iOBgdHCRCvre6Rak9KqXtMjGtCYAAAJFSURBVMV7/z/IdPKenEOSw0MCSdYcx7Hu7m5LpVLl7/GYmT0/P1smk7FIJGLRaNRWVlasVCrVfU6/fWdu8XjcgsFgxX23vr5uZj//3H22bvzkc2dm9vj4aIFAwC4vLyv6urq6sunpaevt7bWBgQFbWlpq6JpSLBYtEAhUXGuhUMju7u5saGjItra2yvsXCgUbGRmxUChkExMTViwWP/T3p3vyX%2BMzM2t0yAMAAPiX8IoQAADAYwQsAAAAjxGwAAAAPEbAAgAA8BgBCwAAwGMELAAAAI8RsAAAADxGwAIAAHVxeHioWCymdDr97VrXdTUzM6POzk6VSqUPbefn50omk%2Brr61MsFtPCwkLFfxHrjYAFAABqbnNzU6urq/L7/d%2Buvb6%2B1uTkpNrb2yva3t7eNDs7q56eHh0fH2t3d1eu62p5edmDUf89AhYAAKi5lpYWbW9vVw1YhUJBiURC4XBYY2Njyufz5banpyetra1pamqqou7h4UGu62p8fFzNzc1qa2tTPB7XxcVFzebyFU0NPToAAPgvJJPJqm03NzdaXFzUxsaGotGozs7OlEql5Pf7FQ6H5TiOJOnk5KSitqOjQ8FgULlcTul0Wi8vL9rf39fg4GCtpvIlPMECAAANlcvlNDw8LMdx1NTUpP7%2BfiUSCe3s7Hxa6/P5lM1mdXBwoEgkIsdx9P7%2BrkwmU4eRV0fAAgAADXV7e6u9vT11dXWVt3w%2Br/v7%2B09rX19fNTc3p9HRUZ2enuro6Eitra2an5%2Bvw8ir%2BwX3wNhSnsBLZAAAAABJRU5ErkJggg%3D%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common5272898482068229639">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">378691200000000000</td>
            <td class="number">1085</td>
            <td class="number">0.6%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">347155200000000000</td>
            <td class="number">866</td>
            <td class="number">0.5%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">373420800000000000</td>
            <td class="number">846</td>
            <td class="number">0.5%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">441849600000000000</td>
            <td class="number">821</td>
            <td class="number">0.5%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">410227200000000000</td>
            <td class="number">782</td>
            <td class="number">0.5%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">383961600000000000</td>
            <td class="number">749</td>
            <td class="number">0.4%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">473385600000000000</td>
            <td class="number">737</td>
            <td class="number">0.4%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">386640000000000000</td>
            <td class="number">732</td>
            <td class="number">0.4%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">381369600000000000</td>
            <td class="number">706</td>
            <td class="number">0.4%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">504921600000000000</td>
            <td class="number">694</td>
            <td class="number">0.4%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (2163)</td>
            <td class="number">164727</td>
            <td class="number">95.4%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme5272898482068229639">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">94608000000000000</td>
            <td class="number">4</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:9%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">94694400000000000</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:35%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97113600000000000</td>
            <td class="number">5</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:11%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97200000000000000</td>
            <td class="number">7</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:15%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">97286400000000000</td>
            <td class="number">49</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">1838246400000000000</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1888185600000000000</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1930435200000000000</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1972252800000000000</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2064268800000000000</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">emp_length<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>12</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>5.5865</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>-1.5</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>10.5</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-2739777887650498343">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAATNJREFUeJzt28FNAzEQQFGCKIki6IkzPVEEPZkG0BeJ5Kxlv3eP4svPzGrj2xhjvAB/er36ALCyt6sPwL7eP7/v/szP18eEkzzOBIEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoFw7J30He5LM58JAkEgEAQCQSAQBAJBIBAEAkEgEAQC4dg36SfzL4L/M0EgCATCFivWIyvDM77n1LVkJ1sEshPPB2uxYkEwQSZ61urHPALZgBDnsWJBEAiE5VYs6wIrWS4Q1nTqD5cVC4JAIAgEgkAgCATCbYwxrj4ErMoEgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgfALsnIeoqj9M5oAAAAASUVORK5CYII%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-2739777887650498343,#minihistogram-2739777887650498343"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-2739777887650498343">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-2739777887650498343"
                                                      aria-controls="quantiles-2739777887650498343" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-2739777887650498343" aria-controls="histogram-2739777887650498343"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-2739777887650498343" aria-controls="common-2739777887650498343"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-2739777887650498343" aria-controls="extreme-2739777887650498343"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-2739777887650498343">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>-1.5</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>0.5</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>2</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>5</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>10.5</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>10.5</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>10.5</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>12</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>8.5</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>3.9298</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.70344</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-1.37</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>5.5865</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>3.4876</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>-0.046929</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>965040</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>15.443</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-2739777887650498343">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3X1c1GW%2B//E3N94OAgOGqYfMcC0RSN3SyMLEY6mb5ZaGN%2BeYZea6mOLJIAyNVpN2lTSzNqm2XHvskZRaKm%2ByG7UOq1n66HBnnJbNDFkFYQghERnm94c/Z3dCHXb30hnw9Xw8fADXNXNdn/mI45vv98uMj8PhcAgAAADG%2BHq6AAAAgPaGgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPP3dAGXi8rKE87PfX19FBJiUXV1vZqbHR6syjvRH/fokXv0yD16dGH0x7220KMrrujmkX05guUBvr4%2B8vHxka%2Bvj6dL8Ur0xz165B49co8eXRj9cY8enR8BCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABjmtQFr5MiRioqKUnR0tPPP0qVLJUl79uzRXXfdpejoaI0ePVrvvPOOy33Xr1%2BvkSNHKiYmRpMmTVJRUZFz7tSpU1qyZImGDh2qwYMHa968eaqurnbOl5WVaebMmRo0aJBiY2O1YsUKNTc3X5oHDQAA2gWvDVi1tbX6/e9/r4KCAuefxYsX69ixY5ozZ44mTpyoffv2KTU1VWlpacrPz5ckffDBB1q9erUyMjK0d%2B9ejRgxQrNnz9YPP/wgSVqxYoUOHDignJwcffTRR2poaNCiRYskSQ6HQ3PnzpXVatXu3bv1xhtvaNu2bVq/fr3H%2BgAAANoerwxYdrtd9fX1CgwMbDH37rvvqk%2BfPpo%2Bfbq6dOmi%2BPh4jRo1Sps3b5Ykbdq0SRMnTtRNN92krl27KjExUZL08ccfq6mpSW%2B//baSkpIUHh6ukJAQpaSkaOfOnTp27JgKCgpUUlKitLQ0BQUFKSIiQrNmzdLGjRsv6eMHAABtm1e%2BVU5tba0cDoeef/557d%2B/X5IUHx%2BvlJQUFRcXa%2BDAgS63j4yM1LZt2yRJxcXFGjdunHPOx8dHAwYMUGFhoSIjI1VXV%2Bdy/4iICHXp0kVFRUWqqKhQ7969FRwc7JwfOHCgDh06pLq6OgUEBLSq/oqKClVWVrqM%2Bft3VVhYmCTJz8/X5SNc0R/36JF79Mg9enRh9Mc9enR%2BXhmwGhsbdf311%2BvGG2/U008/rYqKCs2fP1/p6emy2Wy67rrrXG4fHBzsvI7KZrO5BCRJCgoKUnV1tWw2m/PrvxcYGOic//Hc2a9tNlurA1Z2drbWrl3rMpaYmKh58%2Bb9aN8urVrvckV/3KNH7tEj9%2BjRhdEf9%2BhRS14ZsHr06KE333zT%2BXVAQIAWLlyoX/ziF7rhhhvOeR8fHx%2BXj%2BebPx938/%2BIhIQExcfHu4z5%2B3eVzVYv6UzSDwzsotrak7LbuYD%2Bx%2BiPe/TIPXrkHj26MPrjXlvokdVq8ci%2BXhmwzuXf/u3f1NzcLF9fX9XU1LjM2Ww2hYSESJKsVus55/v376/Q0FBJUk1Njbp27SrpzIXtNTU1Cg0Nld1uP%2Bd9JTnXb42wsDDn6cCzKitPqKnJ9ZvPbm9uMYa/oT/u0SP36JF79OjC6I979KglrwxYBw8e1Ntvv%2B387T5J%2Buabb9SxY0eNGDFCf/zjH11un5%2Bfr5iYGElSdHS0CgsLNWHCBElnLpgvLi7WxIkTFR4eruDgYBUVFalXr16SpJKSEjU2NioqKkqVlZUqLy%2BXzWaT1Wp1rt2vXz9ZLJ5JwAAA/LPGrs7zdAmtti1puKdLMMorr0oLDQ3Vpk2blJWVpcbGRh06dEirV6/WlClTdPfdd%2BvIkSN6/fXX1dDQoO3bt%2BuTTz5RQkKCJGny5MnKycnR3r17VV9fr2effVadO3dWfHy8/Pz8dN9992n16tX67rvvVFVVpYyMDI0ZM0bdu3fXgAEDFBMTo2XLlqm2tlYlJSXKysrStGnTPNwRAADQlnjlEaywsDBlZWVp5cqV%2Bu1vfyur1apx48Zp3rx56tixo9atW6elS5cqMzNTvXr1UmZmpvPC97i4OCUnJys1NVVVVVWKiopSVlaWOnXqJEl65JFHVF9fr3vuuUd2u10jR45Uenq6c%2B/nnntOS5Ys0a233iqLxaKpU6dq6tSpnmgDAABoo3wcDofD00VcDiorTzg/9/f3ldVqkc1Wzznrc6A/7tEj9%2BiRe/TowtpDfzhFKF1xRbeLsq47XnmKEAAAoC0jYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABjWJgLW8uXLde211zq/3rNnj%2B666y5FR0dr9OjReuedd1xuv379eo0cOVIxMTGaNGmSioqKnHOnTp3SkiVLNHToUA0ePFjz5s1TdXW1c76srEwzZ87UoEGDFBsbqxUrVqi5ufniP0gAANBueH3AOnjwoHJzc51fHzt2THPmzNHEiRO1b98%2BpaamKi0tTfn5%2BZKkDz74QKtXr1ZGRob27t2rESNGaPbs2frhhx8kSStWrNCBAweUk5Ojjz76SA0NDVq0aJEkyeFwaO7cubJardq9e7feeOMNbdu2TevXr7/0DxwAALRZXh2wmpub9eSTT2rGjBnOsXfffVd9%2BvTR9OnT1aVLF8XHx2vUqFHavHmzJGnTpk2aOHGibrrpJnXt2lWJiYmSpI8//lhNTU16%2B%2B23lZSUpPDwcIWEhCglJUU7d%2B7UsWPHVFBQoJKSEqWlpSkoKEgRERGaNWuWNm7c6ImHDwAA2iivDlgbN25U586dNX78eOdYcXGxBg4c6HK7yMhIFRYWnnPex8dHAwYMUGFhoQ4fPqy6ujqX%2BYiICHXp0kVFRUUqLi5W7969FRwc7JwfOHCgDh06pLq6uov1MAEAQDvj7%2BkCzuf48eN64YUXtGHDBpdxm82m6667zmUsODjYeR2VzWZzCUiSFBQUpOrqatlsNufXfy8wMNA5/%2BO5s1/bbDYFBAS0qvaKigpVVla6jPn7d1VYWJgkyc/P1%2BUjXNEf9%2BiRe/TIPXp0YfTn0vL3b1999tqAlZGRofvuu0/XXHONysrK3N7ex8fH5eP55t3d34Ts7GytXbvWZSwxMVHz5s1zGQsM7GJsz/aI/rhHj9yjR%2B7RowujP5eG1WrxdAlGeWXA2rNnjwoLC7V8%2BfIWcyEhIaqpqXEZs9lsCgkJkSRZrdZzzvfv31%2BhoaGSpJqaGnXt2lXSmQvba2pqFBoaKrvdfs77nt23tRISEhQfH%2B8y5u/fVTZbvaQzPw0FBnZRbe1J2e38huKP0R/36JF79Mg9enRh9OfSOvt/pGmeCm5eGbDeeecdHT16VHFxcZLOhCBJGjZsmGbOnKn33nvP5fb5%2BfmKiYmRJEVHR6uwsFATJkyQJNntdhUXF2vixIkKDw9XcHCwioqK1KtXL0lSSUmJGhsbFRUVpcrKSpWXl8tms8lqtTrX7tevnyyW1v8FhYWFOU8HnlVZeUJNTa7/QO325hZj%2BBv64x49co8euUePLoz%2BXBrtrcdeecLz8ccf1/vvv6/c3Fzl5uYqKytLkpSbm6s777xTR44c0euvv66GhgZt375dn3zyiRISEiRJkydPVk5Ojvbu3av6%2Bno9%2B%2Byz6ty5s%2BLj4%2BXn56f77rtPq1ev1nfffaeqqiplZGRozJgx6t69uwYMGKCYmBgtW7ZMtbW1KikpUVZWlqZNm%2BbJdgAAgDbGK49gBQUFuVxs3tTUJEm68sorJUnr1q3T0qVLlZmZqV69eikzM9N54XtcXJySk5OVmpqqqqoqRUVFKSsrS506dZIkPfLII6qvr9c999wju92ukSNHKj093bnXc889pyVLlujWW2%2BVxWLR1KlTNXXq1Ev0yAEAQHvg4zh7/g0XVWXlCefn/v6%2Bslotstnq290hURPoj3v0yD165B49urD20J%2Bxq/M8XUKrbUsaflHWveKKbhdlXXe88hQhAABAW0bAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw4wHrLq6OtNLAgAAtCnGA9Ytt9yi1NRUHThwwPTSAAAAbYLxgJWenq7KykpNnz5d48aN02uvvabq6mrT2wAAAHgt4wFrwoQJeuWVV/TJJ59oypQpev/993XbbbcpKSlJeXl5prcDAADwOhftIveQkBD953/%2BpzZu3KiMjAzl5eXpoYce0pgxY7R9%2B/aLtS0AAIDHXbSAVVVVpZdfflljx45VcnKyhgwZorVr12rGjBlKT09XVlbWBe//1VdfacaMGbrhhht00003af78%2BaqoqJAk7dmzR3fddZeio6M1evRovfPOOy73Xb9%2BvUaOHKmYmBhNmjRJRUVFzrlTp05pyZIlGjp0qAYPHqx58%2Ba5nMIsKyvTzJkzNWjQIMXGxmrFihVqbm422BkAANDeGQ9Yn376qebNm6cRI0bo97//ve644w598MEHWrdunUaNGqXJkycrKytLr7766nnXaGxs1IMPPqgbb7xRf/rTn7R161ZVV1crPT1dx44d05w5czRx4kTt27dPqampSktLU35%2BviTpgw8%2B0OrVq5WRkaG9e/dqxIgRmj17tn744QdJ0ooVK3TgwAHl5OToo48%2BUkNDgxYtWiRJcjgcmjt3rqxWq3bv3q033nhD27Zt0/r16023CQAAtGPGA9bDDz%2Bs%2Bvp6rVq1Srt27VJSUpJ69erlcpuYmBiFhYWdd42TJ09qwYIFmj17tjp27KiQkBDdcccd%2BvOf/6x3331Xffr00fTp09WlSxfFx8dr1KhR2rx5syRp06ZNmjhxom666SZ17dpViYmJkqSPP/5YTU1Nevvtt5WUlKTw8HCFhIQoJSVFO3fu1LFjx1RQUKCSkhKlpaUpKChIERERmjVrljZu3Gi6TQAAoB3zN73gjh07FB4ersbGRvn5%2BUmS6uvrZbFYXG737rvvnneNoKAgTZo0SdKZo0rffPON3nrrLY0dO1bFxcUaOHCgy%2B0jIyO1bds2SVJxcbHGjRvnnPPx8dGAAQNUWFioyMhI1dXVudw/IiJCXbp0UVFRkSoqKtS7d28FBwc75wcOHKhDhw6prq5OAQEBrepBRUWFKisrXcb8/bs6Q6Wfn6/LR7iiP%2B7RI/fokXv06MLoz6Xl79%2B%2B%2Bmw8YPn4%2BGj8%2BPFKTEzUmDFjJEnZ2dnKycnRSy%2B9pPDw8FavdeTIEd1%2B%2B%2B2y2%2B1KSEjQ/PnzNXPmTF133XUutwsODnZeR2Wz2VwCknQmsFVXV8tmszm//nuBgYHO%2BR/Pnf3aZrO1OmBlZ2dr7dq1LmOJiYmaN2/ej/bt0qr1Llf0xz165B49co8eXRj9uTSsVov7G7UhxgPW008/rWuuuUZDhgxxjt111136v//7Pz399NN66aWXWr1W7969VVhYqG%2B//VaLFy/WY489dt7b%2Bvj4uHw837y7%2B5uQkJCg%2BPh4lzF//66y2eolnflpKDCwi2prT8pu5wL6H6M/7tEj9%2BiRe/TowujPpXX2/0jTPBXcjAes/fv3a9euXeratatzrHv37lq8eLFGjhz5D6/n4%2BOjq6%2B%2BWsnJyZo4caJGjBihmpoal9vYbDaFhIRIkqxW6znn%2B/fvr9DQUElSTU2Nsz6Hw6GamhqFhobKbref876SnOu3RlhYWItrzCorT6ipyfUfqN3e3GIMf0N/3KNH7tEj9%2BjRhdGfS6O99dj4CU%2BHw6GmpqYW4ydPnmz1yx3s27dP//7v/%2B6yztn73nzzzS4vuyBJ%2Bfn5iomJkSRFR0ersLDQOWe321VcXKyYmBiFh4crODjY5f4lJSVqbGxUVFSUoqOjVV5e7gxVZ9fu169fi2vIAAAAzsd4wBo%2BfLiSk5NVXFys2tpa1dTUaP/%2B/VqwYIFuueWWVq0RGRmpkydPKjMzUydPnlR1dbWef/553XDDDRo/fryOHDmi119/XQ0NDdq%2Bfbs%2B%2BeQTJSQkSJImT56snJwc7d27V/X19Xr22WfVuXNnxcfHy8/PT/fdd59Wr16t7777TlVVVcrIyNCYMWPUvXt3DRgwQDExMVq2bJlqa2tVUlKirKwsTZs2zXSbAABAO%2BbjcDgcJhesqqrSwoULtWfPHpfrmoYNG6aVK1eqe/furVrn4MGD%2BvWvf63CwkL5%2B/tr2LBhWrRokXr06KEvvvhCS5cu1V/%2B8hf16tVLCxcu1OjRo533/e///m9lZWWpqqpKUVFReuqpp/STn/xE0pnX2HrmmWf07rvvym63a%2BTIkUpPT1e3bt0kSUePHtWSJUv02WefyWKxaOrUqZo7d%2B6/3JfKyhPOz/39fWW1WmSz1be7Q6Im0B/36JF79Mg9enRh7aE/Y1e3nbeo25Y0/KKse8UV3S7Kuu4YD1hnlZaW6tChQ3I4HOrbt68iIiIuxjZtBgGr9eiPe/TIPXrkHj26sPbQHwKW5wKW8Yvcz4qIiHB5SYbGxkZJUseOHS/WlgAAAF7BeMD68ssvlZ6erj//%2Bc%2By2%2B0t5g8ePGh6SwAAAK9iPGCdvZ7piSeeUOfOnU0vDwAA4PWMB6xDhw7ps88%2BU6dOnUwvDQAA0CYYf5mGXr166fTp06aXBQAAaDOMH8FauHChMjIylJqa2ur37gO8UVv67Rvp4v0GDgDgH2c8YK1du1ZlZWV6%2B%2B23ZbVaW7zH3//8z/%2BY3hIAAMCrGA9Yt956qzp06GB6WQAAgDbDeMBasGCB6SUBAADaFOMXuUvS//7v/yo1NVX333%2B/pDNv1Lxt27aLsRUAAIDXMR6wPvroI02dOlU2m00HDhyQdOb9/RYvXqxNmzaZ3g4AAMDrGA9YL730klasWKGXXnrJeYF7r1699Nxzz%2Bn11183vR0AAIDXMR6wvvnmG91%2B%2B%2B2S5PIbhLGxsTpy5Ijp7QAAALyO8YDVoUMH1dTUtBg/dOgQb50DAAAuC8YD1m233aa0tDSVlpZKkmw2mz799FMlJSVp5MiRprcDAADwOsYDVmpqqhwOh372s5/p1KlTuvnmmzVr1iz17NlTjz/%2BuOntAAAAvI7x18EKDAzUunXrVFpaqkOHDsnHx0d9%2B/ZV3759TW8FAADglYwHrLMiIiIUERFxsZYHAADwWsYD1i233HLeObvdrj179pjeEgAAwKsYD1gJCQkuL8/Q3NyssrIy5eXlafbs2aa3AwAA8DrGA9YjjzxyzvH8/Hz94Q9/ML0dAACA17ko70V4LjExMSooKLhU2wEAAHjMJQtY3377rb7//vtLtR0AAIDHGD9FOHny5BZjjY2N%2Bstf/qJRo0aZ3g4AAMDrGA9YV199tctF7pLUqVMn3Xvvvbr33ntNbwcAAOB1jAesZ555xvSSAAAAbYrxgLVp0yZ16NChVbedMGGC6e0BAAA8znjAevrpp3Xq1Ck5HA6XcR8fH5cxHx8fAhYAAGiXjAes3/72t3rjjTc0Z84cRUREyG636%2Buvv1ZWVpamT5%2Bu2NhY01sCAAB4FeMBa/ny5Xr11VcVFhbmHBs8eLCefPJJPfjgg9q6davpLQEAALyK8dfBKisrU2BgYIvxoKAglZeXm94OAADA6xgPWH379lVGRoZsNptz7Pvvv1dmZqb69u1rejsAAACvY/wUYVpamubMmaM333xTFotFPj4%2Bqqurk8Vi0QsvvGB6OwAAAK9jPGANGTJEu3bt0u7du3X06FE5HA716NFDcXFxCggIML0dAACA1zEesCSpS5cuGj16tMrLyxUeHn4xtgAAAPBaxq/Bamho0JNPPqnrr79eY8eOlSTV1tbq4Ycf1okTJ0xvBwAA4HWMB6w1a9boyy%2B/1MqVK%2BXr%2B7flT58%2BrV//%2BtemtwMAAPA6xgPWhx9%2BqNWrV2vMmDHON30ODAxURkaGdu7caXo7AAAAr2M8YFVUVOjqq69uMR4aGqq6ujrT2wEAAHgd4wHryiuv1IEDB1qMv//%2B%2B%2BrZs6fp7QAAALyO8d8inDFjhn75y19q4sSJstvt%2Bt3vfqfCwkLt2LFDTzzxhOntAAAAvI7xgDV58mQFBwfrtddeU9euXbVu3Tr17dtXK1eu1JgxY0xvBwAA4HWMB6yqqiqNGTOGMAUAAC5bRq/Bam5u1siRI%2BVwOEwuCwAA0KYYDVi%2Bvr66%2BeabtW3bNpPLAgAAtCnGTxH26tVLy5cvV1ZWlq666ip16NDBZT4zM9P0lgAAAF7FeMD6%2Buuv1bdvX0mSzWYzvTwAAIDXMxawFixYoFWrVmnDhg3OsRdeeEGJiYmmtgAAAGgTjF2D9fHHH7cYy8rKMrU8AABAm2EsYJ3rNwf5bUIAAHA5Mhawzr6xs7ux1iorK9OcOXM0dOhQxcbGKjk5Wd9//70k6eDBg5o8ebJiYmIUFxen1157zeW%2BW7Zs0R133KHo6GjdeeedysvLc845HA6tWrVKw4cP1/XXX68ZM2bou%2B%2B%2Bc87bbDYtWLBAQ4YM0Y033qgnnnhCDQ0N//TjAAAAlx/j70Voypw5cxQcHKydO3cqNzdXpaWl%2Bs1vfqOTJ09q1qxZGjJkiPbs2aM1a9boxRdf1I4dOyRJhYWFSklJ0fz58/X555/r/vvvV2Jioo4ePSpJWr9%2BvXJycvTKK68oLy9P4eHhmjt3rvNo26JFi1RVVaUdO3bovffe08GDB7Vy5UqP9QEAALQ9XhmwTpw4oaioKC1cuFAWi0VhYWG655579MUXX2jXrl06ffq0Hn30UVksFg0aNEgJCQnKzs6WJOXk5CguLk7jxo1T586dNWnSJPXv31%2B5ubmSpE2bNumhhx7SgAEDFBAQoJSUFJWWlurLL79UVVWVdu7cqdTUVHXv3l09evRQUlKScnJy1NjY6MmWAACANsTYbxGeDT3uxlrzOljdunVTRkaGy1h5ebl69uyp4uJiXXfddfLz83PORUZGatOmTZKk4uJixcXFudw3MjJShYWFOnXqlEpLSxUVFeWcCwgI0FVXXaXCwkLV1dXJ399f1157rXN%2B4MCB%2BuGHH/TNN9%2B4jAMAAJyPsYD105/%2BVBUVFW7H/hkFBQXasGGD1q1bpy1btigoKMhlPjg4WDU1NWpubpbNZlNwcLDLfFBQkL7%2B%2BmvV1NTI4XC0uH9QUJCqq6sVFBSkgIAA%2Bfr6usxJUnV1davrraioUGVlpcuYv39XhYWFSZL8/HxdPsIV/fnn%2BPvTr7/H95F79OjC6M%2Bl1d6ew4wFrL9//SuT9u/frzlz5ujRRx9VbGystmzZ8k%2Bt4%2B6C%2B391/u9lZ2dr7dq1LmOJiYmaN2%2Bey1hgYJdWr3k5oj//GKvV4ukSvBLfR%2B7RowujP5dGe3sOM/5K7ibt3LlTCxcu1JIlS3T33XdLkkJDQ3X48GGX29lsNlmtVvn6%2BiokJKTFK8jbbDaFhIQ4b1NTU9NiPjQ0VKGhoTpx4oTsdrvzFOTZtUJDQ1tdd0JCguLj413G/P27ymarl3Tmp6HAwC6qrT0pu7251eteLujPP%2Bfs9xfO4PvIPXp0YfTn0rpYz2GeCm5eG7AOHDiglJQUrVmzRsOHD3eOR0dHa%2BPGjWpqapK//5ny8/PzFRMT45wvKipyWaugoEA/%2B9nP1LFjR/Xv319FRUW68cYbJUk1NTU6fPiwoqOjFR4erubmZpWUlCgyMtK5drdu3XT11Ve3uvawsDDn6cCzKitPqKnJ9R%2Bo3d7cYgx/Q3/%2BMfTq3Pg%2Bco8eXRj9uTTaW4%2B98oRnU1OT0tLSNH/%2BfJdwJUlxcXGyWCzKzMxUfX299u3bpzfffFPTpk2TJE2aNEl5eXnaunWrGhoatGHDBh0%2BfFgTJkyQJE2ZMkWvvPKKvvrqK504cULLli1TVFSUYmJiZLVaNXbsWGVkZOj48eM6cuSIVq1apYSEhBZvWg0AAHA%2BXnkE68svv1RpaameeeYZPfPMMy5z27dv17p167RkyRLFxsYqNDRUycnJGjFihCSpf//%2BWrlypTIzM5WSkqKIiAitW7dO3bt3lyRNnjxZlZWVevDBB1VfX69hw4ZpzZo1zvWfeuoppaena/To0erQoYPGjx%2Bv%2BfPnX7oHDwAA2jwfB%2B9nc0lUVp5wfu7v7yur1SKbrb7dHRI1wVv6M3Z1nvsbeZFtScPd3%2Bgy4i3fR96MHl1Ye%2BhPW3oeu1jPYVcmB0N/AAAQVUlEQVRc0e2irOuOV54iBAAAaMsIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDvPKtcgAAl4e29ErjEu%2BYgNbjCBYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIb5e7oAXD7Grs7zdAntWlvq77ak4Z4uAQAuKo5gAQAAGEbAAgAAMIyABQAAYBgBCwAAwDAucgcAN9rSLxBI/BIB4A04ggUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhvl7ugAAl5%2Bxq/M8XQIAXFQcwQIAADCMgAUAAGCYV58i/PTTT5WSkqJhw4Zp1apVLnNbtmzRmjVrVF5erj59%2Big1NVXDhw%2BXJDkcDq1evVqbN29WXV2dBg8erKVLlyo8PFySZLPZ9Ktf/Uq7d%2B%2BWn5%2Bfbr/9di1evFidO3eWJB08eFBPPfWUiouLFRwcrAceeEAPPPDApX3wAPBP4hQs4HleewTr5Zdf1rJly9SnT58Wc4WFhUpJSdH8%2BfP1%2Beef6/7771diYqKOHj0qSVq/fr1ycnL0yiuvKC8vT%2BHh4Zo7d64cDockadGiRaqqqtKOHTv03nvv6eDBg1q5cqUk6eTJk5o1a5aGDBmiPXv2aM2aNXrxxRe1Y8eOS/fgAQBAm%2Ba1AatTp07avHnzOQNWTk6O4uLiNG7cOHXu3FmTJk1S//79lZubK0natGmTHnroIQ0YMEABAQFKSUlRaWmpvvzyS1VVVWnnzp1KTU1V9%2B7d1aNHDyUlJSknJ0eNjY3atWuXTp8%2BrUcffVQWi0WDBg1SQkKCsrOzL3ULAABAG%2BW1pwinT59%2B3rni4mLFxcW5jEVGRqqwsFCnTp1SaWmpoqKinHMBAQG66qqrVFhYqLq6Ovn7%2B%2Bvaa691zg8cOFA//PCDvvnmGxUXF%2Bu6666Tn5%2Bfy9qbNm1qde0VFRWqrKx0GfP376qwsDBJkp%2Bfr8tHAEDb4O/P8/bF0t5667UB60JsNpuCg4NdxoKCgvT111%2BrpqZGDodDQUFBLearq6sVFBSkgIAA%2Bfr6usxJUnV1tWw2W4v7BgcHq6amRs3NzS73O5/s7GytXbvWZSwxMVHz5s1zGQsM7OL%2BwQIAvIbVavF0Ce1We%2BttmwxY5%2BPj43NR51srISFB8fHxLmP%2B/l1ls9VLOnPkKjCwi2prT8pubzayJwDg4jv7PA7zLlZvPRXc2mTACgkJkc1mcxmz2WwKCQmR1WqVr6%2BvampqWsyHhoYqNDRUJ06ckN1ud54GPLvW2fnDhw%2B3uO/ZdVsjLCzMeTrwrMrKE2pqcg1TdntzizEAgPfiOfviaW%2B9bZMnPKOjo1VUVOQyVlBQoJiYGHXs2FH9%2B/d3ma%2BpqdHhw4cVHR2tyMhINTc3q6SkxDmfn5%2Bvbt266eqrr1Z0dLRKSkrU1NTkMh8TE3PxHxgAAGgX2mTAmjRpkvLy8rR161Y1NDRow4YNOnz4sCZMmCBJmjJlil555RV99dVXOnHihJYtW6aoqCjFxMTIarVq7NixysjI0PHjx3XkyBGtWrVKCQkJ6tChg%2BLi4mSxWJSZman6%2Bnrt27dPb775pqZNm%2BbhRw0AANoKrz1FGB0dLUnOI0kffvihpDNHqvr376%2BVK1cqMzNTKSkpioiI0Lp169S9e3dJ0uTJk1VZWakHH3xQ9fX1GjZsmNasWeNc%2B6mnnlJ6erpGjx6tDh06aPz48Zo/f74kqWPHjlq3bp2WLFmi2NhYhYaGKjk5WSNGjLiUDx8AALRhPo6zr76Ji6qy8oTzc39/X1mtFtls9e3unPOF8OrSANq6bUnDPV3CP6QtPe9erN5ecUW3i7KuO23yFCEAAIA3I2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPat8pB67SlV%2BkFAOBywREsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDBeBwsAgFbitQfRWhzBAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgHUOZWVlmjlzpgYNGqTY2FitWLFCzc3Nni4LAAC0Ef6eLsDbOBwOzZ07V/369dPu3bt1/PhxzZo1S927d9cDDzzg6fIAAEAbwBGsHykoKFBJSYnS0tIUFBSkiIgIzZo1Sxs3bvR0aQAAoI3gCNaPFBcXq3fv3goODnaODRw4UIcOHVJdXZ0CAgLcrlFRUaHKykqXMX//rgoLC5Mk%2Bfn5unwEAOBy5%2B/fvv5PJGD9iM1mU1BQkMvY2a9tNlurAlZ2drbWrl3rMjZ37lw98sgjks4EsPXrX1FCQoIzdP2zvnh6zL90f29UUVGh7OxsI/1pr%2BiRe/TIPXp0YfTHPXp0fu0rLnqJhIQEvfXWWy5/EhISnPOVlZVau3Zti6NcOIP%2BuEeP3KNH7tGjC6M/7tGj8%2BMI1o%2BEhoaqpqbGZcxms0mSQkJCWrVGWFgYSR4AgMsYR7B%2BJDo6WuXl5c5QJUn5%2Bfnq16%2BfLBaLBysDAABtBQHrRwYMGKCYmBgtW7ZMtbW1KikpUVZWlqZNm%2Bbp0gAAQBvhl56enu7pIrzNrbfeqvfff19Lly7V1q1bNWXKFM2cOdPoHhaLRUOHDuWo2HnQH/fokXv0yD16dGH0xz16dG4%2BDofD4ekiAAAA2hNOEQIAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsDyEJvNpuTkZMXGxmro0KH65S9/qb/%2B9a%2BeLsvjysrKNHPmTA0aNEixsbFasWKFmpubPV2WVykrK9OcOXM0dOhQxcbGKjk5Wd9//72ny/JKy5cv17XXXuvpMrzSiy%2B%2BqFtuuUWDBw/WjBkz9N1333m6JK9SVFSk6dOn66c//aluvvlmJScny2azebosj/r000918803a8GCBS3mtmzZojvuuEPR0dG68847lZeX54EKvQsBy0NSU1Nls9m0detWffjhh7Lb7UpNTfV0WR7lcDg0d%2B5cWa1W7d69W2%2B88Ya2bdum9evXe7o0rzJnzhwFBwdr586dys3NVWlpqX7zm994uiyvc/DgQeXm5nq6DK/0hz/8QR9//LGys7O1a9cu9ezZU6%2B//rqny/IadrtdDz/8sAYPHqw9e/Zo69atOn78uC7nt%2B59%2BeWXtWzZMvXp06fFXGFhoVJSUjR//nx9/vnnuv/%2B%2B5WYmKijR496oFLvQcDyAIfDoR49eig5OVlWq1WBgYGaMmWK9u/fr8v5rSELCgpUUlKitLQ0BQUFKSIiQrNmzdLGjRs9XZrXOHHihKKiorRw4UJZLBaFhYXpnnvu0RdffOHp0rxKc3OznnzySc2YMcPTpXilV199VYsXL1bv3r0VFBSkjIwMLV682NNleY3KykodP35c48ePV8eOHRUcHKxRo0apuLjY06V5TKdOnbR58%2BZzBqycnBzFxcVp3Lhx6ty5syZNmqT%2B/ftf9j/gELA8wMfHR0899ZR%2B8pOfOMfKy8t15ZVXysfHx4OVeVZxcbF69%2B6t4OBg59jAgQN16NAh1dXVebAy79GtWzdlZGQoNDTUOVZeXq6ePXt6sCrvs3HjRnXu3Fnjx4/3dCle59ixYzp69Ki%2B/fZb3X777Ro2bJiSkpIu%2B9Nff69Hjx6KjIzUm2%2B%2BqZMnT6q6uloffPCBbrvtNk%2BX5jHTp09Xt27dzjlXXFysgQMHuoxFRkaqsLDwUpTmtQhYXqCsrEzPPfecfvGLX3i6FI%2By2WwKCgpyGTv7NU/%2B51ZQUKANGzZo9uzZni7Faxw/flwvvPDCZX0650KOHj0qHx8fffjhh8rOztYf//hHHTlyhCNYf8fHx0dr1qzRRx995LwetLm5Wf/1X//l6dK8ks1mc/nBWDrz3F1dXe2hirwDAesiyc3N1bXXXnvOP2%2B99ZbzdqWlpfqP//gP/fznP9e9997rwYrR1uzfv18zZ87Uo48%2BqtjYWE%2BX4zUyMjJ033336ZprrvF0KV7p9OnTOn36tB577DFZrVb17NlT8%2BbN04cffqhTp055ujyv0NjYqNmzZ2vcuHE6cOCA8vLyFBAQoMcee8zTpbUpl/MZGUny93QB7dXdd9%2Btu%2B%2B%2B%2B4K3yc/P16xZszRz5kw9/PDDl6gy7xUaGqqamhqXsbNHrkJCQjxRktfauXOnFi5cqCVLlrj9Pruc7NmzR4WFhVq%2BfLmnS/FaZ480BAQEOMd69%2B4th8Ohqqoq9erVy1OleY0//elPKisrU1JSkvz8/GSxWPTII49owoQJqq6u5vnoR0JCQlqcZbDZbJd9nziC5SGHDh3S7NmzlZqaSrj6/6Kjo1VeXu7yDzU/P1/9%2BvWTxWLxYGXe5cCBA0pJSdGaNWsIVz/yzjvv6OjRo4qLi9OwYcN0zz33SJKGDRumLVu2eLg679CnTx8FBASoqKjIOXbkyBH5%2B/srLCzMg5V5D4fD0eLlYU6fPi2JozLnEh0d7fL9JJ25fCEmJsZDFXkHApaH/OpXv9LPf/5zTZgwwdOleI0BAwYoJiZGy5YtU21trUpKSpSVlaVp06Z5ujSv0dTUpLS0NM2fP1/Dhw/3dDle5/HHH9f777%2Bv3Nxc5ebmKisrS9KZU/bx8fEers47dOjQQZMmTdLKlSt19OhRVVZW6oUXXtDdd98tf39OakjSoEGDZLFY9Pzzz6uhoUHff/%2B9Xn75ZQ0ePFhWq9XT5XmdSZMmKS8vT1u3blVDQ4M2bNigw4cPX/b/v/k4LufXBfCQv/71r7rtttvUoUOHFj8N/e53v9ONN97ooco87%2BjRo1qyZIk%2B%2B%2BwzWSwWTZ06VXPnzvV0WV7jiy%2B%2B0LRp09SxY8cWc9u3b1fv3r09UJX3Kisr06hRo1RSUuLpUrxKY2OjnnnmGb333nvy9fXVyJEj9cQTT7icNrzc5efna8WKFTp48KA6dOigoUOHKjU1VVdeeaWnS/OI6OhoSWd%2ByJPkDOMFBQWSpB07digzM1Pl5eWKiIhQWlqabrjhBs8U6yUIWAAAAIZxihAAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPt/qfbUhcL2LogAAAAASUVORK5CYII%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-2739777887650498343">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">10.5</td>
            <td class="number">49479</td>
            <td class="number">28.6%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2.0</td>
            <td class="number">16294</td>
            <td class="number">9.4%</td>
            <td>
                <div class="bar" style="width:33%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.5</td>
            <td class="number">14318</td>
            <td class="number">8.3%</td>
            <td>
                <div class="bar" style="width:29%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.0</td>
            <td class="number">14219</td>
            <td class="number">8.2%</td>
            <td>
                <div class="bar" style="width:29%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">5.0</td>
            <td class="number">13433</td>
            <td class="number">7.8%</td>
            <td>
                <div class="bar" style="width:27%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1.0</td>
            <td class="number">11862</td>
            <td class="number">6.9%</td>
            <td>
                <div class="bar" style="width:24%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">4.0</td>
            <td class="number">11162</td>
            <td class="number">6.5%</td>
            <td>
                <div class="bar" style="width:23%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">6.0</td>
            <td class="number">10784</td>
            <td class="number">6.2%</td>
            <td>
                <div class="bar" style="width:22%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">7.0</td>
            <td class="number">9689</td>
            <td class="number">5.6%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">8.0</td>
            <td class="number">7819</td>
            <td class="number">4.5%</td>
            <td>
                <div class="bar" style="width:16%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (2)</td>
            <td class="number">13686</td>
            <td class="number">7.9%</td>
            <td>
                <div class="bar" style="width:28%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-2739777887650498343">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">-1.5</td>
            <td class="number">7507</td>
            <td class="number">4.3%</td>
            <td>
                <div class="bar" style="width:46%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.5</td>
            <td class="number">14318</td>
            <td class="number">8.3%</td>
            <td>
                <div class="bar" style="width:87%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1.0</td>
            <td class="number">11862</td>
            <td class="number">6.9%</td>
            <td>
                <div class="bar" style="width:73%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">2.0</td>
            <td class="number">16294</td>
            <td class="number">9.4%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3.0</td>
            <td class="number">14219</td>
            <td class="number">8.2%</td>
            <td>
                <div class="bar" style="width:87%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">6.0</td>
            <td class="number">10784</td>
            <td class="number">6.2%</td>
            <td>
                <div class="bar" style="width:22%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">7.0</td>
            <td class="number">9689</td>
            <td class="number">5.6%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">8.0</td>
            <td class="number">7819</td>
            <td class="number">4.5%</td>
            <td>
                <div class="bar" style="width:16%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">9.0</td>
            <td class="number">6179</td>
            <td class="number">3.6%</td>
            <td>
                <div class="bar" style="width:13%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">10.5</td>
            <td class="number">49479</td>
            <td class="number">28.6%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">fico_range_low<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>40</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>699.97</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>625</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>845</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram7776014525002438121">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAAShJREFUeJzt3cENglAUAEExlmQR9uTZnizCnrABs0ETwhdm7ib/snkEfDDN8zyfgI/OWx8ARnbZ%2BgBbud6fX//m9bitcBJGZoJAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQDjsRuEvvt1CtIH4/0wQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAjTHr4P8sub2kfl/1tjMUEgCASCQCAIBIJAINgoHIxvJ47FBIEgEAgCgSAQCAKB4C7WDnhf13pMEAgmyAF51rKcCQLBBGGRo06dXSxMwVpcYkEQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUB4A9jsIZRFf9FJAAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives7776014525002438121,#minihistogram7776014525002438121"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives7776014525002438121">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles7776014525002438121"
                                                      aria-controls="quantiles7776014525002438121" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram7776014525002438121" aria-controls="histogram7776014525002438121"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common7776014525002438121" aria-controls="common7776014525002438121"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme7776014525002438121" aria-controls="extreme7776014525002438121"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles7776014525002438121">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>625</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>660</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>675</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>695</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>715</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>765</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>845</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>220</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>40</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>32.182</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.045977</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>1.1327</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>699.97</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>25.082</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>1.1394</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>120920000</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>1035.7</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram7776014525002438121">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xt4VPWdx/FPLlxyIckkGCs0Ig0PQpKJoHKHaEKhgAV5MCFcdildSlkaCKEqCAZkK5puIRIVbUntSlbbEiFSr1AU8LIpLLashiQ0a1mRW0NiMjEQCCGT2T9cZh0BCfhL5kx4v56HJ8/8fuf8zm/O1%2BPzmXPOnPFzuVwuAQAAwBh/b08AAACgoyFgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDAr09getFdfUpb0/Bkvz9/RQZGaLa2ga1tLi8PZ3rHvWwDmphHdTCOq6lFjfc0K2NZ3VpnMGCV/n7%2B8nPz0/%2B/n7engpEPayEWlgHtbAOX6oFAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADAv09gQAqxqfV%2BztKVyVbVkjvD0FAMD/4QwWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMsHbCeffZZjRw5UgMHDtTs2bN19OhRSdKePXs0adIk2e12jRkzRq%2B%2B%2BqrHegUFBUpOTlZiYqLS0tJUVlbm7jt37pxWrlypwYMHa%2BDAgcrMzFRtba27/9ixY5ozZ44GDBigYcOGac2aNWppaWmfNwwAADoEywas3/3ud9q1a5cKCwv1zjvv6KabbtLGjRt18uRJzZ8/X6mpqdq3b5%2BWLVum7OxslZSUSJLeeust5eXlKScnR3v37tVdd92lefPm6cyZM5KkNWvWaP/%2B/SoqKtLOnTvV2Nio5cuXS5JcLpcWLFggm82md999Vy%2B%2B%2BKK2bdumgoICr%2B0HAADgeywbsH7zm99oxYoV6tmzp8LDw5WTk6MVK1botddeU69evTRr1iwFBQUpJSVFo0eP1pYtWyRJmzdvVmpqqoYOHarg4GBlZGRIknbt2qXm5mZt3bpVWVlZiomJUWRkpJYuXardu3fr5MmTOnDggCoqKpSdna3w8HDFxsZq7ty52rRpkzd3BQAA8DGWDFgnT55UZWWlPv30U40dO1ZDhgxRVlaWHA6HysvLFR8f77F8XFycSktLJemifj8/P/Xv31%2BlpaU6cuSITp8%2B7dEfGxuroKAglZWVqby8XD179lRERIS7Pz4%2BXocPH9bp06fb%2BF0DAICOItDbE7iUyspK%2Bfn56e2331ZhYaEaGxuVmZmpFStWqKGhQf369fNYPiIiwn0flcPh8AhIkhQeHq7a2lo5HA736y8LCwtz93%2B178Jrh8Oh0NDQVs2/qqpK1dXVHm2BgcGKjo5u1frXk4AAf4%2B/uHaBgd98H1IP66AW1kEtrMOXamHJgHX%2B/HmdP39eDz74oGw2myQpMzNTc%2BfO1bBhwy65jp%2Bfn8ffy/VfzpX6r0ZhYaHWr1/v0ZaRkaHMzExj2%2BhowsKCvD0Fn2ezhRgbi3pYB7WwDmphHb5QC0sGrAtnoL58xqhnz55yuVxqbm5WXV2dx/IOh0ORkZGSJJvNdsn%2Bvn37KioqSpJUV1en4OBgSV/c2F5XV6eoqCg5nc5LrivJPX5rpKenKyUlxaMtMDBYDkdDq8e4XgQE%2BCssLEj19WfldPJtzW/CxH9f1MM6qIV1UAvruJZamPzweTUsGbB69eql0NBQlZWVaeTIkZKk48ePKzAwUHfffbdeeeUVj%2BVLSkqUmJgoSbLb7SotLdXkyZMlSU6nU%2BXl5UpNTVVMTIwiIiJUVlamHj16SJIqKirU1NSkhIQEVVdX68SJE3I4HO4zZyUlJerTp49CQlpfoOjo6IsuB1ZXn1JzMwfm5TidLeyfb8jk/qMe1kEtrINaWIcv1MKSFzE7deqktLQ0rV27VpWVlaqurtYzzzyje%2B%2B9V5MnT9bx48e1ceNGNTY2avv27XrvvfeUnp4uSZo2bZqKioq0d%2B9eNTQ06IknnlDXrl2VkpKigIAATZ06VXl5eTp69KhqamqUk5OjcePGqXv37urfv78SExO1evVq1dfXq6KiQvn5%2BZo5c6aX9wgAAPAlljyDJUk//elP9fOf/1yTJk2Sv7%2B/kpOTtXz5coWGhmrDhg169NFHlZubqx49eig3N9d943tSUpKWLFmiZcuWqaamRgkJCcrPz1eXLl0kSQsXLlRDQ4OmTJkip9Op5ORkrVq1yr3dJ598UitXrtSoUaMUEhKiGTNmaMaMGd7YBQAAwEf5uVwul7cncT2orj7l7SlYUmCgv2y2EDkcDZY73Ts%2Br9jbU7gq27JGfOMxrFyP6w21sA5qYR3XUosbbujWxrO6NEteIgQAAPBlBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGGbZgJWcnKyEhATZ7Xb3v0cffVSStGfPHk2aNEl2u11jxozRq6%2B%2B6rFuQUGBkpOTlZiYqLS0NJWVlbn7zp07p5UrV2rw4MEaOHCgMjMzVVtb6%2B4/duyY5syZowEDBmjYsGFas2aNWlpa2udNAwCADsGyAau%2Bvl7//u//rgMHDrj/rVixQidPntT8%2BfOVmpqqffv2admyZcrOzlZJSYkk6a233lJeXp5ycnK0d%2B9e3XXXXZo3b57OnDkjSVqzZo3279%2BvoqIi7dy5U42NjVq%2BfLkkyeVyacGCBbLZbHr33Xf14osvatu2bSooKPDafgAAAL7HkgHL6XSqoaFBYWFhF/W99tpr6tWrl2bNmqWgoCClpKRo9OjR2rJliyRp8%2BbNSk1N1dChQxUcHKyMjAxJ0q5du9Tc3KytW7cqKytLMTExioyM1NKlS7V7926dPHlSBw4cUEVFhbKzsxUeHq7Y2FjNnTtXmzZtatf3DwAAfJslA1Z9fb1cLpeefvppjRw5UiNHjtTKlSvV0NCg8vJyxcfHeywfFxen0tJSSbqo38/PT/3791dpaamOHDmi06dPe/THxsYqKChIZWVlKi8vV8%2BePRUREeHuj4%2BP1%2BHDh3X69Ok2ftcAAKCjCPT2BC6lqalJt912mwYNGqTHHntMVVVVWrRokVatWiWHw6F%2B/fp5LB8REeG%2Bj8rhcHgEJEkKDw9XbW2tHA6H%2B/WXhYWFufu/2nfhtcPhUGhoaKvmX1VVperqao%2B2wMBgRUdHt2r960lAgL/HX1y7wMBvvg%2Bph3VQC%2BugFtbhS7WwZMC68cYb9dJLL7lfh4aG6oEHHtA///M/684777zkOn5%2Bfh5/L9d/OVfqvxqFhYVav369R1tGRoYyMzONbaOjCQsL8vYUfJ7NFmJsLOphHdTCOqiFdfhCLSwZsC7l29/%2BtlpaWuTv76%2B6ujqPPofDocjISEmSzWa7ZH/fvn0VFRUlSaqrq1NwcLCkL25sr6urU1RUlJxO5yXXleQevzXS09OVkpLi0RYYGCyHo6HVY1wvAgL8FRYWpPr6s3I6%2BbbmN2Hivy/qYR3UwjqohXVcSy1Mfvi8GpYMWAcPHtTWrVvd3%2B6TpE8%2B%2BUSdO3fWXXfdpT/84Q8ey5eUlCgxMVGSZLfbVVpaqsmTJ0v64ob58vJypaamKiYmRhERESorK1OPHj0kSRUVFWpqalJCQoKqq6t14sQJORwO2Ww299h9%2BvRRSEjrCxQdHX3R5cDq6lNqbubAvByns4X98w2Z3H/UwzqohXVQC%2BvwhVpY8iJmVFSUNm/erPz8fDU1Nenw4cPKy8vT9OnTde%2B99%2Br48ePauHGjGhsbtX37dr333ntKT0%2BXJE2bNk1FRUXau3evGhoa9MQTT6hr165KSUlRQECApk6dqry8PB09elQ1NTXKycnRuHHj1L17d/Xv31%2BJiYlavXq16uvrVVFRofz8fM2cOdPLewQAAPgSS57Bio6OVn5%2BvtauXatf/vKXstlsmjBhgjIzM9W5c2dt2LBBjz76qHJzc9WjRw/l5ua6b3xPSkrSkiVLtGzZMtXU1CghIUH5%2Bfnq0qWLJGnhwoVqaGjQlClT5HQ6lZycrFWrVrm3/eSTT2rlypUaNWqUQkJCNGPGDM2YMcMbuwEAAPgoP5fL5fL2JK4H1dWnvD0FSwoM9JfNFiKHo8Fyp3vH5xV7ewpXZVvWiG88hpXrcb2hFtZBLazjWmpxww3d2nhWl2bJS4QAAAC%2BjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIb5RMB6/PHHdeutt7pf79mzR5MmTZLdbteYMWP06quveixfUFCg5ORkJSYmKi0tTWVlZe6%2Bc%2BfOaeXKlRo8eLAGDhyozMxM1dbWuvuPHTumOXPmaMCAARo2bJjWrFmjlpaWtn%2BTAACgw7B8wDp48KBeeeUV9%2BuTJ09q/vz5Sk1N1b59%2B7Rs2TJlZ2erpKREkvTWW28pLy9POTk52rt3r%2B666y7NmzdPZ86ckSStWbNG%2B/fvV1FRkXbu3KnGxkYtX75ckuRyubRgwQLZbDa9%2B%2B67evHFF7Vt2zYVFBS0/xsHAAA%2By9IBq6WlRY888ohmz57tbnvttdfUq1cvzZo1S0FBQUpJSdHo0aO1ZcsWSdLmzZuVmpqqoUOHKjg4WBkZGZKkXbt2qbm5WVu3blVWVpZiYmIUGRmppUuXavfu3Tp58qQOHDigiooKZWdnKzw8XLGxsZo7d642bdrkjbcPAAB8VKDpAU%2BfPq3Q0FAjY23atEldu3bVxIkTlZeXJ0kqLy9XfHy8x3JxcXHatm2bu3/ChAnuPj8/P/Xv31%2BlpaWKi4vT6dOnPdaPjY1VUFCQysrKVFVVpZ49eyoiIsLdHx8fr8OHD1/V%2B6qqqlJ1dbVHW2BgsKKjo69uB1wHAgL8Pf7i2gUGfvN9SD2sg1pYB7WwDl%2BqhfGANXLkSI0fP15paWm6/fbbr3mczz77TM8884xeeOEFj3aHw6F%2B/fp5tEVERLjvo3I4HB4BSZLCw8NVW1srh8Phfv1lYWFh7v6v9l147XA4Wh2wCgsLtX79eo%2B2jIwMZWZmtmr961FYWJC3p%2BDzbLYQY2NRD%2BugFtZBLazDF2phPGCtWrVKr7/%2BumbNmqWbb75ZaWlpuvfeexUZGXlV4%2BTk5Gjq1Kn6zne%2Bo2PHjl1xeT8/P4%2B/l%2Bu/0vompKenKyUlxaMtMDBYDkeDsW10FAEB/goLC1J9/Vk5nXyZ4Jsw8d8X9bAOamEd1MI6rqUWJj98Xg3jAWvy5MmaPHmyamtr9cYbb%2BiNN97QunXrlJKSorS0NI0YMeKKY%2BzZs0elpaV6/PHHL%2BqLjIxUXV2dR5vD4XAHOJvNdsn%2Bvn37KioqSpJUV1en4OBgSV/c2F5XV6eoqCg5nc5Lrnthu60VHR190eXA6upTam7mwLwcp7OF/fMNmdx/1MM6qIV1UAvr8IVatNlFzMjISP3jP/6jNm3apJycHBUXF%2BtHP/qRxo0bp%2B3bt3/tuq%2B%2B%2BqoqKyuVlJSkIUOGaMqUKZKkIUOG6NZbb/V47IIklZSUKDExUZJkt9tVWlrq7nM6nSovL1diYqJiYmIUERHhsX5FRYWampqUkJAgu92uEydOuEPVhbH79OmjkBDvJGAAAOB7jJ/BuqCmpkYvv/yyXn75ZR05ckQjR47U1KlTVV1drVWrVunIkSP68Y9/fMl1H3roIS1atMj9urKyUunp6XrllVfU0tKiDRs2aOPGjZo2bZreeecdvffee3rppZckSdOmTdOiRYv03e9%2BV3a7Xc8%2B%2B6y6du2qlJQUBQQEaOrUqcrLy1O/fv0UHBysnJwcjRs3Tt27d1f37t2VmJio1atX65FHHtHf//535efn6yc/%2BUlb7SYAANABGQ9Y77//vjZv3qxdu3bJZrPpvvvu09SpU9WjRw/3MnFxcZo7d%2B5lA1Z4eLjHzebNzc2SpG9961uSpA0bNujRRx9Vbm6uevToodzcXPeN70lJSVqyZImWLVummpoaJSQkKD8/X126dJEkLVy4UA0NDZoyZYqcTqeSk5O1atUq97aefPJJrVy5UqNGjVJISIhmzJihGTNmGN1HAACgY/NzuVwukwP2799fw4cP17Rp09xnjS5l4sSJeu2110xu2tKqq095ewqWFBjoL5stRA5Hg%2BWup4/PK/b2FK7Ktqwr3994JVaux/WGWlgHtbCOa6nFDTd0a%2BNZXZrxM1g7duxQTEyMmpqa3OGqoaHhonuYrqdwBbQHXwqEJsIgAFiZ8Zvc/fz8NHHiRO3atcvdVlhYqHvuuUdHjx41vTkAAADLMR6wHnvsMX3nO9/xeMjohR9mfuyxx0xvDgAAwHKMXyL8y1/%2Bonfeecf9nClJ6t69u1asWKHk5GTTmwMAALAc42ewXC6X%2B1t/X3b27Fm1tHBzIAAA6PiMB6wRI0ZoyZIlKi8vV319verq6vSXv/xFixcv1siRI01vDgAAwHKMXyJcsWKFHnjgAU2ZMsXj9/2GDBmi7Oxs05sDAACwHOMBKyoqSs8//7wOHTqkw4cPy%2BVyqXfv3oqNjTW9KQAAAEtqs5/KiY2NVUxMjPt1U1OTJKlz585ttUkAAABLMB6wPvzwQ61atUp/%2B9vf5HQ6L%2Bo/ePCg6U0CAABYivGAtWrVKnXr1k0PP/ywunbtanp4AAAAyzMesA4fPqz//M//dP%2B4MgAAwPXG%2BGMaevToofPnz5seFgAAwGcYD1gPPPCAcnJydPr0adNDAwAA%2BATjlwjXr1%2BvY8eOaevWrbLZbB7PwpKk//iP/zC9SQAAAEsxHrBGjRqlTp06mR4WAADAZxgPWIsXLzY9JAAAgE8xfg%2BWJH300UdatmyZfvCDH0iSWlpatG3btrbYFAAAgOUYD1g7d%2B7UjBkz5HA4tH//fklSZWWlVqxYoc2bN5veHAAAgOUYD1i/%2BtWvtGbNGv3qV79y3%2BDeo0cPPfnkk9q4caPpzQEAAFiO8YD1ySefaOzYsZLk8Q3CYcOG6fjx46Y3BwAAYDnGA1anTp1UV1d3Ufvhw4f56RwAAHBdMB6w7r77bmVnZ%2BvQoUOSJIfDoffff19ZWVlKTk42vTkAAADLMR6wli1bJpfLpXvuuUfnzp3T8OHDNXfuXN1000166KGHTG8OAADAcow/ByssLEwbNmzQoUOHdPjwYfn5%2Bal3797q3bu36U0BAABYkvGAdUFsbKxiY2PbangAAADLMh6wRo4cedk%2Bp9OpPXv2mN4kAACApRgPWOnp6R6PZ2hpadGxY8dUXFysefPmmd4cAACA5RgPWAsXLrxke0lJiX73u9%2BZ3hwAAIDltMlvEV5KYmKiDhw40F6bAwAA8Jp2C1iffvqpPv/88/baHAAAgNcYv0Q4bdq0i9qampr0P//zPxo9erTpzQEAAFiO8YB1yy23eNzkLkldunTRfffdp/vuu8/05gAAACzHeMD6%2Bc9/bnpIAAAAn2I8YG3evFmdOnVq1bKTJ082vXkAAACvMx6wHnvsMZ07d04ul8uj3c/Pz6PNz8%2BPgAUAADok4wHrl7/8pV588UXNnz9fsbGxcjqd%2Bvjjj5Wfn69Zs2Zp2LBhpjcJAABgKcYf0/D444/rkUceUUJCgoKCghQaGqqBAwfqkUce0aOPPtrqcf76179q9uzZuvPOOzV06FAtWrRIVVVVkqQ9e/Zo0qRJstvtGjNmjF599VWPdQsKCpScnKzExESlpaWprKzM3Xfu3DmtXLlSgwcP1sCBA5WZmana2lp3/7FjxzRnzhwNGDBAw4YN05o1a9TS0vIN9woAALieGA9Yx44dU1hY2EXt4eHhOnHiRKvGaGpq0j/90z9p0KBB%2BtOf/qQ333xTtbW1WrVqlU6ePKn58%2BcrNTVV%2B/bt07Jly5Sdna2SkhJJ0ltvvaW8vDzl5ORo7969uuuuuzRv3jydOXNGkrRmzRrt379fRUVF2rlzpxobG7V8%2BXJJksvl0oIFC2Sz2fTuu%2B/qxRdf1LZt21RQUGBo7wAAgOuB8YDVu3dv5eTkyOFwuNs%2B//xz5ebmqnfv3q0a4%2BzZs1q8eLHmzZunzp07KzIyUt/73vf0t7/9Ta%2B99pp69eqlWbNmKSgoSCkpKRo9erS2bNki6Yub7FNTUzV06FAFBwcrIyNDkrRr1y41Nzdr69atysrKUkxMjCIjI7V06VLt3r1bJ0%2Be1IEDB1RRUaHs7GyFh4crNjZWc%2BfO1aZNm0zvJgAA0IEZvwcrOztb8%2BfP10svvaSQkBD5%2Bfnp9OnTCgkJ0TPPPNOqMcLDw5WWlibpi7NKn3zyiV5%2B%2BWWNHz9e5eXlio%2BP91g%2BLi5O27ZtkySVl5drwoQJ7j4/Pz/1799fpaWliouL0%2BnTpz3Wj42NVVBQkMrKylRVVaWePXsqIiLC3R8fH6/Dhw/r9OnTCg0NbdX8q6qqVF1d7dEWGBis6OjoVq1/PQkI8Pf4i%2BtDYCD1vhKODeugFtbhS7UwHrBuv/12vfPOO3r33XdVWVkpl8ulG2%2B8UUlJSa0OKBccP35cY8eOldPpVHp6uhYtWqQ5c%2BaoX79%2BHstFRES476NyOBweAUn6IrDV1ta6z6qFh4d79IeFhbn7v9p34bXD4Wj1/AsLC7V%2B/XqPtoyMDGVmZrZq/etRWFiQt6eAdmSzhXh7Cj6DY8M6qIV1%2BEItjAcsSQoKCtKYMWN04sQJxcTEXPM4PXv2VGlpqT799FOtWLFCDz744GWXvfD0%2BK8%2BRf6r/Vda34T09HSlpKR4tAUGBsvhaDC2jY4iIMBfYWFBqq8/K6eTLxNcLzgWroxjwzqohXVcSy289YHOeMBqbGxUTk6OioqKJEmlpaWqr6/XAw88oNzcXHXr1u2qxvPz89Mtt9yiJUuWKDU1VXfddZfq6uo8lnE4HIqMjJQk2Wy2S/b37dtXUVFRkqS6ujoFBwdL%2BuISZF1dnaKiouR0Oi%2B5riT3%2BK0RHR190eXA6upTam7mwLwcp7OF/XMdodatx7FhHdTCOnyhFsYvYj711FP68MMPtXbtWvn7///w58%2Bf17/%2B67%2B2aox9%2B/bpu9/9rpqbm91tFx6VMHz4cI/HLkhSSUmJEhMTJUl2u12lpaXuPqfTqfLyciUmJiomJkYREREe61dUVKipqUkJCQmy2%2B06ceKExw36JSUl6tOnj0JCuKQBAABax3jAevvtt5WXl6dx48a5L7uFhYUpJydHu3fvbtUYcXFxOnv2rHJzc3X27FnV1tbq6aef1p133qmJEyfq%2BPHj2rhxoxobG7V9%2B3a99957Sk9PlyRNmzZNRUVF2rt3rxoaGvTEE0%2Boa9euSklJUUBAgKZOnaq8vDwdPXpUNTU1ysnJ0bhx49S9e3f1799fiYmJWr16terr61VRUaH8/HzNnDnT9G4CAAAdmPGAVVVVpVtuueWi9qioKJ0%2BfbpVY4SGhuq5557TwYMHNWrUKE2YMEEhISF64oknFBUVpQ0bNmjr1q0aNGiQ1q1bp9zcXPeN70lJSVqyZImWLVumYcOG6b/%2B67%2BUn5%2BvLl26SJIWLlyoIUOGaMqUKRozZoy6d%2B/u8QDUJ598UqdOndKoUaP0wx/%2BUNOmTdOMGTO%2B%2BY4BAADXDT/XV3808BsaN26cHnvsMd1xxx267bbb9NFHH0mSXn/9da1fv17bt283uTmfUV19yttTsKTAQH/ZbCFyOBosdz19fF6xt6fQYW3LGuHtKVielY%2BN6w21sI5rqcUNN1zdvd%2BmGL/Jffbs2frJT36i1NRUOZ1O/du//ZtKS0u1Y8cOPfzww6Y3BwAAYDnGA9a0adMUERGh559/XsHBwdqwYYN69%2B6ttWvXaty4caY3BwAAYDnGA1ZNTY3GjRtHmAIAANctoze5t7S0KDk5WYZv6wIAAPApRgOWv7%2B/hg8f7v5dQAAAgOuR8UuEPXr00OOPP678/HzdfPPN6tSpk0d/bm6u6U0CAABYivGA9fHHH6t3796S5PFEdAAAgOuFsYC1ePFirVu3Ti%2B88IK77ZlnnlFGRoapTQAAAPgEY/dg7dq166K2/Px8U8MDAAD4DGMB61LfHOTbhAAA4HpkLGBd%2BGHnK7UBAAB0dMZ/7BkAAOB6R8ACAAAwzNi3CM%2BfP6/777//im08BwsAAHR0xgLWHXfcoaqqqiu2AQAAdHTGAtaXn38FAABwPeMeLAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2BR7f3xAAAULklEQVQBAAAYRsACAAAwLNDbE8D1Y3xesbenAABAuyBgAWh3vha2t2WN8PYUAPgYLhECAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwywbsI4dO6b58%2Bdr8ODBGjZsmJYsWaLPP/9cknTw4EFNmzZNiYmJSkpK0vPPP%2B%2Bx7htvvKHvfe97stvt%2Bv73v6/i4v//SrjL5dK6des0YsQI3XbbbZo9e7aOHj3q7nc4HFq8eLFuv/12DRo0SA8//LAaGxvb500DAIAOwbIBa/78%2BYqIiNDu3bv1yiuv6NChQ/rFL36hs2fPau7cubr99tu1Z88ePfXUU3r22We1Y8cOSVJpaamWLl2qRYsW6YMPPtAPfvADZWRkqLKyUpJUUFCgoqIiPffccyouLlZMTIwWLFggl8slSVq%2BfLlqamq0Y8cOvf766zp48KDWrl3rtf0AAAB8jyUD1qlTp5SQkKAHHnhAISEhio6O1pQpU/TnP/9Z77zzjs6fP6/7779fISEhGjBggNLT01VYWChJKioqUlJSkiZMmKCuXbsqLS1Nffv21SuvvCJJ2rx5s370ox%2Bpf//%2BCg0N1dKlS3Xo0CF9%2BOGHqqmp0e7du7Vs2TJ1795dN954o7KyslRUVKSmpiZv7hIAAOBDLBmwunXrppycHEVFRbnbTpw4oZtuuknl5eXq16%2BfAgIC3H1xcXEqLS2VJJWXlys%2BPt5jvAv9586d06FDh5SQkODuCw0N1c0336zS0lKVl5crMDBQt956q7s/Pj5eZ86c0SeffNJWbxcAAHQwPvFTOQcOHNALL7ygDRs26I033lB4eLhHf0REhOrq6tTS0iKHw6GIiAiP/vDwcH388ceqq6uTy%2BW6aP3w8HDV1tYqPDxcoaGh8vf39%2BiTpNra2lbPt6qqStXV1R5tgYHBio6ObvUYAKwjMLD9P4sGBPh7/IX3UAvr8KVaWD5g/eUvf9H8%2BfN1//33a9iwYXrjjTeuaRw/P7827f%2BywsJCrV%2B/3qMtIyNDmZmZrR4DgHXYbCFe23ZYWJDXtg1P1MI6fKEWlg5Yu3fv1gMPPKCVK1fq3nvvlSRFRUXpyJEjHss5HA7ZbDb5%2B/srMjJSDofjov7IyEj3MnV1dRf1R0VFKSoqSqdOnZLT6XRfgrww1pcvV15Jenq6UlJSPNoCA4PlcDS0egwA1uGNYzcgwF9hYUGqrz8rp7Ol3beP/0ctrONaauGtD0iWDVj79%2B/X0qVL9dRTT2nEiP//JXu73a5NmzapublZgYFfTL%2BkpESJiYnu/rKyMo%2BxDhw4oHvuuUedO3dW3759VVZWpkGDBkmS6urqdOTIEdntdsXExKilpUUVFRWKi4tzj92tWzfdcsstrZ57dHT0RZcDq6tPqbmZAxPwRd48dp3OFv7fYRHUwjp8oRaWvIjZ3Nys7OxsLVq0yCNcSVJSUpJCQkKUm5urhoYG7du3Ty%2B99JJmzpwpSUpLS1NxcbHefPNNNTY26oUXXtCRI0c0efJkSdL06dP13HPP6a9//atOnTql1atXKyEhQYmJibLZbBo/frxycnL02Wef6fjx41q3bp3S09PVqVOndt8PAADAN/m5LjwAykL%2B/Oc/a%2BbMmercufNFfdu3b9eZM2e0cuVKlZWVKSoqSj/%2B8Y81ffp09zI7duxQbm6uTpw4odjYWGVnZ%2BvOO%2B909z/99NP6/e9/r4aGBg0ZMkQ/%2B9nP9K1vfUvSF4%2BIWLVqlXbt2qVOnTpp4sSJWrp06SXncjWqq099o/U7gvF5xVdeCLCgbVkjrryQYYGB/rLZQuRwNFj%2Bk3pHRy2s41pqccMN3dp4VpdmyYDVERGwCFjwXQSs6xu1sA5fCliWvEQIAADgywhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwLNDbEwAAqxufV%2BztKVyVbVkjvD0F4LrHGSwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDBLB6z3339fw4cP1%2BLFiy/qe%2BONN/S9731Pdrtd3//%2B91VcXOzuc7lcWrdunUaMGKHbbrtNs2fP1tGjR939DodDixcv1u23365Bgwbp4YcfVmNjo7v/4MGDmjZtmhITE5WUlKTnn3%2B%2Bbd8oAADoUCwbsH79619r9erV6tWr10V9paWlWrp0qRYtWqQPPvhAP/jBD5SRkaHKykpJUkFBgYqKivTcc8%2BpuLhYMTExWrBggVwulyRp%2BfLlqqmp0Y4dO/T666/r4MGDWrt2rSTp7Nmzmjt3rm6//Xbt2bNHTz31lJ599lnt2LGj/d48AADwaZYNWF26dNGWLVsuGbCKioqUlJSkCRMmqGvXrkpLS1Pfvn31yiuvSJI2b96sH/3oR%2Brfv79CQ0O1dOlSHTp0SB9%2B%2BKFqamq0e/duLVu2TN27d9eNN96orKwsFRUVqampSe%2B8847Onz%2Bv%2B%2B%2B/XyEhIRowYIDS09NVWFjY3rsAAAD4KMsGrFmzZqlbt26X7CsvL1d8fLxHW1xcnEpLS3Xu3DkdOnRICQkJ7r7Q0FDdfPPNKi0tVXl5uQIDA3Xrrbe6%2B%2BPj43XmzBl98sknKi8vV79%2B/RQQEHDR2AAAAK0R6O0JXAuHw6GIiAiPtvDwcH388ceqq6uTy%2BVSeHj4Rf21tbUKDw9XaGio/P39Pfokqba2Vg6H46J1IyIiVFdXp5aWFo/1LqeqqkrV1dUebYGBwYqOjr6q9wkA1yIw0LKfnX1SQIC/x194jy/VwicD1uX4%2Bfm1aX9rFRYWav369R5tGRkZyszMNDI%2BAHwdmy3E21PokMLCgrw9BfwfX6iFTwasyMhIORwOjzaHw6HIyEjZbDb5%2B/urrq7uov6oqChFRUXp1KlTcjqd7suAF8a60H/kyJGL1r0wbmukp6crJSXFoy0wMFgOR8NVvU8AuBb8v8asgAB/hYUFqb7%2BrJzOFm9P57p2LbXw1gcOnwxYdrtdZWVlHm0HDhzQPffco86dO6tv374qKyvToEGDJEl1dXU6cuSI7Ha7YmJi1NLSooqKCsXFxUmSSkpK1K1bN91yyy2y2%2B3atGmTmpubFRgY6O5PTExs9fyio6MvuhxYXX1Kzc0cmADaHv%2BvaRtOZwv71iJ8oRbWv4h5CWlpaSouLtabb76pxsZGvfDCCzpy5IgmT54sSZo%2Bfbqee%2B45/fWvf9WpU6e0evVqJSQkKDExUTabTePHj1dOTo4%2B%2B%2BwzHT9%2BXOvWrVN6ero6deqkpKQkhYSEKDc3Vw0NDdq3b59eeuklzZw508vvGgAA%2BAo/14WHQ1mM3W6XJDU3N0uS%2B2zSgQMHJEk7duxQbm6uTpw4odjYWGVnZ%2BvOO%2B90r//000/r97//vRoaGjRkyBD97Gc/07e%2B9S1J0qlTp7Rq1Srt2rVLnTp10sSJE7V06VJ17txZkvTxxx9r5cqVKisrU1RUlH784x9r%2BvTp3%2Bj9VFef%2BkbrdwTj84qvvBCAb2xb1ghvT6FDCQz0l80WIoejwfJnTTq6a6nFDTdc%2BokEbc2yAaujIWARsID2QsAyi4BlHb4UsHzyEiEAAICVEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMCzQ2xMAAJjlSz%2Bszg9To6PiDBYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYTzJHQDgNb701HmJJ8%2Bj9TiDBQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGBXp7AgAA%2BIrxecXensJV2ZY1wttTuG5xBgsAAMAwAtYlHDt2THPmzNGAAQM0bNgwrVmzRi0tLd6eFgAA8BFcIvwKl8ulBQsWqE%2BfPnr33Xf12Wefae7cuerevbt%2B%2BMMfent6AADAB3AG6ysOHDigiooKZWdnKzw8XLGxsZo7d642bdrk7akBAAAfwRmsrygvL1fPnj0VERHhbouPj9fhw4d1%2BvRphYaGXnGMqqoqVVdXe7QFBgYrOjra%2BHwBALgcX7op/60HRl1xmYAAf4%2B/VkbA%2BgqHw6Hw8HCPtguvHQ5HqwJWYWGh1q9f79G2YMECLVy40NxEfdCfHxt3UVtVVZUKCwuVnp5OALUA6mEd1MI6qIV1VFVVqaDgOZ%2BohfUjoA9KT0/Xyy%2B/7PEvPT3d29OypOrqaq1fv/6iM37wDuphHdTCOqiFdfhSLTiD9RVRUVGqq6vzaHM4HJKkyMjIVo0RHR1t%2BWQNAADaDmewvsJut%2BvEiRPuUCVJJSUl6tOnj0JCQrw4MwAA4CsIWF/Rv39/JSYmavXq1aqvr1dFRYXy8/M1c%2BZMb08NAAD4iIBVq1at8vYkrGbUqFH64x//qEcffVRvvvmmpk%2Bfrjlz5nh7Wh1WSEiIBg8ezBlCi6Ae1kEtrINaWIev1MLP5XK5vD0JAACAjoRLhAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbDQ5p599lmNHDlSAwcO1OzZs3X06FEdPXpUt956q%2Bx2u8e/bdu2udcrKChQcnKyEhMTlZaWprKyMi%2B%2BC9/2wQcfXLSvExISdOutt0qS9uzZo0mTJslut2vMmDF69dVXPdanFuZ8XS04LtpfWVmZZs2apTvuuEPDhw/XkiVL5HA4JHFctLfL1cJnjwsX0IZ%2B%2B9vfuu677z7XsWPHXHV1da6HHnrI9bOf/cxVVlbmio%2BPv%2Bx6O3bscA0YMMC1Z88eV0NDg%2Bvpp592jRgxwtXQ0NCOs%2B/YnnnmGdeiRYtclZWVrttuu81VUFDgOnPmjGvnzp0uu93u%2Buijj1wuF7VoDxdqwXHRvpqbm13Dhw93PfHEE65z5865HA6H64c//KErMzOT46KdfV0tfPW44AwW2tRvfvMbrVixQj179lR4eLhycnK0YsUKff755%2BrWrdtl19u8ebNSU1M1dOhQBQcHKyMjQ5K0a9eu9pp6h3bixAkVFBRoyZIleu2119SrVy/NmjVLQUFBSklJ0ejRo7VlyxZJ1KKtfbkWHBftq7q6Wp999pkmTpyozp07KyIiQqNHj1Z5eTnHRTv7ulr46nFBwEKbOXnypCorK/Xpp59q7NixGjJkiLKysuRwOFRfX6%2BWlhbNmzdPgwYN0tixY/X888/L9X%2B/PV5eXq74%2BHj3WH5%2Bfurfv79KS0u99XY6lHXr1ik1NVU9evS4aF9LUlxcnHtfU4u29eVacFy0rxtvvFFxcXF66aWXdPbsWdXW1uqtt97S3XffzXHRzr6uFr56XBCw0GYqKyvl5%2Bent99%2BW4WFhfrDH/6g48ePa8WKFercubP69OmjmTNn6v3339cjjzyi9evXuz8dOhwORUREeIwXHh6u2tpab7yVDuXTTz/V22%2B/rTlz5kj6Yl%2BHh4d7LBMREeHe19Si7Xy1FhwX7cvPz09PPfWUdu7cqQEDBmjYsGFqaWnRT3/6U46LdvZ1tfDV44KAhTZz/vx5nT9/Xg8%2B%2BKBsNptuuukmZWZm6u2339bw4cP129/%2BVklJSeratatGjBih9PR0FRUVSfriYLuUy7Wj9V588UWNGTNGkZGRX7vchX1NLdrOV2uRnJzMcdGOmpqaNG/ePE2YMEH79%2B9XcXGxQkND9eCDD152HY6LtvF1tfDV44KAhTZz4RNFaGiou61nz55yuVyqqam5aPlvf/vb%2BuyzzyRJNptNdXV1Hv0Oh%2BOKoQBX9sc//lHjxo1zv46MjPzafU0t2s5Xa3EpHBdt509/%2BpOOHTumrKwshYSEqHv37lq4cKHeeustderUieOiHX1dLS51JsoXjgsCFtpMr169FBoa6vF12ePHjyswMFAlJSX67W9/67H8J598opiYGEmS3W73uH7udDpVXl6uxMTE9pl8B/Xxxx%2BrqqpKgwcPdrfZ7faLvtJcUlLi3tfUom1cqhbbt2/nuGhHLpdLLS0tHm3nz5%2BXJA0dOpTjoh19XS327t3rk8cFAQttplOnTkpLS9PatWtVWVmp6upqPfPMM7r33nvVpUsX/eIXv1BxcbGam5v1pz/9SVu2bNHMmTMlSdOmTVNRUZH27t2rhoYGPfHEE%2BratatSUlK8/K5828GDB3XTTTd5nFWcOHGijh8/ro0bN6qxsVHbt2/Xe%2B%2B9p/T0dEnUoq1cqhYcF%2B1rwIABCgkJ0dNPP63GxkZ9/vnn%2BvWvf62BAwfq3nvv5bhoR19Xi6CgIN88Lrz5jAh0fOfOnXP9y7/8i2vQoEGuIUOGuB566CHXqVOnXC6Xy7Vp0ybX2LFjXbfddpvrnnvucRUVFXms%2B7vf/c519913u%2Bx2u2v69Omu//7v//bGW%2BhQnnvuOdfkyZMvav/ggw9ckyZNciUkJLjGjh3r2rFjh0c/tTDvcrXguGhfH330kesf/uEfXHfccYdr6NChrszMTNff//53l8vFcdHevq4Wvnhc%2BLlc//c9RwAAABjBJUIAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMOx/AZwOLh44S8MKAAAAAElFTkSuQmCC"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common7776014525002438121">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">680.0</td>
            <td class="number">13161</td>
            <td class="number">7.6%</td>
            <td>
                <div class="bar" style="width:24%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">670.0</td>
            <td class="number">12875</td>
            <td class="number">7.5%</td>
            <td>
                <div class="bar" style="width:23%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">675.0</td>
            <td class="number">12564</td>
            <td class="number">7.3%</td>
            <td>
                <div class="bar" style="width:23%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">690.0</td>
            <td class="number">12290</td>
            <td class="number">7.1%</td>
            <td>
                <div class="bar" style="width:22%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">685.0</td>
            <td class="number">12278</td>
            <td class="number">7.1%</td>
            <td>
                <div class="bar" style="width:22%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">665.0</td>
            <td class="number">11832</td>
            <td class="number">6.8%</td>
            <td>
                <div class="bar" style="width:21%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">695.0</td>
            <td class="number">11174</td>
            <td class="number">6.5%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">660.0</td>
            <td class="number">10712</td>
            <td class="number">6.2%</td>
            <td>
                <div class="bar" style="width:19%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">700.0</td>
            <td class="number">10333</td>
            <td class="number">6.0%</td>
            <td>
                <div class="bar" style="width:19%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">705.0</td>
            <td class="number">9309</td>
            <td class="number">5.4%</td>
            <td>
                <div class="bar" style="width:17%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (30)</td>
            <td class="number">56217</td>
            <td class="number">32.5%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme7776014525002438121">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">625.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">630.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">660.0</td>
            <td class="number">10712</td>
            <td class="number">6.2%</td>
            <td>
                <div class="bar" style="width:83%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">665.0</td>
            <td class="number">11832</td>
            <td class="number">6.8%</td>
            <td>
                <div class="bar" style="width:91%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">670.0</td>
            <td class="number">12875</td>
            <td class="number">7.5%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">825.0</td>
            <td class="number">115</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">830.0</td>
            <td class="number">72</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:62%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">835.0</td>
            <td class="number">26</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:23%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">840.0</td>
            <td class="number">23</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">845.0</td>
            <td class="number">16</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:14%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">home_ownership<br/>
                <small>Categorical</small>
            </p>
        </div><div class="col-md-3">
        <table class="stats ">
            <tr class="">
                <th>Distinct count</th>
                <td>5</td>
            </tr>
            <tr>
                <th>Unique (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (n)</th>
                <td>0</td>
            </tr>
        </table>
    </div>
    <div class="col-md-6 collapse in" id="minifreqtable-4936025553574265820">
        <table class="mini freq">
            <tr class="">
        <th>MORTGAGE</th>
        <td>
            <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 46.9%">
                81081
            </div>
            
        </td>
    </tr><tr class="">
        <th>RENT</th>
        <td>
            <div class="bar" style="width:94%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 44.5%">
                76924
            </div>
            
        </td>
    </tr><tr class="">
        <th>OWN</th>
        <td>
            <div class="bar" style="width:18%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 8.4%">
                &nbsp;
            </div>
            14565
        </td>
    </tr><tr class="other">
        <th>Other values (2)</th>
        <td>
            <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 0.1%">
                &nbsp;
            </div>
            175
        </td>
    </tr>
        </table>
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#freqtable-4936025553574265820, #minifreqtable-4936025553574265820"
           aria-expanded="true" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="col-md-12 extrapadding collapse" id="freqtable-4936025553574265820">
        
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">MORTGAGE</td>
            <td class="number">81081</td>
            <td class="number">46.9%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">RENT</td>
            <td class="number">76924</td>
            <td class="number">44.5%</td>
            <td>
                <div class="bar" style="width:94%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">OWN</td>
            <td class="number">14565</td>
            <td class="number">8.4%</td>
            <td>
                <div class="bar" style="width:18%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">OTHER</td>
            <td class="number">137</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">NONE</td>
            <td class="number">38</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr>
    </table>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">id<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>172745</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>100.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>4110500</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>54734</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>10234817</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-9055646698779244224">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAAvRJREFUeJzt3DFLanEcxvEnkVoaa82lwUVcjDBsaekKjQUuBRG%2BgLbgQlP3DYQE0TuIwGpJ26Kg5RJBkCS0SENTb6Di3OFCIV6f9OI5eq/fDzRk58Rv%2Bfr7QydHgiAIBOCPYv0eABhk8X4P0AuZ75Wu7/n541sIk%2BB/wwYBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBDAIBjIF71ORvHhsBwsIGAQwCAQwCAQwCAQwCAQwCAQwCAQwCAQwCAQwCAQwCAYyBexYL3ev2%2BTU%2B8qhzbBDAIBDA4IjVBY4yw4cNAhgEAhgcsQYM/1E5WAgEHRnWT9DniAUYbJAhxDGuc2wQwGCDhIh36n8fGwQwhnaD8O6OTrBBAGNoNwjC9z/87YQNAhgEAhgEAhgEAhgEAhgEAhgEAhgEAhgjQRAE/R4CGFRsEMAgEMAgEMAgEMAgEPTc5eWl5ubmtLm52dV9i4uLSqVSTV/JZFLlcjmkSb/G4%2B7oqYODAx0dHSmRSHR9b7Vabfq%2B0WioUChofn6%2BV%2BN1jQ2CnhobG7OBVKtV5fN5pdNpLS0t6fT0tO3v2tnZ0cbGhiYmJsIa90tsEPTU2tpa2589Pj5qa2tLe3t7mpmZ0e3trYrFohKJhNLpdNO119fXqtfrKpVKYY9ssUEQmcPDQy0sLCibzSoejyuTySifz%2Bv4%2BLjl2lKppGKxqNHR0T5M%2BokNgsg0Gg1dXFzo/Pz847UgCJTL5Zquq9Vqur%2B/1/7%2BftQjtiAQRCYWi6lQKGh7e9teV6lUlMvlND4%2BHtFk7XHEQmSmpqZUr9ebXnt%2Bftb7%2B3vTa1dXV5qdnY1ytLYIBJFZXl7Wzc2NyuWyXl9fVavVtLKy0nLkenh40PT0dB8n/cTTvOipVColSXp7e5MkxeO/T/F3d3eSpLOzM%2B3u7urp6UmTk5NaXV3V%2Bvr6x/0vLy/KZrM6OTlRMpmMePpWBAIYHLEAg0AAg0AAg0AAg0AAg0AAg0AAg0AAg0AAg0AA4xchW69tQQQH4AAAAABJRU5ErkJggg%3D%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-9055646698779244224,#minihistogram-9055646698779244224"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-9055646698779244224">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-9055646698779244224"
                                                      aria-controls="quantiles-9055646698779244224" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-9055646698779244224" aria-controls="histogram-9055646698779244224"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-9055646698779244224" aria-controls="common-9055646698779244224"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-9055646698779244224" aria-controls="extreme-9055646698779244224"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-9055646698779244224">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>54734</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>498350</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>1339400</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>3494800</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>6636700</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>9175200</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>10234817</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>10180083</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>5297200</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>2978800</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.72469</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-1.2228</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>4110500</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>2652700</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>0.38567</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>710067252482</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>8873400000000</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-9055646698779244224">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzs3XtUVXX%2B//EXcMQLyNWwNMpGR%2BVy0EwzszBoLLW0vo2IlzXlZOb41RB/OZJKDpXFNGaSl2ZkbMyvfUtTMnXyVmmXL6NT5moUMKZIx9tSCA6BKCKH8/uj5ZlOaMH4OZ5z8PlYi8U6n8/en/3Zb7aHF3tv9/FzOBwOAQAAwBh/T08AAACgpSFgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwnwhYzz33nHr06OF8vWvXLo0YMUJWq1WDBw/Wxo0bXZZfuXKlkpKSlJCQoJSUFBUWFjr7zp49q7lz5%2Brmm2/WjTfeqLS0NFVUVDj7jx49qgkTJqh3794aMGCA5s%2Bfr4aGBvfvJAAAaDG8PmAdOHBAGzZscL4%2BefKkJk%2BerJEjR%2BqTTz7RrFmzlJmZqX379kmS3n33XeXk5Cg7O1u7d%2B/WoEGDNGnSJJ0%2BfVqSNH/%2BfO3du1d5eXl6//33VVtbq9mzZ0uSHA6Hpk6dqvDwcH344Yd67bXXtGXLFq1cufLy7zgAAPBZfg6Hw%2BHpSVxMQ0ODRo8eraSkJOXk5Ki4uFjLly/Xpk2bXELX9OnT1b59ez399NN69NFHdf3112vOnDmSvgtNt99%2Bu5544gkNGTJE/fv31/PPP69f/OIXkqSSkhINGzZMH330kU6ePKnU1FTt2rVLYWFhkqQ33nhDr776qrZt23ZJ%2B1JWVn1J65/n7%2B%2BniIggVVTUqKHBa390Poe6ugd1dQ/qah41dQ9vqOtVV7X3yHYtHtlqE61evVpt2rTR8OHDlZOTI0kqKipSXFycy3KxsbHasmWLs3/YsGHOPj8/P8XExKigoECxsbE6deqUy/pdu3ZV27ZtVVhYqNLSUnXu3NkZriQpLi5Ohw4d0qlTpxQcHNykeZeWlqqsrMylzWJpp6ioqOYV4AICAvzl5%2BenVq0CZLdz6dIU6uoe1NU9qKt51NQ9ruS6em3A%2Buabb7R06VKtWrXKpd1ms6lnz54ubWFhYc77qGw2m0tAkqTQ0FBVVFTIZrM5X39fSEiIs/%2BHfedf22y2JgesNWvWaMmSJS5tU6ZMUVpaWpPWb4qQkLbGxsK/UVf3oK7uQV3No6bucSXW1WsDVnZ2tkaNGqWf/exnOnr06E8u7%2Bfn5/L9Yv0/tb4JqampSk5OdmmzWNrJZqu55LEDAvwVEtJWVVVnrri/BtyJuroHdXUP6moeNXUPb6hreHiQR7brlQFr165dKigo0HPPPdeoLyIiQpWVlS5tNptNERERkqTw8PAL9nfv3l2RkZGSpMrKSrVr107Sd/doVVZWKjIyUna7/YLrnt9uU0VFRTW6HFhWVq36enMHl93eYHQ8fIe6ugd1dQ/qah41dY8rsa5e%2Bb8IN27cqBMnTigxMVH9%2B/fXAw88IEnq37%2B/evTo4fLYBUnat2%2BfEhISJElWq1UFBQXOPrvdrqKiIiUkJCg6OlphYWEu6xcXF6uurk7x8fGyWq06fvy4M1SdH7tbt24KCvJMAgYAAL7HKwPWE088oW3btmnDhg3asGGDcnNzJUkbNmzQvffeq2PHjunVV19VbW2ttm7dqo8%2B%2BkipqamSpNGjRysvL0%2B7d%2B9WTU2NXnzxRbVp00bJyckKCAjQqFGjlJOToyNHjqi8vFzZ2dkaMmSIOnTooJiYGCUkJGjevHmqqqpScXGxcnNzNW7cOE%2BWAwAA%2BBivvEQYGhrqcrN5fX29JOnqq6%2BWJC1btkzPPPOMFixYoE6dOmnBggXOG98TExM1c%2BZMzZo1S%2BXl5YqPj1dubq5at24tSXrsscdUU1OjBx54QHa7XUlJScrKynJu66WXXtLcuXN1%2B%2B23KygoSGPHjtXYsWMv054DAICWwKufg9WSmHoOlsXir/DwINlsNVfc9Wx3oq7uQV3dg7qaR03dwxvq6qnnYHnlJUIAAABfRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw7zyOVhomYbm5Ht6Cs2yJX2gp6cAAPBRnMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGeW3A%2BuKLLzR%2B/Hj17dtXt9xyi6ZNm6bS0lIdOXJEPXr0kNVqdfnasmWLc92VK1cqKSlJCQkJSklJUWFhobPv7Nmzmjt3rm6%2B%2BWbdeOONSktLU0VFhbP/6NGjmjBhgnr37q0BAwZo/vz5amhouKz7DgAAfJtXBqy6ujo9/PDD6tevn/72t79p8%2BbNqqioUFZWlqqrq9WqVSvt37/f5Wvo0KGSpHfffVc5OTnKzs7W7t27NWjQIE2aNEmnT5%2BWJM2fP1979%2B5VXl6e3n//fdXW1mr27NmSJIfDoalTpyo8PFwffvihXnvtNW3ZskUrV670WC0AAIDv8cqAdebMGU2fPl2TJk1SYGCgIiIidPfdd%2Burr77St99%2Bq/bt21903bVr12rkyJG65ZZb1K5dO02ZMkWStGPHDtXX12v9%2BvVKT09XdHS0IiIilJGRoZ07d%2BrkyZPav3%2B/iouLlZmZqdDQUHXt2lUTJ07U6tWrL9euAwCAFsDi6QlcSGhoqFJSUiR9d1bp4MGDeuuttzR06FBVVVWpoaFBkyZN0t69exUeHq4xY8Zo/Pjx8vPzU1FRkYYNG%2BYcy8/PTzExMSooKFBsbKxOnTqluLg4Z3/Xrl3Vtm1bFRYWqrS0VJ07d1ZYWJizPy4uTocOHdKpU6cUHBzcpPmXlpaqrKzMpc1iaaeoqKhLKYskKSDA3%2BU73MdiocaXiuPVPairedTUPa7kunplwDrv2LFjuuuuu2S325Wamqpp06bpww8/VLdu3TRu3Di99NJL%2Buyzz5SWlqbg4GClpKTIZrO5BCTpu8BWUVEhm83mfP19ISEhzv4f9p1/bbPZmhyw1qxZoyVLlri0TZkyRWlpac3a/x8TEtLW2Fi4sPDwIE9PocXgeHUP6moeNXWPK7GuXh2wOnfurIKCAv3rX//Sk08%2Bqd/%2B9rdasGCBkpKSnMsMHDhQqampysvLU0pKivz8/C441sXam9rfHKmpqUpOTnZps1jayWarueSxAwL8FRLSVlVVZ2S3c/O9O5n4eV3pOF7dg7qaR03dwxvq6qk/lr06YEnfBZ8uXbpo5syZGjlypObMmaOIiAiXZa699lpt375dkhQeHq7KykqXfpvNpu7duysyMlKSVFlZqXbt2kn67hJkZWWlIiMjZbfbL7iupEbb/DFRUVGNLgeWlVWrvt7cwWW3NxgdD41RX3M4Xt2DuppHTd3jSqyrV14U/eSTT/SLX/xC9fX1zrbzj0r4%2B9//rv/93/91Wf7gwYOKjo6WJFmtVhUUFDj77Ha7ioqKlJCQoOjoaIWFhbk8tqG4uFh1dXWKj4%2BX1WrV8ePHnaFKkvbt26du3bopKIjLRQAAoGm8MmDFxsbqzJkzWrBggc6cOaOKigotXrxYffv2VZs2bfSHP/xB%2Bfn5qq%2Bv19/%2B9jetW7dO48aNkySNHj1aeXl52r17t2pqavTiiy%2BqTZs2Sk5OVkBAgEaNGqWcnBwdOXJE5eXlys7O1pAhQ9ShQwfFxMQoISFB8%2BbNU1VVlYqLi5Wbm%2BscGwAAoCm88hJhcHCwli9frueff1633367LBaL%2Bvfvr2effVYdO3bU7Nmz9fTTT%2BvkyZO69tpr9eSTT%2BoXv/iFJCkxMVEzZ87UrFmzVF5ervj4eOXm5qp169aSpMcee0w1NTV64IEHZLfblZSUpKysLOe2X3rpJc2dO1e33367goKCNHbsWI0dO9YTZQAAAD7Kz%2BFwODw9iStBWVm1kXEsFn%2BFhwfJZqvxuevZQ3PyPT2FZtmSPtDTU/B5vny8ejPqah41dQ9vqOtVV1382Znu5JWXCAEAAHwZAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDvDZgffHFFxo/frz69u2rW265RdOmTVNpaakkadeuXRoxYoSsVqsGDx6sjRs3uqy7cuVKJSUlKSEhQSkpKSosLHT2nT17VnPnztXNN9%2BsG2%2B8UWlpaaqoqHD2Hz16VBMmTFDv3r01YMAAzZ8/Xw0NDZdnpwEAQIvglQGrrq5ODz/8sPr166e//e1v2rx5syoqKpSVlaWTJ09q8uTJGjlypD755BPNmjVLmZmZ2rdvnyTp3XffVU5OjrKzs7V7924NGjRIkyZN0unTpyVJ8%2BfP1969e5WXl6f3339ftbW1mj17tiTJ4XBo6tSpCg8P14cffqjXXntNW7Zs0cqVKz1WCwAA4Hu8MmCdOXNG06dP16RJkxQYGKiIiAjdfffd%2Buqrr7Rp0yZdf/31evDBB9W2bVslJyfrzjvv1Lp16yRJa9eu1ciRI3XLLbeoXbt2mjJliiRpx44dqq%2Bv1/r165Wenq7o6GhFREQoIyNDO3fu1MmTJ7V//34VFxcrMzNToaGh6tq1qyZOnKjVq1d7shwAAMDHeGXACg0NVUpKiiwWixwOh77%2B%2Bmu99dZbGjp0qIqKihQXF%2BeyfGxsrAoKCiSpUb%2Bfn59iYmJUUFCgw4cP69SpUy79Xbt2Vdu2bVVYWKiioiJ17txZYWFhzv64uDgdOnRIp06dcvNeAwCAlsLi6Qn8mGPHjumuu%2B6S3W5Xamqqpk2bpgkTJqhnz54uy4WFhTnvo7LZbC4BSfousFVUVMhmszlff19ISIiz/4d951/bbDYFBwc3ad6lpaUqKytzabNY2ikqKqpJ6/%2BYgAB/l%2B9wH4uFGl8qjlf3oK7mUVP3uJLr6tUBq3PnziooKNC//vUvPfnkk/rtb3970WX9/Pxcvl%2Bs/6fWN2HNmjVasmSJS9uUKVOUlpZmbBshIW2NjYULCw8P8vQUWgyOV/egruZRU/e4Euvq1QFL%2Bi74dOnSRTNnztTIkSM1aNAgVVZWuixjs9kUEREhSQoPD79gf/fu3RUZGSlJqqysVLt27SR9d2N7ZWWlIiMjZbfbL7iuJOf4TZGamqrk5GSXNoulnWy2miaPcTEBAf4KCWmrqqozstv5343uZOLndaXjeHUP6moeNXUPb6irp/5Y9sqA9cknn2j27NnaunWrLJbvpnj%2BUQm33nqr3nrrLZfl9%2B3bp4SEBEmS1WpVQUGB7r//fkmS3W5XUVGRRo4cqejoaIWFhamwsFCdOnWSJBUXF6uurk7x8fEqKyvT8ePHZbPZFB4e7hy7W7duCgpq%2Bg8oKiqq0eXAsrJq1debO7js9gaj46Ex6msOx6t7UFfzqKl7XIl19cqLorGxsTpz5owWLFigM2fOqKKiQosXL1bfvn01fPhwHTt2TK%2B%2B%2Bqpqa2u1detWffTRR0pNTZUkjR49Wnl5edq9e7dqamr04osvqk2bNkpOTlZAQIBGjRqlnJwcHTlyROXl5crOztaQIUPUoUMHxcTEKCEhQfPmzVNVVZWKi4uVm5urcePGebgiAADAl/g5HA6HpydxIQcOHNDzzz%2BvgoICWSwW9e/fX7Nnz1bHjh21Z88ePfPMM/r666/VqVMnzZgxQ4MHD3au%2B8Ybbyg3N1fl5eWKj4/XU089pZ///OeSvnvG1u9//3tt2rRJdrtdSUlJysrKUvv27SVJJ06c0Ny5c/X3v/9dQUFBGjt2rKZOnXrJ%2B1NWVn3JY0jf3XgdHh4km61G9fUNGpqTb2RcNLYlfaCnp%2BDzfni8wgzqah41dQ9vqOtVV7X3yHa9NmC1NAQs30PAunTe8ObaElFX86ipe3hDXT0VsLzyEiEAAIAvI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMK8NWEePHtXkyZN18803a8CAAZo5c6a%2B/fZbNTQ0qGfPnoqPj5fVanV%2BvfLKK85133nnHd19992yWq269957lZ%2Bf7%2BxzOBxauHChBg4cqF69emn8%2BPE6cuSIs99ms2n69Onq06eP%2BvXrpzlz5qi2tvay7jsAAPBtXhuwJk%2BerLCwMO3cuVMbNmxQSUmJ/vCHP6i6uloOh0Pvv/%2B%2B9u/f7/yaMGGCJKmgoEAZGRmaNm2aPv30Uz300EOaMmWKTpw4IUlauXKl8vLytHz5cuXn5ys6OlpTp06Vw%2BGQJM2ePVvl5eXavn27/vrXv%2BrAgQN64YUXPFYHAADge7wyYFVXVys%2BPl4zZsxQUFCQoqKi9MADD2jPnj2qqqqSJIWEhFxw3by8PCUmJmrYsGFq06aNUlJS1L17d23YsEGStHbtWj3yyCOKiYlRcHCwMjIyVFJSos8//1zl5eXauXOnZs2apQ4dOqhjx45KT09XXl6e6urqLtv%2BAwAA32bx9AQupH379srOznZpO378uK655hp9%2B%2B238vPzU2ZmpvLz89W2bVvde%2B%2B9SktLU6tWrVRUVKTExESXdWNjY1VQUKCzZ8%2BqpKRE8fHxzr7g4GBdd911Kigo0KlTp2SxWNSjRw9nf1xcnE6fPq2DBw%2B6tP%2BY0tJSlZWVubRZLO0UFRXV3FI0EhDg7/Id7mOxUONLxfHqHtTVPGrqHldyXb0yYP3Q/v37tWrVKi1btkyS1KtXLyUlJemZZ55RSUmJpk6dqoCAAKWnp8tmsyksLMxl/dDQUH355ZeqrKyUw%2BFQaGhoo/6KigqFhoYqODhY/v7%2BLn2SVFFR0eT5rlmzRkuWLHFpmzJlitLS0pq13z8mJKStsbFwYeHhQZ6eQovB8eoe1NU8auoeV2JdvT5gffbZZ5o8ebIef/xxDRgwQNJ3AeY8q9WqRx99VH/605%2BUnp5%2B0XH8/Px%2BdDuX2v99qampSk5OdmmzWNrJZqtp8hgXExDgr5CQtqqqOiO7veGSx8PFmfh5Xek4Xt2DuppHTd3DG%2BrqqT%2BWvTpg7dy5UzNmzNDcuXN13333XXS5a6%2B9VhUVFXI4HIqIiJDNZnPpt9lsioiIUHh4uPz9/VVZWdmoPzIyUpGRkaqurpbdbldAQICzT5IiIyObPO%2BoqKhGlwPLyqpVX2/u4LLbG4yOh8aorzkcr%2B5BXc2jpu5xJdbVay%2BK7t27VxkZGVq0aJFLuNq1a5eWLl3qsuzBgwfVuXNn%2Bfn5yWq1qrCw0KV///79SkhIUGBgoLp37%2B7SX1lZqcOHD8tqtSo2NlYNDQ0qLi529u/bt0/t27dXly5d3LOjAACgxfHKgFVfX6/MzExNmzZNAwcOdOkLCwvTH//4R23YsEH19fXav3%2B/XnnlFY0bN06SlJKSovz8fG3evFm1tbVatWqVDh8%2BrPvvv1%2BSNGbMGC1fvlxffPGFqqurNW/ePMXHxyshIUHh4eEaOnSosrOz9c033%2BjYsWNauHChUlNT1apVq8teBwAA4Jv8HOcfAGXIqVOnFBwcfElj7NmzR%2BPGjVNgYGCjvq1bt6qoqEiLFy/W4cOHFRUVpdTUVP3617923py%2Bfft2LViwQMePH1fXrl2VmZmpvn37OsdYvHix3njjDdXU1Kh///56%2BumndfXVV0v67hERWVlZ2rFjh1q1aqXhw4crIyPjgnNpjrKy6kta/zyLxV/h4UGy2WpUX9%2BgoTn5P70S/iNb0gf%2B9EL4UT88XmEGdTWPmrqHN9T1qqvae2S7xgNW7969NXToUKWkpKhPnz4mh/ZpBCzfQ8C6dN7w5toSUVfzqKl7eENdPRWwjF8izMrKUllZmR588EENGzZMK1asaNYjDgAAAHyd8YB1//33a/ny5froo480ZswYbdu2TXfccYfS09NdPhMQAACgpXLbTe4RERH61a9%2BpdWrVys7O1v5%2Bfl65JFHNGTIEG3dutVdmwUAAPA4tz0Hq7y8XG%2B99ZbeeustHT58WLfddptGjRqlsrIyZWVl6fDhw3r00UfdtXkAAACPMR6wPv74Y61du1Y7duxQeHi4fvnLX2rUqFHq1KmTc5nY2FhNnDiRgAUAAFok4wHr0Ucf1a233qqFCxcqOTnZ%2BUT070tISDDywccAAADeyHjA2r59u6Kjo1VXV%2BcMVzU1NQoKcv0soE2bNpneNAAAgFcwfpO7n5%2Bfhg8frh07djjb1qxZo3vuuUdHjhwxvTkAAACvYzxgPfvss/rZz37m8pDRESNGyGq16tlnnzW9OQAAAK9j/BLhZ599pg8%2B%2BEDt2rVztnXo0EFPPvmkkpKSTG8OAADA6xg/g%2BVwOFRfX9%2Bo/cyZM2po4OMHAABAy2c8YA0cOFAzZ85UUVGRqqqqVFlZqc8%2B%2B0zTp0/XbbfdZnpzAAAAXsf4JcInn3xSM2bM0AMPPCA/Pz9ne//%2B/ZWZmWl6cwAAAF7HeMCKjIzUihUrVFJSokOHDsnhcOiGG25Q165dTW8KAADAK7nto3K6du2q6Oho5%2Bu6ujpJUmBgoLs2CQAA4BWMB6zPP/9cWVlZ%2Buqrr2S32xv1HzhwwPQmAQAAvIrxgJWVlaX27dtrzpw5atOmjenhAQAAvJ7xgHXo0CH9/e9/V%2BvWrU0PDQAA4BOMP6ahU6dOOnfunOlhAQAAfIbxgDVjxgxlZ2fr1KlTpocGAADwCcYvES5ZskRHjx7V%2BvXrFR4e7vIsLEn6v//7P9ObBAAA8CrGA9btt9%2BuVq1amR4WAADAZxgPWNOnTzc9JAAAgE8xfg%2BWJP3jH//QrFmz9NBDD0mSGhoatGXLFndsCgAAwOsYD1jvv/%2B%2Bxo4dK5vNpr1790qSTpw4oSeffFJr1641vTkAAACvYzxg/elPf9L8%2BfP1pz/9yXmDe6dOnfTSSy/p1VdfNb05AAAAr2M8YB08eFB33XWXJLn8D8IBAwbo2LFjpjcHAADgdYwHrFatWqmysrJR%2B6FDh/joHAAAcEUwHrDuuOMOZWZmqqSkRJJks9n08ccfKz09XUlJSaY3BwAA4HWMB6xZs2bJ4XDonnvu0dmzZ3Xrrbdq4sSJuuaaa/TEE080eZyjR49q8uTJuvnmmzVgwADNnDlT3377rSTpwIEDGj16tBISEpSYmKgVK1a4rPvOO%2B/o7rvvltVq1b333qv8/Hxnn8Ph0MKFCzVw4ED16tVL48eP15EjR5z9NptN06dPV58%2BfdSvXz/NmTNHtbW1l1gVAABwJTEesEJCQrRs2TK98847Wrp0qZYuXaotW7Zo2bJlCg0NbfI4kydPVlhYmHbu3KkNGzaopKREf/jDH3TmzBlNnDhRffr00a5du7Ro0SK9/PLL2r59uySpoKBAGRkZmjZtmj799FM99NBDmjJlik6cOCFJWrlypfLy8rR8%2BXLl5%2BcrOjpaU6dOlcPhkCTNnj1b5eXl2r59u/7617/qwIEDeuGFF0yXCQAAtGBueQ6WJHXt2lV33nmnkpOTdcMNNzRr3erqasXHx2vGjBkKCgpSVFSUHnjgAe3Zs0cffPCBzp07p8cff1xBQUHq3bu3UlNTtWbNGklSXl6eEhMTNWzYMLVp00YpKSnq3r27NmzYIElau3atHnnkEcXExCg4OFgZGRkqKSnR559/rvLycu3cuVOzZs1Shw4d1LFjR6WnpysvL091dXXGawQAAFom409yv%2B222y7aZ7fbtWvXrp8co3379srOznZpO378uK655hoVFRWpZ8%2BeCggIcPbFxsY6n7FVVFSkxMREl3VjY2NVUFCgs2fPqqSkRPHx8c6%2B4OBgXXfddSooKNCpU6dksVjUo0cPZ39cXJxOnz6tgwcPurT/mNLSUpWVlbm0WSztFBUV1aT1f0xAgL/Ld7iPxUKNLxXHq3tQV/OoqXtcyXU1HrBSU1NdHs/Q0NCgo0ePKj8/X5MmTfqPxty/f79WrVrlvPT4w0uNYWFhqqysVENDg2w2m8LCwlz6Q0ND9eWXX6qyslIOh6PR%2BqGhoaqoqFBoaKiCg4Pl7%2B/v0idJFRUVTZ7vmjVrtGTJEpe2KVOmKC0trclj/JSQkLbGxsKFhYcHeXoKLQbHq3tQV/OoqXtciXU1HrAee%2ByxC7bv27dPr7/%2BerPH%2B%2ByzzzR58mQ9/vjjGjBggN55553/aF7fD33u6P%2B%2B1NRUJScnu7RZLO1ks9U0eYyLCQjwV0hIW1VVnZHd3nDJ4%2BHiTPy8rnQcr%2B5BXc2jpu7hDXX11B/LxgPWxSQkJGjWrFnNWmfnzp2aMWOG5s6dq/vuu0%2BSFBkZqcOHD7ssZ7PZFB4eLn9/f0VERMhmszXqj4iIcC7zw%2Bd02Ww2RUZGKjIyUtXV1bLb7c5LkOfHioyMbPK8o6KiGl0OLCurVn29uYPLbm8wOh4ao77mcLy6B3U1j5q6x5VY18t2UfRf//o2TW0pAAAfTUlEQVSX8zELTbF3715lZGRo0aJFznAlSVarVcXFxaqvr3e27du3TwkJCc7%2BwsJCl7H279%2BvhIQEBQYGqnv37i79lZWVOnz4sKxWq2JjY9XQ0KDi4mKXsdu3b68uXbo0d5cBAMAVyvgZrNGjRzdqq6ur09dff60777yzSWPU19crMzNT06ZN08CBA136EhMTFRQUpAULFmjq1KkqLCzUm2%2B%2BqZycHElSSkqKRo4cqc2bNys5OVlr167V4cOHdf/990uSxowZoyVLluiWW25R586dNW/ePMXHxzsD2tChQ5Wdna2FCxfq7NmzWrhwoVJTU9WqVatLKQsAALiC%2BDnOPwDKkCeeeKLR/UqtW7fWz3/%2Bc/3yl79s0sfl7NmzR%2BPGjVNgYGCjvq1bt%2Br06dOaO3euCgsLFRkZqUcffVRjxoxxLrN9%2B3YtWLBAx48fV9euXZWZmam%2Bffs6%2BxcvXqw33nhDNTU16t%2B/v55%2B%2BmldffXVkr57RERWVpZ27NihVq1aafjw4crIyLjgXJqjrKz6ktY/z2LxV3h4kGy2GtXXN2hoTv5Pr4T/yJb0gT%2B9EH7UD49XmPFTdfWl9wVv%2BXfGseoe3lDXq65q75HtGg9YuDAClu/xljd%2BX%2BYNb64tEQHLPI5V9/CGunoqYBm/RLh27domX047f9kOAACgJTEesJ599lmdPXtWPzwx5ufn59Lm5%2BdHwAIAAC2S8YD1xz/%2BUa%2B99pomT56srl27ym6368svv1Rubq4efPBBDRgwwPQmAQAAvIrxgPXcc8/plVdecXkO1I033qjf/e53evjhh7V582bTmwQAAPAqxp%2BDdfToUYWEhDRqDw0N1fHjx01vDgAAwOsYD1g33HCDsrOzXZ6m/u2332rBggW64YYbTG8OAADA6xi/RJiZmanJkyfrzTffVFBQkPz8/HTq1CkFBQVp6dKlpjcHAADgdYwHrD59%2BuiDDz7Qhx9%2BqBMnTsjhcKhjx45KTExUcHCw6c0BAAB4Hbd82HPbtm01ePBgHT9%2BXNHR0e7YBAAAgNcyfg9WbW2tfve736lXr14aOnSoJKmqqkqPPvqoqqvNPM0cAADAmxkPWIsWLdLnn3%2BuF154Qf7%2B/x7%2B3Llzev75501vDgAAwOsYD1jvvfeecnJyNGTIEOeHPoeEhCg7O1s7d%2B40vTkAAACvYzxglZaWqkuXLo3aIyMjderUKdObAwAA8DrGA9bVV1%2BtvXv3Nmrftm2brrnmGtObAwAA8DrG/xfh%2BPHj9d///d8aOXKk7Ha7/vKXv6igoEDbt2/XnDlzTG8OAADA6xgPWKNHj1ZYWJhWrFihdu3aadmyZbrhhhv0wgsvaMiQIaY3BwAA4HWMB6zy8nINGTKEMAUAAK5YRu/BamhoUFJSkhwOh8lhAQAAfIrRM1j%2B/v669dZbtWXLFg0bNszk0AB%2BwtCcfE9Pocm2pA/09BQAwK2MXyLs1KmTnnvuOeXm5uq6665Tq1atXPoXLFhgepMAAABexXjA%2BvLLL3XDDTdIkmw2m%2BnhAQAAvJ6xgDV9%2BnQtXLhQq1atcrYtXbpUU6ZMMbUJAAAAn2DsJvcdO3Y0asvNzTU1PAAAgM8wFrAu9D8H%2Bd%2BEAADgSmQsYJ3/YOefagMAAGjpjH8WIQAAwJWOgAUAAGCYsf9FeO7cOT3%2B%2BOM/2cZzsAAAQEtn7AzWTTfdpNLSUpevC7U1x8cff6xbb71V06dPd2nfvXu3evToIavV6vK1b98%2BSd/dXL9w4UINHDhQvXr10vjx43XkyBHn%2BjabTdOnT1efPn3Ur18/zZkzR7W1tc7%2BAwcOaPTo0UpISFBiYqJWrFhxCZUBAABXGmNnsL7//CsT/vznP2vdunW6/vrrG/VVV1erS5cu2rZt2wXXXblypfLy8rR8%2BXJFR0fr%2Beef19SpU/X222/Lz89Ps2fPVk1NjbZv3y673a7JkyfrhRdeUGZmps6cOaOJEydqxIgReuWVV/Tll19q4sSJ6ty5s%2B666y6j%2BwgAAFomr70Hq3Xr1hcNWN9%2B%2B63at29/0XXXrl2rRx55RDExMQoODlZGRoZKSkr0%2Beefq7y8XDt37tSsWbPUoUMHdezYUenp6crLy1NdXZ0%2B%2BOAD56XNoKAg9e7dW6mpqVqzZo07dxcAALQgXhuwHnzwwYuGqKqqKn377bf61a9%2Bpb59%2B%2Bqee%2B7Rhg0bJElnz55VSUmJ4uPjncsHBwfruuuuU0FBgYqKimSxWNSjRw9nf1xcnE6fPq2DBw%2BqqKhIPXv2VEBAgLM/NjZWBQUFbtpTAADQ0hj/LMLLITg4WNdee63S0tIUExOjHTt2aMaMGYqKitLPfvYzORwOhYaGuqwTGhqqiooKhYaGKjg4WP7%2B/i59klRRUSGbzdZo3bCwMFVWVqqhocFlvYspLS1VWVmZS5vF0k5RUVH/6S47BQT4u3yH%2B1gs1NhdqO2laUnvA95yLLSkmnqTK7muPhmwRo0apVGjRjlfDxs2TNu2bdO6des0c%2BbMi673Uw8%2BNfVg1DVr1mjJkiUubVOmTFFaWpqR8SUpJKStsbFwYeHhQZ6eQovla7XtO2erp6fQYnnbscB7q3tciXX1yYB1Iddee60KCgoUHh4uf39/VVZWuvTbbDZFRkYqMjJS1dXVstvtzsuANptNkpz9hw8fbrTu%2BXGbIjU1VcnJyS5tFks72Ww1/%2BnuOQUE%2BCskpK2qqs7Ibm%2B45PFwcSZ%2BXrgwaovzvOVY4L3VPbyhrp4K8T4ZsFavXq2QkBANGzbM2Xbw4EFFR0crMDBQ3bt3V2Fhofr16ydJqqys1OHDh2W1WhUdHa2GhgYVFxcrNjZWkrRv3z61b99eXbp0kdVq1erVq1VfXy%2BLxeLsT0hIaPL8oqKiGl0OLCurVn29uYPLbm8wOh4ao77uQ21xnrcdC7y3useVWFefvChaX1%2BvefPmqaCgQOfOndM777yjjz76SGPGjJEkjRkzRsuXL9cXX3yh6upqzZs3T/Hx8UpISFB4eLiGDh2q7OxsffPNNzp27JgWLlyo1NRUtWrVSomJiQoKCtKCBQtUU1OjTz75RG%2B%2B%2BabGjRvn4b0GAAC%2BwmvPYFmtVknfhSlJeu%2B99yRJ%2B/fv17hx41RVVaW0tDTZbDbdcMMNWrp0qeLi4iRJo0ePVllZmR5%2B%2BGHV1NSof//%2BWrRokXPsp556SllZWRo8eLBatWql4cOHa9q0aZKkwMBALVu2THPnztWAAQMUGRmpmTNnatCgQZdz9wEAgA/zczgcDk9P4kpQVlZtZByLxV/h4UGy2WpUX9%2BgoTn5RsZFY1vSB3p6Cs3iS8cCtcV53nIs/PC9FWZ4Q12vuuriz810J5%2B8RAgAAODNCFgAAACGee09WAAAeBtfu1zsLZdgr0ScwQIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhlk8PQEAwJVraE6%2Bp6cAuAVnsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYxkflALjs%2BHgUAC2dV5/B%2Bvjjj3Xrrbdq%2BvTpjfreeecd3X333bJarbr33nuVn//vN2yHw6GFCxdq4MCB6tWrl8aPH68jR444%2B202m6ZPn64%2BffqoX79%2BmjNnjmpra539Bw4c0OjRo5WQkKDExEStWLHCvTsKAABaFK8NWH/%2B8581b948XX/99Y36CgoKlJGRoWnTpunTTz/VQw89pClTpujEiROSpJUrVyovL0/Lly9Xfn6%2BoqOjNXXqVDkcDknS7NmzVV5eru3bt%2Buvf/2rDhw4oBdeeEGSdObMGU2cOFF9%2BvTRrl27tGjRIr388svavn375dt5AADg07w2YLVu3Vrr1q27YMDKy8tTYmKihg0bpjZt2iglJUXdu3fXhg0bJElr167VI488opiYGAUHBysjI0MlJSX6/PPPVV5erp07d2rWrFnq0KGDOnbsqPT0dOXl5amurk4ffPCBzp07p8cff1xBQUHq3bu3UlNTtWbNmstdAgAA4KO8NmA9%2BOCDat%2B%2B/QX7ioqKFBcX59IWGxurgoICnT17ViUlJYqPj3f2BQcH67rrrlNBQYGKiopksVjUo0cPZ39cXJxOnz6tgwcPqqioSD179lRAQECjsQEAAJrCJ29yt9lsCgsLc2kLDQ3Vl19%2BqcrKSjkcDoWGhjbqr6ioUGhoqIKDg%2BXv7%2B/SJ0kVFRWy2WyN1g0LC1NlZaUaGhpc1ruY0tJSlZWVubRZLO0UFRXVrP28kIAAf5fvcB%2BLhRoD8G2efh%2B7kn9n%2BWTAuhg/Pz%2B39jfVmjVrtGTJEpe2KVOmKC0tzcj4khQS0tbYWLiw8PAgT08BAC6Jt7yPXYm/s3wyYEVERMhms7m02Ww2RUREKDw8XP7%2B/qqsrGzUHxkZqcjISFVXV8tutzsvA54f63z/4cOHG617ftymSE1NVXJyskubxdJONltNs/bzQgIC/BUS0lZVVWdktzdc8ni4OBM/LwDwJE%2B/j3nD7yxPhUyfDFhWq1WFhYUubfv379c999yjwMBAde/eXYWFherXr58kqbKyUocPH5bValV0dLQaGhpUXFys2NhYSdK%2BffvUvn17denSRVarVatXr1Z9fb0sFouzPyEhocnzi4qKanQ5sKysWvX15g4uu73B6HhojPoC8HXe8j52Jf7O8smLoikpKcrPz9fmzZtVW1urVatW6fDhw7r//vslSWPGjNHy5cv1xRdfqLq6WvPmzVN8fLwSEhIUHh6uoUOHKjs7W998842OHTumhQsXKjU1Va1atVJiYqKCgoK0YMEC1dTU6JNPPtGbb76pcePGeXivAQCAr/DaM1hWq1WSVF9fL0l67733JH13pqp79%2B564YUXtGDBAmVkZKhr165atmyZOnToIEkaPXq0ysrK9PDDD6umpkb9%2B/fXokWLnGM/9dRTysrK0uDBg9WqVSsNHz5c06ZNkyQFBgZq2bJlmjt3rgYMGKDIyEjNnDlTgwYNupy7DwAAfJif4/zTN%2BFWZWXVRsaxWPwVHh4km61G9fUNfOQIAOCitqQP9Oj2f/g7yxOuuurCj3xyN5%2B8RAgAAODNCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGGbx9AQAAIB7DM3J9/QUmszTH0xtGmewAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACG%2BWzASkpKUnx8vKxWq/PrmWeekSTt2rVLI0aMkNVq1eDBg7Vx40aXdVeuXKmkpCQlJCQoJSVFhYWFzr6zZ89q7ty5uvnmm3XjjTcqLS1NFRUVl3XfAACAb/PZgFVVVaX/%2BZ//0f79%2B51fTz75pE6ePKnJkydr5MiR%2BuSTTzRr1ixlZmZq3759kqR3331XOTk5ys7O1u7duzVo0CBNmjRJp0%2BfliTNnz9fe/fuVV5ent5//33V1tZq9uzZntxVAADgY3wyYNntdtXU1CgkJKRR36ZNm3T99dfrwQcfVNu2bZWcnKw777xT69atkyStXbtWI0eO1C233KJ27dppypQpkqQdO3aovr5e69evV3p6uqKjoxUREaGMjAzt3LlTJ0%2BevKz7CAAAfJfF0xP4T1RVVcnhcGjx4sX67LPPJEnJycnKyMhQUVGR4uLiXJaPjY3Vli1bJElFRUUaNmyYs8/Pz08xMTEqKChQbGysTp065bJ%2B165d1bZtWxUWFqpjx45Nml9paanKyspc2iyWdoqKivqP9vf7AgL8Xb4DANASWCwt6/eaTwasuro69erVS/369dOzzz6r0tJSTZs2TVlZWbLZbOrZs6fL8mFhYc77qGw2m8LCwlz6Q0NDVVFRIZvN5nz9fSEhIc26D2vNmjVasmSJS9uUKVOUlpbW5DF%2BSkhIW2NjAQDgaeHhQZ6eglE%2BGbA6duyoN9980/k6ODhYM2bM0G9%2B8xv17dv3guv4%2Bfm5fL9Y/8X8VP/3paamKjk52aXNYmknm62myWNcTECAv0JC2qqq6ozs9oZLHg8AAG9g4nfkhXgquPlkwLqQa6%2B9Vg0NDfL391dlZaVLn81mU0REhCQpPDz8gv3du3dXZGSkJKmyslLt2rWTJDkcDlVWVjr7miIqKqrR5cCysmrV15sLRHZ7g9HxAADwpJb2O80nL3geOHBAzz33nEvbwYMHFRgYqEGDBrk8dkGS9u3bp4SEBEmS1WpVQUGBs89ut6uoqEgJCQmKjo5WWFiYy/rFxcWqq6tTfHy8G/cIAAC0JD4ZsCIjI7V27Vrl5uaqrq5Ohw4dUk5OjsaMGaP77rtPx44d06uvvqra2lpt3bpVH330kVJTUyVJo0ePVl5ennbv3q2amhq9%2BOKLatOmjZKTkxUQEKBRo0YpJydHR44cUXl5ubKzszVkyBB16NDBw3sNAAB8hZ/D4XB4ehL/iU8//VQvvPCC/vnPfyo8PFzDhg1TWlqaAgMDtWfPHj3zzDP6%2Buuv1alTJ82YMUODBw92rvvGG28oNzdX5eXlio%2BP11NPPaWf//znkr67gf73v/%2B9Nm3aJLvdrqSkJGVlZal9%2B/aXNN%2BysupLWv88i8Vf4eFBstlqVF/foKE5%2BUbGBQDAk7akD3TLuFdddWm/v/9TPhuwfA0BCwCAi2tpAcsnLxECAAB4MwIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgXcDRo0c1YcIE9e7dWwMGDND8%2BfPV0NDg6WkBAAAfYfH0BLyNw%2BHQ1KlT1a1bN3344Yf65ptvNHHiRHXo0EG//vWvPT09AADgAziD9QP79%2B9XcXGxMjMzFRoaqq5du2rixIlavXq1p6cGAAB8BGewfqCoqEidO3dWWFiYsy0uLk6HDh3SqVOnFBwc/JNjlJaWqqyszKXNYmmnqKioS55fQIC/y3cAAFoCi6Vl/V4jYP2AzWZTaGioS9v51zabrUkBa82aNVqyZIlL29SpU/XYY49d8vxKS0u1cuVypaamKioqSnueHXLJY%2BK7uq5Zs8ZZV5hBXd2DuppHTd3jSq5ry4qLXiI1NVVvvfWWy1dqaqqRscvKyrRkyZJGZ8hwaaire1BX96Cu5lFT97iS68oZrB%2BIjIxUZWWlS5vNZpMkRURENGmMqKioKy6pAwCAf%2BMM1g9YrVYdP37cGaokad%2B%2BferWrZuCgoI8ODMAAOArCFg/EBMTo4SEBM2bN09VVVUqLi5Wbm6uxo0b5%2BmpAQAAHxGQlZWV5elJeJvbb79d27Zt0zPPPKPNmzdrzJgxmjBhgqen5RQUFKSbb76ZM2qGUVf3oK7uQV3No6bucaXW1c/hcDg8PQkAAICWhEuEAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsLzQ0aNHNWHCBPXu3VsDBgzQ/Pnz1dDQcMFlV65cqaSkJCUkJCglJUWFhYWXeba%2Bozl1ff3113XXXXfpxhtv1PDhw/Xee%2B9d5tn6jubU9byTJ0/qxhtv1OLFiy/TLH1Pc%2BpaUlKicePGqVevXrrjjjv06quvXt7J%2Boim1rShoUEvvfSSkpKSnO8BW7du9cCMfcfHH3%2BsW2%2B9VdOnT//R5RwOhxYuXKiBAweqV69eGj9%2BvI4cOXKZZnmZOeBVGhoaHPfdd5/j8ccfd1RWVjq%2B%2BuorR1JSkuMvf/lLo2W3b9/u6N27t2PXrl2Ompoax%2BLFix0DBw501NTUeGDm3q05dd22bZvjpptucuzdu9dx7tw5x7p16xxxcXGOf/3rXx6YuXdrTl2/b%2BrUqY4%2Bffo4Fi1adJlm6luaU9fa2lpHcnKyY82aNY7a2lrH7t27HUOGDHF89dVXHpi592pOTVetWuW47bbbHF9//bXDbrc7du7c6YiNjXV88cUXHpi598vNzXXcddddjtGjRzvS09N/dNkVK1Y4Bg4c6CgqKnJUV1c7MjMzHSNGjHA0NDRcptlePgQsL/OPf/zD0bNnT4fNZnO2vf7664677rqr0bITJ050zJs3z/m6oaHBMXDgQMemTZsuy1x9SXPqumHDBsfrr7/u0ta/f3/Hxo0b3T5PX9Ocup73wQcfOIYOHep4/PHHCVgX0Zy6rl%2B/3jFhwoTLOT2f1Jyazp4925GWlubSdssttzjefvttt8/TF61cudJRVVXlyMjI%2BMmANWzYMMeKFSucr6urqx1xcXGOvXv3unmWlx%2BXCL1MUVGROnfurLCwMGdbXFycDh06pFOnTjVaNi4uzvnaz89PMTExKigouGzz9RXNqeuIESM0ZswY5%2BuqqiqdOnVK11xzzWWbr69oTl0lqba2Vk8//bSysrJksVgu51R9SnPqumfPHt1www1KS0vTTTfdpGHDhmnz5s2Xe8perzk1veOOO/Tpp5/qiy%2B%2BUH19vd577z2dPXtWN9988%2BWetk948MEH1b59%2B59c7uzZsyopKVF8fLyzLTg4WNddd12L/L1FwPIyNptNoaGhLm3nX9tstkbLfv/N4vyyFRUV7p2kD2pOXb/P4XAoMzNTN954o2666Sa3ztEXNbeuS5cuVb9%2B/fhF9ROaU9cTJ05ow4YNGjlypPLz8zVhwgQ9/vjjOnDgwGWbry9oTk0HDx6s1NRU3XfffYqLi9OMGTOUnZ3NH1mXqLKyUg6H44I/h5b4e4s/IX2Yn59fs9rRPOfOndMTTzyhr776SitXrqSul%2Birr77S%2BvXrtXHjRk9PpUWpr6/XHXfcocTEREnSL3/5S7355pvavHmzYmJiPDw73/T2229rw4YNWr9%2Bvbp166Zdu3bp//2//6drrrlGCQkJnp5ei9QS3185g%2BVlIiMjVVlZ6dJ2/q%2BriIgIl/bw8PALLvvD5dC8ukrfXcqaNGmSjh8/rtdff11XXXXVZZmnr2lOXbOyspSens7x2QTNqWtoaGijyzOdO3fWN998495J%2Bpjm1HTVqlUaNWqUYmNjFRgYqEGDBql///56%2B%2B23L9t8W6Lw8HD5%2B/tf8OcQGRnpoVm5DwHLy1itVh0/ftzllPW%2BffvUrVs3BQUFNVr2%2B9et7Xa7ioqK%2BAvrAppTV4fDoenTpyswMFCvvvpqo8uw%2BLem1vXYsWP69NNPNX/%2BfPXv31/9%2B/fXO%2B%2B8o%2BXLl%2Bu//uu/PDF1r9ac4zUuLq7R41mOHTumzp07X5a5%2Bormvgf88PEN9fX18vfnV%2BalCAwMVPfu3V2O18rKSh0%2BfFhWq9WDM3MPjhYvExMTo4SEBM2bN09VVVUqLi5Wbm6uxo0bJ0kaMmSI9uzZI0kaPXq08vLytHv3btXU1OjFF19UmzZtlJyc7Mld8ErNqeumTZtUXFyshQsXqnXr1p6cttdral2vvvpqffjhh9qwYYPzKzk5WaNHj1Zubq6H98L7NOd4vf/%2B%2B1VcXKzVq1errq5OGzduVGFhoUaMGOHJXfA6zalpUlKS1q1bpy%2B//FJ2u127du3Srl27dMcdd3hwD3zTyZMnNWTIEOezrsaMGaPly5friy%2B%2BUHV1tebNm6f4%2BPgWeWKAe7C80EsvvaS5c%2Bfq9ttvV1BQkMaOHauxY8dKkg4ePKjTp09LkhITEzVz5kzNmjVL5eXlio%2BPV25uLqHgIppa17y8PJ04caLRjdj33Xef5s2bd9nn7e2aUteAgABdffXVLuu1bdtWwcHBXH69iKYer1FRUcrNzdWzzz6r7OxsXXfddXr55Zd13XXXeXL6XqmpNf3Nb36j%2Bvp6TZo0SRUVFerUqZOysrJ02223eXL6Xuv82af6%2BnpJcj6Yef/%2B/Tp37pwOHjyouro6Sd%2BdGCgrK9PDDz%2Bsmpoa9e/fX4sWLfLMxN3Mz%2BFwODw9CQAAgJaES4QAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAADgFT7%2B%2BGPdeuutmj59erPWu/vuu2W1Wl2%2BevbsqfXr17tppj%2BNJ7kDAACP%2B/Of/6x169bp%2Buuvb/a627Ztc3l9%2BPBhjR49Wrfffrup6TUbZ7AAAIDHtW7d%2BkcD1rZt2zR06FD16tVL9957rzZu3HjRsebNm6cJEyaoQ4cO7pruT%2BIMFgAA8LgHH3zwon0lJSV64okn9PLLL6tfv376/PPPNXHiRF1//fXq1auXy7K7du3SP//5Ty1ZssTdU/5RnMECAABe7c0331RycrIGDBggi8Wivn37aujQoXr77bcbLbtkyRJNnDhRgYGBHpjpv3EGC/j/GwWjYBSMglEwqMGjR48YDh48yLBr1y642P///xlsbGxQ1F2/fp3h2rVrDDNnzqS3EzEAAIvVTC1VO58oAAAAAElFTkSuQmCC"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-9055646698779244224">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">1181780</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">9006215</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">9827476</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">7720083</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1430672</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">585912</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">3488909</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">603276</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">904096</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">1121417</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (172735)</td>
            <td class="number">172735</td>
            <td class="number">100.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-9055646698779244224">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">54734</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">55742</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">57245</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">57416</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">58524</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">10234755</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">10234796</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">10234813</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">10234814</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">10234817</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">loan_amnt<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>1183</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.7%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>0</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>11900</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>500</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>35000</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.0%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-5194504174558260982">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAATZJREFUeJzt3NGNgkAUQFE1W9IWYU9%2B29MWsT2NDZgbMEFGPOefZH5uHgMD5zHGOAFPXfZeAMzsZ%2B8F7OX39rf6mv/7dYOVMDMTBIJAIAgEgkAgCASCQCAc4jHvK49sYQkTBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAc4rDiu6w9FOkb9s9ngkAQCASBQBAIBJv0Dfk53eczQSAIBIJAINiDTMa%2BZS4mCITpJoh/XDETEwSCQCAIBIJAIAgEgkAgCASCQCBM96KQ9Xwrvx2BfCHnvZZziwXBBGEzR5hUAmGRbz1E6hYLgkAgnMcYY%2B9FwKxMEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgP9Ukh1uzVjXsAAAAASUVORK5CYII%3D">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-5194504174558260982,#minihistogram-5194504174558260982"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-5194504174558260982">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-5194504174558260982"
                                                      aria-controls="quantiles-5194504174558260982" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-5194504174558260982" aria-controls="histogram-5194504174558260982"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-5194504174558260982" aria-controls="common-5194504174558260982"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-5194504174558260982" aria-controls="extreme-5194504174558260982"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-5194504174558260982">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>500</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>3000</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>6500</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>10000</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>15000</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>25500</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>35000</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>34500</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>8500</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>7208.2</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.60571</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>0.93464</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>11900</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>5618.4</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>1.0593</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>2055700000</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>51958000</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-5194504174558260982">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xl0VFW%2B/v8nAwgkkAmxBaNgWExJhUAzhWAweEEZtRUIiiItIpcvyCA0yCDiBY0tIIjYV2i6EVEviGCjzLbglEbxihpCMK0IHYYGAqkiAxBIsn9/eKmfZQIE3CGV1Pu1Fiur9q6zz/5UTp16OOfUiZ8xxggAAADW%2BFf2BAAAAKobAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsCywsifgK7Kz865qOX9/P4WHByknp0AlJcbyrLyfL9dP7dTua7VLvl0/tVdM7ddfX9fqeOXFESwv5%2B/vJz8/P/n7%2B1X2VCqFL9dP7dTui3y5fmqvXrUTsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAssDKngDgrXouSK3sKVyRTeMSKnsKAID/wxEsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMCyKhGwnnvuOTVv3tz9eMeOHerXr58cDoe6d%2B%2Bu9957z%2BP5y5cvV1JSkmJjYzVgwADt2bPH3VdYWKgZM2aoQ4cOatOmjcaMGaOcnBx3/6FDhzRs2DDFxcUpPj5ec%2BbMUUlJScUXCQAAqg2vD1h79%2B7VunXr3I%2BPHTumkSNHqn///tq5c6emTJmi6dOnKy0tTZL0wQcfaMGCBUpJSdHnn3%2Burl27asSIETp9%2BrQkac6cOdq1a5fWrFmjDz/8UGfPntXUqVMlScYYjR49WmFhYfr444/1xhtvaNOmTVq%2BfPm1LxwAAFRZXh2wSkpK9PTTT2vo0KHutvfff1%2B33HKLhgwZotq1a6tbt26644479M4770iSVq9erf79%2B6tTp06qU6eORo0aJUnatm2bioqK9O6772rcuHGKjIxUeHi4Jk%2BerO3bt%2BvYsWPavXu3MjMzNX36dIWEhCgqKkrDhw/XypUrK6N8AABQRXl1wFq5cqVq1aqlvn37utsyMjIUHR3t8bxWrVopPT29zH4/Pz%2B1bNlS6enpysrKUn5%2Bvkd/VFSUateurT179igjI0ONGjVSaGiouz86OloHDhxQfn5%2BRZUJAACqmcDKnsDFnDhxQq%2B88opWrFjh0e50OtWiRQuPttDQUPd1VE6n0yMgSVJISIhycnLkdDrdj3%2BuXr167v5f9l147HQ6FRwcXK65Hz9%2BXNnZ2R5tgYF11KBBg3It/3MBAf4eP32Nr9d/JQIDq89r5Mu/d1%2BuXfLt%2Bqm9etXutQErJSVFAwcO1K233qpDhw5d9vl%2Bfn4ePy/Wf7nlbVi1apUWLVrk0TZq1CiNGTPmqsesV6/2r51Wlebr9ZdHWFhQZU/BOl/%2Bvfty7ZJv10/t1YNXBqwdO3YoPT1dzz33XKm%2B8PBwuVwujzan06nw8HBJUlhYWJn9zZo1U0REhCTJ5XKpTp06kn66sN3lcikiIkLFxcVlLnthveWVnJysbt26ebQFBtaR01lQ7jEuCAjwV716tZWbe0bFxb73bUZfr/9KXM325a18%2Bffuy7VLvl0/tVdM7ZX1n0%2BvDFjvvfeejh49qsTEREk/hSBJ6tixo4YNG6b169d7PD8tLU2xsbGSJIfDofT0dN1zzz2SpOLiYmVkZKh///6KjIxUaGio9uzZo4YNG0qSMjMzde7cOcXExCg7O1tHjhyR0%2BlUWFiYe%2BymTZsqKKj8v6AGDRqUOh2YnZ2noqKr32iKi0t%2B1fJVna/XXx7V8fXx5d%2B7L9cu%2BXb91F49avfKk51PPvmktmzZonXr1mndunVasmSJJGndunXq06ePDh8%2BrNdee01nz57V5s2b9cknnyg5OVmSNGjQIK1Zs0aff/65CgoK9OKLL6pWrVrq1q2bAgICNHDgQC1YsEAHDx7UyZMnlZKSorvuukv169dXy5YtFRsbq9mzZys3N1eZmZlasmSJBg8eXJkvBwAAqGK88ghWSEiIx8XmRUVFkqTf/OY3kqTFixdr1qxZmjdvnho2bKh58%2Ba5L3xPTEzUpEmTNGXKFJ08eVIxMTFasmSJrrvuOknS448/roKCAt17770qLi5WUlKSZs6c6V7XSy%2B9pBkzZui2225TUFCQHnjgAT3wwAPXqHIAAFAd%2BJkL599QobKz865qucBAf4WFBcnpLKg2h02vRGXW33NB6jVd36%2B1aVxCZU/BGl/e7n25dsm366f2iqn9%2BuvrWh2vvLzyFCEAAEBVRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFgWWNkTgO/ouSC1sqcAAMA1wREsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWeW3A%2Bu677zR06FC1a9dOnTp10tixY3X8%2BHFJ0o4dO9SvXz85HA51795d7733nseyy5cvV1JSkmJjYzVgwADt2bPH3VdYWKgZM2aoQ4cOatOmjcaMGaOcnBx3/6FDhzRs2DDFxcUpPj5ec%2BbMUUlJybUpGgAAVAteGbDOnTunRx55RO3bt9c//vEPbdy4UTk5OZo5c6aOHTumkSNHqn///tq5c6emTJmi6dOnKy0tTZL0wQcfaMGCBUpJSdHnn3%2Burl27asSIETp9%2BrQkac6cOdq1a5fWrFmjDz/8UGfPntXUqVMlScYYjR49WmFhYfr444/1xhtvaNOmTVq%2BfHmlvRYAAKDq8cqAdebMGY0fP14jRoxQzZo1FR4erjvvvFM//PCD3n//fd1yyy0aMmSIateurW7duumOO%2B7QO%2B%2B8I0lavXq1%2Bvfvr06dOqlOnToaNWqUJGnbtm0qKirSu%2B%2B%2Bq3HjxikyMlLh4eGaPHmytm/frmPHjmn37t3KzMzU9OnTFRISoqioKA0fPlwrV66szJcDAABUMV4ZsEJCQjRgwAAFBgbKGKMff/xRa9euVc%2BePZWRkaHo6GiP57dq1Urp6emSVKrfz89PLVu2VHp6urKyspSfn%2B/RHxUVpdq1a2vPnj3KyMhQo0aNFBoa6u6Pjo7WgQMHlJ%2BfX8FVAwCA6iKwsidwKYcPH1aPHj1UXFys5ORkjR07VsOGDVOLFi08nhcaGuq%2BjsrpdHoEJOmnwJaTkyOn0%2Bl%2B/HP16tVz9/%2By78Jjp9Op4ODgcs37%2BPHjys7O9mgLDKyjBg0alGv5nwsI8Pf4CVxMYGD12UZ8ebv35dol366f2qtX7V4dsBo1aqT09HT961//0lNPPaU//OEPF32un5%2Bfx8%2BL9V9ueRtWrVqlRYsWebSNGjVKY8aMueox69Wr/WunhWouLCyosqdgnS9v975cu%2BTb9VN79eDVAUv6Kfg0btxYkyZNUv/%2B/dW1a1e5XC6P5zidToWHh0uSwsLCyuxv1qyZIiIiJEkul0t16tSR9NOF7S6XSxERESouLi5zWUnu8csjOTlZ3bp182gLDKwjp7Og3GNcEBDgr3r1ais394yKi/k2Iy7uarYvb%2BXL270v1y75dv3UXjG1V9Z/Pr0yYO3cuVNTp07V5s2bFRj40xQv3Cqhc%2BfOWrt2rcfz09LSFBsbK0lyOBxKT0/XPffcI0kqLi5WRkaG%2Bvfvr8jISIWGhmrPnj1q2LChJCkzM1Pnzp1TTEyMsrOzdeTIETmdToWFhbnHbtq0qYKCyv8LatCgQanTgdnZeSoquvqNpri45Fctj%2BqvOm4fvrzd%2B3Ltkm/XT%2B3Vo3avPNnZqlUrnTlzRvPmzdOZM2eUk5Ojl19%2BWe3atVPfvn11%2BPBhvfbaazp79qw2b96sTz75RMnJyZKkQYMGac2aNfr8889VUFCgF198UbVq1VK3bt0UEBCggQMHasGCBTp48KBOnjyplJQU3XXXXapfv75atmyp2NhYzZ49W7m5ucrMzNSSJUs0ePDgSn5FAABAVeKVASs4OFhLly7V3r17ddttt6lXr14KCgrSiy%2B%2BqIiICC1evFjvvvuu2rdvr/nz52vevHnuC98TExM1adIkTZkyRfHx8fr666%2B1ZMkSXXfddZKkxx9/XB07dtS9996r7t27q379%2Bpo1a5Z73S%2B99JLy8vJ022236fe//70GDRqkBx54oFJeBwAAUDX5GWNMZU/CF2Rn513VcoGB/goLC5LTWVDlD5v2XJBa2VOo1jaNS6jsKVhTnbb7K%2BXLtUu%2BXT%2B1V0zt119f1%2Bp45eWVR7AAAACqMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACyzHrDy8/NtDwkAAFClWA9YXbp00ZQpU7Rr1y7bQwMAAFQJ1gPWzJkzlZ2drSFDhqhXr15atmyZcnJybK8GAADAa1kPWPfcc4%2BWLl2qTz75RPfff7%2B2bNmi22%2B/XePGjVNqaqrt1QEAAHidCrvIPTw8XA899JBWrlyplJQUpaam6tFHH9Vdd92lzZs3V9RqAQAAKl1gRQ188uRJrV27VmvXrlVWVpa6dOmigQMHKjs7WzNnzlRWVpYee%2Byxilo9AABApbEesD799FOtXr1a27ZtU1hYmO677z4NHDhQDRs2dD%2BnVatWGj58OAELAABUS9YD1mOPPabOnTtr/vz56tatmwICAko9JzY2Vg0aNLC9agAAAK9gPWBt3bpVkZGROnfunDtcFRQUKCgoyON577//vu1VAwAAeAXrF7n7%2Bfmpb9%2B%2B2rZtm7tt1apV6t27tw4ePGh7dQAAAF7HesB69tlndeutt6pt27butn79%2BsnhcOjZZ5%2B1vToAAACvY/0U4VdffaWPPvpIderUcbfVr19fTz31lJKSkmyvDgAAwOtYP4JljFFRUVGp9jNnzqikpMT26gAAALyO9YCVkJCgSZMmKSMjQ7m5uXK5XPrqq680fvx4denSxfbqAAAAvI71U4RPPfWUJk6cqHvvvVd%2Bfn7u9o4dO2r69Om2VwcAAOB1rAesiIgILVu2TPv27dOBAwdkjFGTJk0UFRVle1UAAABeqcL%2BVE5UVJQiIyPdj8%2BdOydJqlmzZkWtEgAAwCtYD1jffPONZs6cqR9%2B%2BEHFxcWl%2Bvfu3Wt7lQAAAF7FesCaOXOm6tatq2nTpqlWrVq2hwcAAPB61gPWgQMH9MUXX%2Bi6666zPTQAAECVYD1gNWzYUOfPnydgAddYzwWplT2Fcts0LqGypwAAFcr6fbAmTpyolJQU5efn2x4aAACgSrB%2BBGvRokU6dOiQ3n33XYWFhXncC0uSPvvsM9urBAAA8CrWA9Ztt92mGjVq2B4WAACgyrAesMaPH297SAAAgCrF%2BjVYkvTtt99qypQpevjhhyVJJSUl2rRpU0WsCgAAwOtYD1gffvihHnjgATmdTu3atUuSdPToUT311FNavXq17dUBAAB4HesB69VXX9WcOXP06quvui9wb9iwoV566SW99tprtlcHAADgdawHrP3796tHjx6S5PENwvj4eB0%2BfNj26gAAALyO9YBVo0YNuVyuUu0HDhzgT%2BcAAACfYD1g3X777Zo%2Bfbr27dsnSXI6nfr00081btw4JSUl2V4dAACA17EesKZMmSJjjHr37q3CwkJ17txZw4cP14033qgnn3zS9uoAAAC8jvX7YNWrV0%2BLFy/Wvn37dODAAfn5%2BalJkyZq0qSJ7VUBAAB4JesB64KoqChFRUVV1PAAAABey3rA6tKly0X7iouLtWPHDturBAAA8CrWA1ZycrLH7RlKSkp06NAhpaamasSIEbZXBwAA4HWsB6zHH3%2B8zPa0tDS99dZbtlcHAADgdSrkbxGWJTY2Vrt3775WqwMAAKg01yxg/etf/9KpU6eu1eoAAAAqjfVThIMGDSrVdu7cOf3444%2B64447bK8OAADA61gPWI0bN/a4yF2SrrvuOt1333267777bK8OAADA61gPWM8//7ztIQEAAKoU6wFr9erVqlGjRrmee88999hePQAAQKWzHrCeffZZFRYWyhjj0e7n5%2BfR5ufnR8ACAADVkvWA9d///d964403NHLkSEVFRam4uFjff/%2B9lixZoiFDhig%2BPt72KgEAALyK9YD13HPP6S9/%2BYsaNGjgbmvTpo2efvppPfLII9q4caPtVQIAAHgV6/fBOnTokOrVq1eqPSQkREeOHLG9OgAAAK9jPWA1adJEKSkpcjqd7rZTp05p3rx5atKkSbnHOXTokEaOHKkOHTooPj5ekyZNct%2BodO/evRo0aJBiY2OVmJioZcuWeSy7YcMG3XnnnXI4HOrTp49SU1PdfcYYzZ8/XwkJCWrdurWGDh2qgwcPuvudTqfGjx%2Bvtm3bqn379po2bZrOnj17tS8HAADwQdYD1vTp07V582Z17txZ7dq1U/v27dWpUyf97W9/05NPPlnucUaOHKnQ0FBt375d69at0759%2B/TCCy/ozJkzGj58uNq2basdO3Zo4cKF%2BtOf/qStW7dKktLT0zV58mSNHTtWX375pR5%2B%2BGGNGjVKR48elSQtX75ca9as0dKlS5WamqrIyEiNHj3afQH%2B1KlTdfLkSW3dulXr16/X3r17NXfuXNsvEwAAqMasX4PVtm1bffTRR/r444919OhRGWN0ww03KDExUcHBweUaIy8vTzExMZo4caKCgoIUFBSke%2B%2B9V6%2B//ro%2B%2BugjnT9/XhMmTFBAQIDi4uKUnJysVatWqUePHlqzZo0SExPVq1cvSdKAAQO0evVqrVu3TiNGjNDq1av16KOPqmXLlpKkyZMnq1OnTvrmm2908803a/v27Xr33XdVv359SdK4ceM0duxYTZo0STVr1rT9cgEAgGrIesCSpNq1a6t79%2B46cuSIIiMjr3j5unXrKiUlxaPtyJEjuvHGG5WRkaEWLVooICDA3deqVSutXr1akpSRkaHExESPZVu1aqX09HQVFhZq3759iomJcfcFBwfr5ptvVnp6uvLz8xUYGKjmzZu7%2B6Ojo3X69Gnt37/fox0AAOBirAess2fPKiUlRWvWrJH00ym73NxcTZw4UfPmzVPdunWveMzdu3drxYoVWrx4sTZs2KCQkBCP/tDQULlcLpWUlMjpdCo0NNSjPyQkRN9//71cLpeMMaWWDwkJUU5OjkJCQhQcHCx/f3%2BPPknKyckp93yPHz%2Bu7Oxsj7bAwDoe36wsr4AAf4%2BfQHUQGHjp7dmXt3tfrl3y7fqpvXrVbj1gLVy4UN98843mzp2rSZMmudvPnz%2BvP/7xj5o9e/YVjffVV19p5MiRmjBhguLj47Vhw4armtcv/z6i7f6fW7VqlRYtWuTRNmrUKI0ZM6bcY/xSvXq1r3pZwNt0n/tpZU/hivzvs3dd83X6%2Bnvel%2Bun9urBesD6%2B9//rsWLF6tJkyaaPHmyJKlevXpKSUm54j/2vH37dk2cOFEzZszQ3XffLUmKiIhQVlaWx/OcTqfCwsLk7%2B%2Bv8PBwj28wXugPDw93P8flcpXqj4iIUEREhPLy8lRcXOw%2BBXlhrIiIiHLPOzk5Wd26dfNoCwysI6ezoNxjXBAQ4K969WorN/eMiotLrnh5AL/e1bx3r5avv%2Bd9uX5qr5jaw8KCrI5XXtYD1vHjx9W4ceNS7REREcrPzy/3OLt27dLkyZO1cOFCJSQkuNsdDodWrlypoqIiBQb%2BNP20tDTFxsa6%2B/fs2eMx1u7du9W7d2/VrFlTzZo10549e9S%2BfXtJksvlUlZWlhwOhyIjI1VSUqLMzEy1atXKPXbdunXLrOliGjRoUOp0YHZ2noqKrn6jKS4u%2BVXLA7h6lfHe8/X3vC/XT%2B3Vo3brJzt/85vfaNeuXaXat2zZohtvvLFcYxQVFWn69OkaO3asR7iSpMTERAUFBWnevHkqKCjQzp079fbbb2vw4MGSfvrWYGpqqjZu3KizZ89qxYoVysrKcv/dw/vvv19Lly7Vd999p7y8PM2ePVsxMTGKjY1VWFiYevbsqZSUFJ04cUKHDx/W/PnzlZycXO4/YA0AAGD9CNbQoUP1//7f/1P//v1VXFysv/71r0pPT9fWrVs1bdq0co3xzTffaN%2B%2BfXr%2B%2Bef1/PPPe/Rt3rxZixcv1owZMxQfH6%2BIiAhNmjRJXbt2lSQ1a9ZMc%2BfO1bx58zR58mRFRUVp8eLF7tsuDBo0SNnZ2XrkkUdUUFCgjh07auHChe7xn3nmGc2cOVPdu3dXjRo11LdvX40dO9bSqwMAAHyBn7lwh02LNm/erGXLlmn//v3y8/NTkyZNNHToUN1117W/UNRbZGfnXdVygYH%2BCgsLktNZUOUPm/ZckHr5JwFeaNO4hMs/yZLq9J6/Gr5cP7VXTO3XX3/ldy%2BwwfoRrJMnT%2Bquu%2B7y6TAFAAB8m9VrsEpKSpSUlKQKOCgGAABQZVgNWP7%2B/urcubM2bdpkc1gAAIAqxfopwoYNG%2Bq5557TkiVLdPPNN5f69t28efNsrxIAAMCrWA9Y33//vZo0aSJJpW74CQAA4AusBazx48dr/vz5WrFihbvtlVde0ahRo2ytAgAAoEqwdg3Wtm3bSrUtWbLE1vAAAABVhrWAVdY3B/k2IQAA8EXWApafn1%2B52gAAAKo763%2BLEAAAwNcRsAAAACyz9i3C8%2BfPa8KECZdt4z5YAACgurMWsH7729/q%2BPHjl20DAACo7qwFrJ/f/woAAMCXcQ0WAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYJm1%2B2ABQHXVc0FqZU/himwal1DZUwB8HkewAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACzjTu5VXFW7wzQAAL6AI1gAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwzKsD1qeffqrOnTtr/Pjxpfo2bNigO%2B%2B8Uw6HQ3369FFqaqq7zxij%2BfPnKyEhQa1bt9bQoUN18OBBd7/T6dT48ePVtm1btW/fXtOmTdPZs2fd/Xv37tWgQYMUGxurxMRELVu2rGILBQAA1YrXBqw///nPmj17tm655ZZSfenp6Zo8ebLGjh2rL7/8Ug8//LBGjRqlo0ePSpKWL1%2BuNWvWaOnSpUpNTVVkZKRGjx4tY4wkaerUqTp58qS2bt2q9evXa%2B/evZo7d64k6cyZMxo%2BfLjatm2rHTt2aOHChfrTn/6krVu3XrviAQBAlea1Aeu6667TO%2B%2B8U2bAWrNmjRITE9WrVy/VqlVLAwYMULNmzbRu3TpJ0urVq/Xoo4%2BqZcuWCg4O1uTJk7Vv3z598803OnnypLZv364pU6aofv36uuGGGzRu3DitWbNG586d00cffaTz589rwoQJCgoKUlxcnJKTk7Vq1apr/RIAAIAqymsD1pAhQ1S3bt0y%2BzIyMhQdHe3R1qpVK6Wnp6uwsFD79u1TTEyMuy84OFg333yz0tPTlZGRocDAQDVv3tzdHx0drdOnT2v//v3KyMhQixYtFBAQUGpsAACA8gis7AlcDafTqdDQUI%2B2kJAQff/993K5XDLGKCQkpFR/Tk6OQkJCFBwcLH9/f48%2BScrJyZHT6Sy1bGhoqFwul0pKSjyWu5jjx48rOzvboy0wsI4aNGhwRXVKUkCAv8dPALicwMCqu7/w5X0etVev2qtkwLoYPz%2B/Cu0vr1WrVmnRokUebaNGjdKYMWOuesx69Wr/2mkB8BFhYUGVPYVfzZf3edRePVTJgBUeHi6n0%2BnR5nQ6FR4errCwMPn7%2B8vlcpXqj4iIUEREhPLy8lRcXOw%2BDXhhrAv9WVlZpZa9MG55JCcnq1u3bh5tgYF15HQWXFGd0k9pvl692srNPaPi4pIrXh6A77mafY238OV9HrVXTO2V9R%2BOKhmwHA6H9uzZ49G2e/du9e7dWzVr1lSzZs20Z88etW/fXpLkcrmUlZUlh8OhyMhIlZSUKDMzU61atZIkpaWlqW7dumrcuLEcDodWrlypoqIiBQYGuvtjY2PLPb8GDRqUOh2YnZ2noqKr32iKi0t%2B1fIAfEd12Ff48j6P2qtH7VXyZOeAAQOUmpqqjRvYMEXIAAARMElEQVQ36uzZs1qxYoWysrJ0zz33SJLuv/9%2BLV26VN99953y8vI0e/ZsxcTEKDY2VmFhYerZs6dSUlJ04sQJHT58WPPnz1dycrJq1KihxMREBQUFad68eSooKNDOnTv19ttva/DgwZVcNQAAqCq89giWw%2BGQJBUVFUmS/v73v0v66UhVs2bNNHfuXM2bN0%2BTJ09WVFSUFi9erPr160uSBg0apOzsbD3yyCMqKChQx44dtXDhQvfYzzzzjGbOnKnu3burRo0a6tu3r8aOHStJqlmzphYvXqwZM2YoPj5eERERmjRpkrp27XotywcAAFWYn7lw901UqOzsvKtaLjDQX2FhQXI6C8o8bNpzQWoZSwHwZZvGJVT2FK7a5fZ51Rm1V0zt119f9i2fKlqVPEUIAADgzQhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwLLCyJwAAsKvngtTKnkK5bRqXUNlTACoER7AAAAAsI2ABAABYRsACAACwjIAFAABgGRe5AwAqTVW6IF/ionyUH0ewAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZf%2BwZAIBqqir9Me3/ffauyp6CVRzBAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFjGjUYBACinqnTjTlQujmABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICVhkOHTqkYcOGKS4uTvHx8ZozZ45KSkoqe1oAAKCKCKzsCXgbY4xGjx6tpk2b6uOPP9aJEyc0fPhw1a9fX7///e8re3oAAKAK4AjWL%2BzevVuZmZmaPn26QkJCFBUVpeHDh2vlypWVPTUAAFBFcATrFzIyMtSoUSOFhoa626Kjo3XgwAHl5%2BcrODj4smMcP35c2dnZHm2BgXXUoEGDK55PQIC/x08AAKqr6vRZR8D6BafTqZCQEI%2B2C4%2BdTme5AtaqVau0aNEij7bRo0fr8ccfv%2BL5HD9%2BXMuXL1VycnKZAe1/n73risesSo4fP65Vq1ZdtP7qjNqp3ddql3y7fl%2Bv/eWXX65WtVefqOhFkpOTtXbtWo9/ycnJVzVWdna2Fi1aVOqImK/w5fqpndp9kS/XT%2B3Vq3aOYP1CRESEXC6XR5vT6ZQkhYeHl2uMBg0aVJsEDgAArhxHsH7B4XDoyJEj7lAlSWlpaWratKmCgoIqcWYAAKCqIGD9QsuWLRUbG6vZs2crNzdXmZmZWrJkiQYPHlzZUwMAAFVEwMyZM2dW9iS8zW233aYtW7Zo1qxZ2rhxo%2B6//34NGzas0uYTFBSkDh06%2BOwRNF%2Bun9qp3Rf5cv3UXn1q9zPGmMqeBAAAQHXCKUIAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYXuzQoUMaNmyY4uLiFB8frzlz5qikpKSyp3XVkpKSFBMTI4fD4f43a9YsSdKOHTvUr18/ORwOde/eXe%2B9957HssuXL1dSUpJiY2M1YMAA7dmzx91XWFioGTNmqEOHDmrTpo3GjBmjnJyca1pbWT799FN17txZ48ePL9W3YcMG3XnnnXI4HOrTp49SU1PdfcYYzZ8/XwkJCWrdurWGDh2qgwcPuvudTqfGjx%2Bvtm3bqn379po2bZrOnj3r7t%2B7d68GDRqk2NhYJSYmatmyZRVbaBkuVvvnn3%2Bu5s2be2wDDodDaWlpkqpH7YcOHdLIkSPVoUMHxcfHa9KkSTp16lS55leR28W1cLHaS0pK1KJFi1Lv/7/85S/VpvbvvvtOQ4cOVbt27dSpUyeNHTtWx48fl1Sx%2Bzdv%2BJy4WO0HDx4s8/2%2BadOmalP7JRl4pZKSEnP33XebCRMmGJfLZX744QeTlJRk/vrXv1b21K5a27ZtzVdffVWq/ejRo6Z169Zm%2BfLl5vTp0%2BbDDz80DofDfPvtt8YYY7Zu3Wri4uLMjh07TEFBgXn55ZdNQkKCKSgoMMYYM2vWLNO7d2%2BTlZVlTp48aYYPH25GjBhxTWv7pSVLlpgePXqYQYMGmXHjxnn07d6920RHR5sNGzaYM2fOmLffftu0bt3a/Pvf/zbGGLNs2TKTkJBgMjIyTF5enpk%2Bfbrp16%2BfKSkpMcYY85//%2BZ/moYceMtnZ2ebo0aPmd7/7nZk1a5YxxpjTp0%2BbhIQE88c//tHk5%2Bebr7/%2B2rRr185s2bLFK2rfunWr6dGjx0WXreq1G2NMnz59zJNPPmny8/PNsWPHzL333mumTp162flV5HZR2bW7XC7TrFkzc/To0TKXq%2Bq1FxYWmvj4eLNo0SJTWFhoTp48aR588EEzcuTICt2/ecPnxKVq37Nnj4mOjr7oslW99sshYHmpb7/91rRo0cI4nU5321tvvXXJDydvVlRUZJo3b26%2B//77Un1//vOfTb9%2B/Tzaxo0bZ5566iljjDHDhw83s2fPdveVlJSYhIQE8/7775vz58%2Bbtm3bmg8%2B%2BMDd/8MPP1xyZ34tLF%2B%2B3OTm5prJkyeXChkzZ840I0eO9GgbMGCAefXVV40xxvTq1cssW7bM3ZeXl2eio6PNrl27zIkTJ0zz5s1NRkaGu//jjz82cXFxprCw0GzcuNF06NDBFBUVufvnzJljHnnkkQqosmyXqn316tXmvvvuu%2BiyVb323Nxc8%2BSTT5oTJ06429544w3To0ePy86vIreLa%2BFStWdlZZlmzZqZ06dPl7lsVa/d5XKZt99%2B25w/f97dtmLFCtO9e/cK3b95w%2BfEpWr/xz/%2BYTp16nTRZat67ZfDKUIvlZGRoUaNGik0NNTdFh0drQMHDig/P78SZ3Z1cnNzZYzRyy%2B/rC5duqhLly6aMWOGCgoKlJGRoejoaI/nt2rVSunp6ZJUqt/Pz08tW7ZUenq6srKylJ%2Bf79EfFRWl2rVrexxqvtaGDBmiunXrltl3qXoLCwu1b98%2BxcTEuPuCg4N18803Kz09XRkZGQoMDFTz5s3d/dHR0Tp9%2BrT279%2BvjIwMtWjRQgEBAaXGvlYuVXtubq5OnTqlhx56SO3atVPv3r21bt06SaoWtdetW1cpKSmKiIhwtx05ckQ33njjZedXkdvFtXCp2k%2BdOiU/Pz9Nnz5dnTp1UlJSkubNm6fz589Lqvq1h4SEaMCAAQoMDJQxRj/%2B%2BKPWrl2rnj17Vuj%2BzRs%2BJy5Ve25urkpKSjRixAi1b99ePXr00LJly2SMqRa1Xw4By0s5nU6FhIR4tF147HQ6K2NKv8q5c%2BfUunVrtW/fXps3b9brr7%2Bur7/%2BWjNnziyz1tDQUPe5dqfT6fEmkn56LXJyctyvxS%2BXr1evnldch1WWS9XjcrlkjCnzd3%2Bh3uDgYPn7%2B3v0SXL3l/Vaulwur7g2ITg4WDfddJOeeOIJffbZZxo1apSmTJmiHTt2VMvad%2B/erRUrVmjEiBGXnV9FbheV4ee1S1Lr1q2VlJSkbdu2aeHChXrvvff0yiuvSKrY98S1dPjwYcXExKhXr15yOBwaO3Zshe7fvOlzoqzaa9asqaZNm2rw4MH69NNP9fTTT2vRokV655133HOsDrVfDAEL18QNN9ygt99%2BWw8%2B%2BKCCg4N16623auLEiVq/fr2KiorKXMbPz8/j58X6L%2BZy/d7m19ZTFeodOHCgli1bpjZt2qhWrVrq1auXunfv7t7hXkxVrP2rr77SsGHDNGHCBMXHx1/1OFVxu/hl7TExMVq1apX69OmjOnXqyOFw6LHHHtOaNWsuOU5Vq71Ro0ZKT0/X5s2b9eOPP%2BoPf/jDRZ9b3fZvZdWelJSkN998U4mJiapVq5YSEhKUnJzs/r1Xl9ovhoDlpSIiIuRyuTzaLqTy8PDwypiSdTfddJNKSkrk7%2B9fZq0X6gwLC7to/4XTET/vN8bI5XJ5nKrwJuHh4aX%2Bh3WhnrCwsIu%2BHhEREYqIiFBeXp6Ki4s9%2BiS5%2B8ta9sK43uimm27SiRMnqlXt27dv12OPPaZp06bp4Ycfds/xUvOryO3iWiqr9rLcdNNNysnJkTGm2tQu/fTh37hxY02aNEnr169XjRo1Kmz/5m2fE7%2BsvawjiBfe71L1qr0s3rnHhRwOh44cOeKx00lLS1PTpk0VFBRUiTO7Onv37tVzzz3n0bZ//37VrFlTXbt2LXW9VFpammJjYyX99Fr8/Dqa4uJiZWRkKDY2VpGRkQoNDfVYPjMzU%2BfOnfO4ZsObOByOUvXu3r1bsbGxqlmzppo1a%2BbR73K5lJWVJYfDoVatWqmkpESZmZnu/rS0NNWtW1eNGzeWw%2BFQZmamx1HBn7%2BWlW3lypXauHGjR9v%2B/fsVGRlZbWrftWuXJk%2BerIULF%2Bruu%2B92t19ufhW5XVwrF6t9x44d7tOBF%2Bzfv1%2BNGjWSn59fla99586d%2Bo//%2BA%2BP3%2B2F09KdO3eusP2bN3xOXKr2L774Qm%2B%2B%2BabH8y%2B836WqX/tlVc619SiPgQMHmieeeMKcOnXKfPfddyYhIcG8%2BeablT2tq3Ls2DETFxdnFi9ebAoLC83%2B/ftN7969zbPPPmtOnDhh2rZta5YtW2bOnDljNm3aZBwOh9m7d68x5v//RtCOHTtMfn6%2BeeGFF8ztt99uzp49a4wxZu7cuaZXr14mKyvLnDhxwgwZMsSMHTu2Mst1K%2BubdJmZmcbhcLi/kv7666%2Bbtm3bmuzsbGOMMf/zP/9jEhISzN69e01ubq6ZMGGCGTBggHv58ePHmwcffNBkZ2ebQ4cOmd69e5sXXnjBGPPTV6aTkpLM888/b/Lz880XX3xh4uLizEcffXTtiv4/ZdW%2BYsUKEx8fb3bv3m3OnTtn1q9fb6Kjo016eroxpurXfv78edOzZ0/zxhtvlOq73Pwqcru4Fi5Ve0ZGhomOjjZ/%2B9vfzPnz501aWprp0qWLee2114wxVb/2vLw807lzZ/P888%2Bb06dPm5MnT5phw4aZBx54oML3b5X9OXGp2rdt22ZiY2PNZ599Zs6fP29SU1NNXFyc%2B5uBVb32yyFgebF///vfZvjw4SY2NtbEx8ebl19%2BubKn9Kvs3LnTDBw40MTFxZmkpCQzZ84c99eov/zyS9OvXz8TExNjevToYbZu3eqx7FtvvWVuv/1243A4zP3332/%2B%2Bc9/uvsKCwvNM888Y9q1a2fatGljnnjiCZObm3tNa/ulmJgYExMTY1q0aGFatGjhfnzBli1bTI8ePUxMTIy5%2B%2B67zZdffumx/MKFC018fLyJjY01w4cPd98PyJifvg7/xBNPmLi4ONO%2BfXvzX//1Xx5fR//nP/9pBg0aZBwOh7n99tvNW2%2B9VfEF/8ylai8pKTGvvPKKSUpKMnFxceZ3v/tdqQBUlWv/8ssvTbNmzdw1//zfoUOHLju/itwuKtrlat%2B6davp27evad26tenevbtZunSpKS4urha1G/NTiHz44YfNb3/7W9OxY0czZswY961iKnL/5g2fE5eqfeXKlaZHjx6mdevWpnfv3mbNmjUey1b12i/Fz5j/%2B74kAAAArOAaLAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACw7P8DduL33ZV3DfwAAAAASUVORK5CYII%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-5194504174558260982">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">10000.0</td>
            <td class="number">14911</td>
            <td class="number">8.6%</td>
            <td>
                <div class="bar" style="width:15%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">12000.0</td>
            <td class="number">10206</td>
            <td class="number">5.9%</td>
            <td>
                <div class="bar" style="width:10%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">15000.0</td>
            <td class="number">8587</td>
            <td class="number">5.0%</td>
            <td>
                <div class="bar" style="width:9%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">8000.0</td>
            <td class="number">7382</td>
            <td class="number">4.3%</td>
            <td>
                <div class="bar" style="width:8%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">6000.0</td>
            <td class="number">6847</td>
            <td class="number">4.0%</td>
            <td>
                <div class="bar" style="width:7%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">20000.0</td>
            <td class="number">6680</td>
            <td class="number">3.9%</td>
            <td>
                <div class="bar" style="width:7%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">5000.0</td>
            <td class="number">6289</td>
            <td class="number">3.6%</td>
            <td>
                <div class="bar" style="width:7%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">16000.0</td>
            <td class="number">3712</td>
            <td class="number">2.1%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">7000.0</td>
            <td class="number">3645</td>
            <td class="number">2.1%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">18000.0</td>
            <td class="number">3269</td>
            <td class="number">1.9%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (1173)</td>
            <td class="number">101217</td>
            <td class="number">58.6%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-5194504174558260982">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">500.0</td>
            <td class="number">5</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">700.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">725.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">750.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">800.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:20%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">34825.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34900.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34925.0</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">34975.0</td>
            <td class="number">6</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">35000.0</td>
            <td class="number">2900</td>
            <td class="number">1.7%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">loan_amnt_to_inc<br/>
                <small>Categorical</small>
            </p>
        </div><div class="col-md-3">
        <table class="stats ">
            <tr class="">
                <th>Distinct count</th>
                <td>3</td>
            </tr>
            <tr>
                <th>Unique (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (n)</th>
                <td>0</td>
            </tr>
        </table>
    </div>
    <div class="col-md-6 collapse in" id="minifreqtable-6722466286584859427">
        <table class="mini freq">
            <tr class="">
        <th>low</th>
        <td>
            <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 78.4%">
                135514
            </div>
            
        </td>
    </tr><tr class="">
        <th>avg</th>
        <td>
            <div class="bar" style="width:28%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 21.5%">
                37139
            </div>
            
        </td>
    </tr><tr class="">
        <th>high</th>
        <td>
            <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 0.1%">
                &nbsp;
            </div>
            92
        </td>
    </tr>
        </table>
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#freqtable-6722466286584859427, #minifreqtable-6722466286584859427"
           aria-expanded="true" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="col-md-12 extrapadding collapse" id="freqtable-6722466286584859427">
        
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">low</td>
            <td class="number">135514</td>
            <td class="number">78.4%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">avg</td>
            <td class="number">37139</td>
            <td class="number">21.5%</td>
            <td>
                <div class="bar" style="width:28%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">high</td>
            <td class="number">92</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr>
    </table>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">loan_status<br/>
                <small>Categorical</small>
            </p>
        </div><div class="col-md-3">
        <table class="stats ">
            <tr class="">
                <th>Distinct count</th>
                <td>2</td>
            </tr>
            <tr>
                <th>Unique (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (n)</th>
                <td>0</td>
            </tr>
        </table>
    </div>
    <div class="col-md-6 collapse in" id="minifreqtable-794032012630797727">
        <table class="mini freq">
            <tr class="">
        <th>Fully Paid</th>
        <td>
            <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 87.6%">
                151353
            </div>
            
        </td>
    </tr><tr class="">
        <th>Charged Off</th>
        <td>
            <div class="bar" style="width:14%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 12.4%">
                &nbsp;
            </div>
            21392
        </td>
    </tr>
        </table>
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#freqtable-794032012630797727, #minifreqtable-794032012630797727"
           aria-expanded="true" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="col-md-12 extrapadding collapse" id="freqtable-794032012630797727">
        
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">Fully Paid</td>
            <td class="number">151353</td>
            <td class="number">87.6%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">Charged Off</td>
            <td class="number">21392</td>
            <td class="number">12.4%</td>
            <td>
                <div class="bar" style="width:14%">&nbsp;</div>
            </td>
    </tr>
    </table>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">percent_bc_gt_75<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>135</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.1%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (%)</th>
                        <td>21.0%</td>
                    </tr>
                    <tr class="alert">
                        <th>Missing (n)</th>
                        <td>36346</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>52.646</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>100</td>
                    </tr>
                    <tr class="alert">
                        <th>Zeros (%)</th>
                        <td>12.5%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram-9025337614255567638">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAATRJREFUeJzt2sFpxDAQQNFsSEkpIj3lvD2liPSkNBA%2BXoOxkN67G3TwZwZbjzHGeAP%2B9X73AWBmH3cfgHV9fv%2B8/Mzv8%2BuCk5xngkAQCASBQBAIBIFAEAgEgUAQCASBQBAIhOmumqxwPYF1mCAQBAJBIBAEAkEgEKb7irU7X/HmYoJAEAgEK9YCXl3LrGTHmSAQBAJBIBAEAkEgEAQCQSAQBAJh2x%2BF7jxxxLaBnOGP9X6sWBBMkA1ZL48zQSCYIBxyZuqswASBIBAIVqwL7bqWrGSJQLyIXMWKBUEgEAQCQSAQBAJBIBAEAkEgEB5jjHH3IWBWJggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiEP3WiIdY6Gc9PAAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives-9025337614255567638,#minihistogram-9025337614255567638"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives-9025337614255567638">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles-9025337614255567638"
                                                      aria-controls="quantiles-9025337614255567638" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram-9025337614255567638" aria-controls="histogram-9025337614255567638"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common-9025337614255567638" aria-controls="common-9025337614255567638"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme-9025337614255567638" aria-controls="extreme-9025337614255567638"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles-9025337614255567638">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>25</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>50</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>80</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>100</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>100</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>100</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>55</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>34.436</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.6541</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-1.2091</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>52.646</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>29.443</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>-0.08618</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>7180800</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>1185.8</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-9025337614255567638">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3X1UlWW%2B//EPsCGVZzBMzaex8QHZaI5maGHqsdTJ8jQqOq4xJ9OGo/kwOjIqOlQaNcrgUWtGs6OOrSk0mnEsH6jU6jA2Zi4Pj3H6kQ4%2BLANhI4giAvv3Rz/3rx1aWNd2743v11ouF9e17%2Bv%2B3l838PG%2Bb2587Ha7XQAAADDG190FAAAAtDQELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY5rEB67PPPtO0adM0YMAA3XvvvZo7d65KS0slSYcOHdIjjzwiq9WqkSNH6u9//7vTtlu3btWwYcMUGxurCRMmKD8/3zF3%2BfJlLV%2B%2BXPfcc4/uvvtuzZkzRxUVFY75U6dOafr06erXr5/i4uK0atUqNTY23pyDBgAALYJHBqy6ujo98cQTGjhwoP7xj39o9%2B7dqqioUEpKir788kslJiZq/PjxOnz4sBYvXqzk5GTl5ORIkt59912tWbNGqamp%2BvjjjzV06FA99dRTunjxoiRp1apVOnr0qDIzM/X%2B%2B%2B%2BrtrZWS5YskSTZ7XbNnj1b4eHh%2BuCDD/Taa69pz5492rp1q9t6AQAAvI%2BP3W63u7uIbzp//ryysrL07//%2B77JYLJKk1157TX/%2B8581ceJE7dq1Szt37nS8fv78%2BQoODtazzz6rmTNnqkuXLlq6dKmkr0LT/fffr9/%2B9rcaNWqUBg0apBdffFH/9m//JkkqLi7WmDFj9OGHH%2BrLL79UQkKCDh06pLCwMEnS66%2B/ri1btmjfvn0/6JjKyqp/0PbX4uvro4iIQFVU1Kix0eP%2BGb0avXUdeuta9Nd16K3ruLK3t98ebHS95vLIM1ihoaGaMGGCLBaL7Ha7vvjiC7311lsaPXq0CgoK1KdPH6fXR0dHKy8vT5KazPv4%2BKh3797Ky8tTSUmJLly44DTfvXt3tW7dWvn5%2BSooKFDHjh0d4UqS%2BvTpoxMnTujChQsuPuob5%2BvrIx8fH/n6%2Bri7lBaH3roOvXUt%2Bus69NZ1WmJvLe4u4NucPn1aDz74oBoaGpSQkKC5c%2Bdq%2BvTp6tWrl9PrwsLCHPdR2Ww2p4AkfRXYKioqZLPZHB9/XUhIiGP%2Bm3NXP7bZbAoKCmpW3aWlpSorK3Mas1jaKCoqqlnbN5efn6/T3zCH3roOvXUt%2Bus69NZ1WmJvPTpgdezYUXl5efrXv/6lZcuW6Te/%2Bc11X%2Bvj4%2BP09/Xmv2t7EzIyMrR%2B/XqnsVmzZmnOnDnG9vF1ISGtXbIu6K0r0VvXor%2BuQ29dpyX11qMDlvRV8OnatasWLVqk8ePHa%2BjQoaqsrHR6jc1mU0REhCQpPDz8mvM9evRQZGSkJKmyslJt2rSR9NU9WpWVlYqMjFRDQ8M1t5XkWL85EhISNHz4cKcxi6WNbLaaZq/RHH5%2BvgoJaa2qqktqaOAnHU2it65Db12L/roOvXUdV/Y2PDzQ6HrN5ZEB6/Dhw1qyZIn27t3ruMn96qMSBg8erLfeesvp9Tk5OYqNjZUkWa1W5eXlady4cZKkhoYGFRQUaPz48erUqZPCwsKUn5%2BvDh06SJKKiopUV1enmJgYlZWV6cyZM7LZbAoPD3esfddddykwsPn/QFFRUU0uB5aVVau%2B3jWfkA0NjS5b%2B1ZHb12H3roW/XUdeus6Lam3HnmxMzo6WpcuXVJaWpouXbqkiooKrVu3TgMGDNDYsWN1%2BvRpbdmyRbW1tdq7d68%2B/PBDJSQkSJImTZqkzMxMffzxx6qpqdEf/vAHtWrVSsOHD5efn58mTpyoNWvW6OTJkyovL1dqaqpGjRqltm3bqnfv3oqNjdWKFStUVVWloqIibdy4UVOmTHFzRwAAgDfxyMc0SFJhYaFefPFF5eXlyWKxaNCgQVqyZInatWunI0eO6LnnntMXX3yhDh06aOHChRo5cqRj29dff10bN25UeXm5YmJi9Mwzz%2BjHP/6xpK%2BesfXCCy9o165damho0LBhw5SSkqLg4K9%2BjPPs2bNavny5/vnPfyowMFA///nPNXv27B98PK54TIPF4qvw8EDZbDUtJvF7CnrrOvTWteiv69Bb13Flb931mAaPDVgtDQHLu9Bb16G3rkV/XYfeuk5LDFgeeYkQAADAmxGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDCPfJI7AAD44UavyXZ3Cc12ZOUod5dgFGewAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADDMYwPWqVOnlJiYqHvuuUdxcXFatGiRzp8/r8bGRvXq1UsxMTGyWq2OP6%2B%2B%2Bqpj23feeUcPPfSQrFarHn74YWVnZzvm7Ha70tPTNWTIEPXt21fTpk3TyZMnHfM2m03z589X//79NXDgQC1dulS1tbU39dgBAIB389iAlZiYqLCwMB04cEA7d%2B5UcXGxfv/736u6ulp2u13vv/%2B%2BcnNzHX%2BmT58uScrLy1NSUpLmzp2rTz75RI8//rhmzZqls2fPSpK2bt2qzMxMbdq0SdnZ2erUqZNmz54tu90uSVqyZInKy8uVlZWlt99%2BW4WFhVq9erXb%2BgAAALyPRwas6upqxcTEaOHChQoMDFRUVJQee%2BwxHTlyRFVVVZKkkJCQa26bmZmp%2BPh4jRkzRq1atdKECRPUo0cP7dy5U5K0Y8cOPfnkk%2Brdu7eCgoKUlJSk4uJiHTt2TOXl5Tpw4IAWL16stm3bql27dpo3b54yMzNVV1d3044fAAB4N4u7C7iW4OBgpaamOo2dOXNG7du31/nz5%2BXj46Pk5GRlZ2erdevWevjhhzVnzhz5%2B/uroKBA8fHxTttGR0crLy9Ply9fVnFxsWJiYhxzQUFB6ty5s/Ly8nThwgVZLBb17NnTMd%2BnTx9dvHhRx48fdxr/NqWlpSorK3Mas1jaKCoq6kZb8a38/Hyd/oY59NZ16K1r0V/Xobeu15J665EB65tyc3O1bds2bdiwQZLUt29fDRs2TM8995yKi4s1e/Zs%2Bfn5ad68ebLZbAoLC3PaPjQ0VJ9//rkqKytlt9sVGhraZL6iokKhoaEKCgqSr6%2Bv05wkVVRUNLvejIwMrV%2B/3mls1qxZmjNnzg0dd3OFhLR2ybqgt65Eb12L/roOvXWdltRbjw9Yn376qRITE7VgwQLFxcVJ%2BirAXGW1WjVz5kz96U9/0rx58667jo%2BPz7fu54fOf11CQoKGDx/uNGaxtJHNVtPsNZrDz89XISGtVVV1SQ0NjUbXvtXRW9eht65Ff12H3rqeK3obHh5odL3m8uiAdeDAAS1cuFDLly/Xo48%2Bet3X3XnnnaqoqJDdbldERIRsNpvTvM1mU0REhMLDw%2BXr66vKysom85GRkYqMjFR1dbUaGhrk5%2BfnmJOkyMjIZtcdFRXV5HJgWVm16utd8wnZ0NDosrVvdfTWdeita9Ff16G3rtOSeuuxFzuPHj2qpKQkrV271ilcHTp0SC%2B99JLTa48fP66OHTvKx8dHVqtV%2Bfn5TvO5ubmKjY1VQECAevTo4TRfWVmpkpISWa1WRUdHq7GxUUVFRY75nJwcBQcHq2vXrq45UAAA0OJ4ZMCqr69XcnKy5s6dqyFDhjjNhYWF6Y9//KN27typ%2Bvp65ebm6tVXX9WUKVMkSRMmTFB2drZ2796t2tpabdu2TSUlJRo3bpwkafLkydq0aZM%2B%2B%2BwzVVdXa8WKFYqJiVFsbKzCw8M1evRopaam6ty5czp9%2BrTS09OVkJAgf3//m94HAADgnXzsVx8A5UGOHDmiKVOmKCAgoMnc3r17VVBQoHXr1qmkpERRUVFKSEjQL3/5S8fN6VlZWUpLS9OZM2fUvXt3JScna8CAAY411q1bp9dff101NTUaNGiQnn32Wd1xxx2SvnpEREpKivbv3y9/f3%2BNHTtWSUlJ16zlRpSVVf%2Bg7a/FYvFVeHigbLaaFnNK1VPQW9eht65Ff13HG3s7ek32d7/IQxxZOcolvb399mCj6zWXRwasloiA5V3orevQW9eiv67jjb0lYLkvYHnkJUIAAABvRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGWdxdAH6YAUv3uruEZtszb4i7SwAA4KbgDBYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMM8NmCdOnVKiYmJuueeexQXF6dFixbp/PnzkqTCwkJNmjRJsbGxio%2BP1%2BbNm522feedd/TQQw/JarXq4YcfVnZ2tmPObrcrPT1dQ4YMUd%2B%2BfTVt2jSdPHnSMW%2Bz2TR//nz1799fAwcO1NKlS1VbW3tzDhoAALQIHhuwEhMTFRYWpgMHDmjnzp0qLi7W73//e126dEkzZsxQ//79dejQIa1du1Yvv/yysrKyJEl5eXlKSkrS3Llz9cknn%2Bjxxx/XrFmzdPbsWUnS1q1blZmZqU2bNik7O1udOnXS7NmzZbfbJUlLlixReXm5srKy9Pbbb6uwsFCrV692Wx8AAID38ciAVV1drZiYGC1cuFCBgYGKiorSY489piNHjujgwYO6cuWKFixYoMDAQPXr108JCQnKyMiQJGVmZio%2BPl5jxoxRq1atNGHCBPXo0UM7d%2B6UJO3YsUNPPvmkevfuraCgICUlJam4uFjHjh1TeXm5Dhw4oMWLF6tt27Zq166d5s2bp8zMTNXV1bmzJQAAwItY3F3AtQQHBys1NdVp7MyZM2rfvr0KCgrUq1cv%2Bfn5Oeaio6O1Y8cOSVJBQYHi4%2BOdto2OjlZeXp4uX76s4uJixcTEOOaCgoLUuXNn5eXl6cKFC7JYLOrZs6djvk%2BfPrp48aKOHz/uNP5tSktLVVZW5jRmsbRRVFRU8xrQTH5%2BHpmPr8ti8Z56r/bW23rsDeita9Ff16G3rteSeuuRAeubcnNztW3bNm3YsEHvvPOOQkNDnebDwsJUWVmpxsZG2Ww2hYWFOc2Hhobq888/V2Vlpex2e5PtQ0NDVVFRodDQUAUFBcnX19dpTpIqKiqaXW9GRobWr1/vNDZr1izNmTOn2Wu0ROHhge4u4YaFhLR2dwktFr11LfrrOvTWdVpSbz0%2BYH366adKTEzUggULFBcXp3feeed7rePj4%2BPS%2Ba9LSEjQ8OHDncYsljay2WqavUZzeFvSN338ruTn56uQkNaqqrqkhoZGd5fTotBb16K/rkNvXc8VvXXXf%2B49OmAdOHBACxcu1PLly/Xoo49KkiIjI1VSUuL0OpvNpvDwcPn6%2BioiIkI2m63JfEREhOM1lZWVTeYjIyMVGRmp6upqNTQ0OC5BXl0rMjKy2XVHRUU1uRxYVlat%2Bvpb%2BxPSG4%2B/oaHRK%2Bv2BvTWteiv69Bb12lJvfXYUyBHjx5VUlKS1q5d6whXkmS1WlVUVKT6%2BnrHWE5OjmJjYx3z%2Bfn5Tmvl5uYqNjZWAQEB6tGjh9N8ZWWlSkpKZLVaFR0drcbGRhUVFTmtHRwcrK5du7roSAEAQEvjkQGrvr5eycnJmjt3roYMGeI0Fx8fr8DAQKWlpammpkaHDx/W9u3bNWXKFEnShAkTlJ2drd27d6u2tlbbtm1TSUmJxo0bJ0maPHmyNm3apM8%2B%2B0zV1dVasWKFYmJiFBsbq/DwcI0ePVqpqak6d%2B6cTp8%2BrfT0dCUkJMjf3/%2Bm9wEAAHgnj7xEeOzYMRUXF%2BuFF17QCy%2B84DS3d%2B9ebdiwQcuXL1dcXJwiIyO1aNEiDR06VJLUo0cPrV69WmlpaUpKSlL37t21YcMGtW3bVpI0adIklZWV6YknnlBNTY0GDRqktWvXOtZ/5plnlJKSopEjR8rf319jx47V3Llzb97BAwAAr%2Bdjv/qETbhUWVm18TUtFl%2BNXP2R8XVdZc%2B8Id/9Ig9hsfgqPDxQNltNi7kfwFPQW9eiv67jjb0dvSb7u1/kIY6sHOWS3t5%2Be7DR9ZrLIy8RAgAAeDMCFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwj/xVOQDgSbzpadjSV0/EBuBenMECAAAwzHjAunDhguklAQAAvIrxgHXfffdp8eLFOnr0qOmlAQAAvILxgJWSkqKysjJNnTpVY8aM0ebNm1VRUWF6NwAAAB7LeMAaN26cNm3apA8//FCTJ0/Wvn379MADD2jevHnKzvauG0UBAAC%2BD5fd5B4REaFf/OIXeuONN5Samqrs7Gw9%2BeSTGjVqlPbu3euq3QIAALidyx7TUF5errfeektvvfWWSkpKdN9992nixIkqKytTSkqKSkpKNHPmTFftHgAAwG2MB6yPPvpIO3bs0P79%2BxUeHq6f/exnmjhxojp06OB4TXR0tGbMmEHAAgAALZLxgDVz5kwNHjxY6enpGj58uPz8/Jq8JjY2VlFRUaZ3DQAA4BGMB6ysrCx16tRJdXV1jnBVU1OjwMBAp9ft2rXL9K4BAAA8gvGb3H18fDR27Fjt37/fMZaRkaGf/vSnOnnypOndAQAAeBzjAWvlypX60Y9%2BpP79%2BzvGHnnkEVmtVq1cudL07gAAADyO8UuEn376qQ4ePKg2bdo4xtq2batly5Zp2LBhpncHAADgcYyfwbLb7aqvr28yfunSJTU2NpreHQAAgMcxHrCGDBmiRYsWqaCgQFVVVaqsrNSnn36q%2BfPn67777jO9OwAAAI9j/BLhsmXLtHDhQj322GPy8fFxjA8aNEjJycmmdwcAAOBxjAesyMhIbd68WcXFxTpx4oTsdru6deum7t27m94VAACAR3LZr8rp3r27OnXq5Pi4rq5OkhQQEOCqXQIAAHgE4wHr2LFjSklJ0f/5P/9HDQ0NTeYLCwtN7xIAAMCjGA9YKSkpCg4O1tKlS9WqVSvTywMAAHg84wHrxIkT%2Buc//6nbbrvN9NIAgGYYsHSvu0totj3zhri7BMAljD%2BmoUOHDrpy5YrpZQEAALyG8YC1cOFCpaam6sKFC6aXBgAA8ArGLxGuX79ep06d0l//%2BleFh4c7PQtLkv77v//b9C4BAAA8ivGAdf/998vf39/0sgAAAF7DeMCaP3%2B%2B6SUBAAC8ivF7sCTpf/7nf7R48WI9/vjjkqTGxkbt2bPHFbsCAADwOMYD1vvvv6%2Bf//znstlsOnr0qCTp7NmzWrZsmXbs2GF6dwAAAB7HeMD605/%2BpFWrVulPf/qT4wb3Dh066D//8z%2B1ZcsW07sDAADwOMYD1vHjx/Xggw9KktNPEMbFxen06dOmdwcAAOBxjAcsf39/VVZWNhk/ceIEvzoHAADcEowHrAceeEDJyckqLi6WJNlsNn300UeaN2%2Behg0bZnp3AAAAHsd4wFq8eLHsdrt%2B%2BtOf6vLlyxo8eLBmzJih9u3b67e//e0NrfXRRx9p8ODBTR798PHHH6tnz56yWq1Of3JyciRJdrtd6enpGjJkiPr27atp06bp5MmTju1tNpvmz5%2Bv/v37a%2BDAgVq6dKlqa2sd84WFhZo0aZJiY2MVHx%2BvzZs3/4COAACAW43x52CFhIRow4YNKi4u1okTJ%2BTj46Nu3bqpW7duN7TOK6%2B8ojfffFNdunRpMlddXa2uXbtq375919x269atyszM1KZNm9SpUye9%2BOKLmj17tv72t7/Jx8dHS5YsUU1NjbKystTQ0KDExEStXr1aycnJunTpkmbMmKFHHnlEr776qj7//HPNmDFDHTt2dNxbBgAA8G1c8hwsSerevbtGjBih4cOH33C4kqTbbrvtugHr/PnzCg4Ovu62O3bs0JNPPqnevXsrKChISUlJKi4u1rFjx1ReXq4DBw5o8eLFatu2rdq1a6d58%2BYpMzNTdXV1OnjwoK5cuaIFCxYoMDBQ/fr1U0JCgjIyMm74GAAAwK3J%2BBms%2B%2B6777pzDQ0NOnToULPWmTp16nXnqqqqdP78ef3iF79QYWGh2rVrp5kzZ%2BrRRx/V5cuXVVxcrJiYGMfrg4KC1LlzZ%2BXl5enChQuyWCzq2bOnY75Pnz66ePGijh8/roKCAvXq1Ut%2Bfn6O%2Bejo6Bt6hldpaanKysqcxiyWNoqKimr2Gs3h5%2BeyfOwSFov31Hu1t97WY29Ab/F1fF3A17Wk3hoPWAkJCU6PZ2hsbNSpU6eUnZ2tp556ysg%2BgoKCdOedd2rOnDnq3bu39u/fr4ULFyoqKko/%2BtGPZLfbFRoa6rRNaGioKioqFBoaqqCgIPn6%2BjrNSVJFRYVsNluTbcPCwlRZWanGxkan7a4nIyND69evdxqbNWuW5syZ830PuUUIDw90dwk3LCSktbtLaLHoLSS%2BLsBZS%2Bqt8YD19NNPX3M8JydHf/nLX4zsY%2BLEiZo4caLj4zFjxmjfvn168803tWjRoutu9/Xg933mmyshIUHDhw93GrNY2shmqzGy/lXelvRNH78r%2Bfn5KiSktaqqLqmhodHd5bQo9BZfx9cFfJ0reuuuEG88YF1PbGysFi9e7LL177zzTuXl5Sk8PFy%2Bvr5NnsVls9kUGRmpyMhIVVdXq6GhwXEZ0GazSZJjvqSkpMm2V9dtjqioqCaXA8vKqlVff2t/Qnrj8Tc0NHpl3d6A3kLi6wKctaTe3rRTIP/61790/vx5I2u98cYb2r17t9PY8ePH1alTJwUEBKhHjx7Kz893zFVWVqqkpERWq1XR0dFqbGxUUVGRYz4nJ0fBwcHq2rWrrFarioqKVF9f7zQfGxtrpHYAANDyGT%2BDNWnSpCZjdXV1%2BuKLLzRixAgj%2B6ivr9eKFSvUuXNn9ezZU1lZWfrwww8dP%2Bk3efJkrV%2B/Xvfee686duyoFStWKCYmxhGSRo8erdTUVKWnp%2Bvy5ctKT09XQkKC/P39FR8fr8DAQKWlpWn27NnKz8/X9u3btWbNGiO1AwCAls94wOratWuTe5luu%2B02/exnP9PPfvazZq9jtVolyXEm6b333pMk5ebmasqUKaqqqtKcOXNks9nUrVs3vfTSS%2BrTp4%2Bkr0JeWVmZnnjiCdXU1GjQoEFau3atY%2B1nnnlGKSkpGjlypPz9/TV27FjNnTtXkhQQEKANGzZo%2BfLliouLU2RkpBYtWqShQ4d%2B/6YAAIBbio/dbre7u4hbQVlZtfE1LRZfjVz9kfF1XWXPvCHuLqHZLBZfhYcHymaraTH3A3gKb%2Bzt6DXZ7i6hxeLrgmt503v3yMpRLunt7bdf/7mZrmT8DNaOHTvk7%2B/frNeOGzfO9O4BAADcznjAWrlypS5fvqxvnhjz8fFxGvPx8SFgAQCAFsl4wPrjH/%2Bo1157TYmJierevbsaGhr0%2Beefa%2BPGjZo6dari4uJM7xIAAMCjGA9Yzz//vF599VWn50Ddfffd%2Bt3vfqcnnniiyeMVAAAAWhrjz8E6deqUQkJCmoyHhobqzJkzpncHAADgcYwHrG7duik1NdXxdHRJOn/%2BvNLS0tStWzfTuwMAAPA4xi8RJicnKzExUdu3b1dgYKB8fHx04cIFBQYG6qWXXjK9OwAAAI9jPGD1799fBw8e1AcffKCzZ8/KbrerXbt2io%2BPV1BQkOndAQAAeByX/LLn1q1ba%2BTIkTpz5ow6derkil0AAAB4LOP3YNXW1up3v/ud%2Bvbtq9GjR0uSqqqqNHPmTFVXm3%2BaOQAAgKcxHrDWrl2rY8eOafXq1fL1/f/LX7lyRS%2B%2B%2BKLp3QEAAHgc4wHrvffe05o1azRq1CjHL30OCQlRamqqDhw4YHp3AAAAHsd4wCotLVXXrl2bjEdGRurChQumdwcAAOBxjAesO%2B64Q0ePHm0yvm/fPrVv39707gAAADyO8Z8inDZtmv7jP/5D48ePV0NDg/7rv/5LeXl5ysrK0tKlS03vDgAAwOMYD1iTJk1SWFiYNm/erDZt2mjDhg3q1q2bVq9erVGjRpneHQAAgMcxHrDKy8s1atQowhQAALhlGb0Hq7GxUcOGDZPdbje5LAAAgFcxGrB8fX01ePBg7dmzx%2BSyAAAAXsX4JcIOHTro%2Beef18aNG9W5c2f5%2B/s7zaelpZneJQAAgEcxHrA%2B//xzdevWTZJks9lMLw8AAODxjAWs%2BfPnKz09Xdu2bXOMvfTSS5o1a5apXQAAAHgFY/dg7d%2B/v8nYxo0bTS0PAADgNYwFrGv95CA/TQgAAG5FxgLW1V/s/F1jAAAALZ3x30UIAABwqyNgAQAAGGbspwivXLmiBQsWfOcYz8ECAAAtnbGA9ZOf/ESlpaXfOQYAANDSGQtYX3/%2BFQAAwK2Me7AAAAAMI2ABAAAYRsACAAAwzPgvewZYoO3YAAAVsElEQVRaigFL97q7hBuyZ94Qd5cAAPh/OIMFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGMZzsIAWYvSabHeX0GxHVo5ydwkA4FIefQbro48%2B0uDBgzV//vwmc%2B%2B8844eeughWa1WPfzww8rO/v/fXOx2u9LT0zVkyBD17dtX06ZN08mTJx3zNptN8%2BfPV//%2B/TVw4EAtXbpUtbW1jvnCwkJNmjRJsbGxio%2BP1%2BbNm117oAAAoEXx2ID1yiuvaMWKFerSpUuTuby8PCUlJWnu3Ln65JNP9Pjjj2vWrFk6e/asJGnr1q3KzMzUpk2blJ2drU6dOmn27Nmy2%2B2SpCVLlqi8vFxZWVl6%2B%2B23VVhYqNWrV0uSLl26pBkzZqh///46dOiQ1q5dq5dffllZWVk37%2BABAIBX89iAddttt%2BnNN9%2B8ZsDKzMxUfHy8xowZo1atWmnChAnq0aOHdu7cKUnasWOHnnzySfXu3VtBQUFKSkpScXGxjh07pvLych04cECLFy9W27Zt1a5dO82bN0%2BZmZmqq6vTwYMHdeXKFS1YsECBgYHq16%2BfEhISlJGRcbNbAAAAvJTH3oM1derU684VFBQoPj7eaSw6Olp5eXm6fPmyiouLFRMT45gLCgpS586dlZeXpwsXLshisahnz56O%2BT59%2BujixYs6fvy4CgoK1KtXL/n5%2BTmtvWPHjmbXXlpaqrKyMqcxi6WNoqKimr1Gc/j5eWw%2BviaLxXvq9bbeeiN6DMk7vy7w3nWdltRbjw1Y38ZmsyksLMxpLDQ0VJ9//rkqKytlt9sVGhraZL6iokKhoaEKCgqSr6%2Bv05wkVVRUyGazNdk2LCxMlZWVamxsdNruejIyMrR%2B/XqnsVmzZmnOnDk3dJwtTXh4oLtLgAcJCWnt7hLgAbzx6wLvXddpSb31yoB1PT4%2BPi6db66EhAQNHz7cacxiaSObrcbI%2Bld5W9I3ffyu5G299UZVVZfU0NDo7jLgZt72dSEkpDXvXRdyRW/dFeK9MmBFRETIZrM5jdlsNkVERCg8PFy%2Bvr6qrKxsMh8ZGanIyEhVV1eroaHBcRnw6lpX50tKSppse3Xd5oiKimpyObCsrFr19bf2J%2BStfvxw1tDQyHsCXvke4L3rOi2pt17533Sr1ar8/HynsdzcXMXGxiogIEA9evRwmq%2BsrFRJSYmsVquio6PV2NiooqIix3xOTo6Cg4PVtWtXWa1WFRUVqb6%2B3mk%2BNjbW9QcGAABaBK8MWBMmTFB2drZ2796t2tpabdu2TSUlJRo3bpwkafLkydq0aZM%2B%2B%2BwzVVdXa8WKFYqJiVFsbKzCw8M1evRopaam6ty5czp9%2BrTS09OVkJAgf39/xcfHKzAwUGlpaaqpqdHhw4e1fft2TZkyxc1HDQAAvIXHXiK0Wq2S5DiT9N5770n66kxVjx49tHr1aqWlpSkpKUndu3fXhg0b1LZtW0nSpEmTVFZWpieeeEI1NTUaNGiQ1q5d61j7mWeeUUpKikaOHCl/f3%2BNHTtWc%2BfOlSQFBARow4YNWr58ueLi4hQZGalFixZp6NChN/PwAQCAF/PYgJWbm/ut8w8%2B%2BKAefPDB684//fTTevrpp685FxwcrLS0tOtu%2B%2BMf/1ivv/568woFAAD4Bq%2B8RAgAAODJCFgAAACGEbAAAAAM89h7sAAALd/oNdnuLuGGHFk5yt0lwEtwBgsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMJ7kDgBAMw1YutfdJcBLcAYLAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGWdxdAIBbz4Cle91dAgC4FGewAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMO89jlYw4YNU1lZmXx8fBxjEydO1LJly3To0CGlpqbq%2BPHjuuOOO/T000/rkUcecbxu69at2rJli8rLy9WzZ0%2BlpKSoT58%2BkqTLly9r5cqV2rt3r65cuaL7779fKSkpioiIuOnH2NKMXpPt7hIAALgpvPYMVlVVlf785z8rNzfX8WfZsmX68ssvlZiYqPHjx%2Bvw4cNavHixkpOTlZOTI0l69913tWbNGqWmpurjjz/W0KFD9dRTT%2BnixYuSpFWrVuno0aPKzMzU%2B%2B%2B/r9raWi1ZssSdhwoAALyMVwashoYG1dTUKCQkpMncrl271KVLF02dOlWtW7fW8OHDNWLECL355puSpB07dmj8%2BPG699571aZNG82aNUuStH//ftXX1%2Buvf/2r5s2bp06dOikiIkJJSUk6cOCAvvzyy5t6jAAAwHt55SXCqqoq2e12rVu3Tp9%2B%2Bqkkafjw4UpKSlJBQYHjct9V0dHR2rNnjySpoKBAY8aMccz5%2BPiod%2B/eysvLU3R0tC5cuOC0fffu3dW6dWvl5%2BerXbt2zaqvtLRUZWVlTmMWSxtFRUV9r%2BO9Hj8/r8zHAABcU0v6vuaVAauurk59%2B/bVwIEDtXLlSpWWlmru3LlKSUmRzWZTr169nF4fFhamiooKSZLNZlNYWJjTfGhoqCoqKmSz2Rwff11ISIhj%2B%2BbIyMjQ%2BvXrncZmzZqlOXPmNHsNAABuNSEhrd1dgjFeGbDatWun7du3Oz4OCgrSwoUL9atf/UoDBgy45jZXb4b/%2Bk3x15q/nu%2Ba/7qEhAQNHz7cacxiaSObrabZazRHS0r6AABUVV1SQ0Oj0TXDwwONrtdcXhmwruXOO%2B9UY2OjfH19VVlZ6TRns9kcPwUYHh5%2BzfkePXooMjJSklRZWak2bdpIkux2uyorKx1zzREVFdXkcmBZWbXq682%2BaQAAaEkaGhpbzPdKrzwFUlhYqOeff95p7Pjx4woICNDQoUOVn5/vNJeTk6PY2FhJktVqVV5enmOuoaFBBQUFio2NVadOnRQWFua0fVFRkerq6hQTE%2BPCIwIAAC2JVwasyMhI7dixQxs3blRdXZ1OnDihNWvWaPLkyXr00Ud1%2BvRpbdmyRbW1tdq7d68%2B/PBDJSQkSJImTZqkzMxMffzxx6qpqdEf/vAHtWrVSsOHD5efn58mTpyoNWvW6OTJkyovL1dqaqpGjRqltm3buvmoAQCAt/Cx2%2B12dxfxfXzyySdavXq1/vd//1fh4eEaM2aM5syZo4CAAB05ckTPPfecvvjiC3Xo0EELFy7UyJEjHdu%2B/vrr2rhxo8rLyxUTE6NnnnlGP/7xjyV9dQP9Cy%2B8oF27dqmhoUHDhg1TSkqKgoODf1C9ZWXVP2j7a7FYfDVy9UfG1wUA4GY7snKUbLYa45cIb7/9h33//r68NmB5GwIWAADX19IClldeIgQAAPBkBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWNdw6tQpTZ8%2BXf369VNcXJxWrVqlxsZGd5cFAAC8hMXdBXgau92u2bNn66677tIHH3ygc%2BfOacaMGWrbtq1%2B%2Bctfurs8AADgBTiD9Q25ubkqKipScnKyQkND1b17d82YMUNvvPGGu0sDAABegjNY31BQUKCOHTsqLCzMMdanTx%2BdOHFCFy5cUFBQ0HeuUVpaqrKyMqcxi6WNoqKijNbq50c%2BBgC0HC3p%2BxoB6xtsNptCQ0Odxq5%2BbLPZmhWwMjIytH79eqex2bNn6%2BmnnzZXqL4Kco/f8bkSEhKMh7dbXWlpqTIyMuitC9Bb16K/rkNvXae0tFTr1q1rUb1tOVHRgyQkJOitt95y%2BpOQkGB8P2VlZVq/fn2Ts2X44eit69Bb16K/rkNvXacl9pYzWN8QGRmpyspKpzGbzSZJioiIaNYaUVFRLSaBAwCAG8cZrG%2BwWq06c%2BaMI1RJUk5Oju666y4FBga6sTIAAOAtCFjf0Lt3b8XGxmrFihWqqqpSUVGRNm7cqClTpri7NAAA4CX8UlJSUtxdhKe5//77tW/fPj333HPavXu3Jk%2BerOnTp7u7rGsKDAzUPffcw9k1F6C3rkNvXYv%2Bug69dZ2W1lsfu91ud3cRAAAALQmXCAEAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwA8GIvv/yy7rvvPt19992aNm2aTp48KUk6dOiQHnnkEVmtVo0cOVJ///vf3VwpcGshYAGAl/rLX/6i/fv3KyMjQwcPHlT79u21ZcsWffnll0pMTNT48eN1%2BPBhLV68WMnJycrJyXF3ycAtgweNAoCXGjFihP7whz%2Bob9%2B%2BTuObNm3Srl27tHPnTsfY/PnzFRwcrGefffZmlwnckjiD5YVOnTql6dOnq1%2B/foqLi9OqVavU2Njo7rK81qlTp5SYmKh77rlHcXFxWrRokc6fPy9JKiws1KRJkxQbG6v4%2BHht3rzZzdV6r%2Beff149e/Z0fMwlrB/myy%2B/1NmzZ/Wvf/1LDz74oAYNGqR58%2BbJZrOpoKBAffr0cXp9dHS08vLy3FSt98nPz9fUqVP1k5/8RIMHD9aiRYtks9kk8d79Pj766CMNHjxY8%2BfPbzL3zjvv6KGHHpLVatXDDz%2Bs7Oxsx5zdbld6erqGDBmivn37Ol0G93QELC9jt9s1e/ZshYeH64MPPtBrr72mPXv2aOvWre4uzWslJiYqLCxMBw4c0M6dO1VcXKzf//73unTpkmbMmKH%2B/fvr0KFDWrt2rV5%2B%2BWVlZWW5u2SvU1hY6HQ2hUtYP9zZs2fl4%2BOj9957TxkZGfrb3/6m06dPa9myZbLZbAoNDXV6fVhYmCoqKtxUrXdpaGjQzJkzdffdd%2BvQoUPavXu3zp07p5SUFN6738Mrr7yiFStWqEuXLk3m8vLylJSUpLlz5%2BqTTz7R448/rlmzZuns2bOSpK1btyozM1ObNm1Sdna2OnXqpNmzZ8sbLr4RsLxMbm6uioqKlJycrNDQUHXv3l0zZszQG2%2B84e7SvFJ1dbViYmK0cOFCBQYGKioqSo899piOHDmigwcP6sqVK1qwYIECAwPVr18/JSQkKCMjw91le5XGxkb97ne/07Rp0xxju3btUpcuXTR16lS1bt1aw4cP14gRI/Tmm2%2B6r1Avc%2BXKFV25ckW/%2Bc1vFB4ervbt22vOnDl67733rruNj4/PTazQe5WVlencuXMaO3asAgICFBYWphEjRqigoID37vdw22236c0337xmwMrMzFR8fLzGjBmjVq1aacKECerRo4fjP2Q7duzQk08%2Bqd69eysoKEhJSUkqLi7WsWPHbvZh3DAClpcpKChQx44dFRYW5hjr06ePTpw4oQsXLrixMu8UHBys1NRURUZGOsbOnDmj9u3bq6CgQL169ZKfn59jjsssN%2B6NN95Qq1atNHbsWMcYl7B%2BuKtfA4KCghxjHTt2lN1uV319vSorK51eb7PZFBERcVNr9Fbt2rVTdHS0tm/frkuXLqmiokLvvvuuHnjgAd6738PUqVMVHBx8zblv6%2Bfly5dVXFysmJgYx1xQUJA6d%2B7sFf0mYHmZa536v/rx1fsD8P3l5uZq27Zteuqpp657maWyspJ73prp3Llzeumll5SSkuI0ziWsH65Lly4KCgpSfn6%2BY%2Bz06dOyWCx64IEHnMYlKScnR7GxsTe7TK/k4%2BOjtWvX6v3333fc69rY2Khf//rXvHcNs9lsTicMpK%2B%2Bp1VUVKiyslJ2u/2a3/O8od8ELOD/%2BfTTTzV9%2BnQtWLBAcXFx7i6nRUhNTdXEiRP1ox/9qFmv5xJW8/n7%2B2vChAlavXq1zp49q7KyMr300kt69NFHNW7cOJ0%2BfVpbtmxRbW2t9u7dqw8//FAJCQnuLtsr1NXV6amnntKYMWN09OhRZWdnKygoSL/5zW%2Buuw3vXbO%2Bq5/e0G8ClpeJjIy85ql/SZz%2B/wEOHDigmTNnaunSpXr88cclXb/X4eHh8vXlU%2Be7HDp0SHl5efrVr37VZC4iIoJLWAb8%2Bte/Vv/%2B/fXII49o7Nix6tatm5YsWaLIyEht2LBBf/3rXzVw4EClp6crLS1NvXr1cnfJXuEf//iHTp06pXnz5ikwMFBt27bV008/rXfffVf%2B/v68dw2KiIhocvXlaj%2Bvfq29Vr%2B/fluHp7K4uwDcGKvVqjNnzji%2B0Utfnfq/6667FBgY6ObqvNPRo0eVlJSktWvXasiQIY5xq9WqN954Q/X19bJYvvpU4TJL8/3973/X2bNnFR8fL0mOn/oZNGiQpk%2Bfrrffftvp9fT2xgUEBGj58uVavnx5k7kBAwY4/eQmms9utze5DeDKlSuSpHvvvVd/%2B9vfnOZ4735/Vqu1yeXs3Nxc/fSnP1VAQIB69Oih/Px8DRw4UJJUWVmpkpISWa1Wd5R7Q/hvuJfp3bu3YmNjtWLFClVVVamoqEgbN27UlClT3F2aV6qvr1dycrLmzp3rFK4kKT4%2BXoGBgUpLS1NNTY0OHz6s7du30%2Btm%2Bu1vf6t9%2B/Zp586d2rlzpzZu3ChJ2rlzpx5%2B%2BGEuYcFj9evXT4GBgVq3bp1qa2t1/vx5vfLKK7r77rv16KOP8t41aMKECcrOztbu3btVW1urbdu2qaSkROPGjZMkTZ48WZs2bdJnn32m6upqrVixQjExMV4RaHmSuxc6e/asli9frn/%2B858KDAzUz3/%2Bc82ePdvdZXmlI0eOaMqUKQoICGgyt3fvXl28eFHLly9Xfn6%2BIiMjNXPmTE2ePNkNlXq/U6dOacSIESoqKpL0Ve%2Bfe%2B45ffHFF%2BrQoYMWLlyokSNHurlK4Cs5OTlatWqVCgsL5e/vr3vuuUeLFy/WHXfcwXv3Bl0921RfXy9JjisCubm5kqSsrCylpaXpzJkz6t69u5KTkzVgwADH9uvWrdPrr7%2BumpoaDRo0SM8%2B%2B6zuuOOOm3wUN46ABQAAYBiXCAEAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAsP8LL5FDu7BXWJwAAAAASUVORK5CYII%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common-9025337614255567638">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">100.0</td>
            <td class="number">28569</td>
            <td class="number">16.5%</td>
            <td>
                <div class="bar" style="width:78%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">21592</td>
            <td class="number">12.5%</td>
            <td>
                <div class="bar" style="width:59%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">50.0</td>
            <td class="number">16685</td>
            <td class="number">9.7%</td>
            <td>
                <div class="bar" style="width:46%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">66.7</td>
            <td class="number">11432</td>
            <td class="number">6.6%</td>
            <td>
                <div class="bar" style="width:32%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">33.3</td>
            <td class="number">9277</td>
            <td class="number">5.4%</td>
            <td>
                <div class="bar" style="width:26%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">75.0</td>
            <td class="number">7375</td>
            <td class="number">4.3%</td>
            <td>
                <div class="bar" style="width:21%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">25.0</td>
            <td class="number">5676</td>
            <td class="number">3.3%</td>
            <td>
                <div class="bar" style="width:16%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">60.0</td>
            <td class="number">4310</td>
            <td class="number">2.5%</td>
            <td>
                <div class="bar" style="width:12%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">40.0</td>
            <td class="number">4208</td>
            <td class="number">2.4%</td>
            <td>
                <div class="bar" style="width:12%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">80.0</td>
            <td class="number">4152</td>
            <td class="number">2.4%</td>
            <td>
                <div class="bar" style="width:12%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (124)</td>
            <td class="number">23123</td>
            <td class="number">13.4%</td>
            <td>
                <div class="bar" style="width:63%">&nbsp;</div>
            </td>
    </tr><tr class="missing">
            <td class="fillremaining">(Missing)</td>
            <td class="number">36346</td>
            <td class="number">21.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-9025337614255567638">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">21592</td>
            <td class="number">12.5%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.2</td>
            <td class="number">3</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.25</td>
            <td class="number">2</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.29</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.33</td>
            <td class="number">17</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">93.3</td>
            <td class="number">3</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">93.7</td>
            <td class="number">3</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">94.1</td>
            <td class="number">2</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">94.4</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">100.0</td>
            <td class="number">28569</td>
            <td class="number">16.5%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">purpose<br/>
                <small>Categorical</small>
            </p>
        </div><div class="col-md-3">
        <table class="stats ">
            <tr class="">
                <th>Distinct count</th>
                <td>14</td>
            </tr>
            <tr>
                <th>Unique (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (%)</th>
                <td>0.0%</td>
            </tr>
            <tr class="ignore">
                <th>Missing (n)</th>
                <td>0</td>
            </tr>
        </table>
    </div>
    <div class="col-md-6 collapse in" id="minifreqtable1627043463234134575">
        <table class="mini freq">
            <tr class="">
        <th>debt_consolidation</th>
        <td>
            <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 54.9%">
                94874
            </div>
            
        </td>
    </tr><tr class="">
        <th>credit_card</th>
        <td>
            <div class="bar" style="width:41%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 22.7%">
                39270
            </div>
            
        </td>
    </tr><tr class="">
        <th>other</th>
        <td>
            <div class="bar" style="width:11%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 5.9%">
                &nbsp;
            </div>
            10214
        </td>
    </tr><tr class="other">
        <th>Other values (11)</th>
        <td>
            <div class="bar" style="width:30%" data-toggle="tooltip" data-placement="right" data-html="true"
                 data-delay=500 title="Percentage: 16.4%">
                28387
            </div>
            
        </td>
    </tr>
        </table>
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#freqtable1627043463234134575, #minifreqtable1627043463234134575"
           aria-expanded="true" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="col-md-12 extrapadding collapse" id="freqtable1627043463234134575">
        
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">debt_consolidation</td>
            <td class="number">94874</td>
            <td class="number">54.9%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">credit_card</td>
            <td class="number">39270</td>
            <td class="number">22.7%</td>
            <td>
                <div class="bar" style="width:41%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">other</td>
            <td class="number">10214</td>
            <td class="number">5.9%</td>
            <td>
                <div class="bar" style="width:11%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">home_improvement</td>
            <td class="number">9817</td>
            <td class="number">5.7%</td>
            <td>
                <div class="bar" style="width:11%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">major_purchase</td>
            <td class="number">4690</td>
            <td class="number">2.7%</td>
            <td>
                <div class="bar" style="width:5%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">small_business</td>
            <td class="number">3342</td>
            <td class="number">1.9%</td>
            <td>
                <div class="bar" style="width:4%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">car</td>
            <td class="number">2616</td>
            <td class="number">1.5%</td>
            <td>
                <div class="bar" style="width:3%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">wedding</td>
            <td class="number">1892</td>
            <td class="number">1.1%</td>
            <td>
                <div class="bar" style="width:2%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">medical</td>
            <td class="number">1877</td>
            <td class="number">1.1%</td>
            <td>
                <div class="bar" style="width:2%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">moving</td>
            <td class="number">1427</td>
            <td class="number">0.8%</td>
            <td>
                <div class="bar" style="width:2%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (4)</td>
            <td class="number">2726</td>
            <td class="number">1.6%</td>
            <td>
                <div class="bar" style="width:3%">&nbsp;</div>
            </td>
    </tr>
    </table>
    </div>
    </div><div class="row variablerow">
        <div class="col-md-3 namecol">
            <p class="h4">revol_util<br/>
                <small>Numeric</small>
            </p>
        </div><div class="col-md-6">
        <div class="row">
            <div class="col-sm-6">
                <table class="stats ">
                    <tr>
                        <th>Distinct count</th>
                        <td>1063</td>
                    </tr>
                    <tr>
                        <th>Unique (%)</th>
                        <td>0.6%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (%)</th>
                        <td>0.1%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Missing (n)</th>
                        <td>144</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (%)</th>
                        <td>0.0%</td>
                    </tr>
                    <tr class="ignore">
                        <th>Infinite (n)</th>
                        <td>0</td>
                    </tr>
                </table>
    
            </div>
            <div class="col-sm-6">
                <table class="stats ">
    
                    <tr>
                        <th>Mean</th>
                        <td>55.829</td>
                    </tr>
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>140.4</td>
                    </tr>
                    <tr class="ignore">
                        <th>Zeros (%)</th>
                        <td>0.7%</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-3 collapse in" id="minihistogram8120722910657558042">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAATFJREFUeJzt3NGJwkAUQFEjlmQR29N%2Bb08WYU9jA3LRQDZDcs6/MD%2BXl0keLmOMcQHeuu59AJjZbe8DHNn99/H1b55/PxuchLVMEAgCgSAQCAKBsHjN%2B7k1l%2B7/4GK/HRMEgkAgCASCQCAIBIJAIAgEgkAgCATCadfdZ/0qzlxMEAgCgXDaR6wj%2BfZx0XLj50wQCAKBIBAIAoEgEAgCgSAQCAKBcIgPhfaq2IoJAkEgEAQCQSAQpruku3AzExMEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIEy3rMj21iyEnvXvSk0QCMsYY%2Bx9CJiVCQJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBALhBZ01HMmx8WU1AAAAAElFTkSuQmCC">
    
    </div>
    <div class="col-md-12 text-right">
        <a role="button" data-toggle="collapse" data-target="#descriptives8120722910657558042,#minihistogram8120722910657558042"
           aria-expanded="false" aria-controls="collapseExample">
            Toggle details
        </a>
    </div>
    <div class="row collapse col-md-12" id="descriptives8120722910657558042">
        <ul class="nav nav-tabs" role="tablist">
            <li role="presentation" class="active"><a href="#quantiles8120722910657558042"
                                                      aria-controls="quantiles8120722910657558042" role="tab"
                                                      data-toggle="tab">Statistics</a></li>
            <li role="presentation"><a href="#histogram8120722910657558042" aria-controls="histogram8120722910657558042"
                                       role="tab" data-toggle="tab">Histogram</a></li>
            <li role="presentation"><a href="#common8120722910657558042" aria-controls="common8120722910657558042"
                                       role="tab" data-toggle="tab">Common Values</a></li>
            <li role="presentation"><a href="#extreme8120722910657558042" aria-controls="extreme8120722910657558042"
                                       role="tab" data-toggle="tab">Extreme Values</a></li>
    
        </ul>
    
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane active row" id="quantiles8120722910657558042">
                <div class="col-md-4 col-md-offset-1">
                    <p class="h4">Quantile statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Minimum</th>
                            <td>0</td>
                        </tr>
                        <tr>
                            <th>5-th percentile</th>
                            <td>11</td>
                        </tr>
                        <tr>
                            <th>Q1</th>
                            <td>38.6</td>
                        </tr>
                        <tr>
                            <th>Median</th>
                            <td>57.9</td>
                        </tr>
                        <tr>
                            <th>Q3</th>
                            <td>75.1</td>
                        </tr>
                        <tr>
                            <th>95-th percentile</th>
                            <td>92.2</td>
                        </tr>
                        <tr>
                            <th>Maximum</th>
                            <td>140.4</td>
                        </tr>
                        <tr>
                            <th>Range</th>
                            <td>140.4</td>
                        </tr>
                        <tr>
                            <th>Interquartile range</th>
                            <td>36.5</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4 col-md-offset-2">
                    <p class="h4">Descriptive statistics</p>
                    <table class="stats indent">
                        <tr>
                            <th>Standard deviation</th>
                            <td>24.413</td>
                        </tr>
                        <tr>
                            <th>Coef of variation</th>
                            <td>0.43729</td>
                        </tr>
                        <tr>
                            <th>Kurtosis</th>
                            <td>-0.68504</td>
                        </tr>
                        <tr>
                            <th>Mean</th>
                            <td>55.829</td>
                        </tr>
                        <tr>
                            <th>MAD</th>
                            <td>20.246</td>
                        </tr>
                        <tr class="">
                            <th>Skewness</th>
                            <td>-0.32716</td>
                        </tr>
                        <tr>
                            <th>Sum</th>
                            <td>9636100</td>
                        </tr>
                        <tr>
                            <th>Variance</th>
                            <td>596.01</td>
                        </tr>
                        <tr>
                            <th>Memory size</th>
                            <td>1.3 MiB</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram8120722910657558042">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAIABJREFUeJzt3Xl0FGW%2B//FPkk5YErJiUJFt4gUJ6QAOyiZRoiAwIh4kBORcFBUYBgzhgkQgMJHFqBBhAGeuDI5yca4GiA4qiyggeiOMIwySBXOQCyLwg0TSTRbWJPX7g0PfaQMY9AnpdN6vczg5/TxVTz3f6u7ik6pKt49lWZYAAABgjG9dTwAAAMDbELAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGG2up5AQ1FUVGp8TF9fH4WHB6q4uFxVVZbx8T0VdVN3Q9BQ65Yabu3UXTt133RTM%2BNj1gRnsOoxX18f%2Bfj4yNfXp66nckNRN3U3BA21bqnh1k7d3lU3AQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADLPV9QSu5ptvvtGLL76o3Nxc2Ww2de/eXbNmzdL58%2Bf1wAMPKCAgwG35l19%2BWQMHDpQkrVq1Sm%2B%2B%2BaZOnTqlDh06KC0tTZ06dZIknT9/XgsWLNDmzZt18eJF9enTR2lpaQoPD5ckHT16VL///e%2B1e/duNWnSREOHDtXUqVPl60sWBUwZuCS7rqdwXTYl967rKQCoZzwyNVy4cEFPPvmk7rrrLn3xxRfauHGjiouLlZaWptLSUvn7%2BysnJ8ft3%2BVw9fHHH2vJkiVKT0/Xrl27dO%2B992r8%2BPE6c%2BaMJGnhwoXas2ePsrKytHXrVp07d04zZ86UJFmWpUmTJiksLEw7duzQW2%2B9pU2bNmnVqlV1ti8AAED945EB6%2BzZs5oyZYrGjx%2BvgIAAhYeH68EHH9S3336r06dPq1mzZlddd%2B3atRo2bJh69Oihpk2bauLEiZKkbdu2qaKiQu%2B9956Sk5PVqlUrhYeHKyUlRdu3b9fJkyeVk5OjgoICpaamKiQkRFFRURo7dqzeeeedG1U6AADwAh55iTAkJEQJCQmSLp1VOnTokN59910NHDhQJSUlqqqq0vjx47Vnzx6FhYVp5MiReuKJJ%2BTj46P8/HwNGjTINZaPj486duyo3NxcRUdHq6yszHW5UJKioqLUpEkT5eXlqbCwUC1btlRoaKirv1OnTjp8%2BLDKysoUFBRUo/kXFhaqqKjIrc1ma6rIyMhfsluq8fPzdfvZUFB3w6rbE9hsN36fN%2BTnu6HWTt3eVbdHBqzLjh07pv79%2B6uyslKJiYmaPHmyduzYodtvv12jRo3SH/7wB%2B3evVtJSUkKCgpSQkKCHA6HW0CSLgW24uJiORwO1%2BN/FRwc7Or/cd/lxw6Ho8YBKzMzU8uXL3drmzhxopKSkq6r/poKDm5SK%2BN6OurGjRIWFlhn227Iz3dDrZ26vYNHB6yWLVsqNzdX3333nWbPnq1nn31WGRkZ6tu3r2uZ3r17KzExUVlZWUpISJCPj88Vx7pae037r0diYqLi4%2BPd2my2pnI4yo1tQ7qU9oODm6ik5KwqK6uMju3JqLth1e0JTL93a6IhP98NtXbqrp266%2BoXJI8OWNKl4NO2bVtNnz5dw4YN06xZs1x/8XfZbbfdpi1btkiSwsLC5HQ63fodDofat2%2BviIgISZLT6VTTpk0lXboE6XQ6FRERocrKyiuuK6naNq8lMjKy2uXAoqJSVVTUzhumsrKq1sb2ZNSNG6Uu93dDfr4bau3U7R088oLnl19%2BqQceeEAVFRWutqqqSzv973//u/7617%2B6LX/o0CG1atVKkmS325Wbm%2Bvqq6ysVH5%2BvmJjY9WqVSuFhoYqLy/P1V9QUKALFy4oJiZGdrtdx48fd4UqSdq3b59uv/12BQbW3SUCAABQv3hkwIqOjtbZs2eVkZGhs2fPqri4WMuWLVO3bt3UuHFjvfzyy8rOzlZFRYW%2B%2BOILrVu3TqNGjZIkjRgxQllZWdq1a5fKy8v1yiuvqHHjxoqPj5efn5%2BGDx%2BuJUuW6Pvvv9epU6eUnp6uAQMGqHnz5urYsaNiY2M1f/58lZSUqKCgQCtWrHCNDQAAUBMeeYkwKChIK1eu1EsvvaQ%2Bffq4Pmh0wYIFatGihWbOnKm5c%2Bfq5MmTuu222zR79mw98MADkqS4uDhNnz5dM2bM0KlTpxQTE6MVK1aoUaNGkqRnnnlG5eXlGjp0qCorK9W3b1%2BlpaW5tv2HP/xBc%2BbMUZ8%2BfRQYGKjHHntMjz32WF3sBgAAUE/5WJZl1fUkGoKiolLjY9psvgoLC5TDUe5V161/CnXX/7r5JPef5k3P9/VqqLVTd%2B3UfdNNV//szNrkkZcIAQAA6jMCFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYba6ngAAeLqBS7LregrXZVNy77qeAtDgcQYLAADAMM5gAV6ivp1lAQBvxhksAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAzz2ID1zTff6IknnlC3bt3Uo0cPTZ48WYWFhZKknTt36uGHH5bdble/fv30/vvvu627atUq9e3bV7GxsUpISFBeXp6r7/z585ozZ47uvvtude3aVUlJSSouLnb1Hz16VE899ZS6dOminj17auHChaqqqroxRQMAAK/gkQHrwoULevLJJ3XXXXfpiy%2B%2B0MaNG1VcXKy0tDSdPHlSEyZM0LBhw/Tll19qxowZSk1N1b59%2ByRJH3/8sZYsWaL09HTt2rVL9957r8aPH68zZ85IkhYuXKg9e/YoKytLW7du1blz5zRz5kxJkmVZmjRpksLCwrRjxw699dZb2rRpk1atWlVn%2BwIAANQ/Hhmwzp49qylTpmj8%2BPEKCAhQeHi4HnzwQX377bf64IMP1KZNG40ePVpNmjRRfHy87r//fq1bt06StHbtWg0bNkw9evRQ06ZNNXHiREnStm3bVFFRoffee0/Jyclq1aqVwsPDlZKSou3bt%2BvkyZPKyclRQUGBUlNTFRISoqioKI0dO1bvvPNOXe4OAABQz9jqegJXEhISooSEBEmXziodOnRI7777rgYOHKj8/Hx16tTJbfno6Ght2rRJkpSfn69Bgwa5%2Bnx8fNSxY0fl5uYqOjpaZWVlbutHRUWpSZMmysvLU2FhoVq2bKnQ0FBXf6dOnXT48GGVlZUpKCioRvMvLCxUUVGRW5vN1lSRkZHXtyN%2Bgp%2Bfr9vPhoK6G1bduH42W/1%2BjTTU1zp1e1fdHhmwLjt27Jj69%2B%2BvyspKJSYmavLkyXrqqad0xx13uC0XGhrquo/K4XC4BSTpUmArLi6Ww%2BFwPf5XwcHBrv4f911%2B7HA4ahywMjMztXz5cre2iRMnKikpqUbrX6/g4Ca1Mq6no27gysLCAut6CkY01Nc6dXsHjw5YLVu2VG5urr777jvNnj1bzz777FWX9fHxcft5tf6fWt%2BExMRExcfHu7XZbE3lcJQb24Z0Ke0HBzdRSclZVVY2nBvxqbth1Y3rZ/pYc6M11Nc6dddO3XX1C4dHByzpUvBp27atpk%2BfrmHDhunee%2B%2BV0%2Bl0W8bhcCg8PFySFBYWdsX%2B9u3bKyIiQpLkdDrVtGlTSZcuQTqdTkVERKiysvKK60pyjV8TkZGR1S4HFhWVqqKidt4wlZVVtTa2J6Nu4Mq85fXRUF/r1O0dPPKC55dffqkHHnhAFRUVrrbLH5XQq1cvt49dkKR9%2B/YpNjZWkmS325Wbm%2Bvqq6ysVH5%2BvmJjY9WqVSuFhoa6rV9QUKALFy4oJiZGdrtdx48fd4Wqy2PffvvtCgz0jlPuAACg9nlkwIqOjtbZs2eVkZGhs2fPqri4WMuWLVO3bt00ePBgHTt2TG%2B%2B%2BabOnTunzZs367PPPlNiYqIkacSIEcrKytKuXbtUXl6uV155RY0bN1Z8fLz8/Pw0fPhwLVmyRN9//71OnTql9PR0DRgwQM2bN1fHjh0VGxur%2BfPnq6SkRAUFBVqxYoVGjRpVx3sEAADUJx4ZsIKCgrRy5Urt379fffr00aBBgxQYGKhXXnlFEREReu211/Tee%2B/prrvu0uLFi5WRkeG68T0uLk7Tp0/XjBkz1LNnT/3zn//UihUr1KhRI0nSM888o%2B7du2vo0KHq16%2Bfmjdvrnnz5rm2/Yc//EGlpaXq06ePxowZoxEjRuixxx6rk/0AAADqJx/Lsqy6nkRDUFRUanxMm81XYWGBcjjKveq69U%2Bh7ivXPXBJdh3MCp5oU3Lvup7CL8J7nLpNuummZsbHrAmPPIMFAABQnxGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMM8NmAdPXpUEyZM0N13362ePXtq%2BvTpOn36tKqqqnTHHXcoJiZGdrvd9e/11193rbthwwY9%2BOCDstvteuihh5Sdne3qsyxLixcvVu/evdW5c2c98cQT%2Bv777139DodDU6ZM0Z133qm77rpLs2bN0rlz525o7QAAoH7z2IA1YcIEhYaGavv27Vq/fr0OHjyol19%2BWaWlpbIsS1u3blVOTo7r31NPPSVJys3NVUpKiiZPnqx//OMfevzxxzVx4kSdOHFCkrRq1SplZWVp5cqVys7OVqtWrTRp0iRZliVJmjlzpk6dOqUtW7boww8/1P79%2B7Vo0aI62w8AAKD%2B8ciAVVpaqpiYGE2bNk2BgYGKjIzU0KFD9dVXX6mkpESSFBwcfMV1s7KyFBcXp0GDBqlx48ZKSEhQ%2B/bttX79eknS2rVr9fTTT6tjx44KCgpSSkqKDh48qL179%2BrUqVPavn27ZsyYoebNm6tFixZKTk5WVlaWLly4cMPqBwAA9ZutridwJc2aNVN6erpb2/Hjx3XLLbfo9OnT8vHxUWpqqrKzs9WkSRM99NBDSkpKkr%2B/v/Lz8xUXF%2Be2bnR0tHJzc3X%2B/HkdPHhQMTExrr6goCC1bt1aubm5Kisrk81mU4cOHVz9nTp10pkzZ3To0CG39mspLCxUUVGRW5vN1lSRkZHXuyuuyc/P1%2B1nQ0HdDatuXD%2BbrX6/Rhrqa526vatujwxYP5aTk6PVq1frtddekyR17txZffv21bx583Tw4EFNmjRJfn5%2BSk5OlsPhUGhoqNv6ISEhOnDggJxOpyzLUkhISLX%2B4uJihYSEKCgoSL6%2Bvm59klRcXFzj%2BWZmZmr58uVubRMnTlRSUtJ11V1TwcFNamVcT0fdwJWFhQXW9RSMaKivder2Dh4fsHbv3q0JEyZo6tSp6tmzp6RLAeYyu92ucePG6T//8z%2BVnJx81XF8fHyuuZ1f2v%2BvEhMTFR8f79ZmszWVw1Fe4zFqws/PV8HBTVRSclaVlVVGx/Zk1N2w6sb1M32sudEa6mudumun7rr6hcOjA9b27ds1bdo0zZkzR0OGDLnqcrfddpuKi4tlWZbCw8PlcDjc%2Bh0Oh8LDwxUWFiZfX185nc5q/REREYqIiFBpaakqKyvl5%2Bfn6pOkiIiIGs87MjKy2uXAoqJSVVTUzhumsrKq1sb2ZNQNXJm3vD4a6mudur2Dx17w3LNnj1JSUrR06VK3cLVz5069%2BuqrbsseOnRILVu2lI%2BPj%2Bx2u/Ly8tz6c3JyFBsbq4CAALVv396t3%2Bl06siRI7Lb7YqOjlZVVZUKCgpc/fv27VOzZs3Utm3b2ikUAAB4HY8MWBUVFUpNTdXkyZPVu3dvt77Q0FD96U9/0vr161VRUaGcnBy9/vrrGjVqlCQpISFB2dnZ2rhxo86dO6fVq1fryJEjeuSRRyRJI0eO1MqVK/XNN9%2BotLRU8%2BfPV0xMjGJjYxUWFqaBAwcqPT1dP/zwg44dO6bFixcrMTFR/v7%2BN3w/AACA%2BsnHuvwBUB7kq6%2B%2B0qhRoxQQEFCtb/PmzcrPz9eyZct05MgRRUZGKjExUWPGjHHdnL5lyxZlZGTo%2BPHjioqKUmpqqrp16%2BYaY9myZXr77bdVXl6u7t27a%2B7cubr55pslXfqIiLS0NG3btk3%2B/v4aPHiwUlJSrjiX61FUVPqL1r8Sm81XYWGBcjjKveq06k%2Bh7ivXPXBJ9hXWQkO0Kbn3Ty/kwXiPU7dJN93UzPiYNeGRAcsbEbDMoW4CFq6NgFU/Ubd3BSyPvskdqEsEFgDAz%2BWR92ABAADUZwQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhmPGCVlZWZHhIAAKBeMR6w7rnnHs2YMUN79uwxPTQAAEC9YDxgpaWlqaioSKNHj9agQYP0xhtvqLi42PRmAAAAPJbxgPXII49o5cqV%2BuyzzzRy5Eh99NFHuu%2B%2B%2B5ScnKzs7GzTmwMAAPA4PpZlWbW9kQ0bNigtLU1lZWVq06aNkpOTNWDAgNrerEcpKio1PqbN5quwsEA5HOWqqKgyPr6nulF1D1zCLwRAbduU3LtaG8c26jbpppuaGR%2BzJmy1NfCpU6f07rvv6t1339WRI0d0zz33aPjw4SoqKlJaWpqOHDmicePG1dbmAQAA6ozxgPX5559r7dq12rZtm8LCwvToo49q%2BPDhuvXWW13LREdHa%2BzYsQQsAADglYwHrHHjxqlXr15avHix4uPj5efnV22Z2NhYRUZGmt40AACARzAesLZs2aJWrVrpwoULrnBVXl6uwMBAt%2BU%2B%2BOAD05sGAADwCMb/itDHx0eDBw/Wtm3bXG2ZmZn6zW9%2Bo%2B%2B//9705gAAADyO8YC1YMEC/epXv9Kdd97panv44Ydlt9u1YMEC05sDAADwOMYvEe7evVuffvqpmjZt6mpr3ry5Zs%2Berb59%2B5reHAAAgMcxfgbLsixVVFRUaz979qyqqhrO53oAAICGy3jA6t27t6ZPn678/HyVlJTI6XRq9%2B7dmjJliu655x7TmwMAAPA4xi8Rzp49W9OmTdPQoUPl4%2BPjau/evbtSU1NNbw4AAMDjGA9YEREReuONN3Tw4EEdPnxYlmWpXbt2ioqKMr0pAAAAj1RrX5UTFRWlVq1auR5fuHBBkhQQEFBbmwQAAPAIxgPW3r17lZaWpm%2B//VaVlZXV%2Bvfv3296kwAAAB7FeMBKS0tTs2bNNGvWLDVu3Phnj3P06FEtWLBAu3fvlp%2Bfn/r06aNZs2YpJCRE%2B/fv1/PPP6/8/HyFhoZqzJgxGjNmjGvdDRs2aOnSpTp%2B/LjatGmjGTNmqHfvS9/YblmWlixZonXr1qmsrExdu3bVvHnzXGfbHA6H5s6dqx07dsjPz0/9%2B/fX7Nmzf1EtAACgYTEesA4fPqy///3vatSo0S8aZ8KECYqJidH27dtVXl6uCRMm6OWXX1ZqaqrGjh2rhx9%2BWK%2B//roOHDigsWPHqmXLlurfv79yc3OVkpKil19%2BWfHx8frggw80ceJEbd68WTfffLNWrVqlrKwsrVy5Uq1atdJLL72kSZMm6W9/%2B5t8fHw0c%2BZMlZeXa8uWLaqsrNSECRO0aNEibtAHAAA1ZvxjGm699VZdvHjxF41RWlqqmJgYTZs2TYGBgYqMjNTQoUP11Vdf6dNPP9XFixc1depUBQYGqkuXLkpMTFRmZqYkKSsrS3FxcRo0aJAaN26shIQEtW/fXuvXr5ckrV27Vk8//bQ6duyooKAgpaSk6ODBg9q7d69OnTql7du3a8aMGWrevLlatGih5ORkZWVlue4hAwAA%2BCnGz2BNmzZN6enpmjFjhoKCgn7WGM2aNVN6erpb2/Hjx3XLLbcoPz9fd9xxh%2BuLpCUpOjpaa9eulSTl5%2BcrLi7Obd3o6Gjl5ubq/PnzOnjwoGJiYlx9QUFBat26tXJzc1VWViabzaYOHTq4%2Bjt16qQzZ87o0KFDbu3XUlhYqKKiIrc2m62pIiMja7YDasjPz9ftZ0PRUOsGvJHNVv193FDf49TtXXUbD1jLly/X0aNH9d577yksLMzts7Ak6X/%2B53%2Bue8ycnBytXr1ar732mjZs2KCQkBC3/tDQUDmdTlVVVcnhcCg0NNStPyQkRAcOHJDT6ZRlWdXWDwkJUXFxsUJCQhQUFCRfX1%2B3PkkqLi6u8XwzMzO1fPlyt7aJEycqKSmpxmNcj%2BDgJrUyrqdrqHUD3iQsLPCqfQ31PU7d3sF4wOrTp4/8/f2Njbd7925NmDBBU6dOVc%2BePbVhw4afNc6Pg57p/n%2BVmJio%2BPh4tzabrakcjvIaj1ETfn6%2BCg5uopKSs6qsbDhfQ9RQ6wa80ZWOiw31PU7dtVP3tUJ8bTIesKZMmWJsrO3bt2vatGmaM2eOhgwZIunSB5keOXLEbTmHw6GwsDD5%2BvoqPDxcDoejWn94eLhrGafTWa0/IiJCERERKi0tVWVlpesS5OWxIiIiajzvyMjIapcDi4pKVVFRO2%2BYysqqWhvbkzXUugFvcq33cEN9j1O3d6iVC55ff/21ZsyYoccff1ySVFVVpU2bNl3XGHv27FFKSoqWLl3qCleSZLfbVVBQ4PaF0vv27VNsbKyrPy8vz22snJwcxcbGKiAgQO3bt3frdzqdOnLkiOx2u6Kjo1VVVaWCggK3sZs1a6a2bdte1/wBAEDDZTxgbd26VY899pgcDof27NkjSTpx4oRmz57tuhH9p1RUVCg1NVWTJ092fX7VZXFxcQoMDFRGRobKy8v15Zdfas2aNRo1apQkKSEhQdnZ2dq4caPOnTun1atX68iRI3rkkUckSSNHjtTKlSv1zTffqLS0VPPnz1dMTIxiY2MVFhamgQMHKj09XT/88IOOHTumxYsXKzEx0ehlTwAA4N18LMuyTA6YkJCgMWPGaNCgQYqNjdW%2BffskSdnZ2XrhhRdqdA/VV199pVGjRl3xa3U2b96sM2fOaM6cOcrLy1NERITGjRunkSNHupbZsmWLMjIydPz4cUVFRSk1NVXdunVz9S9btkxvv/22ysvL1b17d82dO1c333yzpEsfEZGWlqZt27bJ399fgwcPVkpKyi/%2Bip%2BiotJftP6V2Gy%2BCgsLlMNR7lWnVX/Kjap74JLsWhsbwCWbkntXa%2BPYRt0m3XRTM%2BNj1oTxgNWtWzft2rVLNptNnTt31tdffy3p0mXCO%2B%2B8U3v37jW5uXqDgGUOAQvwHgSs/0Pd3hWwjF8i9Pf3r3YTuXTpE975uhkAANAQGA9Y9913n1JTU3Xw4EFJl/4K7/PPP1dycrL69u1renMAAAAex3jAmjFjhizL0m9%2B8xudP39evXr10tixY3XLLbfoueeeM705AAAAj2P8c7CCg4P12muv6eDBgzp8%2BLB8fHzUrl07tWvXzvSmAAAAPJLxgHVZVFSUoqKiamt4AAAAj2U8YN1zzz1X7ausrNTOnTtNbxIAAMCjGA9YiYmJbt/bV1VVpaNHjyo7O1vjx483vTkAAACPYzxgPfPMM1ds37dvn/77v//b9OYAAAA8Tq18F%2BGVxMbGKicn50ZtDgAAoM7csID13Xff6fTp0zdqcwAAAHXG%2BCXCESNGVGu7cOGC/vd//1f333%2B/6c0BAAB4HOMBq23btm43uUtSo0aN9Oijj%2BrRRx81vTkAAACPYzxgvfjii6aHBAAAqFeMB6y1a9fK39%2B/Rss%2B8sgjpjcPAABQ54wHrAULFuj8%2BfOyLMut3cfHx63Nx8eHgAUAALyS8YD1pz/9SW%2B99ZYmTJigqKgoVVZW6sCBA1qxYoVGjx6tnj17mt4kAACARzEesF544QW9/vrrioyMdLV17dpVv//97/Xkk09q48aNpjcJAADgUYx/DtbRo0cVHBxcrT0kJETHjx83vTkAAACPYzxgtWvXTunp6XI4HK6206dPKyMjQ%2B3atTO9OQAAAI9j/BJhamqqJkyYoDVr1igwMFA%2BPj4qKytTYGCgXn31VdObAwAA8DjGA9add96pTz/9VDt27NCJEydkWZZatGihuLg4BQUFmd4cAACAxzEesCSpSZMm6tevn44fP65WrVrVxiYAAAA8lvF7sM6dO6ff//736ty5swYOHChJKikp0bhx41RaWmp6cwAAAB7HeMBaunSp9u7dq0WLFsnX9/%2BGv3jxol566SXTmwMAAPA4xgPWJ598oiVLlmjAgAGuL30ODg5Wenq6tm/fbnpzAAAAHsd4wCosLFTbtm2rtUdERKisrMz05gAAADyO8YB18803a8%2BePdXaP/roI91yyy2mNwcAAOBxjP8V4RNPPKHf/e53GjZsmCorK/WXv/xFubm52rJli2bNmmV6cwAAAB7HeMAaMWKEQkND9cYbb6hp06Z67bXX1K5dOy1atEgDBgwwvTkAAACPYzxgnTp1SgMGDCBMAQCABsvoPVhVVVXq27evLMsyOSwAAEC9YjRg%2Bfr6qlevXtq0aZPJYQEAAOoV45cIb731Vr3wwgtasWKFWrc6YDrAAAAc30lEQVRuLX9/f7f%2BjIyMGo/1%2BeefKyUlRd27d9fixYtd7bt27dLjjz%2BugIAAt%2BX/%2Bte/KjY2VpZlacmSJVq3bp3KysrUtWtXzZs3z/W1PQ6HQ3PnztWOHTvk5%2Ben/v37a/bs2WrcuLEkaf/%2B/Xr%2B%2BeeVn5%2Bv0NBQjRkzRmPGjPm5uwQAADQwxgPWgQMH1K5dO0mXgszP9ec//1nr1q1TmzZtqvWVlpaqbdu2%2Buijj6647qpVq5SVlaWVK1eqVatWeumllzRp0iT97W9/k4%2BPj2bOnKny8nJt2bJFlZWVmjBhghYtWqTU1FSdPXtWY8eO1cMPP6zXX39dBw4c0NixY9WyZUv179//Z9cDAAAaDmMBa8qUKVq8eLFWr17tanv11Vc1ceLEnzVeo0aNtG7dOi1YsEDnz5936zt9%2BrSaNWt21XXXrl2rp59%2BWh07dpQkpaSkqEePHtq7d69at26t7du367333lPz5s0lScnJyZo8ebKmT5%2BuTz/9VBcvXtTUqVPl5%2BenLl26KDExUZmZmQQsAABQI8YC1rZt26q1rVix4mcHrNGjR1%2B1r6SkRKdPn9a///u/a//%2B/WrRooXGjRunIUOG6Pz58zp48KBiYmJcywcFBal169bKzc1VWVmZbDabOnTo4Orv1KmTzpw5o0OHDik/P1933HGH/Pz8XP3R0dFau3ZtjedeWFiooqIitzabrakiIyNrPEZN%2BPn5uv1sKBpq3YA3stmqv48b6nucur2rbmMB60p/OVhbf00YFBSk2267TUlJSerYsaO2bdumadOmKTIyUr/61a9kWZZCQkLc1gkJCVFxcbFCQkIUFBTk9kXUl5ctLi6Ww%2BGotm5oaKicTqeqqqrc1ruazMxMLV%2B%2B3K1t4sSJSkpK%2BrklX1NwcJNaGdfTNdS6AW8SFhZ41b6G%2Bh6nbu9gLGBd/mLnn2ozYfjw4Ro%2BfLjr8aBBg/TRRx9p3bp1mj59%2BnXN8Xr6ayoxMVHx8fFubTZbUzkc5UbGv8zPz1fBwU1UUnJWlZVVRseuDf0WfV7XUwDgYa50XKxvxzZTqLt26r5WiK9Nxm9yryu33XabcnNzFRYWJl9fXzmdTrd%2Bh8OhiIgIRUREqLS0VJWVla7LgJdvxr/cf%2BTIkWrrXh63JiIjI6tdDiwqKlVFRe28YSorq2ptbACoTdc6djXUYxt1e4d6ecHznXfe0caNG93aDh06pFatWikgIEDt27dXXl6eq8/pdOrIkSOy2%2B2Kjo5WVVWVCgoKXP379u1Ts2bN1LZtW9ntdhUUFKiiosKtPzY2tvYLAwAAXsHYGazLf3n3U23X8zlYV1NRUaH58%2BerdevW6tChg7Zs2aLPPvtMmZmZkqSRI0dq%2BfLl6tGjh1q2bKn58%2BcrJibGFZIGDhyo9PR0LV68WOfPn9fixYuVmJgof39/xcXFKTAwUBkZGZo0aZLy8vK0Zs0aLVmy5BfPGwAANAzGAtavf/1rFRYW/mRbTdntdklynUn65JNPJEk5OTkaNWqUSkpKlJSUJIfDoXbt2unVV19Vp06dJF36wumioiI9%2BeSTKi8vV/fu3bV06VLX2M8//7zS0tLUr18/%2Bfv7a/DgwZo8ebIkKSAgQK%2B99prmzJmjnj17KiIiQtOnT9e99977s%2BoAAAANj4/FFwfeEEVFpcbHtNl8FRYWKIejvF5ctx64JLuupwDAw2xK7l2trb4d20yh7tqp%2B6abrv65mbWpXt6DBQAA4MkIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhmq%2BsJ4JfpNmtzXU8BAAD8CGewAAAADCNgAQAAGEbAAgAAMMyjA9bnn3%2BuXr16acqUKdX6NmzYoAcffFB2u10PPfSQsrOzXX2WZWnx4sXq3bu3OnfurCeeeELff/%2B9q9/hcGjKlCm68847ddddd2nWrFk6d%2B6cq3///v0aMWKEYmNjFRcXpzfeeKN2CwUAAF7FYwPWn//8Z82fP19t2rSp1pebm6uUlBRNnjxZ//jHP/T4449r4sSJOnHihCRp1apVysrK0sqVK5Wdna1WrVpp0qRJsixLkjRz5kydOnVKW7Zs0Ycffqj9%2B/dr0aJFkqSzZ89q7NixuvPOO7Vz504tXbpUf/zjH7Vly5YbVzwAAKjXPDZgNWrUSOvWrbtiwMrKylJcXJwGDRqkxo0bKyEhQe3bt9f69eslSWvXrtXTTz%2Btjh07KigoSCkpKTp48KD27t2rU6dOafv27ZoxY4aaN2%2BuFi1aKDk5WVlZWbpw4YI%2B/fRTXbx4UVOnTlVgYKC6dOmixMREZWZm3uhdAAAA6imP/ZiG0aNHX7UvPz9fcXFxbm3R0dHKzc3V%2BfPndfDgQcXExLj6goKC1Lp1a%2BXm5qqsrEw2m00dOnRw9Xfq1ElnzpzRoUOHlJ%2BfrzvuuEN%2Bfn5uY69du7bGcy8sLFRRUZFbm83WVJGRkTUeoyb8/Dw2HwNAjdhs1Y9jl49tDe0YR93eVbfHBqxrcTgcCg0NdWsLCQnRgQMH5HQ6ZVmWQkJCqvUXFxcrJCREQUFB8vX1deuTpOLiYjkcjmrrhoaGyul0qqqqym29q8nMzNTy5cvd2iZOnKikpKTrqhMAvF1YWOBV%2B4KDm9zAmXgO6vYO9TJgXY2Pj0%2Bt9tdUYmKi4uPj3dpstqZyOMqNjH%2BZt6V9AA3PlY6Lfn6%2BCg5uopKSs6qsrKqDWdUN6q6duq8V4mtTvQxY4eHhcjgcbm0Oh0Ph4eEKCwuTr6%2BvnE5ntf6IiAhFRESotLRUlZWVrsuAl8e63H/kyJFq614etyYiIyOrXQ4sKipVRUXDecMAQE1c67hYWVnVII%2Bb1O0d6uUpELvdrry8PLe2nJwcxcbGKiAgQO3bt3frdzqdOnLkiOx2u6Kjo1VVVaWCggJX/759%2B9SsWTO1bdtWdrtdBQUFqqiocOuPjY2t/cIAAIBXqJcBKyEhQdnZ2dq4caPOnTun1atX68iRI3rkkUckSSNHjtTKlSv1zTffqLS0VPPnz1dMTIxiY2MVFhamgQMHKj09XT/88IOOHTumxYsXKzExUf7%2B/oqLi1NgYKAyMjJUXl6uL7/8UmvWrNGoUaPquGoAAFBf%2BFiXPxzKw9jtdklynUmy2S5dzczJyZEkbdmyRRkZGTp%2B/LiioqKUmpqqbt26udZftmyZ3n77bZWXl6t79%2B6aO3eubr75ZklSaWmp0tLStG3bNvn7%2B2vw4MFKSUlRQECAJOnAgQOaM2eO8vLyFBERoXHjxmnkyJG/qJ6iotJftP6V2Gy%2B6rfoc%2BPjAsCNsim5d7U2m81XYWGBcjjKveqS0U%2Bh7tqp%2B6abmhkfsyY8NmB5GwIWAFRHwPo/1O1dAateXiIEAADwZAQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLB6G7D69u2rmJgY2e1217958%2BZJknbu3KmHH35Ydrtd/fr10/vvv%2B%2B27qpVq9S3b1/FxsYqISFBeXl5rr7z589rzpw5uvvuu9W1a1clJSWpuLj4htYGAADqt3obsEpKSvRf//VfysnJcf2bPXu2Tp48qQkTJmjYsGH68ssvNWPGDKWmpmrfvn2SpI8//lhLlixRenq6du3apXvvvVfjx4/XmTNnJEkLFy7Unj17lJWVpa1bt%2BrcuXOaOXNmXZYKAADqmXoZsCorK1VeXq7g4OBqfR988IHatGmj0aNHq0mTJoqPj9f999%2BvdevWSZLWrl2rYcOGqUePHmratKkmTpwoSdq2bZsqKir03nvvKTk5Wa1atVJ4eLhSUlK0fft2nTx58obWCAAA6i9bXU/g5ygpKZFlWVq2bJl2794tSYqPj1dKSory8/PVqVMnt%2BWjo6O1adMmSVJ%2Bfr4GDRrk6vPx8VHHjh2Vm5ur6OholZWVua0fFRWlJk2aKC8vTy1atKjR/AoLC1VUVOTWZrM1VWRk5M%2Bq92r8/OplPgYAF5ut%2BnHs8rGtoR3jqNu76q6XAevChQvq3Lmz7rrrLi1YsECFhYWaPHmy0tLS5HA4dMcdd7gtHxoa6rqPyuFwKDQ01K0/JCRExcXFcjgcrsf/Kjg4%2BLruw8rMzNTy5cvd2iZOnKikpKQajwEADUFYWOBV%2B4KDm9zAmXgO6vYO9TJgtWjRQmvWrHE9DgoK0rRp0/Tb3/5W3bp1u%2BI6Pj4%2Bbj%2Bv1n81P9X/rxITExUfH%2B/WZrM1lcNRXuMxasLb0j6AhudKx0U/P18FBzdRSclZVVZW1cGs6gZ1107d1wrxtaleBqwrue2221RVVSVfX185nU63PofDofDwcElSWFjYFfvbt2%2BviIgISZLT6VTTpk0lSZZlyel0uvpqIjIystrlwKKiUlVUNJw3DADUxLWOi5WVVQ3yuEnd3qFengLZv3%2B/XnjhBbe2Q4cOKSAgQPfee6/bxy5I0r59%2BxQbGytJstvtys3NdfVVVlYqPz9fsbGxatWqlUJDQ93WLygo0IULFxQTE1OLFQEAAG9SLwNWRESE1q5dqxUrVujChQs6fPiwlixZopEjR2rIkCE6duyY3nzzTZ07d06bN2/WZ599psTEREnSiBEjlJWVpV27dqm8vFyvvPKKGjdurPj4ePn5%2BWn48OFasmSJvv/%2Be506dUrp6ekaMGCAmjdvXsdVAwCA%2BqJeXiKMjIzUihUrtGjRIv3pT39SWFiYBg0apKSkJAUEBOi1117TvHnzlJGRoVtvvVUZGRmuG9/j4uI0ffp0zZgxQ6dOnVJMTIxWrFihRo0aSZKeeeYZlZeXa%2BjQoaqsrFTfvn2VlpZWh9UCAID6xseyLKuuJ9EQFBWVGh/TZvNVv0WfGx8XAG6UTcm9q7XZbL4KCwuUw1HuVffk/BTqrp26b7qpmfExa6JeXiIEAADwZAQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADLPV9QQAAA3XwCXZdT2F67IpuXddTwH1BGewAAAADCNgAQAAGEbAAgAAMIyAdQVHjx7VU089pS5duqhnz55auHChqqqq6npaAACgnuAm9x%2BxLEuTJk3S7bffrh07duiHH37Q2LFj1bx5c40ZM6aupwcAAOoBzmD9SE5OjgoKCpSamqqQkBBFRUVp7Nixeuedd%2Bp6agAAoJ7gDNaP5Ofnq2XLlgoNDXW1derUSYcPH1ZZWZmCgoJ%2BcozCwkIVFRW5tdlsTRUZGWl0rn5%2B5GMAuJFstto77l4%2Bpje0Y7u31k3A%2BhGHw6GQkBC3tsuPHQ5HjQJWZmamli9f7tY2adIkPfPMM%2BYmqktB7vGbDygxMdF4ePNkhYWFyszMpO4GgrobVt1Sw629sLBQq1atpG4v4V1x0UMkJibq3XffdfuXmJhofDtFRUVavnx5tbNl3o66qbshaKh1Sw23dur2rro5g/UjERERcjqdbm0Oh0OSFB4eXqMxIiMjvSqFAwCA68MZrB%2Bx2%2B06fvy4K1RJ0r59%2B3T77bcrMDCwDmcGAADqCwLWj3Ts2FGxsbGaP3%2B%2BSkpKVFBQoBUrVmjUqFF1PTUAAFBP%2BKWlpaXV9SQ8TZ8%2BffTRRx9p3rx52rhxo0aOHKmnnnqqrqd1RYGBgbr77rsb3Nk16qbuhqCh1i013Nqp23vq9rEsy6rrSQAAAHgTLhECAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAqoeOHj2qp556Sl26dFHPnj21cOFCVVVV1fW0asXRo0c1YcIE3X333erZs6emT5%2Bu06dPS5L279%2BvESNGKDY2VnFxcXrjjTfqeLa144UXXlCHDh1cj3fu3KmHH35Ydrtd/fr10/vvv1%2BHszPvj3/8o%2B655x517dpVTzzxhL7//ntJ3l13Xl6eRo8erV//%2Btfq1auXpk%2Bf7vrCeW%2Br%2B/PPP1evXr00ZcqUan0bNmzQgw8%2BKLvdroceekjZ2dmuPsuytHjxYvXu3VudO3d2e23UB9eqe/PmzRo8eLC6du2q/v37KzMz061/1apV6tu3r2JjY5WQkKC8vLwbNe1f7Fp1X1ZeXq57771Xzz33nKutvj/fkiQL9UpVVZU1ZMgQa%2BrUqZbT6bS%2B/fZbq2/fvtZf/vKXup5arXjooYes5557ziorK7NOnjxpDR061Jo5c6Z15swZq3fv3tZLL71klZWVWf/85z%2Btbt26WR999FFdT9mo/Px86%2B6777bat29vWZZlnThxwurcubO1atUq68yZM9bWrVstu91uff3113U8UzP%2B%2Bte/Wo8%2B%2Bqh19OhRy%2Bl0Ws8995w1d%2B5cr667oqLC6tWrl/XKK69Y58%2BftxwOhzVmzBgrKSnJ6%2BpesWKF1b9/f2vEiBFWcnKyW19OTo7VqVMna8OGDdbZs2etNWvWWJ07d7b%2B3//7f5ZlWdYbb7xh9e7d28rPz7dKS0ut1NRU6%2BGHH7aqqqrqopTrcq26v/76a8tut1tbt261KioqrM8%2B%2B8zq1KmT9Y9//MOyLMvasmWL1aVLF2vnzp1WeXm5tWzZMqt3795WeXl5XZRyXa5V979KT0%2B37rzzTislJcXVVp%2Bf78s4g1XP5OTkqKCgQKmpqQoJCVFUVJTGjh2rd955p66nZlxpaaliYmI0bdo0BQYGKjIyUkOHDtVXX32lTz/9VBcvXtTUqVMVGBioLl26KDExsdpvfvVZVVWVfv/73%2BuJJ55wtX3wwQdq06aNRo8erSZNmig%2BPl7333%2B/1q1bV3cTNej111/X7Nmz1bJlS4WEhCg9PV2zZ8/26rqLior0ww8/aPDgwQoICFBoaKjuv/9%2B5efne13djRo10rp169SmTZtqfVlZWYqLi9OgQYPUuHFjJSQkqH379lq/fr0kae3atXr66afVsWNHBQUFKSUlRQcPHtTevXtvdBnX7Vp1O51O/fa3v1V8fLz8/PzUp08fdejQQV999ZWkS3UPGzZMPXr0UNOmTTVx4kRJ0rZt225oDT/Hteq%2B7JtvvtGHH36ooUOHurXX5%2Bf7MgJWPZOfn6%2BWLVsqNDTU1dapUycdPnxYZWVldTgz85o1a6b09HRFRES42o4fP65bbrlF%2Bfn5uuOOO%2BTn5%2Bfqi46OVm5ubl1MtVa88847aty4sQYPHuxqy8/PV6dOndyW85a6T548qRMnTui7775T//791b17dyUnJ8vhcHh13S1atFB0dLTWrFmjs2fPqri4WB9//LHuu%2B8%2Br6t79OjRatas2RX7rlXr%2BfPndfDgQcXExLj6goKC1Lp163qxL65Vd1xcnH73u9%2B5HldUVKiwsFC33HKLpOr7xcfHRx07dqz3dUuXLgOmpaVp2rRpCg4OdrXX9%2Bf7MgJWPeNwOBQSEuLWdvnx5Xs2vFVOTo5Wr16t8ePHX3E/hIaGyul0esX9aD/88INeffVVpaWlubVfre7i4uIbOLvaceLECfn4%2BOiTTz5RZmam/va3v%2BnYsWOaPXu2V9ft4%2BOjpUuXauvWra77KquqqvQf//EfXl33jzkcDrdfHKVLx7bi4mI5nU5ZlnXFY5%2B37YtFixYpMDBQAwYMkHTt/VLfZWZmyt/fX4888ohbu7c83wQs1Au7d%2B/WU089palTp6pnz551PZ1al56eruHDh%2BtXv/pVjZb38fGp5RnVvosXL%2BrixYt69tlnFRYWpltuuUVJSUn65JNPrrqON9R94cIFjR8/XoMGDdKePXuUnZ2toKAgPfvss1ddxxvqrqmfqtVb9oVlWVq4cKE%2B/PBDvfrqq2rUqJGkq9dX3%2Bs%2BdeqUli1bVu2XyJ9Sn%2BomYNUzERERcjqdbm2Xz1yFh4fXxZRq3fbt2zVu3DjNmjVLjz/%2BuKSr74ewsDD5%2Btbvl/XOnTuVm5ur3/72t9X6wsPDr1i3Nzz3l39LDwoKcrW1bNlSlmWpoqLCa%2Bv%2B4osvdPToUSUnJyswMFDNmzfXM888o48//lj%2B/v5eW/ePhYeHVzsLf7nWy%2B/rK%2B2Lf72FoL6qqqrSc889p%2B3btyszM1NRUVGuvrCwMK98Dbz44osaPny4W62XecvzXb//J2qA7Ha7jh8/7nYg2rdvn26//XYFBgbW4cxqx549e5SSkqKlS5dqyJAhrna73a6CggJVVFS42vbt26fY2Ni6mKZR77//vk6cOKG4uDh1797ddfNn9%2B7d1aFDh2p/ou0tdbdp00ZBQUFu9R07dkw2m0333Xef19ZtWVa1y9oXL16UJPXo0cNr6/4xu91erdacnBzFxsYqICBA7du3d%2Bt3Op06cuSI7Hb7jZ6qcS%2B88IIOHjyot99%2BWy1btnTrs9vtbvcdVVZWKj8/v96/Bt5//3299dZb6t69u7p3766VK1dqw4YN6t69u9c83wSseqZjx46KjY3V/PnzVVJSooKCAq1YsUKjRo2q66kZV1FRodTUVE2ePFm9e/d264uLi1NgYKAyMjJUXl6uL7/8UmvWrPGK/fDcc8/po48%2B0vr167V%2B/XqtWLFCkrR%2B/Xo99NBDOnbsmN58802dO3dOmzdv1meffabExMQ6nvUv5%2B/vr4SEBC1atEgnTpxQUVGRXn31VQ0ZMkSPPPKI19bdpUsXBQYGatmyZTp37pxOnz6tP//5z%2BratauGDBnitXX/WEJCgrKzs7Vx40adO3dOq1ev1pEjR1z354wcOVIrV67UN998o9LSUs2fP18xMTH1Pmjs3r1b69ev1x//%2BMdq9xxJ0ogRI5SVlaVdu3apvLxcr7zyiho3bqz4%2BPg6mK05O3bs0AcffOA6zo0YMULx8fGuvxr1hufbx7Isq64ngetz4sQJzZkzR3//%2B98VGBioxx57TJMmTarraRn31VdfadSoUQoICKjWt3nzZp05c0Zz5sxRXl6eIiIiNG7cOI0cObIOZlq7jh49qvvvv18FBQWSLu2XefPm6X//93916623atq0aerXr18dz9KMCxcu6MUXX9SHH34oX19f9e3bV7NmzVJQUJBX171v3z4tXLhQ%2B/fvl7%2B/v%2B6%2B%2B27NmDFDN998s1fVffnsw%2BUzzzabTdKlM1WStGXLFmVkZOj48eOKiopSamqqunXr5lp/2bJlevvtt1VeXq7u3btr7ty5uvnmm29wFdfvWnXPnDlT7733nqvtsrvuukt/%2BctfJElvv/22VqxYoVOnTikmJkbPP/%2B8/u3f/u0GVvDz/NTz/a%2BWLVumY8eO6cUXX3Rrq4/P92UELAAAAMO4RAgAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhv1/j/9pPkEnv30AAAAASUVORK5CYII%3D"/>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12" id="common8120722910657558042">
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">1265</td>
            <td class="number">0.7%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">64.6</td>
            <td class="number">301</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">61.5</td>
            <td class="number">296</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">66.5</td>
            <td class="number">293</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">63.0</td>
            <td class="number">291</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">61.3</td>
            <td class="number">289</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">58.3</td>
            <td class="number">287</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">66.6</td>
            <td class="number">282</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">55.9</td>
            <td class="number">281</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">62.6</td>
            <td class="number">280</td>
            <td class="number">0.2%</td>
            <td>
                <div class="bar" style="width:1%">&nbsp;</div>
            </td>
    </tr><tr class="other">
            <td class="fillremaining">Other values (1052)</td>
            <td class="number">168736</td>
            <td class="number">97.7%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
            <div role="tabpanel" class="tab-pane col-md-12"  id="extreme8120722910657558042">
                <p class="h4">Minimum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">0.0</td>
            <td class="number">1265</td>
            <td class="number">0.7%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.1</td>
            <td class="number">118</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:10%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.2</td>
            <td class="number">103</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:9%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.3</td>
            <td class="number">89</td>
            <td class="number">0.1%</td>
            <td>
                <div class="bar" style="width:7%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">0.4</td>
            <td class="number">76</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:6%">&nbsp;</div>
            </td>
    </tr>
    </table>
                <p class="h4">Maximum 5 values</p>
                
    <table class="freq table table-hover">
        <thead>
        <tr>
            <td class="fillremaining">Value</td>
            <td class="number">Count</td>
            <td class="number">Frequency (%)</td>
            <td style="min-width:200px">&nbsp;</td>
        </tr>
        </thead>
        <tr class="">
            <td class="fillremaining">120.2</td>
            <td class="number">2</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:100%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">122.5</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:50%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">127.6</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:50%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">128.1</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:50%">&nbsp;</div>
            </td>
    </tr><tr class="">
            <td class="fillremaining">140.4</td>
            <td class="number">1</td>
            <td class="number">0.0%</td>
            <td>
                <div class="bar" style="width:50%">&nbsp;</div>
            </td>
    </tr>
    </table>
            </div>
        </div>
    </div>
    </div>
        <div class="row headerrow highlight">
            <h1>Sample</h1>
        </div>
        <div class="row variablerow">
        <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
            <table border="1" class="dataframe sample">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>dti</th>
          <th>bc_util</th>
          <th>fico_range_low</th>
          <th>revol_util</th>
          <th>percent_bc_gt_75</th>
          <th>annual_inc</th>
          <th>avg_cur_bal</th>
          <th>purpose</th>
          <th>emp_length</th>
          <th>loan_status</th>
          <th>home_ownership</th>
          <th>loan_amnt</th>
          <th>loan_amnt_to_inc</th>
          <th>earliest_cr_line_age</th>
          <th>avg_cur_bal_to_inc</th>
          <th>avg_cur_bal_to_loan_amnt</th>
          <th>acc_open_past_24mths_groups</th>
        </tr>
        <tr>
          <th>id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>10129454</th>
          <td>4.62</td>
          <td>15.9</td>
          <td>720.0</td>
          <td>24.0</td>
          <td>0.0</td>
          <td>60000.0</td>
          <td>476.0</td>
          <td>debt_consolidation</td>
          <td>4.0</td>
          <td>Fully Paid</td>
          <td>RENT</td>
          <td>12000.0</td>
          <td>low</td>
          <td>126230400000000000</td>
          <td>0.007933</td>
          <td>0.039667</td>
          <td>high</td>
        </tr>
        <tr>
          <th>10148122</th>
          <td>12.61</td>
          <td>83.5</td>
          <td>705.0</td>
          <td>55.7</td>
          <td>100.0</td>
          <td>96500.0</td>
          <td>11783.0</td>
          <td>debt_consolidation</td>
          <td>3.0</td>
          <td>Fully Paid</td>
          <td>MORTGAGE</td>
          <td>12000.0</td>
          <td>low</td>
          <td>323481600000000000</td>
          <td>0.122104</td>
          <td>0.981917</td>
          <td>avg</td>
        </tr>
        <tr>
          <th>10149577</th>
          <td>18.55</td>
          <td>67.1</td>
          <td>745.0</td>
          <td>54.6</td>
          <td>16.7</td>
          <td>325000.0</td>
          <td>53306.0</td>
          <td>debt_consolidation</td>
          <td>5.0</td>
          <td>Fully Paid</td>
          <td>MORTGAGE</td>
          <td>28000.0</td>
          <td>low</td>
          <td>602208000000000000</td>
          <td>0.164018</td>
          <td>1.903786</td>
          <td>high</td>
        </tr>
        <tr>
          <th>10149342</th>
          <td>22.87</td>
          <td>53.9</td>
          <td>730.0</td>
          <td>61.2</td>
          <td>25.0</td>
          <td>55000.0</td>
          <td>9570.0</td>
          <td>debt_consolidation</td>
          <td>10.5</td>
          <td>Fully Paid</td>
          <td>OWN</td>
          <td>27050.0</td>
          <td>avg</td>
          <td>857347200000000000</td>
          <td>0.174000</td>
          <td>0.353789</td>
          <td>avg</td>
        </tr>
        <tr>
          <th>10119623</th>
          <td>13.03</td>
          <td>93.0</td>
          <td>715.0</td>
          <td>67.0</td>
          <td>1.0</td>
          <td>130000.0</td>
          <td>36362.0</td>
          <td>debt_consolidation</td>
          <td>10.5</td>
          <td>Fully Paid</td>
          <td>MORTGAGE</td>
          <td>12000.0</td>
          <td>low</td>
          <td>507513600000000000</td>
          <td>0.279708</td>
          <td>3.030167</td>
          <td>avg</td>
        </tr>
      </tbody>
    </table>
        </div>
    </div>
    </div>



Predictive Modeling
===================

.. code:: ipython3

    def to_xy(dataset):
        y = dataset.pop('loan_status').cat.codes
        X = pd.get_dummies(dataset, drop_first=True)
        return X, y

Initializing Train/Test Sets
----------------------------

Shuffle and Split Data
~~~~~~~~~~~~~~~~~~~~~~

Let's split the data (both features and their labels) into training and
test sets. 80% of the data will be used for training and 20% for
testing.

Run the code cell below to perform this split.

.. code:: ipython3

    X, y = load_and_preprocess_data().pipe(to_xy)
    
    split_data = train_test_split(X, y, test_size=0.20, stratify=y, random_state=11)
    X_train, X_test, y_train, y_test = split_data
    
    train_test_sets = dict(
        zip(['X_train', 'X_test', 'y_train', 'y_test'], [*split_data]))

.. code:: ipython3

    (pd.DataFrame(
        data={'Observations (#)': [X_train.shape[0], X_test.shape[0]],
              'Percent (%)': ['80%', '20%'],
              'Features (#)': [X_train.shape[1], X_test.shape[1]]},
        index=['Training', 'Test'])
     [['Percent (%)', 'Features (#)', 'Observations (#)']])




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Percent (%)</th>
          <th>Features (#)</th>
          <th>Observations (#)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Training</th>
          <td>80%</td>
          <td>34</td>
          <td>138196</td>
        </tr>
        <tr>
          <th>Test</th>
          <td>20%</td>
          <td>34</td>
          <td>34549</td>
        </tr>
      </tbody>
    </table>
    </div>



Classification Models
---------------------

Naive Predictor (Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    dummy_model = LearningModel(
        'Naive Predictor - Baseline', Pipeline([
            ('imp', Imputer(strategy='median')), 
            ('clf', DummyClassifier(strategy='constant', constant=0))]))
    
    dummy_model.fit_and_predict(**train_test_sets)
    
    model_evals = eval_db(dummy_model.eval_report)

Decision Tree Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    tree_model = LearningModel(
        'Decision Tree Classifier', Pipeline([
            ('imp', Imputer(strategy='median')), 
            ('clf', DecisionTreeClassifier(class_weight='balanced', random_state=11))]))
    
    tree_model.fit_and_predict(**train_test_sets)
    tree_model.display_evaluation()
    
    model_evals = eval_db(model_evals, tree_model.eval_report)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FitTime</th>
          <th>Accuracy</th>
          <th>FBeta</th>
          <th>F1</th>
          <th>AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Decision Tree Classifier</th>
          <td>2.0</td>
          <td>0.78558</td>
          <td>0.156742</td>
          <td>0.154531</td>
          <td>0.516244</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

                 precision    recall  f1-score   support
    
              0       0.88      0.87      0.88     30271
              1       0.15      0.16      0.15      4278
    
    avg / total       0.79      0.79      0.79     34549
    



.. image:: output_66_2.png


Random Forest Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    rf_model = LearningModel(
        'Random Forest Classifier', Pipeline([
            ('imp', Imputer(strategy='median')), 
            ('clf', RandomForestClassifier(
                class_weight='balanced_subsample', random_state=11))]))
    
    rf_model.fit_and_predict(**train_test_sets)
    rf_model.display_evaluation()
    
    model_evals = eval_db(model_evals, rf_model.eval_report)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FitTime</th>
          <th>Accuracy</th>
          <th>FBeta</th>
          <th>F1</th>
          <th>AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Random Forest Classifier</th>
          <td>2.0</td>
          <td>0.874671</td>
          <td>0.008998</td>
          <td>0.014117</td>
          <td>0.573608</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

                 precision    recall  f1-score   support
    
              0       0.88      1.00      0.93     30271
              1       0.27      0.01      0.01      4278
    
    avg / total       0.80      0.87      0.82     34549
    



.. image:: output_68_2.png


Blagging Classifier
~~~~~~~~~~~~~~~~~~~

Base Estimator -> RF
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    blagging_pipeline = Pipeline([
        ('imp', Imputer(strategy='median')), 
        ('clf', BlaggingClassifier(
            random_state=11, n_jobs=-1,
            base_estimator=RandomForestClassifier(
                class_weight='balanced_subsample', random_state=11)))])
    
    blagging_model = LearningModel('Blagging Classifier (RF)', blagging_pipeline)
    
    blagging_model.fit_and_predict(**train_test_sets)
    blagging_model.display_evaluation()
    
    model_evals = eval_db(model_evals, blagging_model.eval_report)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FitTime</th>
          <th>Accuracy</th>
          <th>FBeta</th>
          <th>F1</th>
          <th>AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Blagging Classifier (RF)</th>
          <td>2.0</td>
          <td>0.719181</td>
          <td>0.344518</td>
          <td>0.270746</td>
          <td>0.645877</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

                 precision    recall  f1-score   support
    
              0       0.90      0.76      0.83     30271
              1       0.20      0.42      0.27      4278
    
    avg / total       0.82      0.72      0.76     34549
    



.. image:: output_71_2.png


Base Estimator -> ExtraTrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    blagging_clf = BlaggingClassifier(
        random_state=11, n_jobs=-1,
        base_estimator=ExtraTreesClassifier(
            criterion='entropy', class_weight='balanced_subsample', 
            max_features=None, n_estimators=60, random_state=11))
    
    blagging_model = LearningModel(
        'Blagging Classifier (Extra Trees)', Pipeline([
            ('imp', Imputer(strategy='median')), 
            ('clf', blagging_clf)]))
    
    blagging_model.fit_and_predict(**train_test_sets)
    blagging_model.display_evaluation()
    
    model_evals = eval_db(model_evals, blagging_model.eval_report)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FitTime</th>
          <th>Accuracy</th>
          <th>FBeta</th>
          <th>F1</th>
          <th>AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Blagging Classifier (Extra Trees)</th>
          <td>19.0</td>
          <td>0.749718</td>
          <td>0.309224</td>
          <td>0.259611</td>
          <td>0.645899</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

                 precision    recall  f1-score   support
    
              0       0.90      0.81      0.85     30271
              1       0.20      0.35      0.26      4278
    
    avg / total       0.81      0.75      0.78     34549
    



.. image:: output_73_2.png


Evaluating Model Performance
----------------------------

Feature Importance (via RandomForestClassifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    rf_top_features = LearningModel('Random Forest Classifier',
        Pipeline([('imp', Imputer(strategy='median')), 
                  ('clf', RandomForestClassifier(max_features=None,
                      class_weight='balanced_subsample', random_state=11))]))
                      
    rf_top_features.fit_and_predict(**train_test_sets)

.. code:: ipython3

    rf_top_features.display_top_features(top_n=15)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Feature</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>dti</td>
          <td>0.115602</td>
        </tr>
        <tr>
          <th>2</th>
          <td>earliest_cr_line_age</td>
          <td>0.115381</td>
        </tr>
        <tr>
          <th>3</th>
          <td>revol_util</td>
          <td>0.109714</td>
        </tr>
        <tr>
          <th>4</th>
          <td>annual_inc</td>
          <td>0.099535</td>
        </tr>
        <tr>
          <th>5</th>
          <td>loan_amnt</td>
          <td>0.080153</td>
        </tr>
        <tr>
          <th>6</th>
          <td>bc_util</td>
          <td>0.077465</td>
        </tr>
        <tr>
          <th>7</th>
          <td>fico_range_low</td>
          <td>0.071342</td>
        </tr>
        <tr>
          <th>8</th>
          <td>avg_cur_bal_to_loan_amnt</td>
          <td>0.062594</td>
        </tr>
        <tr>
          <th>9</th>
          <td>avg_cur_bal_to_inc</td>
          <td>0.052817</td>
        </tr>
        <tr>
          <th>10</th>
          <td>avg_cur_bal</td>
          <td>0.050275</td>
        </tr>
        <tr>
          <th>11</th>
          <td>emp_length</td>
          <td>0.047003</td>
        </tr>
        <tr>
          <th>12</th>
          <td>percent_bc_gt_75</td>
          <td>0.034591</td>
        </tr>
        <tr>
          <th>13</th>
          <td>home_ownership_RENT</td>
          <td>0.009037</td>
        </tr>
        <tr>
          <th>14</th>
          <td>purpose_credit_card</td>
          <td>0.008536</td>
        </tr>
        <tr>
          <th>15</th>
          <td>purpose_debt_consolidation</td>
          <td>0.007986</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: ipython3

    rf_top_features.plot_top_features(top_n=10)



.. image:: output_78_0.png


Model Selection
---------------

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    display(model_evals)



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FitTime</th>
          <th>Accuracy</th>
          <th>FBeta</th>
          <th>F1</th>
          <th>AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Naive Predictor - Baseline</th>
          <td>0.0</td>
          <td>0.876176</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.500000</td>
        </tr>
        <tr>
          <th>Decision Tree Classifier</th>
          <td>2.0</td>
          <td>0.785580</td>
          <td>0.156742</td>
          <td>0.154531</td>
          <td>0.516244</td>
        </tr>
        <tr>
          <th>Random Forest Classifier</th>
          <td>2.0</td>
          <td>0.874671</td>
          <td>0.008998</td>
          <td>0.014117</td>
          <td>0.573608</td>
        </tr>
        <tr>
          <th>Blagging Classifier (RF)</th>
          <td>2.0</td>
          <td>0.719181</td>
          <td>0.344518</td>
          <td>0.270746</td>
          <td>0.645877</td>
        </tr>
        <tr>
          <th>Blagging Classifier (Extra Trees)</th>
          <td>19.0</td>
          <td>0.749718</td>
          <td>0.309224</td>
          <td>0.259611</td>
          <td>0.645899</td>
        </tr>
      </tbody>
    </table>
    </div>


Optimal Model
~~~~~~~~~~~~~

.. code:: ipython3

    blagging_model = LearningModel('Blagging Classifier (Extra Trees)', 
        Pipeline([('imp', Imputer(strategy='median')), 
                  ('clf', BlaggingClassifier(
                      base_estimator=ExtraTreesClassifier(
                          criterion='entropy', class_weight='balanced_subsample', 
                          max_features=None, n_estimators=60, random_state=11), 
                      random_state=11, n_jobs=-1))]))
    
    blagging_model.fit_and_predict(**train_test_sets)

Optimizing Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

ToDo: Perform GridSearch...
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results:
^^^^^^^^

.. code:: ipython3

    (pd.DataFrame(data={'Benchmark Predictor': [0.7899, 0.1603, 0.5203],
                       'Unoptimized Model': [0.7499, 0.2602, 0.6463],
                       'Optimized Model': ['', '', '']}, 
                 index=['Accuracy Score', 'F1-score', 'AUC'])
     [['Benchmark Predictor', 'Unoptimized Model', 'Optimized Model']])




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Benchmark Predictor</th>
          <th>Unoptimized Model</th>
          <th>Optimized Model</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Accuracy Score</th>
          <td>0.7899</td>
          <td>0.7499</td>
          <td></td>
        </tr>
        <tr>
          <th>F1-score</th>
          <td>0.1603</td>
          <td>0.2602</td>
          <td></td>
        </tr>
        <tr>
          <th>AUC</th>
          <td>0.5203</td>
          <td>0.6463</td>
          <td></td>
        </tr>
      </tbody>
    </table>
    </div>



Conclusion \*Pending
====================

References
==========
