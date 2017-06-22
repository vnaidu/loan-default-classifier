import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display as ipy_display
import warnings
#%config InlineBackend.figure_format='retina'
#%matplotlib inline
sns.set()


def summary__X_numeric(df_features):
    X_numeric = df_features.select_dtypes(exclude=['O'])
    descr_X_numeric = X_numeric.describe().T
    summary = descr_X_numeric[['min', 'max', 'mean']].copy()
    summary['median'] = X_numeric.median(skipna=True)
    summary['std'] = descr_X_numeric['std']
    summary['var'] = X_numeric.var(skipna=True)
    summary = summary.drop(summary.index[summary.isnull().sum(axis=1)==len(summary.columns)], axis=0).round(2)
    return summary


def summary__X_objs(df_features):
    summary = (df_features
               .select_dtypes(include=['O'])
               .describe().T
               .drop('count', axis=1)
               .sort_values('freq', ascending=False))
    summary[['unique', 'freq']] = ((summary[['unique', 'freq']])
                                               .apply(lambda x: x.astype(np.int), axis=1))
    summary['freq_pct'] = np.around((summary.freq / len(df_features)), 4)
    return summary


def display__summary__X_objs(df_features):
    summary = summary__X_objs(df_features)
    summary.columns = ['Unique (#)', 'Top Factor', 'Observations (#)', 'Frequency (%)']
    summary_styled = (
        summary.style
        .set_caption('Exploratory Factor Analysis')
        .background_gradient(cmap=sns.light_palette("orange", as_cmap=True), low=.2, high=.4, subset=['Unique (#)'])
        .background_gradient(cmap=sns.light_palette("green", as_cmap=True), low=0, high=1, subset=['Observations (#)'])
        .background_gradient(cmap=sns.light_palette("green", as_cmap=True), low=0, high=1, subset=['Frequency (%)'])
        .applymap(lambda s: 'background-color: #ffd17b' if any([i.isdigit() for i in s]) else 'background-color: #83c983',
                  subset=['Top Factor'])
        .format({'Unique (#)': '{:,}', 'Observations (#)': '{:,}', 'Frequency (%)': '{:.2%}'}))

    return summary_styled


def incomplete_stats(lc):
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    missing_data = pd.DataFrame(index=lc.columns)
    missing_data['Null'] = lc.isnull().sum()
    missing_data['NA_or_Missing'] = (
        lc.apply(lambda col: (col.str.contains('(^$|n/a|^na$|^%$)', case=False)
                              .sum())).fillna(0).astype(int))
    missing_data['Incomplete'] = ((missing_data.Null + missing_data.NA_or_Missing) / len(lc))
    incomplete_features = ((missing_data[(missing_data > 0).any(axis=1)])
                           .sort_values('Incomplete', ascending=False))
    return incomplete_features

def display__incomplete_stats(dataset):
    incomplete_features = incomplete_stats(dataset)
    df_incomplete = (incomplete_features.style
     .set_caption('Missing')
     .background_gradient(cmap=sns.light_palette("orange", as_cmap=True), low=0, high=1, subset=['Null', 'NA_or_Missing'])
     .background_gradient(cmap=sns.light_palette("red", as_cmap=True),
                          low=0, high=.6, subset=['Incomplete'])
     .format({'Null': '{:,}', 'NA_or_Missing': '{:,}', 'Incomplete': '{:.1%}'}))
    return df_incomplete

def plot__incomplete_stats(dataset):
    incomplete_features = incomplete_stats(dataset)
    #return incomplete_features
    incomplete_features.Incomplete = incomplete_features.Incomplete * 100
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure()
    canvas = FigureCanvas(fig)
    _ = sns.barplot(x=incomplete_features.index.tolist(),
            y=incomplete_features.Incomplete.tolist())
    for item in _.get_xticklabels():
        item.set_rotation(45)
    _.set(xlabel='Feature', ylabel='Incomplete (%)', title='Features with Missing or Null Values')
    #fig = plt.figure(figsize=(8, 6));
    #fig.plot(_)
    plt.show()
    return canvas.print_figure('test')


def miscodings__null_as_strings(dataset):
    stats = dataset.apply(lambda feature: len(feature.str.extract('(^$|n/a|^na$|^%$)', expand=False).dropna()))
    if (stats.sum()==0):
        return pd.DataFrame(data=[['','']], columns=['missing_count', 'encoded_as'], index=['No Features'])
    missing_objs = pd.DataFrame(stats[stats>0], columns=['missing_count'])
    missing_objs['encoded_as'] = (dataset[missing_objs.index]
         .apply(lambda feature: feature.str.extract('(^$|n/a|^na$|^%$)', expand=False).dropna().unique().tolist()))
    return missing_objs

def miscodings__digits_in_objs(dataset):
    df_objs = dataset.select_dtypes(include=['O'])
    objs_with_digits = df_objs.apply(lambda col: col.str.contains('\d').sum())
    digit_objs = pd.DataFrame({'Count': objs_with_digits[objs_with_digits>0]})
    digit_objs['Percentage'] = digit_objs/len(df_objs)
    return digit_objs

def display__miscodings__digits_in_objs(dataset, style=True):
    digit_objs = miscodings__digits_in_objs(dataset)
    if (len(digit_objs) == 0):
        style = False
    if (style == False):
        return digit_objs
    else:
        styled_df = (digit_objs.style
         .set_caption('Object Features Containing Digits')
         .applymap(lambda x: 'background-color: red' if x ==len(digit_objs) else 'background-color: orange', subset=['Count'])
         .applymap(lambda x: 'background-color: red' if x == 1 else 'background-color: orange', subset=['Percentage'])
         .format({'Count': '{:,}', 'Percentage': '{:.2%}'}))
        return styled_df

def samples__miscodings__digits_in_objs(dataset):
    df_objs = dataset.select_dtypes(include=['O'])
    digit_objs = miscodings__digits_in_objs(dataset)
    if (len(digit_objs) != 0):
        samples = df_objs[digit_objs.index][:4].T
        return samples

def display_report__possible_miscodings(dataset):
    stats__digits_in_objs = display__miscodings__digits_in_objs(dataset)
    samples__digits_in_objs = samples__miscodings__digits_in_objs(dataset)
    stats__null_as_strings = miscodings__null_as_strings(dataset)
    display(stats__digits_in_objs)

def eda_display(to_display, dataset):
    if (to_display == 'incomplete features'):
        disp = display__incomplete_stats(dataset)
        return ipy_display(disp)
    else:
        return 'invalid option: ' + to_display

def eda_plot(to_plot, dataset):
    if (to_plot == 'incomplete features'):
        plot__incomplete_stats(dataset)
        return #ipy_display(plot)
    else:
        return 'invalid option: ' + to_plot
