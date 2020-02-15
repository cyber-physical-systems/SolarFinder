import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn.violinplot API
# https://seaborn.pydata.org/generated/seaborn.violinplot.html
# col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']
# col_names = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']
# col_names = ['id', 'location', 'image', 'size', 'pole', 'gray mean', 'gray standard deviation', 'blue mean', 'green mean', 'red mean', 'blue standard deviation', 'green standard deviation', 'red standard deviation', 'square similarity', 'width height ratio', 'area ratio', 'number of curves', 'number of corners', 'number of corners less 90', 'number of corners less 70', 'label', 'vgg_pro', 'vgg_class']
col_names = ['id', 'location', 'image', 'size', 'pole', 'gray_mean', 'gray_std_deviation', 'blue_mean', 'green_mean', 'red_mean', 'blue_std_deviation', 'green_std_deviation', 'red_std_deviation', 'square_similarity', 'width_height_ratio', 'area_ratio', 'number_of_curves', 'number_of_corners', 'corners_less_90', 'corners_less_70', 'label', 'vgg_pro', 'vgg_class']

data = pd.read_csv("./data/final/split/feature_17_all.csv", names=col_names)

data = data.dropna()

# print(data[:5])
# print(data.shape)

g_plot_outputDir = './output/location1-7/violinplot/'

positive_sample_set = data[data['label'] == 1.0]
negative_sample_set = data[data['label'] == 0.0]
# random_sample_set = data[(data['label'] != 0.0) & (data['label'] != 1.0)]


analysis_features = ['size', 'gray_mean', 'gray_std_deviation', 'blue_mean', 'green_mean', 'red_mean', 'blue_std_deviation', 'green_std_deviation', 'red_std_deviation', 'square_similarity', 'width_height_ratio', 'area_ratio', 'number_of_curves', 'number_of_corners', 'corners_less_90', 'corners_less_70']
# analysis_features = ['mean']


labels_to_draw = ['25%','75%']

def draw_single_label(plot, pos, value):
    plot.text(
            pos, 
            value, 
            str(np.round(value, 2)), 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=30,
            color='white',
            bbox=dict(facecolor='#445A64')
            )

def get_whiskers(feature_array):
    Q1, median, Q3 = np.percentile(np.asarray(feature_array), [25, 50, 75])

    IQR = Q3 - Q1

    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR

    upper_wisk_set = np.compress(feature_array <= hival, feature_array)
    lower_wisk_set = np.compress(feature_array >= loval, feature_array)
    upper_wisk = np.max(upper_wisk_set)
    lower_wisk = np.min(lower_wisk_set)

    return [lower_wisk, upper_wisk]

palette = sns.color_palette(["#e69138", "#3d85c6"])

for analysis_feature in analysis_features:

    data_whis = get_whiskers(data[analysis_feature])

    positive_sample_set_description = positive_sample_set[analysis_feature].describe()
    print('positive_sample_set:')
    print(positive_sample_set_description)
    positive_whis = get_whiskers(positive_sample_set[analysis_feature])
    print(positive_whis[0])
    print(positive_whis[1])

    negative_sample_set_description = negative_sample_set[analysis_feature].describe()
    print('negative_sample_set:')
    print(negative_sample_set_description)
    negative_whis = get_whiskers(negative_sample_set[analysis_feature])
    print(negative_whis[0])
    print(negative_whis[1])

    data_to_show = data.loc[(data[analysis_feature] > data_whis[0]) & (data[analysis_feature] < data_whis[1])]
    
    # Generate boxplot
    # sns.set(font_scale = font_scale_value)
    # sns.set_context(rc={'xtick.major.size': 6.0, 'ytick.minor.size': 4.0, 'legend.fontsize': 22.0, 'ytick.major.width': 1.25, 'axes.labelsize': 24.0, 'ytick.minor.width': 1.0, 'xtick.minor.width': 1.0, 'font.size': 24.0, 'grid.linewidth': 1.0, 'axes.titlesize': 24.0, 'axes.linewidth': 1.25, 'patch.linewidth': 1.0, 'ytick.labelsize': 22.0, 'xtick.labelsize': 10.0, 'lines.linewidth': 1.5, 'ytick.major.size': 6.0, 'lines.markersize': 6.0, 'xtick.major.width': 1.25, 'xtick.minor.size': 4.0})
    # sns.set_context(rc={'axes.titlesize': 'large', 'grid.linewidth': 0.8, 'lines.markersize': 6.0, 'xtick.major.size': 3.5, 'xtick.major.width': 0.8, 'ytick.major.size': 3.5, 'ytick.minor.width': 0.6, 'axes.linewidth': 0.8, 'xtick.labelsize': 'medium', 'patch.linewidth': 1.0, 'ytick.labelsize': 'medium', 'xtick.minor.size': 2.0, 'font.size': 10.0, 'legend.fontsize': 'medium', 'lines.linewidth': 1.5, 'ytick.minor.size': 2.0, 'xtick.minor.width': 0.6, 'axes.labelsize': 'medium', 'ytick.major.width': 0.8})
    sns.set(rc={'figure.figsize':(10, 6)})
    sns.set_context(rc={'axes.titlesize': 22.0, 'axes.labelsize': 50.0, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})
    # print(sns.plotting_context())

    sns_violinplot = sns.violinplot(x='label', y=analysis_feature, data=data_to_show, showfliers=False, split=False, palette=palette)
    # sns_boxplot = sns.boxplot(x='label', y=analysis_feature, data=data)
    sns.despine(offset=10, trim=True);
    for l in labels_to_draw:
        draw_single_label(sns_violinplot, 1, positive_sample_set_description[l])
        draw_single_label(sns_violinplot, 0, negative_sample_set_description[l])

    for l in positive_whis:
        draw_single_label(sns_violinplot, 1, l)

    for l in negative_whis:
        draw_single_label(sns_violinplot, 0, l)

    # sns_violinplot.set_title(analysis_feature)

    # ADDED: Extract axes.
    sns_violinplot.set_xlabel('')

    fig = sns_violinplot.get_figure()
    fig.savefig(g_plot_outputDir + analysis_feature + '_violinplot.png')
    
    plt.show()
    # break
