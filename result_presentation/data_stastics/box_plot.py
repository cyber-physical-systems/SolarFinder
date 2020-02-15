import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn.boxplot API
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
# Understanding Boxplots
# https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

# col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']
col_names = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']

data = pd.read_csv("./data/final/split/feature_17_all.csv", names=col_names)

data = data.dropna()

# print(data[:5])
# print(data.shape)

<<<<<<< HEAD
g_plot_outputDir = './solarpanel/output/final/split/boxplot/'
=======
g_plot_outputDir = ''
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62

positive_sample_set = data[data['label'] == 1.0]
negative_sample_set = data[data['label'] == 0.0]
# random_sample_set = data[(data['label'] != 0.0) & (data['label'] != 1.0)]

analysis_features = ['size', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']
# analysis_features = ['size']

labels_to_draw = ['25%','75%']

def draw_label(plot, label_type):
    labels = [negative_sample_set_description[label_type], positive_sample_set_description[label_type]]
    labels_text = [str(np.round(s, 2)) for s in labels]

    pos = range(len(labels_text))

    for tick,label in zip(pos, plot.get_xticklabels()):
        plot.text(
            pos[tick], 
            labels[tick], 
            labels_text[tick], 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=10,
            color='white',
            bbox=dict(facecolor='#445A64'))

def draw_single_label(plot, pos, value):
    plot.text(
            pos, 
            value, 
            str(np.round(value, 2)), 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=20,
            color='white',
            bbox=dict(facecolor='#445A64'))

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

    sns.set(font_scale = 2)

    # Generate boxplot
    sns_boxplot = sns.boxplot(x='label', y=analysis_feature, data=data, showfliers=False, palette=palette)
    # sns_boxplot = sns.boxplot(x='label', y=analysis_feature, data=data)

    for l in labels_to_draw:
        draw_single_label(sns_boxplot, 1, positive_sample_set_description[l])
        draw_single_label(sns_boxplot, 0, negative_sample_set_description[l])

    for l in positive_whis:
        draw_single_label(sns_boxplot, 1, l)

    for l in negative_whis:
        draw_single_label(sns_boxplot, 0, l)

    sns_boxplot.set_title(analysis_feature+'_distribution_boxplot')

    fig = sns_boxplot.get_figure()
    fig.savefig(g_plot_outputDir + analysis_feature + '_boxplot.png')
    plt.show()
