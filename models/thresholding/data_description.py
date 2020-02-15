import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import csv

# col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']
col_names = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']

data = pd.read_csv("./feature_17_all.csv", names=col_names)
data = data.dropna()

g_outputDir = './output/final/split/'
csv_path = g_outputDir + 'feature_description.csv'

positive_sample_set = data[data['label'] == 1.0]
negative_sample_set = data[data['label'] == 0.0]

analysis_features = ['size', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']
# analysis_features = ['size']

labels = ['mean', 'std', 'min', 'max', '50%', '25%','75%']

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

csv_header = ['feature', 'mean', 'std', 'min', 'max', 'median', '25%', '75%', '0.35%', '99.65%']
with open(csv_path, 'a') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()
csv_file.close()

output = {}

for analysis_feature in analysis_features:

    positive_sample_set_description = positive_sample_set[analysis_feature].describe()
    print('positive_sample_set:')

    row_name = str(analysis_feature+'_pos')
    
    for l in labels:
        output[l] = positive_sample_set_description[l]
    
    positive_whis = get_whiskers(positive_sample_set[analysis_feature])
    output['0.35%'] = positive_whis[0]
    output['99.65%'] = positive_whis[1]

    print(output)

    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([row_name, output['mean'], output['std'], output['min'], output['max'], output['50%'], output['25%'], output['75%'], output['0.35%'], output['99.65%']])
    csv_file.close()


    negative_sample_set_description = negative_sample_set[analysis_feature].describe()
    print('negative_sample_set:')
    row_name = str(analysis_feature+'_neg')

    for l in labels:
        output[l] = negative_sample_set_description[l]
    
    negative_whis = get_whiskers(negative_sample_set[analysis_feature])
    output['0.35%'] = negative_whis[0]
    output['99.65%'] = negative_whis[1]

    print(output)

    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([row_name, output['mean'], output['std'], output['min'], output['max'], output['50%'], output['25%'], output['75%'], output['0.35%'], output['99.65%']])
    csv_file.close()

    # input('Press ENTER to continue...')


