import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import csv

# dataset = 'split'
dataset = 'non-split'


if dataset == 'split':
    col_names = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']
    #split training data path
    training_data_csv_path = "./feature_17_all.csv"

    g_outputDir = './final/split/'
    csv_path = g_outputDir + 'split_data_hard_filters.csv'
elif dataset == 'non-split':
    col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']
    # non-split training data path
    training_data_csv_path = "./final/non_split/feature_train_all.csv"

    g_outputDir = './output/final/non_split/'
    csv_path = g_outputDir + 'non_split_data_hard_filters.csv'
else:
    print('No dataset is selected.')
    exit()

data = pd.read_csv(training_data_csv_path, names=col_names)
data = data.dropna()

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

csv_header = ['feature', 'filter_type', 'accept_zone', 'reject_zone', 'unsure_zone']
with open(csv_path, 'a') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()
csv_file.close()

hard_filters = {}

for analysis_feature in analysis_features:

    hard_filters[analysis_feature] = {}

    positive_sample_set_description = positive_sample_set[analysis_feature].describe()
    print('positive_sample_set:')

    positive_output = {}
    
    for l in labels:
        positive_output[l] = positive_sample_set_description[l]
    
    positive_whis = get_whiskers(positive_sample_set[analysis_feature])
    positive_output['0.35%'] = positive_whis[0]
    positive_output['99.65%'] = positive_whis[1]

    print(positive_output)

    ############

    negative_sample_set_description = negative_sample_set[analysis_feature].describe()
    print('negative_sample_set:')

    negative_output = {}

    for l in labels:
        negative_output[l] = negative_sample_set_description[l]
    
    negative_whis = get_whiskers(negative_sample_set[analysis_feature])
    negative_output['0.35%'] = negative_whis[0]
    negative_output['99.65%'] = negative_whis[1]

    print(negative_output)

    NU = negative_output['99.65%']
    NL = negative_output['0.35%']
    PU = positive_output['99.65%']
    PL = positive_output['0.35%']

    if NU == PU and NL == PL:
        hard_filters[analysis_feature]['filter_type'] = 'equal'
        hard_filters[analysis_feature]['accept_zone'] = []
        hard_filters[analysis_feature]['reject_zone'] = []
        hard_filters[analysis_feature]['unsure_zone'] = [[NL, NU]]
    elif NU >= PU and NL <= PL:
        hard_filters[analysis_feature]['filter_type'] = 'contain'
        hard_filters[analysis_feature]['accept_zone'] = []
        hard_filters[analysis_feature]['reject_zone'] = [[NL, PL], [PU, NU]]
        hard_filters[analysis_feature]['unsure_zone'] = [[PL, PU]]
    elif NU < PU and NU > PL and NL < PL:
        hard_filters[analysis_feature]['filter_type'] = 'intersect-1over0'
        hard_filters[analysis_feature]['accept_zone'] = [[NU, PU]]
        hard_filters[analysis_feature]['reject_zone'] = [[NL, PL]]
        hard_filters[analysis_feature]['unsure_zone'] = [[PL, NU]]
    elif NL > PL and NL < PU and NU > PU:
        hard_filters[analysis_feature]['filter_type'] = 'intersect-0over1'
        hard_filters[analysis_feature]['accept_zone'] = [[PL, NL]]
        hard_filters[analysis_feature]['reject_zone'] = [[PU, NU]]
        hard_filters[analysis_feature]['unsure_zone'] = [[NL, PU]]
    else:
        hard_filters[analysis_feature]['filter_type'] = 'undefine'
        hard_filters[analysis_feature]['accept_zone'] = []
        hard_filters[analysis_feature]['reject_zone'] = []
        hard_filters[analysis_feature]['unsure_zone'] = []

    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([analysis_feature, str(hard_filters[analysis_feature]['filter_type']), str(hard_filters[analysis_feature]['accept_zone']), str(hard_filters[analysis_feature]['reject_zone']), str(hard_filters[analysis_feature]['unsure_zone'])])
    csv_file.close()
    
print(hard_filters)


    # input('Press ENTER to continue...')


