import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import csv

# dataset = 'split'
dataset = 'non-split'

# Generate hard filters

if dataset == 'split':
    col_names = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']
    #split training data path
    training_data_csv_path = "./data/final/split/feature_17_all.csv"
elif dataset == 'non-split':
    col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'vgg_pro', 'vgg_class']
    # non-split training data path
    training_data_csv_path = "./data/final/non_split/feature_train_all.csv"
else:
    print('No dataset is selected.')
    exit()

data = pd.read_csv(training_data_csv_path, names=col_names)
data = data.dropna()

positive_sample_set = data[data['label'] == 1.0]
negative_sample_set = data[data['label'] == 0.0]

analysis_features = ['size', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']
# analysis_features = ['size']

number_of_features = len(analysis_features) + 1

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

hard_filters = {}

for analysis_feature in analysis_features:

    hard_filters[analysis_feature] = {}

    positive_sample_set_description = positive_sample_set[analysis_feature].describe()

    positive_output = {}
    
    for l in labels:
        positive_output[l] = positive_sample_set_description[l]
    
    positive_whis = get_whiskers(positive_sample_set[analysis_feature])
    positive_output['0.35%'] = positive_whis[0]
    positive_output['99.65%'] = positive_whis[1]

    ############

    negative_sample_set_description = negative_sample_set[analysis_feature].describe()
    
    negative_output = {}

    for l in labels:
        negative_output[l] = negative_sample_set_description[l]
    
    negative_whis = get_whiskers(negative_sample_set[analysis_feature])
    negative_output['0.35%'] = negative_whis[0]
    negative_output['99.65%'] = negative_whis[1]

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
        hard_filters[analysis_feature]['filter_type'] = 'intersect'
        hard_filters[analysis_feature]['accept_zone'] = [[NU, PU]]
        hard_filters[analysis_feature]['reject_zone'] = [[NL, PL]]
        hard_filters[analysis_feature]['unsure_zone'] = [[PL, NU]]
    elif NL > PL and NL < PU and NU > PU:
        hard_filters[analysis_feature]['filter_type'] = 'intersect'
        hard_filters[analysis_feature]['accept_zone'] = [[PL, NL]]
        hard_filters[analysis_feature]['reject_zone'] = [[PU, NU]]
        hard_filters[analysis_feature]['unsure_zone'] = [[NL, PU]]
    else:
        hard_filters[analysis_feature]['filter_type'] = 'undefine'
        hard_filters[analysis_feature]['accept_zone'] = []
        hard_filters[analysis_feature]['reject_zone'] = []
        hard_filters[analysis_feature]['unsure_zone'] = []
    # input('Press ENTER to continue...')
print(hard_filters)

print('start testing...')

# Test data

def filter(feature_value, filters):

    feature_value = float(feature_value)

    possibility = 0.5

    if len(filters['accept_zone']) != 0:
        for r in filters['accept_zone']:
            if feature_value >= float(r[0]) and feature_value <= float(r[1]):
                possibility = 1
                return possibility

    if len(filters['reject_zone']) != 0:
        for r in filters['reject_zone']:
            if feature_value >= float(r[0]) and feature_value <= float(r[1]):
                possibility = 0
                return possibility

    return possibility

if dataset == 'split':
    g_output_dir = './output/final/split/'
    output_csv_path = g_output_dir + 'split_810_test_result.csv'

    g_test_data_dir = './data/final/split/'
    test_data_csv_path = g_test_data_dir + 'feature_810_all.csv'
    
    output_csv_header = ['id', 'location', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'hard_pred_label', 'hard_pred_pos']
elif dataset == 'non-split':
    g_output_dir = './output/final/non_split/'
    output_csv_path = g_output_dir + 'non_split_test_result.csv'

    g_test_data_dir = './data/final/non_split/'
    test_data_csv_path = g_test_data_dir + 'feature_test_all.csv'

    output_csv_header = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'b_mean', 'g_mean', 'r_mean', 'b_stddev', 'g_stddev', 'r_stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label', 'hard_pred_label', 'hard_pred_pos']
else:
    print('No dataset is selected.')
    exit()

with open(output_csv_path, 'a') as output_csv_file:
    writer = csv.DictWriter(output_csv_file, fieldnames=output_csv_header)
    writer.writeheader()
output_csv_file.close()

with open(test_data_csv_path, newline='') as test_data_csv_file:
    reader = csv.DictReader(test_data_csv_file)
    for row in reader:
        predict_label = 0
        predict_possibility = 0

        total_possibility = 0
        
        test_result = {}
        
        test_result['id'] = row['id']
        if dataset == 'split':
            test_result['location'] = row['location']
        test_result['image'] = row['image']
        test_result['pole'] = row['pole']
        test_result['label'] = row['label']
        
        for analysis_feature in analysis_features:
            test_result[analysis_feature] = filter(row[analysis_feature], hard_filters[analysis_feature])
            if test_result[analysis_feature] == 1:
                predict_label = 1
            
            total_possibility += test_result[analysis_feature]
            # input('Press ENTER to continue...')
        
        test_result['hard_pred_label'] = predict_label
        
        if predict_label == 1:
            test_result['hard_pred_pos'] = 1
        else:
            total_possibility += float(row['pole']) / 2
            test_result['hard_pred_pos'] = total_possibility / number_of_features
        
        with open(output_csv_path, 'a') as output_csv_file:
            writer = csv.writer(output_csv_file)
            if dataset == 'split':
                writer.writerow([ test_result['id'], test_result['location'], test_result['image'], test_result['size'], test_result['pole'], test_result['mean'], test_result['stddev'], test_result['b_mean'], test_result['g_mean'], test_result['r_mean'], test_result['b_stddev'], test_result['g_stddev'], test_result['r_stddev'], test_result['square'], test_result['ratiowh'], test_result['ratioarea'], test_result['approxlen'], test_result['numangle'], test_result['numangle90'], test_result['numangle70'], test_result['label'], test_result['hard_pred_label'], test_result['hard_pred_pos']])
            if dataset == 'non-split':
                writer.writerow([ test_result['id'], test_result['image'], test_result['size'], test_result['pole'], test_result['mean'], test_result['stddev'], test_result['b_mean'], test_result['g_mean'], test_result['r_mean'], test_result['b_stddev'], test_result['g_stddev'], test_result['r_stddev'], test_result['square'], test_result['ratiowh'], test_result['ratioarea'], test_result['approxlen'], test_result['numangle'], test_result['numangle90'], test_result['numangle70'], test_result['label'], test_result['hard_pred_label'], test_result['hard_pred_pos']])
        output_csv_file.close()

test_data_csv_file.close()

print('finished')


