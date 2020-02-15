import math
import csv


def metric(panel_panel, panel_nopanel,nopanel_panel,nopanel_nopanel):
    metric = {}
    TP = panel_panel
    FN = panel_nopanel
    FP = nopanel_panel
    TN = nopanel_nopanel
    ACCURACY = float((TP + TN)/(TP + FP + FN + TN))
    PRECISION = float(TP/(TP + FP))
    RECALL = float(TP/(TP + FN))
    F1 = float(2*PRECISION*RECALL/(PRECISION + RECALL))
    MCC = float((TP * TN - FP * FN)/ math.sqrt((TP + FP) * (FN + TN) * (FP + TN) * (TP + FN)))
    SPECIFICITY = float(TN/(TN + FP))
    metric['TP'] = float(TP/(TP + FN))
    metric['FN']  = float(FN /(TP + FN))
    metric['TN'] = float(TN /(TN + FP))
    metric['FP']  =float(FP /(TN + FP))
    metric['ACCURACY'] = ACCURACY
    metric['PRECISION'] =PRECISION
    metric['RECALL']= RECALL
    metric['F1'] = F1
    metric['MCC'] = MCC
    metric['SPECIFICITY'] = SPECIFICITY
    metric['description'] = 'vgg pure nosplit'
    print(metric)
    csvpath = './solarpanel/svm/metric.csv'
    with open(csvpath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([metric['description'],metric['TP'],metric['FN'],metric['TN'],metric['FP'],metric['ACCURACY'],metric['PRECISION'],metric['RECALL'],metric['F1'],metric['MCC'],metric['SPECIFICITY']])
    csvfile.close()

#  call function by the number panel_panel, panel_nopanel, nopanel_panel,nopanel_nopanel
# for exmaple
metric(603,276,8671,15396)