import math
import csv


def metric(des,panel_panel, panel_nopanel,nopanel_panel,nopanel_nopanel):
    metric = {}
    TP = int(panel_panel)
    FN = int(panel_nopanel)
    FP = int(nopanel_panel)
    TN = int(nopanel_nopanel)
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
    metric['description'] = des
    print(metric)
    csvpath = './result1.csv'
    with open(csvpath,  'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([metric['description'],metric['TP'],metric['FN'],metric['TN'],metric['FP'],metric['ACCURACY'],metric['MCC'],metric['F1'],metric['SPECIFICITY'],metric['PRECISION'],metric['RECALL']])
    csvfile.close()

def main():
    with open('./result.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metric(row['des'],row['TP'],row['FN'],row['FP'],row['TN'])
    csvfile.close()
main()
