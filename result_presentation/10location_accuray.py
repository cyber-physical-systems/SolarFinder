import pandas as pd
<<<<<<< HEAD
df = pd.read_csv("./data/feature_test_all_vgg_svm_linear.csv")

for i in range(1,11):
    data = pd.read_csv('./finaltest/data/10locations/location' + str(i) + '.csv')
=======
df = pd.read_csv("")
for i in range(1,11):
    data = pd.read_csv('' + str(i) + '.csv')
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62
    y_predict = data.linear_nosplit_class
    y_test =data.label
    print(confusion_matrix(y_test, y_predict))
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict, labels=[0,1]).ravel()
<<<<<<< HEAD
    with open('./finaltest/data/10locations/10location.csv', 'a') as csvfile:
=======
    with open('', 'a') as csvfile:
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62
        writer = csv.writer(csvfile)
        writer.writerow(['location'+str(i),tn,fp,fn,tp])
    csvfile.close()
