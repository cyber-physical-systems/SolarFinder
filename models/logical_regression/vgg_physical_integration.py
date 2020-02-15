import csv
physical_feature_path = './location17/contour_all.csv'
vgg_predict_path = './location17/vgg_predict.csv'
lr_path =  './location17/lr.csv'

with open(lr_path, 'a') as csvfile:
    myFields = ['id', 'location', 'image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label',]
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()

with open(physical_feature_path, newline='') as phyfile:
    contour = {}
    reader = csv.DictReader(phyfile)
    for phy in reader:
        contour = phy
        
        with open(vgg_predict_path, newline='') as vggfile:
            reader = csv.DictReader(vggfile)
            for vgg in reader:
                if (vgg['id'] ==contour['id']):    
                    contour['prediction'] = vgg['prediction']
                    contour['prediction_class'] = vgg['prediction_class']       
        vggfile.close()
        with open(lr_path, 'a') as lrfile:
            writer = csv.writer(lrfile)
            writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['prediction'],contour['prediction_class'],contour['label']])
        lrfile.close()
phyfile.close()

        
        
        
        
        
        
        
        




