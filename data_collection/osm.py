# this file is used to generate the url to download images from google static map,
# for every house, we store the url to download the house original images and house mask.
# the input is osm file, output is json which store the ulr and csv document which store the id and location of houses.
import csv
import datetime
import glob as gb
import json
import os
import xml.dom.minidom
import xml.dom.minidom

# used to calculate the download images time
start = datetime.datetime.now()
# osm_path which is the osm file location
osm_path = gb.glob("/*.osm")
for osm in osm_path:
    dom = xml.dom.minidom.parse(osm)
    num = osm.split("/")[-1]
    num = os.path.splitext(num)[0]
    # dom = xml.dom.minidom.parse('./0.osm')
    root = dom.documentElement
    nodelist = root.getElementsByTagName('node')
    waylist = root.getElementsByTagName('way')
    node_dic = {}

    url_prefix1 = 'https://maps.googleapis.com/maps/api/staticmap?zoom=20&size=400x400&scale=4&maptype=hybrid&path=color:0xff0000ff%7Cweight:5%7Cfillcolor:0xff0000ff'
    url_prefix2 = 'https://maps.googleapis.com/maps/api/staticmap?zoom=20&size=400x400&scale=4&maptype=hybrid&path=color:0x00000000%7Cweight:5%7Cfillcolor:0x00000000'
    url_suffix = '&key=AIzaSyA7UVGBz0YP8OPQnQ9Suz69_u1TUSDukt8'

    for node in nodelist:
        node_id = node.getAttribute('id')
        node_lat = float(node.getAttribute('lat'))
        node_lon = float(node.getAttribute('lon'))
        node_dic[node_id] = (node_lat, node_lon)
    url = []
    location = {}
    csv_lat = 0
    csv_lon = 0
    num_img = 0
    # json used to store the url of images downloading
    with open(os.path.join('./10house/house1/', format(str(num)) + '.json'), 'w') as json_file:
        for way in waylist:
            taglist = way.getElementsByTagName('tag')
            build_flag = False
            for tag in taglist:
                # choose the attribute to be building,
                if tag.getAttribute('k') == 'building':
                    build_flag = True
            if build_flag:
                ndlist = way.getElementsByTagName('nd')
                s = ""
                for nd in ndlist:
                    nd_id = nd.getAttribute('ref')
                    if nd_id in node_dic:
                        node_lat = node_dic[nd_id][0]
                        node_lon = node_dic[nd_id][1]
                        g = nd_id
                        csv_lat = node_dic[nd_id][0]
                        csv_lon = node_dic[nd_id][1]
                        print(g)
                        s += '%7C' + str(node_lat) + '%2C' + str(node_lon)
                # secret = 'pSRLFZI7ujDivoNjR-Vz7GR6F4Q='
                url1 = url_prefix1 + s + url_suffix
                # url1 = sign_url(url1, secret)
                url2 = url_prefix2 + s + url_suffix
                # url2 = sign_url(url2, secret)
                test_dict = {"id": g, "mask": url1, "image": url2}
                url.append(test_dict)
                location[g] = str(csv_lat) + ',' + str(csv_lon)
                num_img = num_img + 1
        json_str = json.dumps(url)
        json_file.write(json_str)
        json_file.close()
        # csv document used to store the house id and location( latitude and longtitude)
        csv_path = "./10house/house1/house1.csv"
        with open(csv_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in location.items():
                writer.writerow([key, value])
        csv_file.close()
end = datetime.datetime.now()
print(end - start)
