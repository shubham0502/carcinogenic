import pytesseract as py
import cv2
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import json
from flask import Flask, request, render_template, jsonify, make_response
from PIL import Image as ima

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
def clean(text):
    return re.sub('[^A-Za-z0-9" "]+', '', text)
    app = Flask(__name__)
# @app.route('/send', methods=['POST', 'GET'])
# def send():
#     if request.method == 'POST':
#         postdata = request.form
#         file_name = postdata['filename']
#         print("file name: ====================== {}".format(file_name))
#         file = str(file_name)
#         path = ".\\static\\" + file
#         return render_template('/send.html')
image = cv2.imread('../input/testimage/test1.jpg')
image = cv2.resize(image, (400, 400))
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
df = py.image_to_data(gray, output_type = 'data.frame')
df = df[df['conf'] != -1] 
df['text'] = df['text'].apply(lambda x: x.strip())
df = df[df['text']!=""]
df['text'] = df['text'].apply(lambda x: x.lower())
df = df[df.text.str.len() > 3]
df.text = df.text.replace('\*','', regex = True)
df.text = df.text.apply(clean)
#formatting of dataset file
# d = pd.read_csv('../input/chemicalcsv2/chemicalsimp2.csv')
# d = d.drop(['Unnamed: 2','ID_v5'], axis = 1)
# d = d.dropna(axis = 0)
# d["Chemical Name"] = d["Chemical Name"].apply(clean)
# d["Chemical Name"] = d["Chemical Name"].apply(lambda x: x.lower())
# d = d.append({'Chemical Name': 'acesulfame potassium'}, ignore_index = True)
# d.to_csv('chemicalcarc1', index = False)
c = pd.read_csv('../input/chemcarc1/chemicalcarc1')
c = c.append({'Chemical Name': 'neotame'}, ignore_index = True)
chem = c["Chemical Name"].tolist()
shifted_text_col = list(df['text'].iloc[1:])
shifted_text_col.append("")
df['text_2row'] = df['text'] + " " + shifted_text_col
i = 0
chemical = []
while i < len(df):   
    if df['text_2row'].iloc[i] in chem:
        chemical.append(df.text_2row.iloc[i])
        i += 1
    elif df['text'].iloc[i] in chem:
        chemical.append(df.text.iloc[i])
        i += 1
    else:
        i += 1
if chemical == []:
    diction = {"harmful ingredients" : "none"}
else:
    diction = {"harmful ingredients" : chemical}
jason_object = json.dumps(diction, indent = 3)
with open("output.json", "w") as outfile:
    outfile.write(jason_object)