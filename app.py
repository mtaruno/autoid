import streamlit as st
from PIL import Image
import pytesseract as pt
from textblob import TextBlob
import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt

# """
#         --psm N
#         Set Tesseract to only run a subset of layout analysis and assume a certain form of image. The options for N are:

#         0 = Orientation and script detection (OSD) only.
#         1 = Automatic page segmentation with OSD.
#         2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
#         3 = Fully automatic page segmentation, but no OSD. (Default)
#         4 = Assume a single column of text of variable sizes.
#         5 = Assume a single uniform block of vertically aligned text.
#         6 = Assume a single uniform block of text.
#         7 = Treat the image as a single text line.
#         8 = Treat the image as a single word.
#         9 = Treat the image as a single word in a circle.
#         10 = Treat the image as a single character.
#         11 = Sparse text. Find as much text as possible in no particular order.
#         12 = Sparse text with OSD.
#         13 = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

#         --oem N
#         Specify OCR Engine mode. The options for N are:

#         0 = Original Tesseract only.
#         1 = Neural nets LSTM only.
#         2 = Tesseract + LSTM.
#         3 = Default, based on what is available.
#     """
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.sidebar.markdown("""<img style=' align:center; margin-left: auto;margin-right: auto;width: auto;' src="C:/Users/Shendy/Desktop/TEATH-master/static/Icon/topan_logo.png">""",unsafe_allow_html=True)
# display: block;
st.sidebar.markdown("""<style>body {background-color: #2C3454;color:white;}</style><body></body>""", unsafe_allow_html=True)
st.markdown("""<h2 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>KTP OCR</h2>""",unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: center;color: #2C3454;margin-top:30px;margin-bottom:-20px;'>Select Image</h2>", unsafe_allow_html=True)

image_file = st.sidebar.file_uploader("", type=["jpg","png","jpeg"])


def extract(img):
    # slide=st.sidebar.slider("Select Page Segmentation Mode",1,14)
    # conf=f"-l ind --oem 3 --psm 6 {slide}"
    st.set_option('deprecation.showfileUploaderEncoding', False)
    conf = r'--dpi 300 --oem 1 --psm 6'
    
    # image = cv.resize(img, (50 * 16, 500))
    # img2 = cv.resize(img, None, fx=0.5, fy=0.5)
    # dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    # img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) 
    # id_number = return_id_number(image, img_gray)

    # fill = cv.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
    # th, threshed = cv.threshold(fill, 127, 255, cv.THRESH_TRUNC)

    # Remove noise using Gaussian Blur
    blur = cv.GaussianBlur(img, (5, 5), 0)

    # Further noise removal (Morphology)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel, iterations=1)

    # Segmentation
    gray = cv.cvtColor(opening, cv.COLOR_RGB2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    # ret, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # TRUNC
    # th1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    # th, threshed = cv.threshold(fill, 127, 255, cv.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # close = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    # result = 255 - close    
    # salvaged_text = pytesseract.image_to_string((threshed), config=custom_config, lang="ind")
    # Morph open to remove noise
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # kernel1 = np.ones((5,5),np.uint8)

    # opening = cv.morphologyEx(threshed, cv.MORPH_OPEN, kernel, iterations=1)
    text = pt.image_to_string(thresh, config = conf, lang='ind')
    
    # for word in text.split("\n"):
    #     if "”—" in word:
    #         word = word.replace("”—", ":")


    if(text!=""):
        st.markdown("<h1 style='color:yellow;'>Extracted Text</h1>", unsafe_allow_html=True)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        
        # ktp = {'Provinsi':text.splitlines()[0],
        #        'Kabupaten / Kota' : text.splitlines()[1],
        #        'NIK' : text.splitlines()[2],
        #        'Nama' : text.splitlines()[3],
        #        'Tempat/Tgl Lahir' : text.splitlines()[4],
        #        'Jenis Kelamin' : text.splitlines()[5],
        #        'Alamat' : text.splitlines()[6],
        #        'RT/RW' : text.splitlines()[7],
        #        'Kel/Desa' : text.splitlines()[8],
        #        'Kecamatan' : text.splitlines()[9],
        #        'Agama' : text.splitlines()[10],
        #        'Status Perkawinan' : text.splitlines()[11],
        #        'Pekerjaan' : text.splitlines()[12],
        #        'Kewarganegaraan' : text.splitlines()[13],
        #        'Berlaku Hingga' : text.splitlines()[14],
        #        }
        # print(pytesseract.image_to_string(photo))

        slot1 = st.empty()
        # slot1.markdown(f"{text}", unsafe_allow_html=True)
        for word in text.split("\n"):
            if "”—" in word:
                word = word.replace("”—", ":")
        slot1.markdown(f"{text}", unsafe_allow_html=True)

if image_file is not None:
    st.markdown("<h1 style='color:yellow;'>Uploaded Image</h1>", unsafe_allow_html=True)
    st.image(image_file, width = 400)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    # radio = st.sidebar.radio("Select Action",('Text Extraction',))
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    # if(radio == "Text Extraction"):
        # extract(img)
    extract(img)  