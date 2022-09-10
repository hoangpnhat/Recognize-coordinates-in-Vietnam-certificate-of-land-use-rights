# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import sys
import matplotlib.pyplot as plt
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import tensorflow as tf
import pandas as pd

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.io.reader import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page

DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large", "linknet_resnet18_rotation"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "master", "sar_resnet31"]

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
def sort_by_heigh(coordinates):
    n = len(coordinates)
    # Traverse through all array elements
    for i in range(n):
        swapped = False
        # Last i elements are already
        #  in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to
            # n-i-1. Swap if the element
            # found is greater than the
            # next element
            if coordinates[j]['geometry'][0][1] > coordinates[j+1]['geometry'][0][1] :
                coordinates[j], coordinates[j+1] = coordinates[j+1], coordinates[j]
                swapped = True
        # IF no two elements were swapped
        # by inner loop, then break
        if swapped == False:
            break
def find_coordinates(json_export):
    X_coordinates = []
    Y_coordinates = []
    # for page in json_export['page_idx']:
    for block in json_export['blocks']:
        for line in block['lines']:
            for word in line['words']:
                if isfloat(word['value']):
                    string = str(word['value'])
                    if float(string)> 1100000 and float(string) < 1500000:
                        # count+=1
                        X_coordinates.append(word)
                        print(word['value'])
                    if float(string)> 100000 and float(string) < 700000:
                        # count+=1
                        Y_coordinates.append(word)
                        print(word['value'])
    return X_coordinates,Y_coordinates

def pair_coordinates(X_coordinates,Y_coordinates):
    epsilon =0.003
    table =[]
    X_coordinates_not_pair = X_coordinates.copy()
    Y_coordinates_not_pair = Y_coordinates.copy()
    for X_cor in X_coordinates:
        y1 = X_cor['geometry'][0][1]
        for Y_cor in Y_coordinates:
            y2= Y_cor['geometry'][0][1]
            minus = abs(y2 - y1)
            # print(X_cor,Y_cor, minus)
            if minus <= epsilon:
                # if float(X_cor['value']) < float(Y_cor['value']):
                #   X_cor, Y_cor = Y_cor, X_cor
                table.append([X_cor['value'],Y_cor['value']])
                # print('remove')
                X_coordinates_not_pair.remove(X_cor)
                Y_coordinates_not_pair.remove(Y_cor)
                continue
    return table,X_coordinates_not_pair,Y_coordinates_not_pair
def main():

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("docTR: Document Text Recognition")
    # For newline
    st.write('\n')
    # Instructions
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    cols[3].subheader("Page reconstitution")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        cols[0].image(doc[page_idx])

    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", DET_ARCHS)
    reco_arch = st.sidebar.selectbox("Text recognition model", RECO_ARCHS)

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Analyze page"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner('Loading model...'):
                predictor = ocr_predictor(
                    det_arch, reco_arch, pretrained=True,
                    assume_straight_pages=(det_arch != "linknet_resnet18_rotation")
                )

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.det_predictor.pre_processor([doc[page_idx]])
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
                seg_map = cv2.resize(seg_map.numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([doc[page_idx]])
                fig = visualize_page(out.pages[0].export(), doc[page_idx], interactive=False)
                cols[2].pyplot(fig)

                # Page reconsitution under input page
                page_export = out.pages[0].export()
                if det_arch != "linknet_resnet18_rotation":
                    img = out.pages[0].synthesize()
                    cols[3].image(img, clamp=True)

                # Display JSON
                st.markdown("\nHere are your analysis results in JSON format:")
                st.json(page_export)
                #Display Table coordinates
                st.markdown("\nHere are your coordinates:")
                X_coordinates,Y_coordinates=find_coordinates(page_export)
                sort_by_heigh(X_coordinates)
                sort_by_heigh(Y_coordinates)
                table,X_coordinates_not_pair,Y_coordinates_not_pair = pair_coordinates(X_coordinates=X_coordinates,Y_coordinates=Y_coordinates)
                st.write("Table of Coordinates: ",pd.DataFrame(table,columns =['X', 'Y']))


if __name__ == '__main__':
    main()
