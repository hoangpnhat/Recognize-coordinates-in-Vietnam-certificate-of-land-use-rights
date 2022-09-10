# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List

from app.schemas import OCROut
from app.vision import predictor
from fastapi import APIRouter, File, UploadFile, status
from doctr.io.reader import DocumentFile
import json

from doctr.io import decode_img_as_tensor

router = APIRouter()

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

# @router.post("/", response_model=List[OCROut], status_code=status.HTTP_200_OK, summary="Perform OCR")
@router.post("/", status_code=status.HTTP_200_OK, summary="Perform OCR")

async def perform_ocr(file: UploadFile = File(...)):
    """Runs docTR OCR model to analyze the input image"""
    print(file.filename)
    img = DocumentFile.from_pdf(file.file.read())
    out = predictor([img[0]])
    page_export = out.pages[0].export()
    X_coordinates,Y_coordinates=find_coordinates(page_export)
    sort_by_heigh(X_coordinates)
    sort_by_heigh(Y_coordinates)
    table,X_coordinates_not_pair,Y_coordinates_not_pair = pair_coordinates(X_coordinates=X_coordinates,Y_coordinates=Y_coordinates)
    jsonString = json.dumps(table)
    print(jsonString)

    # return [OCROut(box=(*word.geometry[0], *word.geometry[1]), value=word.value)
    #         for word in out.pages[0].blocks[0].lines[0].words]
    return jsonString
