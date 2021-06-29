from unittest import result

from django.shortcuts import render
from django.urls import reverse
from django.template import RequestContext
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import CVForm, OjDtForm, objCtForm
from .models import CV


from django.core.files.storage import default_storage
from .CVFun import cannyFilter, objDetect, faceDetect
from .ObjectCounting import counting
from django.conf import settings
import numpy as np
import cv2
import PIL
from PIL import Image
import urllib
from django.http import JsonResponse
import json
import base64
import os
import os.path
from os import path

def readImage(img):
    # img = img.image
    img = img.read()
    img = np.asarray(bytearray(img), dtype="uint8")
    imgResult = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return imgResult

def filter(request):
    
    form = CVForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        obj = request.FILES['image']

        img = readImage(obj)
        
        canny = cannyFilter(img, 100)

        outUrl = os.path.join(settings.MEDIA_ROOT / 'images', 'ouput.jpg')
        cv2.imwrite(outUrl, canny)
        if path.exists(settings.MEDIA_ROOT / 'images/input.jpg'):
            os.remove(settings.MEDIA_ROOT / 'images/input.jpg')
        obj.name = 'input.jpg'
        inImg = CV(image=obj)
        inImg.save()
        out = CV(imageResult=outUrl)

        inUrl = inImg.image.url
        outUrl = out.imageResult.url
        print("in: " + inUrl + "\n out:" + outUrl)
        data = True
        return render(request, 'index.html', { 'data': data, 'inUrl': inUrl, 'outUrl': outUrl})
    else:
        form = CVForm()

    return render(request, 'index.html', )


def objDt(request):
    if request.method == 'POST':
        form = OjDtForm(request.POST, request.FILES)
        if form.is_valid():
            img1 = request.FILES['image1']
            img2 = request.FILES['image2']
            image1 = readImage(img1)
            image2 = readImage(img2)
            obj = objDetect(image1, image2)

            outUrl = os.path.join(settings.MEDIA_ROOT / 'images', 'ouput.jpg')
            cv2.imwrite(outUrl, obj)
            img1Name = 'image1.jpg'
            img2Name = 'image2.jpg'
            upload(img1, img1Name)
            upload(img2, img2Name)
            data = True
            return render(request, 'objDetect.html', {'data': data})

    return render(request, 'objDetect.html')


def objCounting(request):
    if request.method == 'POST':
        form = objCtForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            img = readImage(image)
            choice = request.POST['noise']
            print("xxx  "+choice+"\n")
            choice = int(choice)
            count = counting(img, choice)

            #save image
            if path.exists(settings.MEDIA_ROOT / 'images/input.jpg'):
                os.remove(settings.MEDIA_ROOT / 'images/input.jpg')
            inputName = 'input.jpg'
            inUrl = upload(image, inputName)

            outUrl = os.path.join(settings.MEDIA_ROOT / 'images', 'ouput.jpg')
            step1 = os.path.join(settings.MEDIA_ROOT / 'images', 'step1.jpg')
            step2 = os.path.join(settings.MEDIA_ROOT / 'images', 'step2.jpg')
            step3 = os.path.join(settings.MEDIA_ROOT / 'images', 'step3.jpg')
            # step4 = os.path.join(settings.MEDIA_ROOT / 'images', 'step4.jpg')
            # step5 = os.path.join(settings.MEDIA_ROOT / 'images', 'step5.jpg')

            cv2.imwrite(outUrl, count[0])
            cv2.imwrite(step1, count[1])
            cv2.imwrite(step2, count[2])
            cv2.imwrite(step3, count[3])
            # cv2.imwrite(step4, count[5])
            # cv2.imwrite(step5, count[6])
            data = True
            return render(request, 'objectCounting.html', {'data': data,"count": count[4], "choice": choice})

    return render(request, 'objectCounting.html')


def faceDt(request):
    form = CVForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        obj = request.FILES['image']

        img = readImage(obj)

        face = faceDetect(img)
        # face = readImage(face)

        # face = img
        # face = cannyFilter(img)
        outUrl = os.path.join(settings.MEDIA_ROOT / 'images', 'ouput.jpg')
        cv2.imwrite(outUrl, face)
        if path.exists(settings.MEDIA_ROOT / 'images/input.jpg'):
            os.remove(settings.MEDIA_ROOT / 'images/input.jpg')
        obj.name = 'input.jpg'
        inImg = CV(image=obj)
        inImg.save()
        out = CV(imageResult=outUrl)

        inUrl = inImg.image.url
        outUrl = out.imageResult.url
        data = True
        return render(request, 'faceDetect.html', {'data': data, 'inUrl': inUrl, 'outUrl': outUrl})
    else:
        form = CVForm()

    return render(request, 'faceDetect.html', )


def upload(f, nameFile):

    file = open(os.path.join(settings.MEDIA_ROOT / 'images', nameFile), 'wb+')
    for chunk in f.chunks():
        file.write(chunk)
    return settings.MEDIA_ROOT / 'images' / nameFile
