from datetime import date
from PIL import Image
import requests
from io import BytesIO
from datetime import date, timedelta, datetime
from PIL import Image, UnidentifiedImageError
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import csv
from copy import copy
import cv2 as cv
import data




carte = mpimg.imread("carte_test.png")
cols = {1:[209,251,252],    ## Couleurs de l'echelle d'intensite de pluie (mm/h)
       2:[97,219,241],
       3:[76,147,240],
       4:[23,38,192],
       5:[0,141,3],
       6:[12,255,0],
       7:[255,249,0],
       8:[255,145,0],
       9:[232,0,0],
       10:[232,0,230],
       11:[255,175,254]}



def preproc_from2 (start, finish):
    for (an, mois, jour, heure, minute) in data.iteration_15min(start, finish):
        print (an, mois, jour, heure, minute)
        date_save = f'{an}_{mois}_{jour}_{heure}{minute}'
        try :
            img = data.open_data(date_save)
            X_preproc, Y_preproc = preproc_data(img)
            folder = "X_preproc"
            data.save_data(X_preproc, folder, date_save)
            folder = "Y_preproc"
            data.save_data(Y_preproc, folder, date_save)
        except (SyntaxError, OSError) as e :
            print (date_save, '--> Fichier vide !')
    pass



def preproc_data(img):
    """ Passe par ttes les fct de preproc pour une image
    """
    ## Retirer la carte :
    img = retirer_carte_fond (img, carte)
    img = retirer_txt (img)
    ## Zoom sur lla zone d'interet
    img_gray = colors2grays (img)
    img_gray = lissage_image(img_gray)
    img_zoomX = crop_image (img_gray, 'France_Nord')
    img_zoomY = crop_image (img_gray, 'IDF')
    return img_zoomX,img_zoomY


def retirer_carte_fond (img, carte):
    # Calcul de la diff entre l'image radar et la carte
    im_diff= np.asarray(img)- np.asarray(carte)

    # Restitution de leur vrai valeur aux pixels non proches de 0
    M =np.ones((img.shape[0], img.shape[1], 3))
    M[im_diff<0.1]=0
    img_radar = M*img

    return img_radar

def retirer_txt (img):
    """Mettre zone de txt en haut a gauche de l'image a 0"""
    img[0:100,0:200,:] = 0
    return img


def colors2grays (img):
    """Transforme les vraies couleurs en niveau de gris"""
    gray_image_3c = copy(img)
    if np.max(img) <= 1. :
        gray_image_3c = copy(img)*255

    gray_image_1c = copy(gray_image_3c[:,:,0])
    gray_image_1c[:,:] = 0

    tolerance = 65  ## l

    for i in  range(1,12):
        col_lo=np.array([x-tolerance for x in cols[i]])
        col_hi=np.array([x+tolerance for x in cols[i]])

        mask=cv.inRange(gray_image_3c,col_lo,col_hi)
        gray_image_1c[mask>0]=i/11

    return gray_image_1c


def crop_image (img, zone) :
    ## Zoom sur la zone d'interet
    if zone == 'France_Nord' :
        limite = [30,450,100,750]    ## Limites : [H_min, H_max, L_min, L_max]
    elif zone == 'IDF':
        limite = [190,265,400,510]    ## Limites : [H_min, H_max, L_min, L_max]
    else :
        print("Unknown area : Area should be in ('France_Nord', 'IDF')")
    img_zoom = img[limite[0]:limite[1],limite[2]:limite[3]]
    return img_zoom

def lissage_image(img):
    img = img.astype('float32')
    img = cv.medianBlur(img, 5)
    return img


if __name__ == '__main__' :

    start = datetime(2017, 5, 1, 1)
    finish = datetime(2017, 9, 30, 14, 30)

    #for (an, mois, jour, heure, minute) in data.iteration_15min(start, finish):
    print(preproc_from2 (start, finish))



    # toto =(time_iteration(start_date, end_date))
    # pd.DataFrame(toto).to_pickle('unexiting_files')
    # print(toto)
