import requests
from io import BytesIO
from datetime import date, timedelta, datetime
from PIL import Image, UnidentifiedImageError
import pandas as pd
import matplotlib.image as mpimg
import h5py
import numpy as np

def iteration_15min(start, finish):
    ## Generateur de (an, mois, jour, heure, minute)
     while finish > start:
        start = start + timedelta(minutes=15)
        yield (start.strftime("%Y"),
               start.strftime("%m"),
               start.strftime("%d"),
               start.strftime("%H"),
               start.strftime("%M")
               )


def scrapping_images (start, finish) :
    """Scrape images radar en ligne toutes les 15 min
    entre deux dates donnees sous forme de datetime.datetime
    Sauvegarde les dates pour lesquelles la page n'existe pas.  """
    missing_times = []
    for (an, mois, jour, heure, minute) in iteration_15min(start, finish):
        ## url scrapping :
        url = (f"https://static.infoclimat.net/cartes/compo/{an}/{mois}/{jour}"
            f"/color_{jour}{heure}{minute}.jpg")
        date_save = f'{an}_{mois}_{jour}_{heure}{minute}'

        try :
            open_save_data(url, date_save)

        except UnidentifiedImageError :
            print (date_save, ' --> Missing data')
            missing_times.append(date_save)
    ## Save missing data list :
    missing_data_name = f'missing_datetimes_{start.strftime("%Y")}\
        {start.strftime("%m")}{start.strftime("%d")}_to_{finish.strftime("%Y")}\
            {finish.strftime("%m")}{finish.strftime("%d")}'
    pd.DataFrame(missing_times).to_pickle(missing_data_name)
    print(missing_times)



def open_save_data(url, date_save):
    ## Ouvre l'image pointee par url
    ## Enregistre l'image avec l'extention date_save

    print(url, date_save)

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    img.save( f"/mnt/d/data_programmation/raw_images/radar{date_save}.png")
    pass

def open_data(date_save):
    print('Open '+date_save)
    img = mpimg.imread(f"/mnt/d/data_programmation/raw_images/radar{date_save}.png")
    return img

def save_data(img, folder, date_save):
    ## Save as image :
    print('save image')
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save( f"/mnt/d/data_programmation/{folder}/radar_preproc{date_save}.png")
    pass

# def save_data_hdf(img, date_save):
#     hf = h5py.File(f'nuages_gris_{date_save}.h5', 'w')
#     hf.create_dataset('dataset_1', data=img)
#     hf.close()


if __name__ == '__main__' :

    start = datetime(2014, 5, 1, 1)
    finish = datetime(2014, 8, 30, 14, 30)

    scrapping_images (start, finish)


    # start = datetime(2017, 5, 1, 1)
    # finish = datetime(2017, 9, 30, 14, 30)
