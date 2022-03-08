import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K
import os

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox


# Collect all filenames
def collect_filenames():
    """
    This function returns a list of all the filenames in a folder
    The returned list is sorted (ascending)
    """
    filenames = []
    for filepath in os.listdir('/content/drive/MyDrive/Projet_MeteoStat/Data/X_preproc/'):
        filenames.append(filepath)
    return sorted(filenames)

# Panda DF Creation
def create_df_all_images(sorted_filenames):
    """
    Input : List
    Returns : DataFrame

    This function returns a dataframe with one row for each filename from input variable
    Row order is the same as input list order
    The DF will have Name, NameYear, Month, Day, Time and FileFormat
    """
    series_name = pd.Series(sorted_filenames)
    df_all = pd.DataFrame(series_name, columns=["Filename"])
    df_all[['Name','NameYear','Month', 'Day', 'Time']] = df_all['Filename'].str.split('_',expand=True)
    df_all[['Time', 'FileFormat']] = df_all['Time'].str.split('.', expand=True)
    # df_all.groupby(['Month', 'Day']).count()
    return df_all

# Split DataFrame into train and test
def trainer_test_split(df, frac=0.2):

    """
    Input : DataFrame and optional frac of how much percentage should be test set
            frac default value = 0.2
    Returns : DataFrame train and DataFrame test

    The split works so that the test rows are taken from the tail of the dataframe
    """
    test = df.tail(int(len(df.index)*frac))
    train = df.drop(index=test.index)

    return train, test


def generate_one_set_names(first_index,df):
    """
    Input : index value to start the set, and dataframe
    Returns : One set of 20 sorted filenames ordered from the index value and forward

    The split works so that the test rows are taken from the tail of the dataframe
    """

    sorted_filenames = []
    for i in range(first_index,first_index+20):
      sorted_filenames.append(pd.Series(df['Filename'])[i])

    return sorted_filenames

def generate_all_set_names(number_of_sets,df):
    """
    Input : Number of sets to generate (INT) and Dataframe
    Returns : a long list of all filenames ordered set by set

    How it works, random variable that selects index in DF, and sends the index to
    generate_one_set_names function. Thereafter it keeps the returned list in a
    large list
    """


    all_filenames = []
    min_index = min(list(df.index))
    max_index = max(list(df.index))-20
    for i in range(number_of_sets):
        num = random.randint(min_index,max_index)
        tmp = generate_one_set_names(num, df)
        all_filenames.extend(tmp)
    return all_filenames

def create_sets(sorted_filenames):

    """
    Input : A list of sorted filenames
    Returns : A numpy array of all the images corresponding to the filename
              Array output shape example: (30,20,95, 120) corresponding to 30 sets
              of 20 images with 95 x 120 pixels

    Currently the function generates gray scale out of colors and the folder where
    the images are found is hardcoded in the function
    """

    instances = []

    temporary_list = []

    # Load in the images
    counter = 0

    for name in sorted_filenames:

        gray_image = cv2.imread('/content/drive/MyDrive/Projet_MeteoStat/Data/X_preproc/{}'.format(name))


        gray_image = gray_image[::5,::5,:]
        gray_image = gray_image.dot([0.07, 0.72, 0.21])
        # temporary_list = np.append(temporary_list,gray_image)
        temporary_list.append(gray_image)

        if counter>=19:
            # print(name)
            # instances = np.append(instances, temporary_list,axis=1)
            instances.append(temporary_list)
            counter=-1
            temporary_list = []
            # temporary_list = np.array(temporary_list)
        counter+=1


    return np.array(instances)


def arrange_training_data(input_array):

    """
    Input : numpy array
    Returns : train array and validation array, values has been divided by 255

    Train validation split is 80/20
    """

    # Swap the axes representing the number of frames and number of data samples.
    dataset = np.expand_dims(input_array, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.8 * dataset.shape[0])]
    val_index = indexes[int(0.8 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255
    return train_dataset,val_dataset



def create_shifted_frames(data):
    """
    Note: function name remains but it does not shift frames anymore, it just takes each
    set of 20 images and puts ten first as X and ten last as y

    Input : A dataset of sets of 20 images
    Returns : X and y for each set - qty is 10 for both

    """

    X = data[:,  : 10, :, :]
    y = data[:, 10 : 20, :, :]
    return X, y

def visualize_one_set(train_dataset):

    """
    Input : A dataset (coming from arrange_training_data)
    Returns : Visualizes all 20 images from a random set within the input

    """

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one random data example.
    data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    # Print information and display the figure.
    print(f"Displaying frames for example {data_choice}.")
    plt.show()



def create_model(X_train):

    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *X_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(16, 16),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(8, 8),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(4, 4),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ConvLSTM2D(
    #     filters=64,
    #     kernel_size=(1, 1),
    #     padding="same",
    #     return_sequences=True,
    #     activation="relu",
    # )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="linear", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["Accuracy"])

    model.summary()
    return model

def train_model(model, X_train, y_train,X_val,y_val,epochs=5,batch_size=5):

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


    # Fit the model to the training data.
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )
    return model

def visualize_one_pred(val_dataset, model):

    # Select a random example from the validation dataset.
    example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]


    # Predict a new set of 10 frames.
    for _ in range(10):
        # Extract the model's prediction and post-process it.
        new_prediction = model.predict(np.expand_dims(frames, axis=0))

        new_prediction = np.squeeze(new_prediction, axis=0)

        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame), axis=0)

    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    new_frames = frames[10:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Display the figure.
    plt.show()

def generate_gifs(val_dataset, model):

    # Select a few random examples from the dataset.
    examples = val_dataset[np.random.choice(range(len(val_dataset)), size=5)]

    # Iterate over the examples and predict the frames.
    predicted_videos = []
    for example in examples:
        # Pick the first/last ten frames from the example.
        frames = example[:10, ...]
        print("Example : ",example.shape) # Example shape (20, 84, 130, 1)
        print("Frames : ",frames.shape) # Frames shape :  (10, 84, 130, 1)
        original_frames = example[10:, ...]
        new_predictions = np.zeros(shape=(10, *frames[0].shape))

        # Predict a new set of 10 frames.
        for i in range(10):
            # Extract the model's prediction and post-process it.
            frames = example[: 10 + i + 1, ...]
            print("Frames input to model : ",frames.shape) # (11, 84, 130, 1)
            new_prediction = model.predict(np.expand_dims(frames, axis=0))
            print("Output model predict : ",new_prediction.shape) # Output model predict :  (1, 11, 84, 130, 1)
            new_prediction = np.squeeze(new_prediction, axis=0)
            print("Output after Squeeze : ",new_prediction.shape) # (11, 84, 130, 1)
            predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

            print("Output after expand dims : ",new_prediction.shape) # (11, 84, 130, 1)

            # AJ temporary white augmentation and black if below 0.25
            # predicted_frame = np.where(predicted_frame<0.001,0,predicted_frame*2)

            # Extend the set of prediction frames.
            new_predictions[i] = predicted_frame

        # Create and save GIFs for each of the ground truth/prediction images.
        for frame_set in [original_frames, new_predictions]:
            # Construct a GIF from the selected video frames.
            current_frames = np.squeeze(frame_set)
            current_frames = current_frames[..., np.newaxis] * np.ones(3)
            current_frames = (current_frames * 255).astype(np.uint8)
            current_frames = list(current_frames)

            # Construct a GIF from the frames.
            with io.BytesIO() as gif:
                imageio.mimsave(gif, current_frames, "GIF", fps=5)
                predicted_videos.append(gif.getvalue())

    # Display the videos.
    print(" Truth\tPrediction")
    for i in range(0, len(predicted_videos), 2):
        # Construct and display an `HBox` with the ground truth and prediction.
        box = HBox(
            [
                widgets.Image(value=predicted_videos[i]),
                widgets.Image(value=predicted_videos[i + 1]),
            ]
        )
        display(box)

#### LOSS FUNCTIONS AND METRICS ####

def mse_zoomIDF_Loss(y_true, y_pred):

    """Mse Loss function for IDF zone
    """
    y_true_sliced =y_true[:,:,20:30,50:70,:]
    y_pred_sliced =y_pred[:,:,20:30,50:70,:]

    return K.mean((y_true_sliced-y_pred_sliced)**2)


def correlationLoss(y_true,y_pred):

    """Correlation Loss function for IDF
    """

    y_true_sliced =y_true[:,:,20:30,50:70,:]
    y_pred_sliced =y_pred[:,:,20:30,50:70,:]

    x, y =y_true_sliced[:8,:,:,:], y_pred_sliced[:8,:,:,:]

    cov = tf.reduce_sum(tf.reduce_sum( (x ) * (y ), axis=2), axis=2)
    x_sum2 = tf.experimental.numpy.nanmean(tf.reduce_sum(tf.reduce_sum( (x**2 ) , axis=2), axis=2))
    y_sum2 = tf.experimental.numpy.nanmean(tf.reduce_sum(tf.reduce_sum( (y**2 ), axis=2), axis=2))

    epsilon = 0.000001

    corr = cov / (tf.sqrt(x_sum2 * y_sum2)+epsilon)

    return -corr


# Metrics class
class Metrics(keras.callbacks.Callback):

    def __init__(self, X_val, y_val):
        super().__init__()
        self.validation_data = [X_val, y_val]

    def on_train_begin(self, logs={}):
        self._data = []

    def calc_mse(self, img_1, img_2):
        return  np.mean((img_1-img_2)**2)

    def calc_corr(self, img1, img2):
        cm = np.corrcoef(img1.flat, img2.flat)
        return cm[0, 1]

    def metrics_to_baseline(self, y_true, y_pred):
        """ Calcule les metriques = mse, corr coef
        en pourcentage de la baseline, et en moyenne culumative
        pour chaque pas de temps (à t2, retourne moy(t2, t1)/moy(t2_baseline, t1_baseline)*100"""

        y_true =y_true[:,:,20:30,50:70,:] # Slicing the map over IDF
        y_pred =y_pred[:,:,20:30,50:70,:] # Slicing the map over IDF

        df = pd.read_csv("/content/drive/MyDrive/Projet_MeteoStat/code/Baselines.csv")
        mse_cumul_baseline = df['mse_baseline_cumul']
        corr_cumul_baseline = df['corr_baseline_cumul']

        list_mse, list_corr = {}, {}
        for i in range (y_pred.shape[1]):
            list_mse[i] = self.calc_mse(y_true[0,i,:,:,0], y_pred[0,i,:,:,0]).item()
            list_corr[i] = self.calc_corr(y_true[0,i,:,:,0], y_pred[0,i,:,:,0]).item()

        mse_cumul, corr_cumul = [], []
        dum = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        for i in range(1,len(list_mse)):
            indexs = dum[:i]
            mse_cumul.append( np.nanmean([list_mse[x] for x in indexs]) )
            corr_cumul.append( np.nanmean([list_corr[x] for x in indexs]) )

        result_corr = [0,0,0,0,0,0,0,0,0,0]
        result_mse  = [0,0,0,0,0,0,0,0,0,0]
        epsilon = .000000001
        for i in range(len(mse_cumul)) :
            result_mse[i] = 100* mse_cumul[i] / (mse_cumul_baseline[i]+epsilon)
            result_corr[i] = 100* corr_cumul[i] / (corr_cumul_baseline[i]+epsilon)

        self.mse_cumul_prop = result_mse
        self.corr_cumul_prop = result_corr
        return self.mse_cumul_prop, self.corr_cumul_prop



    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        ## On run le calcul de metrics custo :
        self.metrics_to_baseline(y_val, y_predict)

        self._data.append({
            'mse_perso': self.metrics_to_baseline(y_val, y_predict)[0],
            'corr_perso': self.metrics_to_baseline(y_val, y_predict)[1],
        })

        print ("— val_mse_cumul_prop: %f — val_corr_cumul_prop: %f" %(self.mse_cumul_prop[8], self.corr_cumul_prop[8]))
        return



    def get_data(self):
        return self._data
