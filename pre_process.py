import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Iterate through the bird folder to create a dataframe of the form filepaths labels
def create_dataframe(input_dir):
    filepaths=[]
    labels=[]
    classlist=os.listdir(input_dir)
    for klass in classlist:
        classpath=os.path.join(input_dir,klass)
        if os.path.isdir(classpath):
            flist=os.listdir(classpath)
            for f in flist:
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)                   
    Fseries= pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)

    return df

# Split your data into training, validating and test sets
def split_data(df, train_split, test_split):
    dummy_split = test_split/(1-train_split)
    train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)
    test_df, valid_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)

    return train_df, valid_df, test_df

# Create the train, validate and test generators
def create_generators(train_df, valid_df, test_df, height, width, channels, batch_size):
    img_shape=(height, width, channels)
    img_size=(height, width)
    length=len(test_df)
    test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=30],reverse=True)[0]  
    test_steps=int(length/test_batch_size)
    print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)
    def scalar(img):
        return img/127.5-1  # scale pixel between -1 and +1
    gen=ImageDataGenerator(preprocessing_function=scalar)
    train_gen=gen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen=gen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode='rgb', shuffle=True, batch_size=batch_size)
    test_gen=gen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen, test_batch_size, test_steps