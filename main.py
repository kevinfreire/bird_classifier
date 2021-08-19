import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import os
import pre_process as pp
import model_info as info

bird_dir=r'./Birds500'

train_split = .8
test_split = .1

height=128
width=128
channels=3
img_shape=(height, width, channels)
batch_size=40

df = pp.create_dataframe(bird_dir)

train_df, valid_df, test_df = pp.split_data(df, train_split, test_split)

print(df.head())
print(df['labels'].value_counts())

print ('train_df length: ', len(train_df), ' test_df length: ', len(test_df), '  validate_df length: ', len(valid_df))

train_gen, valid_gen, test_gen, test_batch_size, test_steps = pp.create_generators(train_df, valid_df, test_df, height, width, channels, batch_size)
classes=list(train_gen.class_indices.keys())
class_count=len(classes)

# Create the model
model_name="InceptionResNetV2"
base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
x=base_model.output
x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
x=Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
x=Dropout(rate=.45, seed=123)(x)        
output=Dense(class_count, activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy']) 

# Initiate custom callback and train the model
epochs =10
patience= 1 # number of epochs to wait to adjust lr if monitored value does not improve
stop_patience =3 # number of epochs to wait before stopping training if monitored value does not improve
threshold=.9 # if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor=.5 # factor to reduce lr by
dwell=True # experimental, if True and monitored metric does not improve on current epoch set  modelweights back to weights of previous epoch
freeze=False # if true free weights of  the base model

callbacks=[info.LRA(model=model,patience=patience,stop_patience=stop_patience, threshold=threshold, factor=factor,dwell=dwell, model_name=model_name, freeze=freeze, initial_epoch=0 )]
info.LRA.tepochs=epochs  # used to determine value of last epoch for printing
history=model.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=callbacks,  validation_data=valid_gen, validation_steps=None,  shuffle=False,  initial_epoch=0)

info.tr_plot(history,0)
save_dir=r'./'
subject='birds'
acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1]*100
msg=f'accuracy on the test set is {acc:5.2f} %'
info.print_in_color(msg, (0,255,0),(55,65,80))
save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
save_loc=os.path.join(save_dir, save_id)
model.save(save_loc)

print_code=0
preds=model.predict(test_gen) 
info.print_info( test_gen, preds, print_code, save_dir, subject )  