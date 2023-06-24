from tensorflow import keras
import tensorflow as tf

from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,\
     BatchNormalization

img_res = (180, 180)

train_data, val_data = tf.keras.utils.image_dataset_from_directory(
    "data/PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=img_res,
    batch_size=128,
)


# ## Build the model
model = keras.Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(180,180,3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1,activation='sigmoid'),
])

model.summary()

epochs = 3

callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),]

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.fit( train_data, epochs=epochs,
    callbacks=callbacks,
    validation_data=val_data,
)

model.save("CNN-Model.h5", include_optimizer=True)