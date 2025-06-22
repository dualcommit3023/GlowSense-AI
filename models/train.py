from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.mobilenet_skin import build_mobilenet
from models.utils import get_dataloaders

train_loader, val_loader = get_dataloaders()
model = build_mobilenet(input_shape=(224, 224, 3), num_classes=4)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('outputs/checkpoints/best_model.h5', save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(train_loader,
          validation_data=val_loader,
          epochs=30,
          callbacks=[checkpoint, early_stop])
