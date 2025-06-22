import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_acne_dataloaders(dataset_path='datasets/acne/preprocessed_acne', image_size=(224, 224), batch_size=32):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    data = []
    for img_file in image_files:
        label_file = os.path.join(labels_dir, img_file.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_file):
            with open(label_file, 'r') as lf:
                label = lf.read().strip()
                data.append({'image': os.path.join(images_dir, img_file), 'label': label})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Train-val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Generators
    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    val_gen = ImageDataGenerator(rescale=1./255)

    train_loader = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = val_gen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
