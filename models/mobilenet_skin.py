from tensorflow.keras.application import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.feras.models import Model 

def build_mobilenet(input_shape=(224,224,3),num_classes=4):
    base_model=MobileNetV2(include_top=False,input_shape=input_shape,weights='imagenet')
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.4)(x)
    x=Dense(128,activation='tanh')(x)
    predictions=Dense(num_classes,activation='softmax')(x)

    model=Model(inputs=base_model.input,outputs=predictions)
    return model