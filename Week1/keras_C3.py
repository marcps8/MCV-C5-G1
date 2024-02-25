from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = 'keras_model'
WEIGHTS_DIR = f'/ghome/group01/group01/project23-24-01/Task4/weights/{MODEL_NAME}.h5'
RESULTS_DIR = '/ghome/group01/group01/project23-24-01/Task4/results'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'

NUM_CLASSES = 8
BATCH_SIZE = 64
IMG_SIZE = (64, 64)
EPOCHS = 250

def save_confusion_matrix(model, classes, test_loader, save_path):
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                conf_matrix[label, prediction] += 1
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
def get_datasets():
    train_data_generator = ImageDataGenerator(
        rescale=1./255    
    )

    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/train/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_data_generator = ImageDataGenerator(
        rescale=1./255
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/test/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    test_data_generator = ImageDataGenerator(
        rescale=1./255
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR_GLOBAL+'/test/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    return train_dataset, validation_dataset, test_dataset

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_metrics(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{RESULTS_DIR}/{MODEL_NAME}_accuracy.jpg')
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{RESULTS_DIR}/{MODEL_NAME}_loss.jpg')
    plt.close()

train_dataset, validation_dataset, test_dataset = get_datasets()
model = build_model()

print('Start training...\n')
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=0,
    callbacks=[]
)

model.save_weights(WEIGHTS_DIR)
plot_metrics(history)

# Compute confusion matrix

classes = os.listdir(DATASET_DIR + '/train')
save_confusion_matrix(model, classes, test_dataset, f'{RESULTS_DIR}/{MODEL_NAME}_confusion_matrix.jpg')
