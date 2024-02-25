from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = 'keras_model'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'
WEIGHTS_DIR = f'/ghome/group01/MCV-C5-G1/Week1/weights/{MODEL_NAME}.pt'
RESULTS_DIR = '/ghome/group01/MCV-C5-G1/Week1/results'

NUM_CLASSES = 8
BATCH_SIZE = 64
IMG_SIZE = (64, 64)
EPOCHS = 250

def save_confusion_matrix(model, classes, test_data, save_path):
    # Extracting images and labels from test data
    images, labels = test_data

    # Making predictions
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()



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