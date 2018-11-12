

import os
from PIL import Image
import cv2
import numpy as np
import time
import itertools

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.layers import Dense, LeakyReLU, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
plt.interactive(False)


class SETI(object):

    def __init__(self):

        self.sclasses = ["brightpixel",
                         "narrowband",
                         "narrowbanddrd",
                         "noise",
                         "squarepulsednarrowband",
                         "squiggle",
                         "squigglesquarepulsednarrowband"]
        """
        self.sclasses = ["brightpixel", "noise"]
        """
        self.model_name = "seti_model"
        self.epochs = 100
        self.learning_rate = 0.0005
        self.batch_size = 50
        self.output_classes = len(self.sclasses)
        self.loss = "categorical_crossentropy"
        self.parameter_scaling = 36
        self.regularizer = 0.0 #1.0e-7
        self.model_location = r"C:\Users\Paperspace\IdeaProjects\deeplearning\src\model"
        self.tensorboard = "./tensorboard"
        self.channels = 1
        self.preprocess = True
        self.gaussian_blurr = False
        self.histogram_equalize = False
        self.bitwise_not = False
        self.augument = True
        self.augument_size = 200
        self.image_width_original = 512
        self.image_hieght_original = 384
        self.image_width = 384
        self.image_hieght = 384
        self.input_shape = (self.image_width, self.image_hieght, self.channels)

    def get_data(self, primary_dir, augument=False):
        X = np.empty(shape=(0, self.image_width, self.image_hieght, self.channels))
        Y = np.empty(shape=(0, ))
        _global_index = -1
        for sclass in self.sclasses:
            # loop through directories
            sclass_dir = os.path.join(primary_dir, sclass)
            print("Sclass: {}".format(sclass))
            _x = []
            _y = []
            for index, filename in enumerate(os.listdir(sclass_dir)):
                if index % 100 == 0:
                    print("Pre-processing {}th image in {} class".format(index, sclass))
                _global_index += 1
                # _image = np.asarray(Image.open(os.path.join(sclass_dir, filename)), dtype=np.float32)
                _image = cv2.imread(os.path.join(sclass_dir, filename))
                _image = self.process(_image)
                # into local list
                _x.insert(index, _image)
                _y.insert(index, sclass)
            if augument:
                x_aug = self.augument_images(np.array(_x))
                y_aug = [sclass]*len(x_aug)
                _x.extend(x_aug.tolist())
                _y.extend(y_aug)
                _global_index += self.augument_size
                print("Augumented {} images for the class: \"{}\"".format(self.augument_size, sclass))
            _x = np.array(_x)
            _y = np.array(_y)
            print("yshape: {}".format(_y.shape))
            # into global list
            X = np.append(X, _x, axis=0)
            Y = np.append(Y, _y, axis=0)
            print("Data Extraction complete for class: \"{}\"".format(sclass))
            print("Global Index: {}".format(_global_index))
        return X, Y

    def process(self, image):
        image = cv2.resize(image, (self.image_width, self.image_hieght))
        if self.channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.gaussian_blurr:
            image = cv2.GaussianBlur(image, (3, 3), 1)
        if self.histogram_equalize:
            image = cv2.equalizeHist(image.astype(np.uint8))
        if self.channels == 1 and self.bitwise_not:
            image = cv2.bitwise_not(image)
        if self.preprocess:
            image = self.preprocess_image(image)
        image = image/255
        image = image.reshape(self.image_width, self.image_hieght, self.channels)
        return image

    def preprocess_image(self, image):
        mean = np.mean(image)
        std = np.std(image)
        clipped = np.clip(image, mean-3.5*std, mean+3.5*std)
        morphed = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, kernel=np.ones((2, 2), dtype=np.float32))
        sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
        sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
        blended = cv2.addWeighted(src1=sobelx, alpha=0.8, src2=sobely, beta=0.2, gamma=0)

        """
        # adjusted calculations
        mu = image.mean(axis=(0, 1))
        sigma = image.std()
        kernel_shape = [33, 11]
        threshold = 3.0
        factor = np.product(kernel_shape)
        by9_std = np.sqrt((sigma*sigma)/factor)

        # convolution
        image_blur = (convolve(image, np.ones(kernel_shape), mode="same", method="direct")/factor).astype(np.uint8)
        image_blur[image_blur < (mu+threshold*by9_std)] = 0
        image_blur[image_blur > 0] = 1
        return image_blur
        """
        return blended

    def augument_images(self, images):
        datagen = ImageDataGenerator(width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     shear_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     rotation_range=10)
        datagen.fit(images)
        images_batch = next(datagen.flow(x=images, y=None, batch_size=self.augument_size))
        return images_batch

    def model(self):

        M = self.parameter_scaling
        L1 = self.regularizer

        # model architecture
        _model = Sequential()

        # convolution 1
        _model.add(Conv2D(M, (3, 3), input_shape=self.input_shape, kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 2
        _model.add(Conv2D(2*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 3
        _model.add(Conv2D(3*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 4
        _model.add(Conv2D(4*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 5
        _model.add(Conv2D(5*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 6
        _model.add(Conv2D(6*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # convolution 7
        _model.add(Conv2D(7*M, (3, 3), kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.3))

        # flattening layer
        _model.add(Flatten())

        # first dense layer
        _model.add(Dense(units=7*M, kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(Dropout(0.5))

        # second dense layer
        _model.add(Dense(units=7*M, kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(Dropout(0.5))

        # third dense layer
        _model.add(Dense(units=7*M, kernel_regularizer=l1(L1)))
        _model.add(LeakyReLU(alpha=0.1))
        _model.add(Dropout(0.5))

        # output layer
        _model.add(Dense(self.output_classes, activation="softmax"))

        # optimizer
        _model.compile(Adam(lr=self.learning_rate), loss=self.loss, metrics=["accuracy"])
        print(_model.summary())
        return _model

    def fit(self, model, images_train, labels_train, images_val, labels_val):

        # callbacks
        h_callbacks = [
            callbacks.TensorBoard(
                log_dir=self.tensorboard,
                histogram_freq=10,
                write_graph=True,
                write_images=True
            )
        ]

        with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # fit the data
            history = model.fit(images_train, labels_train,
                                epochs=self.epochs,
                                validation_data=(images_val, labels_val),
                                shuffle=True,
                                batch_size=self.batch_size,
                                verbose=1,
                                callbacks=h_callbacks)

        # write model config
        with open(os.path.join(self.model_location, "{}.json".format(self.model_name)), "w") as model_json:
            model_json.write(model.to_json())
            print("Saved Model json to disk")
        # save weights to h5
        model.save_weights(os.path.join(self.model_location, "{}.h5".format(self.model_name)))
        print("Saved Model weights to disk")
        return history

    def train(self, images_train, labels_train, images_val, labels_val):

        print("Training Images Shape: {}".format(images_train.shape))
        print("Training Labels Shape: {}".format(labels_train.shape))
        print("Validation Images Shape: {}".format(images_val.shape))
        print("Validation Labels Shape: {}".format(labels_val.shape))
        """
        _model = resnet.ResnetBuilder.resnet_50(input_shape=self.input_shape,
                                                output_classes=len(self.sclasses))
        """
        _model = self.model()
        _model.compile(loss="categorical_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
        self.fit(_model, images_train, labels_train, images_val, labels_val)
        return _model

    def test(self, model, images, labels):

        # encode labels
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded = encoder.transform(labels)
        _labels = to_categorical(encoded, len(self.sclasses))
        score = model.evaluate(images, _labels, verbose=0)
        print("Test Score: {}".format(score[0]))
        print("Test Accuracy: {}".format(score[1]))
        return score

    def show_random_image(self, image_np, labels_np):
        index = np.random.randint(0, image_np.shape[0] - 1)
        print(index)
        image = image_np[index]
        image = image.reshape(self.image_width, self.image_hieght)
        print(image.shape)
        plt.imshow(image, cmap="gray")
        plt.title(labels_np[index])
        plt.axis("off")
        plt.show(block=True)

    def plot_confusion_matrix(self, cm, classes, normalize=False, cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


if __name__ == "__main__":

    tick = time.time()
    seti = SETI()

    X_train, y_train = seti.get_data(primary_dir=os.path.join(os.getcwd(), r"C:\Users\Paperspace\IdeaProjects\seti", "train"), augument=seti.augument)
    # encode labels
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_train = encoder.transform(y_train)
    y_train = to_categorical(encoded_train, seti.output_classes)
    train_time = time.time()
    print("Pre Processing time for Train Images:  {} seconds".format(train_time-tick))

    X_val, y_val = seti.get_data(primary_dir=os.path.join(os.getcwd(), r"C:\Users\Paperspace\IdeaProjects\seti", "valid"))
    encoder_val = LabelEncoder()
    encoder_val.fit(y_val)
    encoded_test = encoder_val.transform(y_val)
    y_val = to_categorical(encoded_test, seti.output_classes)
    val_time = time.time()
    print("Pre Processing time for Validation Images:  {} seconds".format(val_time-train_time))

    model = seti.train(X_train, y_train, X_val, y_val)

    X_test, y_test = seti.get_data(primary_dir=os.path.join(os.getcwd(), r"C:\Users\Paperspace\IdeaProjects\seti", "test"))
    #seti.test(model, X_test, y_test)

    y_pred = model.predict_classes(X_test)
    encoder_test = LabelEncoder()
    encoder_test.fit(y_test)
    encoded_test = encoder_test.transform(y_test)
    y_test = to_categorical(encoded_test, seti.output_classes)
    y_test = np.argmax(y_test, axis=1)

    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("\n\n")
    print("Classification Report")
    print(classification_report(y_test, y_pred, digits=5))

    print("Total Execution time:  {} seconds".format(time.time()-tick))














