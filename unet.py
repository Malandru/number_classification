import segmentation_models as sm
import visualize
from tensorflow import keras

# Configuration parameters
BACKBONE = 'efficientnetb0'
BATCH_SIZE = 1
preprocess_input = sm.get_preprocessing(BACKBONE)


class Inference:
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size  # (height, width)
        # Create model
        self.model = sm.Unet(backbone_name=BACKBONE, encoder_weights='imagenet', encoder_freeze=True)
        self.model.compile('Adam', loss=sm.losses.categorical_crossentropy, metrics=['accuracy'])
        # Training variables (N, H, W, C)
        self.x_train = None  # numpy array with shape (X, H, W, 3) ==> X images with 3 color channels
        self.y_train = None  # numpy array with shape (Y, H, W, 1) ==> Y images with mask result
        self.epochs = 0

    def load_model(self, model_path=''):
        if len(model_path) <= 0:
            self.model = keras.models.load_model(model_path)

    def set_train_params(self, x, y, epochs):
        self.x_train = x
        self.y_train = y
        self.epochs = epochs

    def train(self):
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=BATCH_SIZE, epochs=self.epochs)

    def test_prediction(self, x_test, y_test):
        predictions = self.model.predict(x_test)  # returns numpy array with shape (L, H, W, 1)
        random_index = 0

        test_input = x_test[random_index]  # numpy array with shape (H, W, 3)
        test_output = predictions[random_index]  # numpy array with shape (H, W, 2)
        expected_output = y_test[random_index]  # numpy array with shape (H, W, 2)

        visualize.prepare_figure(test_input)
        visualize.prepare_figure(test_output)
        visualize.prepare_figure(expected_output)

    def save_model(self, filename):
        self.model.save(filename)
