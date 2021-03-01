from number_dataset import load_number_dataset
from unet import Inference
import visualize


def main():
    (x_train, y_train), (x_test, y_test) = load_number_dataset(50)
    inference = Inference()
    inference.set_train_params(x_train, y_train, epochs=10)
    inference.train()

    inference.test_prediction(x_test, y_test)
    visualize.show_prepared_figures()
    inference.save_model("numbers.h5")


if __name__ == '__main__':
    main()
