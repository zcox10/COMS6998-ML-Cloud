from models.model_1 import ModelOne
from models.model_2 import ModelTwo
from models.model_3 import ModelThree


def main():
    # m = ModelOne(device_str="cuda")
    # m = ModelTwo(device_str="cuda")
    m = ModelThree(device_str="cuda")
    m.train_model()


if __name__ == "__main__":
    main()
