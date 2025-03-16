from models.model_1 import ModelOne
from models.model_2 import ModelTwo
from models.model_3 import ModelThree


def main():
    # # ModelOne: CUDA
    # p1 = ModelOne(device_str="cuda")
    # p1.train_model()

    # ModelTwo: CUDA
    p2 = ModelTwo(device_str="cuda")
    p2.train_model()

    # # ModelThree: CUDA
    # p3 = ModelThree(device_str="cuda")
    # p3.train_model()


if __name__ == "__main__":
    main()
