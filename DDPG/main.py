from ddpg import DDPG
import keras.backend as K
import tensorflow as tf


if __name__ == "__main__":
    ddpg = DDPG()
    ddpg.run()