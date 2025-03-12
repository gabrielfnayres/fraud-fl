
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.fed_traning import fed_train

if __name__ == '__main__':
    fed_train()