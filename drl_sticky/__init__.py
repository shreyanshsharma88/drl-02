"""drl_sticky - Group 84, DRL Assignment II (Set A).

Tabular Q-learning vs SARSA on MountainCar-v0 with a sticky actuator (attempted action changes are
rejected with probability 0.2).
Every component (wrapper, discretizer, agents, training, evaluation) is written
from scratch with NumPy; Gymnasium is used only to provide the base MountainCar-v0 physics.
"""

__version__ = "1.0.0"
