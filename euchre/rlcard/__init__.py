import sys

name = "rlcard"
__version__ = "0.2.5"

# Preserve the original absolute import style used throughout this fork.
sys.modules.setdefault('rlcard', sys.modules[__name__])

from euchre.rlcard.envs import make
