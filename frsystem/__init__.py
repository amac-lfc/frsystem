from .version import __version__
from .frs import FaceRecognitionSystem

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    'FaceRecognitionSystem',
]