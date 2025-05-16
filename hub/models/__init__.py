from .llava import LlavaModel
from .minigpt import MiniGPTModel
from .deepseek import DeepseekModel
from .internvl import IntervlModel
from .pixtral import PixtralModel
from .omni import OmniModel
from .fuyu import FuyuModel

MODEL_REGISTRY = {
    "llava": LlavaModel,
    "minigpt": MiniGPTModel,
    "deepseek": DeepseekModel,
    "intervl": IntervlModel,
    "pixtral": PixtralModel,
    "omni": OmniModel,
    "fuyu": FuyuModel,
}