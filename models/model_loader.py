from vit_b import ViTBase
from vit_l import ViTLarge
from deit_b import DeiTBase

def load_model(model_name, num_classes):
    """
    Unified interface for loading ViT and DeiT models.
    
    :param model_name: Name of the model ('vit_b', 'vit_l', 'deit_b').
    :param num_classes: Number of output classes for the model.
    :return: Loaded model instance.
    """
    if model_name == 'vit_b':
        return ViTBase(num_classes).load_model()
    elif model_name == 'vit_l':
        return ViTLarge(num_classes).load_model()
    elif model_name == 'deit_b':
        return DeiTBase(num_classes).load_model()
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from ['vit_b', 'vit_l', 'deit_b'].")
