import torch

from ..modules.attn_wrapper import get_attn_wrapper
from ..modules.unet_wrapper import get_unet_wrapper
from ..modules.ref_controller import RefController
from ..modules.block_type import BlockType


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def is_transformer_block(module):
    return isinstance_str(module, 'BasicTransformerBlock')


def is_temporal_block(module):
    return isinstance_str(module, 'TemporalTransformerBlock')


def is_named_module_transformer_block(named_module):
    if is_transformer_block(named_module[1]):
        return True
    elif is_temporal_block(named_module[1]):
        return True
    return False


def get_norm1_shape(block_module):
    if is_transformer_block(block_module):
        return block_module.norm1.normalized_shape[0]
    elif is_temporal_block(block_module):
        return block_module.norms[0].normalized_shape[0]
    raise ValueError('Bad block!', block_module)


def get_block_attention(block_module):
    if is_transformer_block(block_module):
        return block_module.attn1
    elif is_temporal_block(block_module):
        return block_module.attention_blocks[0]


def sort_block(module):
    return -get_norm1_shape(module)


def setup_ref_attn(model, attn_type):
    # TODO: attn_type for full/mid_out
    model = model.model.diffusion_model
    block_modules = list(filter(is_named_module_transformer_block, model.named_modules()))
    block_modules = list(map(lambda x: x[1], block_modules))
    

    ref_controller = RefController()

    for i, block_module in enumerate(block_modules):
        if is_transformer_block(block_module):
            attn = get_block_attention(block_module)
            if not hasattr(attn, 'ref_mode'):
                attn.__class__ = get_attn_wrapper(attn.__class__)
            attn.ref_attn_weight = float(i) / float(len(block_modules))
            ref_controller.add_module(attn)

    block_modules = sorted(block_modules, key=sort_block)
    for i, block_module in enumerate(block_modules):
        attn.ref_attn_weight = float(i) / float(len(block_modules))

    output_modules = list(filter(is_named_module_transformer_block, model.output_blocks.named_modules()))
    output_modules = list(map(lambda x: x[1], output_modules))
    for i, block_module in enumerate(output_modules):
        attn = block_module.attn1
        attn.block_type = BlockType.OUTPUT
        attn.block_idx = i

    temp_transformers = list(filter(lambda x: isinstance_str(x, 'TemporalTransformer3DModel'), model.named_modules()))

    for _, transformer_module in temp_transformers:
        ref_controller.add_temporal_transformer(transformer_module)

    return ref_controller


def setup_ref_unet(model, attn_type):
    if not hasattr(model.model.diffusion_model, 'is_ref'):
        ref_controller = setup_ref_attn(model, attn_type)
        model.model.diffusion_model.__class__ = get_unet_wrapper(model.model.diffusion_model.__class__, ref_controller)
        return ref_controller
    return None
    