import torch

import comfy.model_management

from .ref_config import RefConfig
from ..modules.ref_controller import RefController, RefMode
from ..utils.noise_utils import add_noise


def get_base_model(model):
    base_model = model
    while hasattr(base_model, 'inner_model'):
        base_model = base_model.inner_model
    return base_model


def prepare_ref_latents(model, ref_latent):
    base_model = get_base_model(model)
    ref_latent = ref_latent['samples']
    ref_latent = base_model.process_latent_in(ref_latent)
    device = comfy.model_management.get_torch_device()
    ref_latent = ref_latent.to(device)
    return ref_latent


def create_model_wrapper(model, sigmas, ref_latent, extra_args, ref_config:RefConfig):
    # extra_args: ['cond', 'uncond', 'cond_scale', 'model_options', 'seed', 'denoise_mask']
    ref_latent = prepare_ref_latents(model, ref_latent)

    # Get ref_controller
    model_options = extra_args.get('model_options', {})
    ref_controller: RefController = model_options.get('ref_controller', {})

    ref_conds = []
    for cond in extra_args['cond']:
        cond = {**cond}
        if 'control' in cond:
            del cond['control']
        ref_conds.append(cond)

    ref_unconds = []
    for cond in extra_args['uncond']:
        cond = {**cond}
        if 'control' in cond:
            del cond['control']
        ref_unconds.append(cond)
    
    # Setup ref
    ref_controller.clear_modules()
    is_cfg = extra_args.get('cond_scale', 7) > 1
    ref_controller.set_cfg(is_cfg)
    ref_controller.set_style_fidelity(ref_config.style_fidelity)
    ref_controller.set_attention_bound(ref_config.attention_bound)

    def model_sample(latents, sigma, **kwargs):
        ref_controller.get_default_temporal_values()
        # Do Ref Sampling
        ref_controller.set_temporal_values_to(1, 1)
        ref_controller.set_mode(RefMode.WRITE)
        ref_latent_noised = add_noise(ref_latent, torch.randn_like(ref_latent), sigma[0])
        ref_sigma = torch.tensor([sigma[0]]*len(ref_latent)).to(sigmas.device)

        model(ref_latent_noised, ref_sigma, **{**kwargs, 'cond': ref_conds, 'uncond': ref_unconds})

        # Do Latent Sampling
        ref_controller.set_temporal_values_to_default()
        ref_controller.set_mode(RefMode.READ)
        latents = model(latents, sigma, **kwargs)

        # Clear refs
        ref_controller.clear_modules()
        ref_controller.set_mode(RefMode.OFF)
        return latents


    class RefModelWrapper:
        inner_model = model.inner_model
        def __call__(self, *args, **kwargs):
            return model_sample(*args, **kwargs)
        
    return RefModelWrapper()


def create_sampler(sample_fn, inversion_latent, ref_config):
    @torch.no_grad()
    def sample(model, latents, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        # src_latents = prepare_src_latents(model, src_latent_image)
        model_wrapper = create_model_wrapper(model, sigmas, inversion_latent, extra_args, ref_config)
        output = sample_fn(model_wrapper, latents, sigmas, extra_args=extra_args, callback=callback, disable=disable)
        del model_wrapper
        return output
    
    return sample