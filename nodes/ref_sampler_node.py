from copy import deepcopy

import comfy.samplers

from ..utils.ref_sampling_utils import create_sampler
from ..utils.ref_config import RefConfig


class RefSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "ref_latents": ("LATENT",),
            "sampler": (comfy.samplers.SAMPLER_NAMES, ),
            "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "attention_bound": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "flow"

    def build(self, ref_latents, sampler, style_fidelity, attention_bound):

        ref_config = RefConfig(style_fidelity, attention_bound)

        sampler = comfy.samplers.ksampler(sampler)
        sampler_fn = create_sampler(sampler.sampler_function, ref_latents, ref_config)
        
        sampler.sampler_function = sampler_fn

        return (sampler, )
    



class RefSamplerCustomNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "ref_latents": ("LATENT",),
            "sampler": ("SAMPLER",),
            "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "attention_bound": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "flow"

    def build(self, ref_latents, sampler, style_fidelity, attention_bound):
        sampler = deepcopy(sampler)

        ref_config = RefConfig(style_fidelity, attention_bound)
        
        sampler_fn = create_sampler(sampler.sampler_function, ref_latents, ref_config)
        
        sampler.sampler_function = sampler_fn

        return (sampler, )