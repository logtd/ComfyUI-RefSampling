from ..utils.ref_sampling_utils import prepare_ref_latents
from ..utils.ref_content_config import RefContentConfig


class ApplyRefContentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "ref_latents": ("LATENT",),
            "enabled": ("BOOLEAN", {"default": True}),
            "fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "attention_bound": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self,
              model,
              ref_latents,
              enabled,
              fidelity,
              attention_bound,
              start_percent,
              end_percent):
        transformer_options = model.model_options.get(
            'transformer_options', {})
        model.model_options['transformer_options'] = transformer_options
        if not enabled:
            if 'ref_content_config' in transformer_options:
                del transformer_options['ref_content_config']
            return (model, )

        ref_latents = prepare_ref_latents(model, ref_latents)
        sampling = model.model.model_sampling

        ref_config = RefContentConfig(
            sampling,
            ref_latents,
            enabled,
            fidelity,
            attention_bound,
            start_percent,
            end_percent)

        transformer_options['ref_content_config'] = ref_config

        return (model, )
