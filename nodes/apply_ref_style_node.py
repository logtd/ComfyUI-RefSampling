import torch
from ..utils.ref_sampling_utils import prepare_ref_latents
from ..utils.ref_style_config import RefStyleConfig


class ApplyRefStyleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "ref_latents": ("LATENT",),
            "enabled": ("BOOLEAN", {"default": True}),
            "style_positive": ("CONDITIONING",),
            "style_negative": ("CONDITIONING",),
            "attention_count": ("INT", {"default": 4, "min": 0, "max": 10, "step": 1}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self,
              model,
              ref_latents,
              enabled,
              style_positive,
              style_negative,
              attention_count,
              start_percent,
              end_percent):
        transformer_options = model.model_options.get(
            'transformer_options', {})
        model.model_options['transformer_options'] = transformer_options
        if not enabled:
            if 'ref_style_config' in transformer_options:
                del transformer_options['ref_style_config']
            return (model, )

        ref_latents = prepare_ref_latents(model, ref_latents)
        sampling = model.model.model_sampling

        style_prompt = torch.cat([style_negative[0][0], style_positive[0][0]])

        ref_config = RefStyleConfig(
            sampling,
            ref_latents,
            enabled,
            style_prompt,
            attention_count,
            start_percent,
            end_percent)

        transformer_options['ref_style_config'] = ref_config

        return (model, )
