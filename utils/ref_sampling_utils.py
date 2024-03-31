import comfy.model_management


def get_base_model(model):
    base_model = model
    while hasattr(base_model, 'inner_model'):
        base_model = base_model.inner_model
    return base_model


def prepare_ref_latents(model, ref_latent):
    base_model = model.model # get_base_model(model)
    ref_latent = ref_latent['samples'].clone()
    ref_latent = base_model.process_latent_in(ref_latent)
    device = comfy.model_management.get_torch_device()
    ref_latent = ref_latent.to(device)
    return ref_latent
