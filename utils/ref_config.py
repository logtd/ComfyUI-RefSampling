


class RefConfig:
    def __init__(self, sampling, ref_latent, style_fidelity, attention_bound):
        self.sampling = sampling
        self.ref_latent = ref_latent
        self.style_fidelity = style_fidelity
        self.attention_bound = attention_bound