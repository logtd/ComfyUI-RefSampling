import torch

from .ref_controller import RefMode


def get_attn_wrapper(cls):
    class RefAttention(cls):
        ref_mode = RefMode.OFF
        ref_bank = None
        is_cfg = True
        style_fidelity = 0.5
        ref_attn_weight = 0.5
        attention_bound = 1.0
        hidden_states = None

        def forward(self, *args, **kwargs):
            if self.ref_mode == RefMode.OFF:
                return super().forward(*args, **kwargs)
            
            return self._ref_forward(*args, **kwargs)
        
        def _ref_forward(self, norm_hidden_states, context, **kwargs):
            if self.ref_mode == RefMode.WRITE:
                self.ref_bank = norm_hidden_states.detach().clone()
                return super().forward(norm_hidden_states, context, **kwargs)
            if self.ref_mode == RefMode.READ:
                if self.attention_bound > self.ref_attn_weight:
                    if len(self.ref_bank) == 2:
                        batch_size = len(norm_hidden_states) // len(self.ref_bank)
                        uc_bank = self.ref_bank[0].unsqueeze(0)
                        uc_bank = torch.cat([uc_bank] * batch_size)
                        c_bank = self.ref_bank[1].unsqueeze(0)
                        c_bank = torch.cat([c_bank] * batch_size)
                        bank = torch.cat([uc_bank, c_bank])
                    elif len(self.ref_bank) == 1:
                        bank = torch.cat([self.ref_bank] * len(norm_hidden_states))
                    else:
                        bank = self.ref_bank
                    bank_hidden_states = torch.cat([bank, norm_hidden_states], dim=1)
                    attn_output_uc = (
                        super().forward(
                            norm_hidden_states,
                            bank_hidden_states,
                            **kwargs
                        )
                    )
                    attn_output = attn_output_uc
                    if self.is_cfg and self.style_fidelity > 0:
                        attn_output_c = attn_output_uc.clone()
                        uc_mask =  torch.Tensor(
                                [1] * (norm_hidden_states.shape[0] // 2)
                                + [0] * (norm_hidden_states.shape[0] // 2)
                            ).to(norm_hidden_states.device).bool()
                        attn_sub =  super().forward(
                                norm_hidden_states[uc_mask],
                                norm_hidden_states[uc_mask].clone(),
                                **kwargs
                            )
                        attn_output_c[uc_mask] = attn_sub
                        attn_output = self.style_fidelity * attn_output_c + (1.0 - self.style_fidelity) * attn_output_uc
                else:
                    attn_output = super().forward(norm_hidden_states, context, **kwargs)
                return attn_output
            
    return RefAttention
