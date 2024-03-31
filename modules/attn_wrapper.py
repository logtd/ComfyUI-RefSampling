import torch

from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked
from comfy.ldm.util import default

from .ref_controller import RefMode




def get_attn_wrapper(cls):
    class RefAttention(cls):
        ref_mode = RefMode.OFF
        normal_mode = RefMode.OFF
        kv_mode = RefMode.OFF
        normal_bank = None
        k_bank = None
        v_bank = None
        is_cfg = True
        content_fidelity = 0.5
        ref_attn_weight = 1.0
        attention_bound = 1.0
        hidden_states = None
        block_type = None
        block_idx = None

        def _attention_mechanism(self, q, k, v, mask):
            if mask is None:
                out = optimized_attention(q, k, v, self.heads)
            else:
                out = optimized_attention_masked(q, k, v, self.heads, mask)
            return self.to_out(out)

        def _get_qkv(self, x, context=None, value=None):
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            if value is not None:
                v = self.to_v(value)
                del value
            else:
                v = self.to_v(context)
            return q,k,v
        
        def clean(self):
            self.normal_bank = None
            self.k_bank = None
            self.v_bank = None

        def forward(self, *args, **kwargs):
            if self.ref_mode == RefMode.OFF:
                return super().forward(*args, **kwargs)
            
            return self._ref_forward(*args, **kwargs)
        
        def _ref_forward(self, norm_hidden_states, context, **kwargs):
            context = default(context, norm_hidden_states)
            value = kwargs.get('value', None)
            value = default(value, context)

            if self.ref_mode == RefMode.WRITE:
                if self.normal_mode == RefMode.WRITE and self.attention_bound > self.ref_attn_weight:
                    self.normal_bank = norm_hidden_states.detach().clone()
                if self.kv_mode == RefMode.WRITE:
                    q,k,v = self._get_qkv(norm_hidden_states, context, **kwargs)
                    self.k_bank = k.detach().clone()
                    self.v_bank = v.detach().clone()
                    mask = kwargs['mask'] if 'mask' in kwargs else None
                    return self._attention_mechanism(q,k,v,mask=mask)
                else:
                    return super().forward(norm_hidden_states, context, **kwargs)
            if self.ref_mode == RefMode.READ:
                normal_bank = None
                if self.normal_mode == RefMode.READ and self.attention_bound > self.ref_attn_weight:
                    normal_bank = self.normal_bank
                    self.normal_bank = None
                
                if normal_bank is not None:
                    context = torch.cat([context, normal_bank], dim=1)
                    value = torch.cat([value, normal_bank], dim=1)

                
                q, k, v = self._get_qkv(norm_hidden_states, context, value)

                if self.kv_mode == RefMode.READ:
                    k = self.k_bank
                    v = self.v_bank
                    self.k_bank = None
                    self.v_bank = None
                
                mask = kwargs['mask'] if 'mask' in kwargs else None
                attn_output = self._attention_mechanism(q,k,v, mask=mask)
                
                if normal_bank is not None and self.is_cfg and self.content_fidelity > 0:
                        attn_output_c = attn_output.clone()
                        uc_mask =  torch.Tensor(
                                [1] * (norm_hidden_states.shape[0] // 2)
                                + [0] * (norm_hidden_states.shape[0] // 2)
                            ).to(norm_hidden_states.device).bool()
                        attn_sub =  super().forward(
                                norm_hidden_states[uc_mask].clone(),
                                norm_hidden_states[uc_mask].clone(),
                                **kwargs
                            )
                        attn_output_c[uc_mask] = attn_sub
                        return self.content_fidelity * attn_output_c + (1.0 - self.content_fidelity) * attn_output
                return attn_output
            
    return RefAttention
