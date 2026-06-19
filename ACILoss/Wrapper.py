import torch
from .Head import *
import torch.nn as nn

class InformerPipeline(nn.Module):
    def __init__(self, hf_informer_decoder, emb_size=512, out_size=1):
        super().__init__()
        # 1. House the raw Hugging Face decoder
        self.decoder = hf_informer_decoder
        
        # 2. House your custom regression head inside the same container
        self.regression_head = RegressionHead(emb_size=emb_size, out_size=out_size)

    def forward(self, encoder_hidden_states, target_embeddings):
        """
        Args:
            encoder_hidden_states: Rich 512-dim tensor from your encoder [B, Context_Len, 512]
            target_embeddings: Real 3-dim target sequence (history token + future covariates) [B, Prediction_Len, 3]
        """
        # Pass both real tensors directly into the Hugging Face core
        decoder_outputs = self.decoder(
            inputs_embeds=target_embeddings,
            encoder_hidden_states=encoder_hidden_states
        )
        
        # Pass the resulting hidden states straight into your regression head
        final_output = self.regression_head(decoder_outputs.last_hidden_state)
        
        return final_output