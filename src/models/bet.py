import torch
import torch.nn as nn

from src.models.action_encoder_decoder import EncoderDecoder
from src.models.action_head import Head
from src.models.gpt import GPTConfig, GPT

class BeT(nn.Module):

    def __init__(self, 
                observation_dim, 
                embedding_dim, 
                num_transformer_layers, 
                num_attention_heads, 
                action_dim, 
                num_bins, 
                sequence_length, 
                actions, 
                device="cuda"
                ):
        super(BeT, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.sequence_length = sequence_length
        
        self.device = device
        
        # Initializing the Encoder Decoder which is based on the K-Means
        self.encoderDecoder = EncoderDecoder(action_dim, num_bins, actions, device)

        # Initializing the GPT model to be used for sequence to sequence modeling
        self.gpt = GPT(
                        GPTConfig(
                        num_bins, 
                        sequence_length, 
                        input_size=observation_dim, 
                        n_embd=embedding_dim, 
                        n_layer=num_transformer_layers,
                        n_head=num_attention_heads 
                        )
                    ).to(device)
                
        # Initializing the Head that takes the sequence output of the GPT to return the action bin and action residual
        self.head = Head(num_bins, action_dim).to(device)

    def forward(self, observations_history, train_data=False):
        
        # (Batch Size, Sequence Lenght, Number of Action Bins)
        gpt_logits = self.gpt(observations_history)

        head_output = self.head(gpt_logits)
        # Predicted Sequence Action Bins: (Batch_Size, Sequence_Length, 1)
        selected_seq_action_bins = head_output["selected_seq_action_bins"]
        # Predicted Sequence Action Residuals: (Batch_Size, Sequence_Length, Action_dim)
        selected_seq_action_residuals = head_output["selected_seq_action_residuals"]
        # Predicted Sequence Action Bins Logits: (Batch_Size, Sequence_Length, Number_Action_Bins)
        seq_action_bins_logits = head_output["seq_action_bins_logits"]
        # Predicted Sequence Action Residuals per bin
        seq_action_residuals = head_output["seq_action_residuals"]

        # Predicted Action: (Batch_Size, Action_dim)
        predicted_action = self.encoderDecoder.decode(selected_seq_action_bins[:,-1,0],selected_seq_action_residuals[:,-1,:]) # selects only the last element of the sequence

        # No Training: Inference Only (No Loss Calculation)
        if not train_data:
            return predicted_action
        # Training: Return Predicted Action and Training Required Data
        else:
            return {"seq_action_bins_logits":seq_action_bins_logits, "seq_action_residuals":seq_action_residuals}
    
    def create_optimizer(self, weight_decay, learning_rate, betas):
        optimizer = self.gpt.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )
        optimizer.add_param_group({"params": self.head.parameters()})
        return optimizer

if __name__ == "__main__":
    num_bins = 2
    batch_size = 2
    total_action_batch_size = 5
    sequence_length = 10
    action_dim = 3
    total_actions = 100
    observation_dim=10
    embedding_dim = 5
    num_transformer_layers = 1
    num_attention_heads = 1

    DEVICE = "cpu"
    
    actions_collection = torch.rand((total_actions,action_dim)).to(DEVICE)
    bet = BeT(
            observation_dim, embedding_dim, num_transformer_layers, 
            num_attention_heads, action_dim, num_bins, 
            sequence_length, actions_collection, device=DEVICE
            )

    observations_history = torch.rand((batch_size,sequence_length,observation_dim)).to(DEVICE)
    
    predicted_action = bet(observations_history)
    print("Action Shape: ", predicted_action.shape)