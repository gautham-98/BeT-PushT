import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(self, num_bins, action_dim, drop_out=0.1):
        super(Head, self).__init__()

        self.num_bins = num_bins
        self.action_dim = action_dim

        linear_in_dim = num_bins
        linear_out_dim = num_bins*(action_dim+1)
        
        self.layer = nn.Linear(linear_in_dim,linear_out_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, transformer_logits):
        
        batch_size, sequence_length, num_bins = transformer_logits.shape
        # Transformer Logits: (Batch_Size , Sequence_Length , Number_Action_Bins)

        x = self.layer(transformer_logits)
        action_data = self.dropout(x)
        # Action Data: (Batch_Size , Sequence_Length , Number_Action_Bins*(Action_dim+1))

        seq_action_bins_logits, all_seq_action_residuals = torch.split(action_data, [num_bins, num_bins * self.action_dim], dim=-1)
        # Sequence Action Bins Logits:   (Batch_Size , Sequence_Length, Number_Action_Bins)
        # All Sequence Action Residuals: (Batch_Size , Sequence_Length, Number_Action_Bins*Action_dim)
        
        # Softmaxing the seq_action_bins_logits action bins logits to get the probability per bin at every sequence index
        seq_action_bins_probs = torch.softmax(seq_action_bins_logits,dim=-1)
        selected_seq_action_bins = torch.multinomial(seq_action_bins_probs.view(-1, num_bins), num_samples=1)
        selected_seq_action_bins = selected_seq_action_bins.reshape((batch_size,sequence_length,1))
        # Sequence Action Bins: (Batch_Size, Sequence_Length, 1)

        # Keeping the action resiudals for the selected action bin
        seq_action_residuals = all_seq_action_residuals.reshape((batch_size, sequence_length, num_bins, self.action_dim))
        flat_all_seq_action_residuals = all_seq_action_residuals.reshape((batch_size*sequence_length, num_bins, self.action_dim))
        flat_seq_action_residuals = flat_all_seq_action_residuals[torch.arange(flat_all_seq_action_residuals.shape[0]), selected_seq_action_bins.flatten()]
        selected_seq_action_residuals = flat_seq_action_residuals.reshape((batch_size, sequence_length, self.action_dim))
        # Sequence Action Residuals: (Batch_Size , Sequence_Length, Action_dim)

        return {"selected_seq_action_bins": selected_seq_action_bins, "selected_seq_action_residuals": selected_seq_action_residuals, "seq_action_residuals": seq_action_residuals, "seq_action_bins_logits": seq_action_bins_logits,}