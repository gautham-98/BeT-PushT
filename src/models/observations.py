import torch.nn as nn
import torch
from src.models.resnet import resnet18

class ImageStateObservation(nn.Module):
    """ A simple module that passes the observed image through resnet and combines the feature vector with states
        The module can also be used as resnet encoder alone without using the observation.states by setting use_states = False.
        The module can combine image with state via simple concatenation and MLP modules.
    """
    
    def __init__(self, use_states=True, dropout=0.1):
        super().__init__() 
        self.resnet = resnet18()
        self.use_states = use_states
        if use_states:
          self.state_projection = nn.Sequential(
              nn.Linear(2, 4),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.Linear(4, 16),
              nn.GELU(),
              nn.Dropout(dropout)
          )
          
          self.image_projection = nn.Sequential(
              nn.Linear(512, 128),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.Linear(128, 64),   # final embedding size
              nn.GELU(),
              nn.Dropout(dropout)
          )
    
    def forward(self, images, states):
        # get image features
        B,T,C,H,W = images.shape
        images = images.reshape(B*T, C, H, W)
        image_embeddings = self.resnet(images)
        
        if self.use_states:
            # project states so that they talk the resnet language
            _,_, C_state = states.shape
            states = states.reshape(B*T, C_state)
            states = self.state_projection(states)
            image_embeddings = self.image_projection(image_embeddings)
            output = torch.cat([image_embeddings, states], dim=-1)
            # reshape output to batch, sequence, dim=80
            output = output.reshape(B,T,-1)

        else:
            output = image_embeddings.reshape(B,T,-1)
        
        return output

if __name__ == "__main__" :
    from src.data.dataloader import get_dataloaders
    
    trainloader, testloader = get_dataloaders()
    
    batch = next(iter(trainloader)) # get a single batch of data
    
    images, states, actions = batch["observation.image"], batch["observation.state"], batch["action"]
    
    observation_history_maker = ImageStateObservation(True)
    observation_history = observation_history_maker(images, states)
    pass