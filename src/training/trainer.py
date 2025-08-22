import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb

from src.models.bet import BeT
from src.models.observations import ImageStateObservation
from src.training.losses import FocalLoss, MultiTaskLoss

class Trainer:
    
    def __init__(
                    self, 
                    observation_module: ImageStateObservation,
                    bet:BeT, 
                    trainloader:DataLoader,
                    valloader:DataLoader, 
                    epochs=10, 
                    learning_rate=1e-4, 
                    weight_decay=0.2, 
                    betas=[0.9,0.999],
                    gamma=2.0,
                    residual_loss_scale=10,
                    eval_interval = 5,
                    ckpt_dir = "./checkpoints",
                    run_name = "bet_training"
                    ):
            self.observation_module = observation_module
            self.bet = bet
            self.trainloader = trainloader
            self.valloader = valloader
            self.epochs = epochs
            self.optimizer = self.get_optimizer(learning_rate, weight_decay, betas)
            self.focal_loss = FocalLoss(gamma)
            self.mt_loss = MultiTaskLoss() 
            self.residual_loss_scale = residual_loss_scale
            
            self.eval_interval = eval_interval
            self.ckpt_dir = ckpt_dir
            os.makedirs(self.ckpt_dir, exist_ok=True)
            
            wandb.init(project="BeT", name=run_name, config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": betas,
                "gamma": gamma,
                "residual_loss_scale": residual_loss_scale,
            })
        
    def train(self):
        training_step = 0
        for epoch in tqdm(range(self.epochs), desc="epochs", position=0):
            # Training
            self.observation_module.train()
            self.bet.train()
            
            train_loss = 0.0
            train_bin_loss = 0.0
            train_residual_loss = 0.0
            for batch in tqdm(self.trainloader, desc="training", position=1, leave=False):            
                loss, action_bins_loss, action_residual_loss = self._step(batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bet.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.observation_module.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()
                train_bin_loss += action_bins_loss.item()
                train_residual_loss += action_residual_loss.item()
                
                wandb.log({"training_loss": loss.item(), "training_bin_loss": action_bins_loss.item(), "training_residual_loss":action_residual_loss.item(), "training_step": training_step, "epoch":epoch})
                
                training_step += 1
                
            train_loss /= len(self.trainloader)
            train_bin_loss /= len(self.trainloader)
            train_residual_loss /= len(self.trainloader)
            
            # log
            wandb.log({"train_loss": train_loss, "train_bin_loss":train_bin_loss, "train_residual_loss":train_residual_loss, "epoch": epoch})

            # Validation 
            if epoch%self.eval_interval == 0:
                self.observation_module.eval()
                self.bet.eval()
                val_loss = 0.0
                val_bin_loss = 0.0
                val_residual_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.valloader, desc="validating", position=2, leave=False):
                        loss, bin_loss, residual_loss = self._step(batch)
                        val_loss += loss.item()
                        val_bin_loss += bin_loss.item()
                        val_residual_loss += residual_loss.item()

                val_loss /= len(self.valloader)
                val_bin_loss /= len(self.valloader)
                val_residual_loss /= len(self.valloader)
                
                # log
                wandb.log({"val_loss": val_loss, "val_bin_loss":val_bin_loss, "val_residual_loss":val_residual_loss, "epoch": epoch})
                
                tqdm.write(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # save checkpoints
                self.save_checkpoint(epoch, val_loss)
                
            
    def _step(self, batch):
        # extract the inputs and outputs from batch
        observations_images_history, observation_states_history, actions_history = batch["observation.image"].to(self.bet.device), batch["observation.state"].to(self.bet.device), batch["action"].to(self.bet.device)
        
        # get the target bins and residuals for each action in action history
        target_seq_action_bins, target_seq_action_residuals = self.bet.encoderDecoder.encode(actions_history) # B,T,1; B, T, Action_dimension
        target_seq_action_bins, target_seq_action_residuals =  target_seq_action_bins.to(self.bet.device), target_seq_action_residuals.to(self.bet.device)
        
        # fuse image with state
        observations_history = self.observation_module(observations_images_history.to(self.bet.device), observation_states_history.to(self.bet.device))
        # get the predicted logits and residuals for the observation
        preds = self.bet(observations_history, train_data=True)
        predicted_seq_action_bins_logits, predicted_seq_action_residuals = preds["seq_action_bins_logits"], preds["seq_action_residuals"]
        
        # loss
        ## bin losses
        predicted_seq_action_bins_logits = predicted_seq_action_bins_logits.reshape((-1, predicted_seq_action_bins_logits.shape[-1]))
        target_seq_action_bins = target_seq_action_bins.reshape((-1,1))
        action_bins_loss = self.focal_loss(predicted_seq_action_bins_logits, target_seq_action_bins)
        
        ## residual losses
        predicted_seq_action_residuals = predicted_seq_action_residuals.reshape(-1, predicted_seq_action_residuals.shape[-2], predicted_seq_action_residuals.shape[-1]) #B*T,BINS,ACTION_DIM
        action_residual_loss = self.mt_loss(predicted_seq_action_residuals, target_seq_action_bins, target_seq_action_residuals)
        
        ## total loss
        loss = action_bins_loss + action_residual_loss * self.residual_loss_scale
        
        return loss, action_bins_loss, action_residual_loss
    
    def get_optimizer(self, learning_rate, weight_decay, betas):
        optimizer = self.bet.create_optimizer(learning_rate, weight_decay, betas)
        optimizer.add_param_group({"params": self.observation_module.parameters()})
        return optimizer
    
    def save_checkpoint(self, epoch, val_loss):
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "bet_state_dict": self.bet.state_dict(),
            "observation_state_dict": self.observation_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss
        }, ckpt_path)
        print(f"Saved checkpoint at {ckpt_path}")

    
if __name__ == "__main__":
    import torch
    
    from src.data.dataloader import get_dataloaders
    from src.training.trainer import Trainer
    from src.models.bet import BeT
    from src.models.observations import ImageStateObservation
    
    # load data
    batch_size = 2
    sequence_length = 5
    DEVICE = "cpu"
    
    trainloader, valloader = get_dataloaders(batch_size=batch_size, h=sequence_length)
    action_collection = torch.stack(trainloader.dataset.hf_dataset['action'], dim=0)
    
    # set input dimensions
    observation_dim=514
    num_bins = 6
    action_dim = 2
    embedding_dim = 5
    
    # model hyperparameters
    num_transformer_layers = 1
    num_attention_heads = 1

    # init models
    observation_module = ImageStateObservation(use_cross_attention=False, use_states=True).to(DEVICE)
    bet = BeT(observation_dim, embedding_dim, num_transformer_layers, num_attention_heads, action_dim, num_bins, sequence_length, action_collection, device=DEVICE)
    
    # training
    epochs = 30
    trainer = Trainer(observation_module, bet, trainloader, valloader, epochs)
    
    trainer.train()