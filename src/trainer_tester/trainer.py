import torch
import logging
import time
import collections

from typing import Tuple



class Trainer:
    
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_loader,
                 train_metric_tracker,
                 val_loader,
                 val_metric_tracker,
                 device,
                 wandb_logger,
                 save_dir,
                 batch_size,
                 val_freq,
                 early_stopping_patience) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.train_metric_tracker = train_metric_tracker
        self.val_loader = val_loader
        self.val_metric_tracker = val_metric_tracker
        self.device = device
        self.wandb_logger = wandb_logger
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.early_stopping_patience = early_stopping_patience
        
        
        # Initialize other variables
        self.num_train_data = len(train_loader.dataset)
        self.num_val_data = len(val_loader.dataset)
        self.best_val_loss = float('inf')
        self.best_mIoU = 0.0     
            

    def train(self, epochs) -> None:
        
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_freq parameter, validation is not performed every epoch.
        """     
                
        # Initialize patience counter for early stopping
        patience_counter = 0
        
        # Training loop
        for epoch_idx in range(1, epochs + 1):
            
            starting_time = time.time()

            # Train for one epoch
            train_loss, train_dice_score, train_mIoU = self._train_epoch()

            # check if validation should be performed
            if epoch_idx % self.val_freq == 0:

                # validate
                val_loss, val_dice_score, val_mIoU = self._val_epoch()

                # Check if the current per class accuracy and validation loss is better than the best
                if val_mIoU > self.best_mIoU:
                    self.best_mIoU = val_mIoU

                    # Save the model as the best per class accuracy model
                    self.model.save(self.save_dir, f"best_val_mIoU")
                                
                # Check if the current validation loss is better than the best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0  # Reset patience counter

                    # Save the model as the best validation loss model
                    self.model.save(self.save_dir, f"best_val_loss")
                else:
                    patience_counter += 1  # Increment patience counter
                    
                # If the patience counter reaches the threshold, stop training early
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered. Training stopped after {epoch_idx} epochs.")
                    break
            
            # Step the learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            # Log the current metrics
            logging.info(
                f"Epoch {epoch_idx:2}/{epochs} completed in {(time.time() - starting_time):4.2f} seconds | "
                f"Train Loss: {train_loss:6.4f} | Train Dice Score: {train_dice_score:6.4f} | Train mIoU: {train_mIoU:6.4f} | "
                f"Val Loss: {val_loss:6.4f} | Val Dice Score: {val_dice_score:6.4f} | Val mIoU: {val_mIoU:6.4f}"
            )
               
            if self.wandb_logger:
                
                self.wandb_logger.log({
                    f"train_loss": train_loss,
                    f"train_dice_score": train_dice_score,
                    f"train_mIoU": train_mIoU,
                    f"val_loss": val_loss if (epoch_idx + 1) % self.val_freq == 0 else None,
                    f"val_dice_score": val_dice_score if (epoch_idx + 1) % self.val_freq == 0 else None,
                    f"val_mIoU": val_mIoU if (epoch_idx + 1) % self.val_freq == 0 else None,
                })
                            
        # save the final model
        self.model.save(self.save_dir, f"terminal")
    
    
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.
        """
        
        # Reset the training metric
        self.train_metric_tracker.reset()
        
        # Initialize the epoch loss
        epoch_loss = 0.0
        
        # Training loop
        self.model.train()
        
        # Loop over the training data set
        for _, (inputs, targets) in enumerate(self.train_loader):
            
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(self.device), targets.to(self.device).long()
            # Squueze the target tensor if it has a channel dimension [BatchSize, 1, H, W] -> [BatchSize, H, W]
            targets = targets.squeeze(1)
            
            # Get the batch size
            batch_size = inputs.shape[0]
            
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Calculate the loss
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # Adjust learning rate
            self.optimizer.step()

            # Update the loss
            epoch_loss += ( loss.item() * batch_size )
            
            # Update the training metric [DiceScore, IntersectOverUnion]
            self.train_metric_tracker.update(outputs.cpu(), targets.cpu())
                    
                    
        self.lr_scheduler.step()

        # Calculate average loss for the epoch
        epoch_loss /= self.num_train_data
        
        # Calculate training metrics       
        train_metrics = self.train_metric_tracker.calculate_metric()        
        dice_score, mIoU = train_metrics[0], train_metrics[1]
        
         
        return epoch_loss, dice_score, mIoU
    
    
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        
        # Reset the validation metric
        self.val_metric_tracker.reset()
        
        # Initialize the epoch loss
        epoch_loss = 0.0
        
        # Validation loop
        self.model.eval()
        
        with torch.no_grad():
            
            for _, (inputs, targets) in enumerate(self.val_loader):
                
                # Move inputs and targets to the specified device
                inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                # Squueze the target tensor if it has a channel dimension [BatchSize, 1, H, W] -> [BatchSize, H, W]
                targets = targets.squeeze(1)
                
                batch_size = inputs.shape[0]

                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']
                else:
                    targets = targets.squeeze(1)
                    
                # Calculate the loss
                loss = self.loss_fn(outputs, targets)
                
                # Update the loss
                epoch_loss += ( loss.item() * batch_size )

                # Update the validation metric [DiceScore, IntersectOverUnion]
                self.val_metric_tracker.update(outputs.cpu(), targets.cpu())

        # Calculate average loss for the epoch
        epoch_loss /= self.num_val_data
        
        # Calculate validation metrics
        val_metrics = self.val_metric_tracker.calculate_metric()
        dice_score, mIoU = val_metrics[0], val_metrics[1]
        

        return epoch_loss, dice_score, mIoU
