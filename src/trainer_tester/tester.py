import torch
import os



class Tester:
    
    def __init__(self, model, loss_fn, test_loader, test_metric, wandb_logger, save_dir, device) -> None:
        
        """
        Initialize the Tester class.

        Args:
        - model (torch.nn.Module): The model to test.
        - loss_fn (torch.nn.Module): The loss function.
        - test_loader (DataLoader): The DataLoader for the test dataset.
        - test_metric: The metric to evaluate the test performance.
        - wandb_logger: Logger to record the results.
        - save_dir (str): Directory to save results.
        - device (torch.device): Device to run the test on.
        """
        
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.test_metric = test_metric
        self.wandb_logger = wandb_logger
        self.save_dir = save_dir
        self.device = device
                
    
    def test(self, test_checkpoint_type) -> None:
        
        """
        Test the model on the test data set and return the loss, mean accuracy and mean per class accuracy.
        
        test_loader: The test data set to test the model on.
        """

        # Load the model checkpoint
        self.model.load(os.path.join(self.save_dir, test_checkpoint_type))
        
        # Initialize the test loss, true and predicted labels, and the test metric
        self.test_metric.reset()
        
        # Initialize the test loss
        test_loss = 0.0

        # Set the model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
        
            for inputs, targets in self.test_loader:

                inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                # Squueze the target tensor if it has a channel dimension [BatchSize, 1, H, W] -> [BatchSize, H, W]
                targets = targets.squeeze(1)
                
                # Get the batch size
                batch_size = inputs.shape[0]
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                        
                # Update the loss
                test_loss += ( loss.item() * batch_size )
                
                # Update the test metric
                self.test_metric.update(outputs.cpu(), targets.cpu())


        # Calculate average loss for the test set
        test_loss /= len(self.test_loader.dataset)
        
        # Calculate test metrics
        test_metric = self.test_metric.calculate_metric()
        dice_score, mIoU = test_metric[0], test_metric[1]
        
        
        # Log test metrics
        if self.wandb_logger:
            
            self.wandb_logger.log({
                f"test_loss_{test_checkpoint_type}": test_loss,
                f"test_dice_score_{test_checkpoint_type}": dice_score,
                f"test_mIoU_{test_checkpoint_type}": mIoU
            })