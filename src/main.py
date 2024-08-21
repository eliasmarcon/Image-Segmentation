import argparse
import torch
import logging
import os

# Own modules
import utils_main

from dataset import utils_dataset
from dataset.utils_dataset import DatasetType

from logger.utils_wandb import initialize_wandb

from metrics import utils_metrics

from models import utils_models

from trainer_tester.trainer import Trainer
from trainer_tester.tester import Tester


logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)


# Set the environment variable to silence wandb
os.environ["WANDB_SILENT"] = "true"


# Set seed for reproducibility
torch.manual_seed(100)




def main(args):
    
    # Create the model run name
    model_run_name = f"{args.model_type}_{args.learning_rate}_{args.weight_decay}_{args.gamma}"
    save_dir = utils_main.create_save_dir(runs_path = args.save_model_path, model_run_name = model_run_name)
    
    
    # Create training, validation and test data
    train_data, val_data, test_data = utils_dataset.get_cityscapes_datasets(base_path = args.base_dataset_path)
    train_loader = utils_dataset.create_dataloaders(DatasetType.TRAIN, train_data, args.batch_size)
    val_loader = utils_dataset.create_dataloaders(DatasetType.VAL, val_data, args.batch_size)
    test_loader = utils_dataset.create_dataloaders(DatasetType.TEST, test_data, args.batch_size)
    
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}\n")
    
    # Create the model
    model = utils_models.create_model(model_type = args.model_type)
    model.to(device)
    
        
    ############################## Metrics ################################
    
    # Create the loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay) 
    
    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    # Create the wandb logger
    wandb_logger = initialize_wandb(args)
    
    # create metrics
    train_metric_tracker, val_metric_tracker, test_metric_tracker = utils_metrics.create_metric_trackers()
    
    
    ############################## Metrics ################################
    
    logging.info('*' * 250)
    logging.info(f"Config: Model Type - {args.model_type} | Batch Size - {args.batch_size} | "
                 f"Learing Rate - {args.learning_rate} | Weight Decay - {args.weight_decay} | Gamma - {args.gamma} |")
    logging.info('*' * 250)
    
    ############################## Training ################################
    
    ## Create Trainer
    trainer = Trainer(
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
                        args.batch_size,
                        args.val_freq,
                        args.early_stopping_patience
                    )
    
    logging.info(f"Start training...\n")
    
    trainer.train(args.num_epochs)
    
    logging.info("\nTraining finished.")  
            
    ############################## Testing ################################
        
    tester = Tester(model, loss_fn, test_loader, test_metric_tracker, wandb_logger, save_dir, device)
    
    logging.info("\nTesting the model...")
    
    for model_checkpoint in utils_main.TEST_CHECKPOINTS:
        
        tester.test(model_checkpoint)
        logging.info(f"    ---- Model checkpoint {model_checkpoint} done")

    logging.info("\nTesting finished.")
    
    # Finish the wandb logger
    wandb_logger.finish()




if __name__ == "__main__":
    
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Training')
    
    # Add an argument for the base save model path
    parser.add_argument('-s', '--save_model_path', default="./saved_models", type=str,
                        help='base path to save the model (default: ./saved_models)')
    
    # Add an argument for the base dataset path
    parser.add_argument('-d', '--base_dataset_path', default="./cityscapes_data", type=str,
                        help='base path to the dataset (default: ./cityscapes_data)')
    
    # Add an argument for specifying the number of epochs
    parser.add_argument('-e', '--num_epochs', default=80, type=int,
                        help='number of epochs to train the model (default: 80)')
    
    # Add an argument for specifying the batch_size
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size to train model (default: 16)')
    
    # Add an argument for the model type
    parser.add_argument('-m', '--model_type', default="resnet_50", type=str,
                        help='model type to train [resnet, segformer, unet] (default: resnet_50)')
    
    # Add an argument for specifying the learning rate
    parser.add_argument('-l', '--learning_rate', default=0.0001, type=float,
                        help='learning rate to train model (default: 0.0001)')
    
    # Add an argument for specifying the weight decay
    parser.add_argument('-w', '--weight_decay', default=0.0001, type=float,
                        help='weight decay to train model (default: 0.0001)')
    
    # Add an argument for gamma value
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='gamma value for learning rate scheduler (default: 0.99)')
    
    # Add an argument for specifying the val_frequency of model
    parser.add_argument('-f', '--val_freq', default=1, type=str,
                        help='validation frequency to run validation (default: 1)')
    
    # Add an argument for specifying the early stopping patience
    parser.add_argument('-p', '--early_stopping_patience', default=20, type=str,
                        help='early stopping patience to stop training (default: 20)')
   
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the train function with the parsed arguments
    main(args)