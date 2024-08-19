
import argparse

from logger.wandb_logger import WandBLogger


def initialize_wandb(args : argparse.Namespace) -> WandBLogger:
    
    if "resnet" in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name="ResNet") if WandBLogger else None
    
    elif "cnn" in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name="CNN") if WandBLogger else None
        
    elif "vit" in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name="ViT") if WandBLogger else None
    
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented, so logger cannot be created.")
    
    logger.log({
        f"augmentation": str(args.data_augmentation),
        f"scheduler": args.scheduler,
        f"learning_rate": args.learning_rate,
        f"weight_decay": args.weight_decay,
        f"gamma": args.gamma
    })
    
    return logger