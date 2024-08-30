import torch
import wandb

import logger.config_wandb as config_wandb

from typing import Optional, Dict, Any
from wandb.wandb_run import Run


class WandBLogger:
    """
    A class for logging experiments using Weights & Biases (WandB).

    Attributes:
        enabled (bool): Flag to enable or disable logging.
        run (Optional[wandb.wandb_run.Run]): The WandB run instance.
        run_name (Optional[str]): Custom name for the WandB run.
    """

    def __init__(self, 
                 enabled: bool = True, 
                 model: Optional[torch.nn.Module] = None, 
                 run_name: Optional[str] = None,
                 group_name: Optional[str] = None,
                 project_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the WandBLogger instance.

        Args:
            enabled (bool): Whether to enable WandB logging.
            model (Optional[torch.nn.Module]): The model to watch.
            run_name (Optional[str]): Custom name for the WandB run.
            group_name (Optional[str]): Group name for the WandB run.
            project_name (Optional[str]): Project name for the WandB run.
            config (Optional[Dict[str, Any]]): Configuration dictionary for the run.
        """
        self.enabled = enabled
        self.run: Optional[Run] = None
        self.run_name = run_name

        if self.enabled:
            try:
                wandb.login(key=config_wandb.WANDB_KEY, relogin=True)
                
                self.run = wandb.init(
                    entity=config_wandb.WANDB_ENTITY,
                    project=project_name or config_wandb.WANDB_PROJECT,
                    group=group_name,
                    name=run_name or wandb.util.generate_id(),
                    config=config
                )

                if model is not None:
                    self.watch(model)
                    
            except Exception as e:
                print(f"Failed to initialize WandB: {e}")
                self.enabled = False

    def watch(self, model: torch.nn.Module, log_freq: int = 1, log_graph: bool = True) -> None:
        """
        Watches the model to log gradients, parameters, and optionally the model graph.

        Args:
            model (torch.nn.Module): The model to watch.
            log_freq (int): Frequency of logging gradients and parameters.
            log_graph (bool): Whether to log the model graph.
        """
        if self.enabled:
            try:
                wandb.watch(model, log="all", log_freq=log_freq, log_graph=log_graph)
            except Exception as e:
                print(f"Failed to watch model: {e}")

    def log(self, log_dict: Dict[str, Any], commit: bool = True, step: Optional[int] = None) -> None:
        """
        Logs metrics to WandB.

        Args:
            log_dict (Dict[str, Any]): Dictionary of metrics to log.
            commit (bool): Whether to commit the log entry.
            step (Optional[int]): Step number for logging.
        """
        if self.enabled:
            try:
                wandb.log(log_dict, commit=commit, step=step)
            except Exception as e:
                print(f"Failed to log metrics: {e}")

    def finish(self) -> None:
        """
        Finishes the WandB run.
        """
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Failed to finish WandB run: {e}")