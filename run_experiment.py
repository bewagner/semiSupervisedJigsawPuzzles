from pathlib import Path
from jigsaw_pretrain_stl10 import pretrain_stl10
from supervised_train_pretrained_stl10 import train_classify_stl10
import time
if __name__ == '__main__':
    description = "Experiment"
    
    n_epochs = 100

    # Pretrain the jigsaw model
    pretrain_model, pretrain_logger = pretrain_stl10(number_of_epochs=n_epochs, experiment_description=description)

    # Freeze the first layers and fine tune the jigsaw model with a new top
    supervised_model, supervised_logger = train_classify_stl10(number_of_epochs=n_epochs,
                                                               train_type='classify_pretrained',
                                                               model_path=Path(pretrain_logger.name),
                                                               freeze_weights=True,
                                                               load_model=True)

    # Train a model with frozen random weights and a trainable top
    random_model, random_logger = train_classify_stl10(number_of_epochs=n_epochs,
                                                       train_type='classify_random',
                                                       load_model=False,
                                                       model_path=Path(pretrain_logger.name),
                                                       freeze_weights=True)
