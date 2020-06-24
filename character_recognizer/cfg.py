# Image parameters
maintain_aspect_ratio = False
activation = "relu"  # {relu, tanh}
optimizer = "rmsprop"  # {rmsprop, nadam, adam, adagrad}

# Regularization
batch_norm = True

dropout = True
drop_hidden = 0.25  # {0.2, 0.25, 0.5}
drop_output = 0.5  # {0.2, 0.25, 0.5}

early_stopping = True
monitor = "loss"  # {loss, val_loss}
patience = 3  # {1, 2, 3, 4}

# Hyperparameters
epochs = 30
init_learning_rate = 0.01
batch_size = 32