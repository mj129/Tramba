def adjust_learning_rate(optimizer, epoch, decay_epochs, base_lr, decay_factors):
    """Adjusts the learning rate of the optimizer.
    Args:
        optimizer: The optimizer to adjust.
        epoch: The current epoch number.
        decay_epochs: List of epochs at which to decay the learning rate.
        base_lr: The initial learning rate.
        decay_factors: List of factors by which to decay the learning rate.
    """
    assert len(decay_epochs) == len(decay_factors)
    if epoch in decay_epochs:
        index = decay_epochs.index(epoch)
        optimizer.param_groups[1]['lr'] = base_lr * decay_factors[index]
        optimizer.param_groups[0]['lr'] = base_lr * decay_factors[index] * 0.1
        print(f'Learning rate adjusted to {optimizer.param_groups[1]["lr"]}!!')

    return optimizer.param_groups[1]["lr"]
