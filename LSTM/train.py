

def train_step(model, dataloader, optimizer, validation=False):
    """
    For each batch in the dataloader, we get the input and label, zero the gradients, compute the output
    and loss, backpropagate the loss, and update the weights
    
    :param model: the model we are training
    :param dataloader: the dataloader for the dataset we want to train on
    :param optimizer: the optimizer used to update the weights of the model
    :param validation: whether we are training or validating, defaults to False (optional)
    :return: The total loss over the entire epoch
    """

    
    total_loss = 0
        
    # Iterate over batches using the dataloader
    for batch_index, batch in enumerate(dataloader):
        
        label = batch['labels']
        inp = batch['token_ids']
        optimizer.zero_grad()
        
        out = model.forward(inp)
        loss = model.loss_fn(out, label, len(batch))
        if not validation:
            loss.backward()
            optimizer.step()

        total_loss += (loss.item())
        
        # val_loss_epoch += loss_fn(out,label).item()


    # At the end of each epoch, record and display the loss over all batches in train and val set
    total_loss= total_loss/len(dataloader)
    # val_loss_epoch = val_loss_epoch/len(train_dataloader)
    

        
    return total_loss