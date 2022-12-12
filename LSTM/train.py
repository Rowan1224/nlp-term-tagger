

def train_step(model, dataloader, optimizer, validation=False):

    
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