def evaluate(epoch, loader, model, evaluator, label='VAL'):
    model.eval()
    for X, t in tqdm(loader):
        preds = model(X)
        batch_loss = loss(preds, t)
        evaluator.update(
            loss=batch_loss.cpu().detach().numpy(), 
            probabilities=preds.cpu().detach().numpy(), 
            targets=t.cpu().detach().numpy()
        )
    evaluator.log_metrics(epoch)
    evaluator.next_epoch()

def train(epoch, loader, model, evaluator, optimizer):
    model.train()
    for X, t in tqdm(loader):
        optimizer.zero_grad()
        preds = model(X)
        batch_loss = loss(preds, t)
        evaluator.update(
            loss=batch_loss.cpu().detach().numpy(), 
            probabilities=preds.cpu().detach().numpy(), 
            targets=t.cpu().detach().numpy()
        )
        # update weights
        batch_loss.backward()
        optimizer.step()
    evaluator.log_metrics(epoch)
    evaluator.next_epoch()