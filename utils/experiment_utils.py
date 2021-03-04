from tqdm import tqdm

def evaluate(epoch, loader, model, evaluator, loss, label='VAL'):
    model.eval()
    for X, t in tqdm(loader):
        preds = model(X)
        batch_loss = loss(preds, t)
        evaluator.update(
            loss=batch_loss.cpu().detach().numpy(), 
            probabilities=preds.cpu().detach().numpy(), 
            targets=t.cpu().detach().numpy()
        )
        # break
    evaluator.log_metrics()
    new_metrics = evaluator.next_epoch()
    return new_metrics

def train(epoch, loader, model, evaluator, optimizer, loss, scheduler=None):
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
        if scheduler:
            scheduler.step()
        # break
    evaluator.log_metrics()
    new_metrics = evaluator.next_epoch()
    return new_metrics