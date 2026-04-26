def eval_step(model, images, labels, loss_func, device, return_pred=False):
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = loss_func(outputs, labels)

    _, pred = outputs.max(1)
    correct = (pred == labels).sum().item()

    if return_pred:
        return loss.item(), correct, pred
    else:
        return loss.item(), correct