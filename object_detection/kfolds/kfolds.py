def kfolds(train_dataset, num_folds, num_epochs, model):
    # Configuration options
    loss_function = torch.nn.CrossEntropyLoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits = num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, sampler = train_subsampler)
        val_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, sampler= val_subsampler)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr= 0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # alternative optimizer
        #optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            # train for one epoch, printing every 10 iterations
            engine.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            engine.evaluate(model, val_data_loader, device = device)
            
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
