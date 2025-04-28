import optuna
from sklearn.metrics import matthews_corrcoef
from torch import optim, nn

from deeplearning.GRU import GRU
from deeplearning.CNN import CNN

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )

def objective(trial, train_loader, test_loader, args):
    # Generate the model
    device = args['device']
    epochs = args['epochs']

    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 1000)
    p = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    args['n_layers'] = n_layers
    args['hidden_dim'] = hidden_dim
    args['dropout'] = p
    args['lr'] = lr

    dlmodel = CNN(args)
    # dlmodel.update_model(args)

    # Generate the optimizers
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = optim.Adam(dlmodel.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # Training of the model
    for epoch in range(epochs):
        # model.train()
        for X, y in train_loader:
            X = X.float().to(device)
            y = y.long().to(device)

            optimizer.zero_grad()
            y_hat = dlmodel(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        # Validation of the model
        predict_y, true_y = dlmodel.predict_CNN(dlmodel, test_loader)
        mcc = matthews_corrcoef(true_y, predict_y)
        trial.report(mcc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mcc

def tune_model(train_loader, test_loader, args):
    # deep learning model
    study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False,
                                directions=['maximize'], sampler=optuna.samplers.TPESampler())

    study.optimize(lambda trial: objective(trial, train_loader, test_loader, args), n_trials=30, n_jobs=-1)  # callbacks=[logging_callback],

    args.update(study.best_params)
    # args = args.to(args['device'])
    return args

