import logging
from d2l import torch as d2l

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.config_space import loguniform, randint
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import RandomSearch

def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from syne_tune import Reporter
    from d2l import torch as d2l

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter()
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize the state of Trainer
            trainer.fit(model=model, data=data)
        else:
            trainer.fit_epoch()
        validation_error = trainer.validation_error().cpu().detach().numpy()
        report(epoch=epoch, validation_error=float(validation_error))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    n_workers = 2  # Needs to be <= the number of available GPUs

    max_wallclock_time = 12 * 60  # 12 minutes

    mode = "min"
    metric = "validation_error"

    config_space = {
        "learning_rate": loguniform(1e-2, 1),
        "batch_size": randint(32, 256),
        "max_epochs": 10,
    }
    initial_config = {
        "learning_rate": 0.1,
        "batch_size": 128,
    }

    trial_backend = PythonBackend(
        tune_function=hpo_objective_lenet_synetune,
        config_space=config_space,
    )

    scheduler = RandomSearch(
        config_space,
        metric=metric,
        mode=mode,
        points_to_evaluate=[initial_config],
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        print_update_interval=int(max_wallclock_time * 0.6),
    )

    tuner.run()

    d2l.set_figsize()
    tuning_experiment = load_experiment(tuner.name)
    tuning_experiment.plot()

    d2l.set_figsize([6, 2.5])
    results = tuning_experiment.results

    for trial_id in results.trial_id.unique():
        df = results[results["trial_id"] == trial_id]
        d2l.plt.plot(
            df["st_tuner_time"],
            df["validation_error"],
            marker="o"
        )

    d2l.plt.xlabel("wall-clock time")
    d2l.plt.ylabel("objective function")

    