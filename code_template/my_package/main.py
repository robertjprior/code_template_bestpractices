#training/optimization pipelines

import pandas as pd
from pathlib import Path
from argparse import Namespace
import warnings

from config import config
from . import data
from my_package import utils
from config.config import logger
warnings.filterwarnings("ignore")



def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info("✅ Saved data!")


import json
from my_package import train, utils

"""
#original train_model before optimize
def train_model(args_fp): #This function calls for a train() function inside our train.py script:
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    logger.info(json.dumps(performance, indent=2))"""
import joblib
import tempfile

def train_model(args_fp, experiment_name, run_name):
    """Train a model given arguments. wrapper for train function from train.py \
        such that this will train a final model after optimization has been done \
        and use mlflow to log metrics. MLFlow logs from optimize but I believe only \
        some very high level metrics. """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
    """There's a lot more happening inside our train_model() function but it's necessary in order to store all the metrics, parameters and artifacts. We're also going to update the train() function inside train.py so that the intermediate metrics are captured:"""


import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback
#optimize params setup in train.py under objective function
def optimize(args, study_name, num_trials):
    """Optimize hyperparameters."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1") #passing MLFlow tracking uri to optuna MLflowCallback to pass info 
    #https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html#optuna.integration.MLflowCallback
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback]) #replaces manually adding trail with custom params setups etc https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

"""optimize output when run from run.py:
A new study created in memory with name: optimization
...
Best value (f1): 0.8497010532479641
Best hyperparameters: {
    "analyzer": "char_wb",
    "ngram_max_range": 6,
    "learning_rate": 0.8616849162496086,
    "power_t": 0.21283622300887173
}"""


from my_package import data, predict, train, utils
#KEY
def predict_tag(text, run_id=None):
    """Predict tag for text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction

def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance
    }