"""
Train the model

Train the model using the sample data

By: Julian Bolivar
Version: 1.0.0
Date:  2023-05-29
Revision 1.0.0 (2023-05-29): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import os
import platform

# ML Imports
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from data import process_data
from model import train_model, compute_model_metrics, inference, compute_slices
from model import compute_confusion_matrix

# Main Logger
logHandler = None
logger = None
logLevel_ = logging.INFO


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="model_trainner",
                            description="Train the model")
    parser.add_argument("-d",
        "--data", 
        type=str,
        help="Data Path",
        default="../../data/census_clean.csv",
        required=False
    )
    parser.add_argument("-m",
        "--modelpath", 
        type=str,
        help="Path where the model is saved",
        default="../../model",
        required=False
    )
    parser.add_argument("-s",
        "--slicefile", 
        type=str,
        help="File where the slices test result are saved",
        default="./slice_output.txt",
        required=False
    )
    return parser.parse_args()


def remove_if_exists(filename):
    """
    Delete a file if it exists.
    input:
        filename: str - path to the file to be removed
    output:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)


def main(args):
    """
     Run the main function

     args: command line arguments
    """

    global logger

    # load in the data.
    datapath = args.data
    data = pd.read_csv(datapath)

    # separate train and test slices
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Proces the test data with the process_data function.
    # Set train flag = False - We use the encoding from the train set
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # check if trained model already exists
    savepath = args.modelpath
    filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']
    
    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(savepath,filename[0])):
            model = pickle.load(open(os.path.join(savepath,filename[0]), 'rb'))
            encoder = pickle.load(open(os.path.join(savepath,filename[1]), 'rb'))
            lb = pickle.load(open(os.path.join(savepath,filename[2]), 'rb'))
    
    # Else Train and save a model.
    else:
        model = train_model(X_train, y_train)
        # save model  to disk 
        pickle.dump(model, open(os.path.join(savepath,filename[0]), 'wb'))
        pickle.dump(encoder, open(os.path.join(savepath,filename[1]), 'wb'))
        pickle.dump(lb, open(os.path.join(savepath,filename[2]), 'wb'))
        logger.info(f"Model saved to disk: {savepath}")
    
    
    # evaluate trained model on test set
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    
    logger.info(f"Classification target labels: {list(lb.classes_)}")
    logger.info(
        f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")
    
    cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))
    
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Compute performance on slices for categorical features
    # save results in a new txt file
    slice_savepath = args.slicefile
    remove_if_exists(slice_savepath)
    
    # iterate through the categorical features and save results to log and txt file
    for feature in cat_features:
        performance_df = compute_slices(test, feature, y_test, preds)
        performance_df.to_csv(slice_savepath,  mode='a', index=False)
        logger.info(f"Performance on slice {feature}")
        logger.info(performance_df)


if __name__ == '__main__':

    computer_name = platform.node()
    script_name = "model_trainner"
    loggPath = os.path.join(".","log")
    if not os.path.isdir(loggPath):
        try:
            # mode forced due security
            mode = 0o770
            os.mkdir(loggPath, mode=mode)
        except OSError as error:
            print(error)
            exit(-1)
    LogFileName = os.path.join(loggPath, 
                               computer_name + '-' + script_name + '.log')
    # Configure the logger
    logger = log.getLogger(script_name)  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler(LogFileName, 
                                                  maxBytes=10485760, 
                                                  backupCount=10)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                datefmt='%Y/%m/%d %H:%M:%S')
    logHandler.setFormatter(logFormatter)
    # Add handler to logger
    if 'logHandler' in globals():
        logger.addHandler(logHandler)
    else:
        logger.debug("logHandler NOT defined (001)")
    # Set Logger Lever
    logger.setLevel(logLevel_)
    # Start Running
    logger.debug("Running... (002)")
    args = build_argparser()
    main(args)
    logger.debug("Finished. (003)")
