from utils.models import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str="[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"run_logs.log"),level=logging.INFO,format=logging_str,filemode='a')

def main(df,ETA,EPOCHS):
    

    df = pd.DataFrame(df)

    logging.info(f"this is dataframe:{df}")

    X,y = prepare_data(df)

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename="and.model")
    save_plot(df, "and.png", model)

if __name__ == "__main__":
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}   
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info("start trainng")
        main(AND,ETA,EPOCHS)
        logging.info("end trainng")
    except Exception as e:
        logging.exception(e)
        raise e