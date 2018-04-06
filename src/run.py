import json
import pandas as pd
from time import strftime

from model import CookingModel


def main():
    print("Loading train and test data...", end=' ')
    with open('../_data/train.json', 'r') as f:
        train = json.load(f)
    with open('../_data/test.json', 'r') as f:
        test = json.load(f)
    print("Done.")

    print("Fitting model...", end=' ')
    cmodel = CookingModel()
    cmodel.fit(train, test)
    print("Done.")

    print("Predicting...", end=' ')
    df_pred = cmodel.predict(test, to_kaggle=True)
    print("Done.")

    save_path = '../_data/{}_submission.csv'\
                .format(strftime('%y%m%d_%H%M%S'))
    print("Saving to {}...".format(save_path), end=' ')
    df_pred.to_csv(save_path, index=False)
    print("Done.")

    print("All done.")

if __name__ == "__main__":
    main()
