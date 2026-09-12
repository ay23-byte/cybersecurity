import sys
import os
import pandas as pd
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

MODEL_PATH = os.path.join("finals_model", "model.pkl")
PREPROCESSOR_PATH = os.path.join("finals_model", "preprocessor.pkl")
TEST_CSV = os.path.join("valid_data", "test.csv")

if __name__ == '__main__':
    try:
        print('Loading model...')
        model = load_object(MODEL_PATH)
        preproc = load_object(PREPROCESSOR_PATH)
        nm = NetworkModel(preprocessor=preproc, model=model)
        print('Reading test CSV...')
        df = pd.read_csv(TEST_CSV)
        print('First row:')
        print(df.iloc[0])
        print('Running predict...')
        preds = nm.predict(df)
        print('Predictions:', preds)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('Error:', e)
