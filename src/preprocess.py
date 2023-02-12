import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import logging
import os

if (len(sys.argv) != 3):
    logging.error("Arguments Error \n")
    sys.stderr.write("Usage: \t python3 preprocess.py data/ output/")
    sys.exit(1)

# arguments

input_file = sys.argv[1]
output_loc = sys.argv[2]


def build_csv(input, output):

    dftrain = pd.read_csv(input)

    # feature engineering

    # a feature which indicates if the age is =>50 or not
    dftrain['isGreater50'] = (dftrain['age']/dftrain['age']
                              ).where(dftrain['age'] >= 50, 0)

    # a feature which indicates if the sample smokes or fromerly smoked at age 50 or greater
    dftrain['Greater50_smoked'] = (dftrain['age']/dftrain['age']).where(((dftrain['age'] >= 50) & (
        (dftrain['smoking_status'] == 'smokes') | (dftrain['smoking_status'] == 'formerly smoked'))), 0)

    # a feature indicates if the avg glucose level is greater 96 or not
    dftrain['AvgGlucose96'] = (
        dftrain['age']/dftrain['age']).where(dftrain['avg_glucose_level'] > 96, 0)

    # label encoding

    le = LabelEncoder()

    cat_columns = [col for col in dftrain.columns if dftrain[col].dtype == "O"]

    for col in cat_columns:
        dftrain[col] = le.fit_transform(dftrain[col])

    # saving the file in feature folder
    if not (os.path.exists(output)):
        os.mkdir(output)

    output_loc = os.path.join(output, "preprocessed.csv")
    dftrain.to_csv(output_loc, index=False)
    sys.stderr.write(f"File preprocessed to {output_loc} dir\n")


if __name__ == '__main__':
    build_csv(input_file, output_loc)
