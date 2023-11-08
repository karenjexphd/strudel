# run_strudel.py

# Train the Strudel model on the specified training data set(s) (saus by default)
# Cross-validate against the training dataset if no test data set specified
# Use trained model to generate predictions for test data set if one is specified 

# Script modified by KJ to overcome limitations of original script:

  # script did not run if test data set was specified (via -t flag)
  # script expected training data set and test data set in same directory

import argparse
import os.path

import sys

import pandas
from pebble import ProcessPool
# n.b. pebble.ProcessPool is used to perform concurrent tasks, and process_pebble_results is used to concatenate the output

from strudel.cstrudel import CStrudel, create_cell_feature_vector, combine_feature_vector
from strudel.lstrudel import LStrudel, create_line_feature_vector
from strudel.classification import CrossValidation
from strudel.data import load_data
from strudel.utility import process_pebble_results

sys.path.insert(0, 'strudel')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser for line & cell classification script.')
    parser.add_argument('-d', default='saus', help="Specify the training dataset. Default 'saus' or 'all' uses all data-sets.")
    parser.add_argument('-t', help='Specify the test dataset. If not given, apply cross-validation on the training dataset.')
    parser.add_argument('-f', default='./data/', help='Path of the training data files.')
    parser.add_argument('-o', default='./result/', help='Path where all experiment result files are stored.')
    parser.add_argument('-i', default=0, type=int, help='The number of this iteration.')
    parser.add_argument('-p', default='./data/', help='Path of the test data files.')    # KJ: parameter added for location of test data

    args = parser.parse_args()
    print(args)

    training_dataset_name = args.d
    test_dataset_name = args.t
    output_dir = args.o
    train_file_path = args.f
    test_file_path = args.p
    n_iter = str(args.i)

    max_workers = int(os.cpu_count() * 0.5)
    max_tasks = 10

    true_labels = None
    pred_labels = None
    algorithm_type = None
    description = None

    # <train_file_path>/<training_dataset_name>.jl.gz file will be used for training
    train_dataset_path = os.path.join(train_file_path, training_dataset_name + '.jl.gz')

    # load training dataset into 'dataset' json dictionary
    training_dataset = load_data(dataset_path=train_dataset_path)

    # cross-validation results will be output to <output_dir>/<training_dataset_name>_result.csv
    output_path = os.path.join(output_dir, training_dataset_name + '_result.csv')

    if test_dataset_name is not None:

        # output of line classification will be written to <output_dir>/<test_dataset_name>_lstrudel.csv
        lines_output_path = os.path.join(output_dir, test_dataset_name + '_lstrudel.csv')

        # output of cell classification will be written to <output_dir>/<test_dataset_name>_cstrudel.csv
        cells_output_path = os.path.join(output_dir, test_dataset_name + '_cstrudel.csv')

        # <test_file_path>/<test_dataset_name>.jl.gz file will be processed
        test_dataset_path = os.path.join(test_file_path, test_dataset_name + '.jl.gz')

        # load test dataset into 'dataset' json dictionary
        test_dataset = load_data(dataset_path=test_dataset_path)

    results = []

    #    Calculate line classification features for (training) dataset by processing dataset's ['table_array']
    print('Creating lstrudel feature vector...')
    with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
        optional_line_feature_vectors = pool.map(create_line_feature_vector, training_dataset).result()

    train_line_fvs_dataset = process_pebble_results(optional_line_feature_vectors)

    #    N.B. line_fvs_dataset ("line strudel feature vector") is a json object containing
    #         file_name, table_id, line_number, calculated line features (f_xxx), labels
    #         (content of labels comes from the ['line_annotation'] section of the dataset)

    #    Calculate cell classification features for (training) dataset by processing dataset's ['table_array']
    print('Creating cstrudel feature vector...')
    with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
        optional_cell_feature_vectors = pool.map(create_cell_feature_vector, training_dataset).result()

    train_cell_fvs_dataset = process_pebble_results(optional_cell_feature_vectors)

    #    N.B. the cell_fvs_dataset cell_feature_vector is a concatenation of 
    #      - cell_profile_df   (metadata: file_name, sheet_name, row_index, column_index) 
    #      - feature_vector_df (features and their calculated values) 
    #      - cell_label_df     (annotations from the dataset)

    if test_dataset_name is None:

        # Train model and perform cross-validation on the training dataset.

        # Perform cross-validation for line classification 
        print('Cross-validating lstrudel...')
        cv = CrossValidation(n_splits=10)
        line_cv_results = cv.cross_validate(train_line_fvs_dataset, LStrudel.algo_name)        # results of LStrudel().fit()

        # Combine (merge) the cell feature vector with the line cross-validation results to generate cfvs_dataset
        train_cell_fvs_dataset = combine_feature_vector(train_cell_fvs_dataset, line_cv_results)

        # Perform cross-validation for cell classification 
        print('Cross-validating cstrduel...')
        results = cv.cross_validate(train_cell_fvs_dataset, CStrudel.algorithm)        # results of CStrudel().fit()

        # ** CHECK THIS ! NEED LINE FROM THE ORIGINAL FILE **
        pandas.DataFrame.to_csv(results, train_cell_fvs_dataset, index=False)

    else:   # train model on training dataset and predict line and cell classes in specified test dataset
    
        # Calculate line classification features for (test) dataset
        print('Creating lstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_line_feature_vectors = pool.map(create_line_feature_vector, test_dataset).result()

        test_line_fvs_dataset = process_pebble_results(optional_line_feature_vectors)
        
        # Calculate cell classification features for (test) dataset
        print('Creating cstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_cell_feature_vectors = pool.map(create_cell_feature_vector, test_dataset).result()

        test_cell_fvs_dataset = process_pebble_results(optional_cell_feature_vectors)

        # predict line classification
        print('Predicting line classes with lstrudel...')
        cv = CrossValidation(n_splits=10)
        line_cv_classification = cv.train_classify(train_line_fvs_dataset, test_line_fvs_dataset, LStrudel.algo_name)

        # predict cell classification
        print('Predicting line classes with lstrudel...')
        cv = CrossValidation(n_splits=10)
        cell_cv_classification = cv.train_classify(train_cell_fvs_dataset, test_cell_fvs_dataset, CStrudel.algorithm)

        # 4. write the output in csv format

        pandas.DataFrame.to_csv(results, line_cv_classification, index=False)
        pandas.DataFrame.to_csv(results, cell_cv_classification, index=False)