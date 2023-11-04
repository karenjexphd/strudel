# Modified version of run_strudel.py to overcome limitations:
# 1. script did not run if argument specified with -t flag (test data set)
# 2. script expected training data set and test data set in same directory

# Goal:
# 1. Train model in same way as it was trained in the tests (one-off)
# 2. Run against selected test dataset (Troy in our case) using trained model (each time)

import argparse
import os.path

# from create_database import load
import sys

import pandas
from pebble import ProcessPool

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

    # add required argument(s):
    parser.add_argument('-p', default='./data/', help='Path of the test data files.')

    args = parser.parse_args()
    print(args)

    # parse additional argument(s):
    test_file_path = args.p

    training_dataset_name = args.d
    test_dataset_name = args.t
    output_dir = args.o
    train_file_path = args.f
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

    results = []

    if test_dataset_name is None:

        # No test dataset specified. Cross-validation will be performed on the training dataset.

        # results will be output to <output_dir>/<training_dataset_name>_result.csv
        output_path = os.path.join(output_dir, training_dataset_name + '_result.csv')

        # Calculate line classification features for (training) dataset 
        #   Processes dataset's ['table_array'] (actual datafile contents) to calculate line features
        #   n.b. Pebble.ProcessPool is used to perform concurrent tasks
        
        print('Creating lstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_line_feature_vectors = pool.map(create_line_feature_vector, training_dataset).result()

        # concatenate the output from the concurrent tasks using process_pebble_results

        # line_fvs_dataset ("line strudel feature vector") is a json object containing
        #   file_name, table_id, line_number, calculated line features (f_xxx), labels
        #   (content of labels comes from the ['line_annotation'] section of the dataset)

        line_fvs_dataset = process_pebble_results(optional_line_feature_vectors)

        # Perform cross-validation for line classification 
        #   uses cross_validate() method of CrossValidation class from strudel.classification
        #   input: dataset = line_fvs_dataset, model = lstrudel
        #   uses StratifiedKFold from sklearn.model_selection on selection of data 
        #   output: results of LStrudel().fit()

        print('Cross-validating lstrudel...')
        cv = CrossValidation(n_splits=10)
        line_cv_results = cv.cross_validate(line_fvs_dataset, LStrudel.algo_name)

        # Calculate the cell classification features for the training dataset 
        #   Processes (training) dataset's ['table_array'] (the actual datafile contents) to calculate the cell features
        #   Creates cell_feature_vector as concatenation of 
        #     cell_profile_df (metadata: file_name, sheet_name, row_index, column_index) 
        #     feature_vector_df (features and their calculated values) 
        #     cell_label_df (annotations from the dataset)

        print('Creating cstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_cell_feature_vectors = pool.map(create_cell_feature_vector, training_dataset).result()

        cell_fvs_dataset = process_pebble_results(optional_cell_feature_vectors)

        # Combine (merge) the cell feature vector with the line cross-validation results to generate cfvs_dataset

        cell_fvs_dataset = combine_feature_vector(cell_fvs_dataset, line_cv_results)

        # Perform cross-validation for cell classification 
        #   uses cross_validate() method of CrossValidation class from strudel.classification
        #   input: dataset = cell_fvs_dataset, model = cstrudel
        #   uses StratifiedKFold from sklearn.model_selection on selection of data 
        #   output: results of CStrudel().fit()

        print('Cross-validating cstrduel...')
        results = cv.cross_validate(cell_fvs_dataset, CStrudel.algorithm)


    else:
        # A test dataset has been selected.
        # Train the model against the selected training dataset(s) 
        # and use it to predict line and cell classes in the test dataset

        # output of line classification will be written to <output_dir>/<test_dataset_name>_lstrudel.csv
        lines_output_path = os.path.join(output_dir, test_dataset_name + '_lstrudel.csv')
        # output of cell classification will be written to <output_dir>/<test_dataset_name>_cstrudel.csv
        cells_output_path = os.path.join(output_dir, test_dataset_name + '_cstrudel.csv')

        # <test_file_path>/<test_dataset_name>.jl.gz file will be processed
        test_dataset_path = os.path.join(test_file_path, test_dataset_name + '.jl.gz')

        # load test dataset into 'dataset' json dictionary
        test_dataset = load_data(dataset_path=test_dataset_path)

        # 1. Train model on training dataset(s)

            # Don't want to repeat the code for this part - want to use the pre-trained model
    
        # 2. Calculate the line classification features for (test) dataset 
        
        print('Creating lstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_line_feature_vectors = pool.map(create_line_feature_vector, test_dataset).result()

        # concatenate the output from the concurrent tasks using process_pebble_results
        line_fvs_dataset = process_pebble_results(optional_line_feature_vectors)

        # 4. Predict line classes in the test dataset

        #   create classify() method to replace cross_validate() method of strudel.classification CrossValidation class
        #   Will call new predict() method for Lstrudel class based on fit() that will uses RandomForestClassifier's predict_proba()

        #   input: dataset = line_fvs_dataset, model = lstrudel
        #   output: results of LStrudel().predict()

        print('Predicting line classes with lstrudel...')
        cv = CrossValidation(n_splits=10)               
        line_cv_classification = cv.classify(line_fvs_dataset, LStrudel.algo_name)  
        
        # 5. Calculate the cell classification features for the test dataset 

        #   Processes (test) dataset's ['table_array'] (the actual datafile contents) to calculate the cell features
        #   Creates cell_feature_vector as concatenation of 
        #     cell_profile_df (metadata: file_name, sheet_name, row_index, column_index) 
        #     feature_vector_df (features and their calculated values) 
        #     cell_label_df (annotations from the dataset)

        print('Creating cstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_cell_feature_vectors = pool.map(create_cell_feature_vector, test_dataset).result()

        cell_fvs_dataset = process_pebble_results(optional_cell_feature_vectors)

        # 6. Predict cell classes in the test dataset

        #   create classify() method to replace cross_validate() method of strudel.classification CrossValidation class
        #   Will call new predict() method for Cstrudel class based on fit() that will uses RandomForestClassifier's predict_proba()

        #   input: dataset = line_fvs_dataset, model = cstrudel
        #   output: results of CStrudel().predict()

        print('Predicting line classes with lstrudel...')
        cv = CrossValidation(n_splits=10)               
        cell_cv_classification = cv.classify(cell_fvs_dataset, CStrudel.algorithm)  


        # 4. write the output in csv format

        pandas.DataFrame.to_csv(results, line_cv_classification, index=False)
        pandas.DataFrame.to_csv(results, cell_cv_classification, index=False)