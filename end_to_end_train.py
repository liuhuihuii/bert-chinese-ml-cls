import sys
import os
from utils import generate_concat_file, params_train, load_data, produce_model

# ## global var
# competency_question_id = sys.argv[1]  # for example, '20_4'
# data_path = sys.argv[2]  # relative or absolute path, the folder where txt data stores
# print("Successfully set global variables.")

competency_question_id = "1_3"
data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
has_best_param = False             # need to train to find the best param


def main():
    if not os.path.exists(data_path + '/optimal_concat_%s.txt' % competency_question_id):
        generate_concat_file(data_path, competency_question_id)

    print('optimal_concat_%s.txt already exists, enter training.....' % competency_question_id)

    if not has_best_param:
        data_x, y = params_train(data_path, competency_question_id)
    else:
        # just load data
        data_x, y = load_data(data_path, competency_question_id)

    # ask user for selected ttsplit random_state, n_estimators, rf random_state
    nnd = int(input("Enter selected train-test split random state(nnd):"))
    i = int(input("Enter selected n_estimators for random forest(i):"))
    j = int(input("Enter selected random state for random forest(j):"))
    produce_model(data_x, y, data_path, competency_question_id, nnd, i, j)


if __name__ == '__main__':
    main()
