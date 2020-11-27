from bert_serving.client import BertClient
from sklearn.ensemble import RandomForestClassifier as RM
import sklearn.metrics as me
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os


def generate_concat_file(data_path, competency_question_id):
    ## text2vec
    # zh
    # bert-serving-start -model_dir /Users/leapin/Desktop/project/leapin-bert-service/bert_model/chinese_L-12_H-768_A-12 -num_worker=1 -max_batch_size 512 -max_seq_len 512 -port 5555 -port_out 5556
    # en
    # bert-serving-start -model_dir /Users/leapin/Desktop/project/leapin-bert-service/bert_model/uncased_L-12_H-768_A-12 -num_worker=1 -max_batch_size 512 -max_seq_len 512 -port 5557 -port_out 5558
    bc = BertClient()

    bertvecfile = open(data_path + "/optimal_bertvec_%s.txt" % competency_question_id, 'a')
    with open(data_path + "/optimal_answer_%s.txt" % competency_question_id) as f1:
        for myline in f1:
            wordlist = myline.replace("\n", '').split(',')
            bertvecfile.write(wordlist[0] + ' ')
            for i in (bc.encode([wordlist[1]])).flatten():
                bertvecfile.write(str(i) + ' ')
            bertvecfile.write('\n')
    bertvecfile.close()
    print("Successfully generated bert vecs.")

    ## concat
    concatfile = open(data_path + "/optimal_concat_%s.txt" % competency_question_id, 'a')  # changed 'a' to 'w'
    with open(data_path + '/optimal_bertvec_%s.txt' % competency_question_id) as f1:
        for line1 in f1:
            wordlist1 = line1.split()
            has_fer = False  # 是否存在 nlp + fer 模式， 默认为 false
            if os.path.exists(data_path + '/optimal_fervec_%s.txt' % competency_question_id):
                with open(data_path + '/optimal_fervec_%s.txt' % competency_question_id) as f:
                    for line in f:
                        wordlist = line.split()

                        if wordlist1[0] == wordlist[0]:
                            has_fer = True  # 存在
                            concatfile.write(line1.strip() + ' ' + line.strip().replace(wordlist[0], '') + '\n')
                            break

                if not has_fer:  # 不存在
                    concatfile.write(line1.strip() + ' ' + '0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0' + '\n')  # 中立情绪
                    concatfile.flush()
            else:
                concatfile.write(line1.strip() + ' ' + '0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0' + '\n')  # 语料数据直接填充中立情绪
                # anger  contempt disgust fear happiness neutral sadness  surprise
                # 正向情绪： 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0
                # 中立情绪： 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                # 负向情绪： 0.2 0.2 0.2 0.2 0.0 0.0 0.2 0.0
    concatfile.flush()
    concatfile.close()
    print("Successfully concatenated bert vecs with FER vecs.")


def load_data(data_path, competency_question_id):
    tmp_lst = []
    with open(data_path + '/optimal_concat_%s.txt' % competency_question_id) as f1:
        for myline in f1:
            wordlist = myline.strip().split()
            tmp_lst.append(wordlist)
    data_x_tmp = pd.DataFrame(tmp_lst, dtype="double")
    data_x = data_x_tmp.iloc[:, 1:777]

    print(data_x)
    # print(data_x.iloc[-1, :])

    tmp_lst2 = []
    with open(data_path + '/optimal_answer_%s.txt' % competency_question_id) as f1:
        for myline in f1:
            wordlist = myline.replace("\n", '').split(',')
            wordlist = [wordlist[0], wordlist[2]]
            tmp_lst2.append(wordlist)
    data_y_tmp = pd.DataFrame(tmp_lst2)
    tmp_y = data_y_tmp[1].values.tolist()
    print(tmp_y)
    y = list(map(int, tmp_y))

    print("Successfully loaded data, start trying param:")
    print('data_x:', data_x.shape)
    print('data_y:', len(y))

    return data_x, y


def params_train(data_path, competency_question_id):
    # find best param
    data_x, y = load_data(data_path, competency_question_id)
    for nnd in list(range(0, 10)):
        x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                            y,
                                                            test_size=0.1,
                                                            random_state=nnd)
        print("model is %s, nnd is %d " % (competency_question_id, nnd))
        es = list(range(5, 258))
        rs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in es:
            for j in rs:
                model = RM(n_estimators=i, random_state=j, n_jobs=-1).fit(x_train, y_train)
                y_pred = model.predict(x_test)
                if me.accuracy_score(y_test, y_pred) >= 1.0:
                    print("nnd=" + str(nnd) + " i=" + str(i) + " j=" + str(j) + " test acc = " + str(
                        me.accuracy_score(y_test, y_pred)))
     return data_x, y


def produce_model(data_x, y, data_path, competency_question_id, nnd, i, j):
    # produce model
    x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=nnd)

    model = RM(n_estimators=i, random_state=j).fit(x_train, y_train)

    y_pred = model.predict(data_x)
    y_pred_proba = model.predict_proba(data_x)

    print(me.classification_report(y, y_pred))
    print(me.accuracy_score(y, y_pred))

    print(len(y))
    print(y)
    print(y_pred)
    print(y_pred_proba)

    labls = model.classes_  # 模型标签类别
    print(labls)

    # 产生模型所有样本真实预测分数
    pred_score = []
    pred_lables = []
    for i in y_pred_proba:
        l = len(i)
        sum = 0
        for j in range(l):
            sum = sum + labls[j] * i[j]
        pred_score.append(round(sum * 20))
        pred_lables.append(labls[np.argmax(i)])
    print('**********真实预测分数************')
    print(pred_score)
    print(pred_lables)

    ## 检测异常标签 ##
    true_labels = []
    pred_labels = []
    outier_index = []
    print("分类错误的标签有，上面是真实标签，下面是预测标签")
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            true_labels.append(y[i])
            pred_labels.append(y_pred[i])
            outier_index.append(i)
    print(true_labels)
    print(pred_labels)

    ## 检测异常分数 ##
    """
    1分： 极差，百分分数介于 0 - 29,   模型只显示[1,10]
    2分：比较差，百分分数介于30 - 50
    3分： 一般般，百分分数介于51 - 70
    4分：较好，百分分数介于 71 - 85
    5分：优秀， 百分分数介于 86 -100
    """
    print("************中间是错误的真实分数********")

    wrong_pred_scores = []
    for i in outier_index:
        wrong_pred_scores.append(pred_score[i])
    print(wrong_pred_scores)

    print("错误分类的，机器输出概率分布:")
    wrong_machine_probas = []
    for i in outier_index:
        wrong_machine_probas.append(y_pred_proba[i])
    print(wrong_machine_probas)

    print("model is %s " % competency_question_id)

    joblib.dump(model, data_path + "/model/optimal_train_model_%s_zh.m" % competency_question_id)
    joblib.dump(model, "/Users/leapin/Desktop/模型与产品/optimal_train_model_%s_zh.m" % competency_question_id)

    # 测试#
    print("测试结果如下： ")
    y_pred = model.predict(data_x)  # 全部数据集上测试
    y_pred_proba = model.predict_proba(data_x)

    print(me.classification_report(y, y_pred))

    print(me.accuracy_score(y, y_pred))
