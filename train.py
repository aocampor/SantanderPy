#!/usr/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report


import cPickle

labelnames = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
              'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
              'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
              'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
              'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
              'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
              'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
              'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
              ]

params = {}
param = {'max_depth': 4, 'objective': 'binary:logistic', 'learning_rate': 0.21,
         'n_estimators': 10, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 5,
         'reg_alpha': 2.14, 'reg_lambda': 0.7, 'scale_pos_weight': 3.3}

limit_rows = 1000000
maxiter = 500
rentacut = 3e7
referencetime = datetime.date(2015, 1, 1)
cp = 4e5
to = 0.0001
alph = 0.01
solvy = 'newton-cg' #'sag' 'lbfgs' 'liblinear'
printing = True
plotting = False
dictrainings = {}
kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

def PlotConfusion(conf_arr, title):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'AB'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(title + '.png', format='png')

def Regression(train, labels, topred, label):
    clfa = Train(train, labels, label)
    dictrainings[label] = clfa
    return Predict(clfa, topred)

def MakeRegressionsLoop(dfa, dfalabels):
    preds = {}
    accurs = []
    confMatr = []
    for item in labelnames:
        #if(item == 'ind_cco_fin_ult1'):
            if printing:
                print "Training " + item
            try:
                pre = Regression(dfa[1:limit_rows / 2],
                              dfalabels[item][1:limit_rows / 2],
                              dfa[limit_rows / 2:limit_rows], item
                             )
                acc = accuracy_score(dfalabels[item][limit_rows / 2:limit_rows], pre)
                matr = confusion_matrix(dfalabels[item][limit_rows / 2:limit_rows], pre)
                if plotting:
                    PlotConfusion(matr, item)
                if printing:
                    print "Predictions: "
                    print pre
                    print "Accuracy: "
                    print acc
                    print "Confusion Matrix: "
                    #pd.crosstab(pd.Series(dfalabels[item][limit_rows / 2:limit_rows], name='Actual'),
                    #            pd.Series(pre, name='Predicted'),
                    #            margins=True)
                    print matr
                    print(classification_report(dfalabels[item][limit_rows / 2:limit_rows], pre, target_names=['class0', 'class1']))
                #preds.append(pre)
                preds[item] = pre
                accurs.append(acc)
                confMatr.append(matr)
            except ValueError as e:
                print "There was an error with the regression " + item
                print e
                preds.append(0)
                accurs.append(-1)
                confMatr.append(0)
    if(plotting):            
        PlotList(accurs)
    SavePickle("parameters.pickle", params)
    return preds, accurs, confMatr

def PlotList(accurs):
    figu = plt.figure()
    x= []
    for i in range( len(accurs) ):
        x.append(i)
    plt.plot(x, accurs,'ro')
    plt.xticks(x, labelnames, rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('accuracy.png', format='png')

def Modify(df, label):
    df[label] = pd.to_numeric(df[label], errors='coerce')
    df = df[df[label] >= 0]
    return df

def Predict(clfa, topred):
    return clfa.predict(topred)

def Preprocessing(dfa):
    dfa = dfa.sample(frac=1)
    dfa = Modify(dfa, 'antiguedad')
    dfa = Modify(dfa, 'age')
    dfa = Modify(dfa, 'ind_nuevo')
    dfa = Modify(dfa, 'renta')

    dfa = dfa.drop('ult_fec_cli_1t', 1)
    dfa = dfa.drop('conyuemp', 1)
    dfa = dfa.drop('indrel_1mes', 1)
    dfa = dfa.drop('indresi', 1)
    dfa = dfa.drop('tipodom', 1)

    #dfa = dfa.drop('ind_empleado', 1)
    #dfa = dfa.drop('indext', 1)
    #dfa = dfa.drop('indfall', 1)
    #dfa = dfa.drop('pais_residencia', 1)

    dfa = dfa[dfa['renta'] < rentacut]
    #dfa = dfa[dfa['age'] < 120]

    dfa['fecha_dato'] = pd.to_datetime(dfa['fecha_dato'], errors="coerce") - referencetime
    dfa['fecha_alta'] = pd.to_datetime(dfa['fecha_alta'], errors="coerce") - referencetime

    dfa['fecha_dato'] = dfa['fecha_dato'].map(lambda x: x.astype('timedelta64[D]') / np.timedelta64(1, 'D'))
    dfa['fecha_alta'] = dfa['fecha_alta'].map(lambda x: x.astype('timedelta64[D]') / np.timedelta64(1, 'D'))

    dfa = dfa.dropna()

    tocategory = [
        #'fecha_dato',
        'ind_empleado',
        'pais_residencia',
        'sexo',  # 'fecha_alta',
        'tiprel_1mes', #'indresi',
        'indext',
        'canal_entrada', 'indfall',
        'nomprov',
        'segmento'
    ]
    for it in tocategory:
        dfa[it] = pd.Categorical.from_array(dfa[it]).codes

    return dfa

def PrintScatters(df):
    listavar = list(df)
    for item in listavar:
        if printing:
            print 'Printing ' + item
        df.plot(kind='scatter', x='renta', y=item)
        plt.savefig(item + '.png')

def ReadCSV(namecsv):
    df = pd.read_csv(namecsv,
                     dtype={"sexo": str,
                            "ind_nuevo": str,
                            "ult_fec_cli_1t": str,
                            "indext": str
                            },
                     nrows=limit_rows
                     )
    return df

def ScanningRegressionLoop(df):
    for cp in [4e5]:  # [1e-7, 1e-5, 1e-3, 1e-1, 1, 1e3, 1e5, 1e7]:
        for to in [1, 0.0001, 0.000001]:
            pre = Regression(cp, to, df[1:limit_rows / 2], dflabels[item][1:limit_rows / 2],
                                df[limit_rows / 2:limit_rows])
            acc = accuracy_score(dflabels[labelnames[2]][limit_rows / 2:limit_rows], pre)
            if printing:
                print cp, to, acc
                print confusion_matrix(dflabels[labelnames[2]][limit_rows / 2:limit_rows], pre)

def SplitLabels(dfa):
    # diclabels = {}
    dflabels = dfa.ix[:, 19:43]
    dfclients = dfa['ncodpers']
    for item in labelnames:
        # diclabels[item] = df[item]
        dfa = dfa.drop(item, 1)
    dfa = dfa.drop('ncodpers',1)
    return dfa, dflabels, dfclients

def Gaussian(label):
    if (label == 'ind_ahor_fin_ult1' or label == 'ind_aval_fin_ult1' or label == 'ind_ctju_fin_ult1' or
        label == 'ind_ctma_fin_ult1' or label == 'ind_ctop_fin_ult1' or label == 'ind_fond_fin_ult1' or
        label == 'ind_hip_fin_ult1' or label == 'ind_plan_fin_ult1' or label == 'ind_pres_fin_ult1' or
        label == 'ind_valo_fin_ult1' or label == 'ind_viv_fin_ult1'
        ):
        return GaussianNB(priors=[0.9, 0.1])
    elif label == 'ind_cco_fin_ult1':
        return GaussianNB(priors=[0.35,0.65])
    elif (label == 'ind_cder_fin_ult1' or label == 'ind_deme_fin_ult1' or label == 'ind_reca_fin_ult1' or
                  label == 'ind_tjcr_fin_ult1'
          ):
        return GaussianNB(priors=[0.8, 0.2])

    elif (label == 'ind_cno_fin_ult1' or label == 'ind_dela_fin_ult1' or label == 'ind_ecue_fin_ult1' or
                  label == 'ind_nomina_ult1' or label == 'ind_nom_pens_ult1' or label == 'ind_recibo_ult1'
          ):
        return GaussianNB(priors=[0.6, 0.4])
    elif label == 'ind_ctpp_fin_ult1':
        return GaussianNB(priors=[0.7, 0.3])
    else:
        return GaussianNB(priors=[0.4, 0.6])

def Bernoulli():
    return BernoulliNB(alpha=0)

def SDG(label):
    return SGDClassifier(alpha=alph,loss="hinge", penalty="l2", n_iter=maxiter)

def Logistic():
    return linear_model.LogisticRegression(penalty='l2', C=cp, tol=to,
                                              solver=solvy, multi_class="multinomial",
                                              max_iter=maxiter)

def GPC():
    return GaussianProcessClassifier(kernel)

def AccPlot(train, labels, n_estimators_range, param_name, label):
    nsplits = 4

    if(printing):
        print "Training " + label + " For " + param_name

    cv = StratifiedKFold(n_splits=nsplits)
    nonzeros = np.count_nonzero(labels)
    print nonzeros, len(labels)
    if(nonzeros > 0):
        train_scores, test_scores = validation_curve(
            XGBClassifier(**params[label]), train, labels, param_name=param_name,
            param_range=n_estimators_range, cv=cv, scoring='precision'
        )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    max_train_score_mean = 0
    max_train_score_std = 0
    max_test_score_mean = 0
    max_test_score_std = 0

    min_train_score_mean = 1
    min_train_score_std = 1
    min_test_score_mean = 1
    min_test_score_std = 1

    entry = -1
    for i in range( len(n_estimators_range) ):
        newmax = False
        if(max_train_score_mean <= train_scores_mean[i]):
            max_train_score_mean = train_scores_mean[i]
            max_train_score_std = train_scores_std[i]
            max_test_score_mean = test_scores_mean[i]
            max_test_score_std = test_scores_std[i]
            newmax = True
        if(min_train_score_mean >= train_scores_mean[i]):
            min_train_score_mean = train_scores_mean[i]
            min_train_score_std = train_scores_std[i]
            min_test_score_mean = test_scores_mean[i]
            min_test_score_std = test_scores_std[i]
        if( test_scores_mean[i] + test_scores_std[i] >= train_scores_mean[i] ):
            if(newmax):
                entry = i

    params[label][param_name] = n_estimators_range[entry]
    if printing:
        print "Entry " + str(n_estimators_range[entry])
        print "Max train "+ str(max_train_score_mean)
        print "Max test " + str(max_test_score_mean)
        print "Max train error " + str(max_train_score_std)
        print "Max test error " + str(max_test_score_std)

    if(plotting):    
        fig = plt.figure(figsize=(10, 6), dpi=100)

        plt.title("Validation Curve with XGBoost")
        plt.xlabel(param_name)
        plt.ylabel("precision")
        plt.ylim(min_train_score_mean - 3*min_train_score_std , max_train_score_mean + 3*max_train_score_std)

        plt.plot(n_estimators_range,
             train_scores_mean,
             label="Training score",
             color="r")

        plt.plot(n_estimators_range,
             test_scores_mean,
             label="Cross-validation score",
             color="g")

        plt.fill_between(n_estimators_range,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2, color="r")

        plt.fill_between(n_estimators_range,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.2, color="g")

        plt.axhline(y=1, color='k', ls='dashed')

        plt.legend(loc="best")
        plt.savefig( param_name + '.png')

def TunningParam(train, labels, label):
    n_estimators_range = np.linspace(1, 20, 20).astype('int')
    AccPlot(train, labels, n_estimators_range, 'n_estimators', label)
    n_estimators_range = np.linspace(1, 20, 20).astype('int')
    AccPlot(train, labels, n_estimators_range, 'max_depth', label)
    n_estimators_range = np.linspace(0.01, 10, 20).astype('float')
    AccPlot(train, labels, n_estimators_range, 'min_child_weight', label)
    n_estimators_range = np.linspace(0, 20, 10).astype('float')
    AccPlot(train, labels, n_estimators_range, 'scale_pos_weight', label)
    n_estimators_range = np.linspace(0.01, 1, 15).astype('float')
    AccPlot(train, labels, n_estimators_range, 'learning_rate', label)
    n_estimators_range = np.linspace(0, 5, 20).astype('float')
    AccPlot(train, labels, n_estimators_range, 'gamma', label)
    n_estimators_range = np.linspace(0, 10, 15).astype('float')
    AccPlot(train, labels, n_estimators_range, 'reg_alpha', label)
    n_estimators_range = np.linspace(0, 10, 15).astype('float')
    AccPlot(train, labels, n_estimators_range, 'reg_lambda', label)

def Train(train, labels, label):
    try:
        #clfa = Gaussian(label)
        #clfa = Bernoulli()
        #clfa = SDG(label)
        #clfa = GPC()
        #TunningParam(train, labels, label)
        clfa = XGBClassifier(**params[label])
        clfa.fit(train, labels)
        #xgb.plot_importance(clfa, importance_type='gain', xlabel='Gain')
        #xgb.plot_importance(clfa)
        #plt.savefig('importance.png')
        #xgb.plot_tree(clfa)
        #plt.savefig('tree.pdf')
    except ValueError as e:
        print "There was an error while fitting "
        print e
    return clfa

def SavePickle(name, clfdic):
    with open(name, "wb") as output_file:
        cPickle.dump(clfdic, output_file)

def TrainAndDump(df, dflabels):
    clfdic = TrainModels(df, dflabels)
    SavePickle("trainings.pickle",clfdic)
    #with open(r"trainings.pickle", "wb") as output_file:
    #    cPickle.dump(clfdic, output_file)

def TrainModels(df, dflabels):
    clflist = {}
    for item in labelnames:
        if printing:
            print item
        try:
            clflist[item] = Train(df, dflabels[item], item)
        except ValueError as e:
            print "There was an error with the regression " + item
            print e
    return clflist

def printParams(params):
    for item in params:
        print item, params[item]

def LoadPickle(strin):
    with open(strin, "rb") as input_file:
        return cPickle.load(input_file)

if __name__ == "__main__":
    matplotlib.style.use('ggplot')
    if printing:
        print 'starting............'
        print 'Training with '+str(float(limit_rows)/2)+" events"

    df = ReadCSV('samples/train_ver2.csv')
    df = Preprocessing(df)
    df, dflabels, dfclients = SplitLabels(df)

    try:
        params = LoadPickle("parameters.pickle")
    except:
        for label in labelnames:
            params[label] = param

    if(printing):        
        printParams(params)
            
    TrainAndDump(df, dflabels)

    #PrintScatters(df)
    #preds, accurs, confmatrix = MakeRegressionsLoop(df, dflabels)

    #outdf = pd.DataFrame( {'ncodpers': dfclients[limit_rows / 2:limit_rows], labelnames[0]:preds[0],
    #                        labelnames[1]: preds[1], labelnames[2]: preds[2], labelnames[3]: preds[3],
    #                        labelnames[4]: preds[4], labelnames[5]: preds[5], labelnames[6]: preds[6],
    #                        labelnames[7]: preds[7], labelnames[8]: preds[8], labelnames[9]: preds[9],
    #                        labelnames[10]: preds[10], labelnames[11]: preds[11], labelnames[12]: preds[12],
    #                        labelnames[13]: preds[13], labelnames[14]: preds[14], labelnames[15]: preds[15],
    #                        labelnames[16]: preds[16], labelnames[17]: preds[17], labelnames[18]: preds[18],
    #                        labelnames[19]: preds[19], labelnames[20]: preds[20], labelnames[21]: preds[21],
    #                        labelnames[22]: preds[22], labelnames[23]: preds[23] })
    #print outdf

    # ScanningRegressionLoop(df)

    # df.plot(kind='scatter',x='age', y='renta');
    # plt.plot( diclabels[labelnames[2]], prediction,"o")
