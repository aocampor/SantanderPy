#!/usr/python
import pandas as pd
import numpy as np
import matplotlib
import datetime
import cPickle

labelnames = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
             'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
             'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
             'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
             'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

referencetime = datetime.date(2015,1,1)

def ReadCSV(namecsv):
    df = pd.read_csv(namecsv,
                     dtype={"sexo":str,
                            "ind_nuevo":str,
                            "ult_fec_cli_1t":str,
                            "indext":str
                            }
                    )
    return df

def SplitLabels(df):
    dfclients = df['ncodpers']
    df = df.drop('ncodpers', 1)
    dflabels = []
    try:
        #dflabels = df.ix[:, 19:43]
        #print dflabels.describe()
        for item in labelnames:
            df = df.drop(item, 1)
        return df, dflabels, dfclients
    except ValueError as e:
        print "Error in labels"
        print e
    return df, dflabels, dfclients

def Preprocessing(df):
    df = df.sample(frac=1)
    df = Modify( df, 'antiguedad' )
    df = Modify( df, 'age' )
    df = Modify( df, 'ind_nuevo' )
    df = Modify( df, 'renta' )

    df = df.drop('ult_fec_cli_1t',1)
    df = df.drop('conyuemp',1)
    df = df.drop('indrel_1mes', 1)
    df = df.drop('indresi', 1)
    df = df.drop('tipodom', 1)

    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], errors="coerce") - referencetime
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors="coerce") - referencetime

    df['fecha_dato'] = df['fecha_dato'].map(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1,'D'))
    df['fecha_alta'] = df['fecha_alta'].map(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1,'D'))

    df = df.fillna(0)

    tocategory = [ #'fecha_dato',
                    'ind_empleado', 'pais_residencia',
                    'sexo', #'fecha_alta',
                    'tiprel_1mes', #'indresi',
                    'indext', 'canal_entrada', 'indfall', 'nomprov',
                    'segmento'
                ]
    for it in tocategory:
        df[it] = pd.Categorical.from_array(df[it]).codes

    #df['fecha_dato'] = df['fecha_dato'].astype('category')

    return df

def Modify( df, label ):
    df[label] = pd.to_numeric(df[label], errors='coerce')
    return df

def Predict(clf, topred):
    return clf.predict(topred)

def LoadPickle(strin):
    with open(strin, "rb") as input_file:
        return cPickle.load(input_file)

def GetPredictions(clf, df):
    predict = {}
    for item in labelnames:
        try:
            predict[item] = Predict(clfsdic[item], dft)

        except ValueError as e:
            print "Error predicting "
            print e
    return predict

def PrintOutput(dfclients, predictions, productos, probabilities):
    ncodpers = dfclients.tolist()
    predkeys = list(predictions.keys())
    outfile = open("predicted.csv","w")
    outfile.write("ncodpers,added_products\n")
    print probabilities
    for i in range(len(ncodpers)):
        nco = ncodpers[i]
        try:
            temporals = productos[nco]
        except:
            temporals = []
        toprint = ""
        prob = 0
        products = []
        for item in predkeys:
            flag = False
            for item1 in temporals:
                if (item1 == item ):
                    flag = True
            if predictions[item][i] == 1 :#and not flag:
                products.append(item)
        for item in products:
            toprint = toprint + item + " "
            #print probabilities[item]
            #if (probabilities[item] > prob):
            #    prob = probabilities[item]
            #    toprint = item
        #if toprint == "":
        #    toprint = "ind_cco_fin_ult1"
        outfile.write(str(nco) + "," + toprint + "\n")
    outfile.close()

if __name__ == "__main__":
    dft = ReadCSV('samples/test_ver2.csv')
    #dft = ReadCSV('samples/train_20160528.csv')
    dft = Preprocessing(dft)
    dft, dflabels, dfclients = SplitLabels(dft)
    clfsdic = LoadPickle("trainings.pickle")
    products = LoadPickle("products.pickle")
    probabilities = LoadPickle("probabilities.pickle")
    print probabilities
    predictions = GetPredictions(clfsdic, dft)
    PrintOutput(dfclients, predictions, products, probabilities)

