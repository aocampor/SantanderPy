import pandas as pd
import cPickle

labelnames = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
              'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
              'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
              'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
              'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
              'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
              'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
              'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

def ReadCSV(namecsv):
    df = pd.read_csv(namecsv,
                     dtype={"sexo": str,
                            "ind_nuevo": str,
                            "ult_fec_cli_1t": str,
                            "indext": str
                            }
                     )
    return df

def Dump(obj, fileto):
    with open(fileto, "wb") as output_file:
        cPickle.dump(obj, output_file)

if __name__ == "__main__":
    df = ReadCSV('samples/train_20160528.csv')
    dic = {}
    dic['ncodpers'] = df['ncodpers'].tolist()
    #print len(dic['ncodpers'])
    #dic['added_products'] = []
    dicprods = {}
    dicprobs = {}
    for item in labelnames:
        dic[item] = df[item].tolist()
        print item, df[item].sum(axis=0), len(df[item].tolist()), float( df[item].sum(axis=0))/len(df[item].tolist())
        dicprobs[item] = float( df[item].sum(axis=0))/len(df[item].tolist())
    for i in range(len(dic['ncodpers'])):
        #addprod = []
        dicprods[dic['ncodpers'][i]] = []
        for item in labelnames:
            if( dic[item][i] == 1 ):
                dicprods[dic['ncodpers'][i]].append(item)
                #addprod.append(item)
        #dic['added_products'].append(addprod)
    #print dicprods
    Dump(dicprods, "products.pickle")
    Dump(dicprobs, "probabilities.pickle")
