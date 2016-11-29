import pandas as pd

def ReadCSV(namecsv):
    df = pd.read_csv(namecsv,
                     dtype={"sexo":str,
                            "ind_nuevo":str,
                            "ult_fec_cli_1t":str,
                            "indext":str
                            }
                    )
    return df


if __name__ == "__main__":
    text = []
    fil = open("predicted.csv")
    for line in fil:
        text.append(line)
    fil.close()
    df = ReadCSV("samples/train_20160528.csv")
    sum = 0
    lines = 0

    for line in text:
        lines = lines + 1
        if lines % 100000 == 0:
            print lines
        licomma = line.rsplit(',')
        lispace = licomma[1].rsplit(' ')
        if licomma[0] != 'ncodpers':
            dfa = df[ df['ncodpers'] == int(licomma[0]) ]
            pk = 0
            for it in lispace:
                pk = pk + dfa[ it.rstrip('\n') ].tolist()[0]
            sum = sum + float(pk) / len(lispace)
    print sum, lines
    print "score " + str(float(sum)/lines)