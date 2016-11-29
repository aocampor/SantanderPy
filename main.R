library(dplyr)
library(e1071)
library("xgboost", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.3")
library(VGAM)

df <- fread("/home/aocampor/DataScience/Santander/samples/train_ver2.csv")
df = df[ ! is.na(df$antiguedad) ]
df = df[ df$antiguedad > 0 ]
#df = df[ df$age < 100 ]
#df = df[ df$age > 5 ]
#df = df[ df$indrel < 90]
df = df[ ! is.na( df$cod_prov ) ]
df = df[ ! is.na( df$renta ) ]
#df = df[df$renta < 27000000]
#df$indrel <- NULL  
df$tipodom <- NULL
for(i in names(df)){
  if( typeof( df[[i]] ) == "character" ){
    df[[i]] <- factor( df[[i]] )
  }
}

dfrandom <- df[sample(nrow(df)),]
#dfrandlabels <- dfrandom$ncodpers
#dfrandom$ncodpers <- NULL
dfran1 <- dfrandom[1:5000000, ]
dfran2 <- dfrandom[5000001:10810753, ]

dfl1 <- dfran1$ind_ahor_fin_ult1
dfran1$ind_ahor_fin_ult1 <- NULL
dfl2 <- dfran1$ind_aval_fin_ult1
dfran1$ind_aval_fin_ult1 <- NULL
dfl3 <- dfran1$ind_cco_fin_ult1
dfran1$ind_cco_fin_ult1 <- NULL
dfl4 <- dfran1$ind_cder_fin_ult1
dfran1$ind_cder_fin_ult1 <- NULL
dfl5 <- dfran1$ind_cno_fin_ult1
dfran1$ind_cno_fin_ult1 <- NULL
dfl6 <- dfran1$ind_ctju_fin_ult1
dfran1$ind_ctju_fin_ult1 <- NULL
dfl7 <- dfran1$ind_ctma_fin_ult1
dfran1$ind_ctma_fin_ult1 <- NULL
dfl8 <- dfran1$ind_ctop_fin_ult1
dfran1$ind_ctop_fin_ult1 <- NULL
dfl9 <- dfran1$ind_ctpp_fin_ult1
dfran1$ind_ctpp_fin_ult1 <- NULL
dfl10 <- dfran1$ind_deco_fin_ult1
dfran1$ind_deco_fin_ult1 <- NULL
dfl11 <- dfran1$ind_deme_fin_ult1
dfran1$ind_deme_fin_ult1 <- NULL
dfl12 <- dfran1$ind_dela_fin_ult1
dfran1$ind_dela_fin_ult1 <- NULL
dfl13 <- dfran1$ind_ecue_fin_ult1
dfran1$ind_ecue_fin_ult1 <- NULL
dfl14 <- dfran1$ind_fond_fin_ult1
dfran1$ind_fond_fin_ult1 <- NULL
dfl15 <- dfran1$ind_hip_fin_ult1
dfran1$ind_hip_fin_ult1 <- NULL
dfl16 <- dfran1$ind_plan_fin_ult1
dfran1$ind_plan_fin_ult1 <- NULL
dfl17 <- dfran1$ind_pres_fin_ult1
dfran1$ind_pres_fin_ult1 <- NULL
dfl18 <- dfran1$ind_reca_fin_ult1
dfran1$ind_reca_fin_ult1 <- NULL
dfl19 <- dfran1$ind_tjcr_fin_ult1
dfran1$ind_tjcr_fin_ult1 <- NULL
dfl20 <- dfran1$ind_valo_fin_ult1
dfran1$ind_valo_fin_ult1 <- NULL
dfl21 <- dfran1$ind_viv_fin_ult1
dfran1$ind_viv_fin_ult1 <- NULL
dfl22 <- dfran1$ind_nomina_ult1
dfran1$ind_nomina_ult1 <- NULL
dfl23 <- dfran1$ind_nom_pens_ult1
dfran1$ind_nom_pens_ult1 <- NULL
dfl24 <- dfran1$ind_recibo_ult1
dfran1$ind_recibo_ult1 <- NULL

df2l1 <- dfran2$ind_ahor_fin_ult1
dfran2$ind_ahor_fin_ult1 <- NULL
df2l2 <- dfran2$ind_aval_fin_ult1
dfran2$ind_aval_fin_ult1 <- NULL
df2l3 <- dfran2$ind_cco_fin_ult1
dfran2$ind_cco_fin_ult1 <- NULL
df2l4 <- dfran2$ind_cder_fin_ult1
dfran2$ind_cder_fin_ult1 <- NULL
df2l5 <- dfran2$ind_cno_fin_ult1
dfran2$ind_cno_fin_ult1 <- NULL
df2l6 <- dfran2$ind_ctju_fin_ult1
dfran2$ind_ctju_fin_ult1 <- NULL
df2l7 <- dfran2$ind_ctma_fin_ult1
dfran2$ind_ctma_fin_ult1 <- NULL
df2l8 <- dfran2$ind_ctop_fin_ult1
dfran2$ind_ctop_fin_ult1 <- NULL
df2l9 <- dfran2$ind_ctpp_fin_ult1
dfran2$ind_ctpp_fin_ult1 <- NULL
df2l10 <- dfran2$ind_deco_fin_ult1
dfran2$ind_deco_fin_ult1 <- NULL
df2l11 <- dfran2$ind_deme_fin_ult1
dfran2$ind_deme_fin_ult1 <- NULL
df2l12 <- dfran2$ind_dela_fin_ult1
dfran2$ind_dela_fin_ult1 <- NULL
df2l13 <- dfran2$ind_ecue_fin_ult1
dfran2$ind_ecue_fin_ult1 <- NULL
df2l14 <- dfran2$ind_fond_fin_ult1
dfran2$ind_fond_fin_ult1 <- NULL
df2l15 <- dfran2$ind_hip_fin_ult1
dfran2$ind_hip_fin_ult1 <- NULL
df2l16 <- dfran2$ind_plan_fin_ult1
dfran2$ind_plan_fin_ult1 <- NULL
df2l17 <- dfran2$ind_pres_fin_ult1
dfran2$ind_pres_fin_ult1 <- NULL
df2l18 <- dfran2$ind_reca_fin_ult1
dfran2$ind_reca_fin_ult1 <- NULL
df2l19 <- dfran2$ind_tjcr_fin_ult1
dfran2$ind_tjcr_fin_ult1 <- NULL
df2l20 <- dfran2$ind_valo_fin_ult1
dfran2$ind_valo_fin_ult1 <- NULL
df2l21 <- dfran2$ind_viv_fin_ult1
dfran2$ind_viv_fin_ult1 <- NULL
df2l22 <- dfran2$ind_nomina_ult1
dfran2$ind_nomina_ult1 <- NULL
df2l23 <- dfran2$ind_nom_pens_ult1
dfran2$ind_nom_pens_ult1 <- NULL
df2l24 <- dfran2$ind_recibo_ult1
dfran2$ind_recibo_ult1 <- NULL

#month1 = df[ df$fecha_dato == "2015-01-28"]
#month2 = df[ df$fecha_dato == "2015-02-28"]
#month3 = df[ df$fecha_dato == "2015-03-28"]
#month4 = df[ df$fecha_dato == "2015-04-28"]
#month5 = df[ df$fecha_dato == "2015-05-28"]
#month6 = df[ df$fecha_dato == "2015-06-28"]
#month7 = df[df$fecha_dato == "2015-07-28"]
#month8 = df[df$fecha_dato == "2015-08-28"]
#month9 = df[df$fecha_dato == "2015-09-28"]
#month10 = df[df$fecha_dato == "2015-10-28"]
#month11 = df[df$fecha_dato == "2015-11-28"]
#month12 = df[df$fecha_dato == "2015-12-28"]
#month13 = df[df$fecha_dato == "2016-01-28"]
#month14 = df[df$fecha_dato == "2016-02-28"]
#month15 = df[df$fecha_dato == "2016-03-28"]
#month16 = df[df$fecha_dato == "2016-04-28"]
#month17 = df[df$fecha_dato == "2016-05-28"]

#nb_model <- naiveBayes(dfl1~., data = dfran1 )
nb_model <- naiveBayes(dfran1, dfl1 )
prediction <- predict(nb_model, dfran2)

fit <- vglm( dfl1~., family=multinomial, data=dfran1 )
#xgb <- xgboost(data = dfran2, 
#               label = df2l1, 
#               eta = 0.1,
#               max_depth = 15, 
#               nround=25, 
#               subsample = 0.5,
#               colsample_bytree = 0.5,
#               seed = 1,
#               eval_metric = "merror",
#               objective = "multi:softprob",
#               num_class = 12,
#               nthread = 3
#)

plot(df2l1, prediction)

temp <- df$ind_cco_fin_ult1
hist(dfl1[ dfl1 > 0]) 
plot( density(dfl1[ dfl1 > 0]) )

