import pandas as pd
for i in range(532):
    df=pd.read_json(r"/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Experiment/features/"+"feature_extraction"+str(i)+'.json')
    #for j in range(768): 
     #   df1=pd.DataFrame()
      #  for k in range(16):
            #df1.loc[k,str(j)]=df.loc[k,'encoder'][j]
    print(len(df.loc[0,'encoder']))
        #print(df1)
       # df=pd.concat((df,df1),axis=1)
    #df.drop('encoder',axis="columns")
    #df.to_csv(r"features.dat",mode='a',header=False,index=False)