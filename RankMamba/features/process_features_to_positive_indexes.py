# Reading text data from a .dat file
#This is the second step to deal with positive indexes of those features
for i in range(5):

    #We first modify the training features
    with open('train_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row
            features=modified_row.split(' ')[2:]
            features_plus=[str(int(feature.split(':')[0])+1)+':'+feature.split(':')[1] for feature in features]
                       
            modified_row= modified_row.replace(modified_row.split(' ',maxsplit=2)[2:][0],'')+" ".join(features_plus)
            modified_rows.append(modified_row)
    with open('train_feature_modified'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)

    #Then we modify test data
    with open('test_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row
            features=modified_row.split(' ')[2:]
            features_plus=[str(int(feature.split(':')[0])+1)+':'+feature.split(':')[1] for feature in features]
                       
            modified_row= modified_row.replace(modified_row.split(' ',maxsplit=2)[2:][0],'')+" ".join(features_plus)
            modified_rows.append(modified_row)

    with open('test_feature_modified'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)