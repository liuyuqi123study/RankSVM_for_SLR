#This file is used to reorder this 
# Reading text data from a .dat file
#This is the second step to deal with positive indexes of those features
for i in range(5):

    #We first modify the training features
    with open('train_feature_modified'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=sorted(rows,key=lambda x:int(x.split(' ')[1].replace('qid:','')))
    with open('train_feature_modified'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)

    #Then we modify test data
    with open('test_feature_modified'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=sorted(rows,key=lambda x:int(x.split(' ')[1].replace('qid:','')))

    with open('test_feature_modified'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)