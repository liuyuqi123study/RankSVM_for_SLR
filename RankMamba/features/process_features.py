# Reading text data from a .dat file
for i in range(5):

    #We first modify the training features
    with open('train_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row.strip('"')
            #We write it into some specific format
            modified_row=modified_row.split(' ',maxsplit=1)[0][-2]+' '+modified_row.split(' ',maxsplit=1)[1]
            modified_rows.append(modified_row)
    with open('train_feature'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)

    #Then we modify test data
    with open('test_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row.strip('"')
            #We write it into some specific format
            modified_row=modified_row.split(' ',maxsplit=1)[0][-2]+' '+modified_row.split(' ',maxsplit=1)[1]
            modified_rows.append(modified_row)
    with open('test_feature'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)