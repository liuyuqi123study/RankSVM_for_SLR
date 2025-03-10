# Reading text data from a .dat file
#This is the second step to deal with positive indexes of those features
for i in range(5):

    #We first modify the training features
    with open('train_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row
            positions = [i for i, c in enumerate(row) if c == ':'] 
            for position in positions[1:]:
                modified_row[i-1]=str(int(modified_row[i-1])+1)
            modified_rows.append(modified_row)
    with open('train_feature'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)

    #Then we modify test data
    with open('test_feature'+str(i)+'.dat', 'r') as file:
        rows = file.readlines()
        modified_rows=[]
        for row in rows:
            modified_row=row
            positions = [i for i, c in enumerate(row) if c == ':'] 
            for position in positions[1:]:
                modified_row[i-1]=str(int(modified_row[i-1])+1)
            modified_rows.append(modified_row)
    with open('test_feature'+str(i)+'.dat','w') as file:
        file.writelines(modified_rows)