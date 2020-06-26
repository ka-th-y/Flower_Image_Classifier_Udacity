### Working area to try out code

with open ("save_progress.txt", "r") as myfile:
    data=myfile.readlines()
    data1 = data[0].strip()
    data2 = data[1].strip()
    #data1 = data1.split("\n")[0]
    
    print(str(data1))
    print(str(data2))
    
 

