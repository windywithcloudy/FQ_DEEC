count =0
f = open('E:\code\\test.txt',"r",encoding="UTF-8")
#print(f"{f.read()}")
for line in f.readlines():
    a = str(line)
    #print(f"{a}")
    b=a.replace("\n"," ")
    c=b.split(" ")    
    #print(f"{c}")
    for elements in c:
        #print(f"{elements}")
        if elements == 'itheima':
            count+=1

f.close()
print(f"itheima出现次数为{count}")
count =0
with open('E:\code\\test.txt',"r",encoding="UTF-8") as f:
    count =f.read().count('itheima')
print(f"{count}")