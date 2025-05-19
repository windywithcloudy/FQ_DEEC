with open('E:\code\\bill.txt',"r",encoding="UTF-8") as f:
    g= open('E:\code\\bill.txt.bak',"w",encoding="UTF-8")
    h = open('E:\code\\bill1.txt',"a",encoding="UTF-8")
    for line in f.readlines():
        b=line.strip()
        c=b.split(",")
        #print(f"{c}")        
        if c[4] != "测试":
            h.write(line)
            h.flush()
g.close()
h.close()
