""" # i=1
# j=1

# for i in range(1,10):
#     for j in range(1,i+1):
#         print(f"{j}*{i}={j*i}\t",end='')
#     print()
 """ 
import random


total = 10000
i =1

for i in range(1,21):
    num = random.randint(1,10)
    if num <5:
        print(f"员工{i},绩效分{num},不发工资，下一位")
        continue
    else:
        total -=1000
        print(f"员工{i},绩效分{num},发工资1000，账户余额还剩{total}")
    
    if total <=0:
        print("工资发完了，等下个月吧")
        break