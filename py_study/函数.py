money = 5000000
user_name = input("请输入用户名")
flag = True
def check(money):
    print(f"您的余额还剩{money}元")

def cun(money):
    add = int(input("请输入存款金额："))
    money +=add
    print(f"您的余额还剩{money}元")

def qu(money):
    increase =int(input("请输入取款金额："))
    money -= increase
    print(f"您的余额还剩{money}元")

def main(flag):
    print("程序开始运行：1代表存款，2代表取款，3代表查询余额,4代表退出")
    while flag:
        i= int(input("请输入数字"))
        if i==1:
            check(money)
        elif i==2:
            cun(money)
        elif i==3:
            qu(money)
        elif i==4:
            print("已退出")
            flag = False
        else:
            print("输入错误，将退出")
            flag = False

main(flag)