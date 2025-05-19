import random

num = random.randint(1,10)
guess_num = int(input("请输入第一次猜测的数字"))

if guess_num > num:
    print("猜大了，第二次了")
    guess_num =int(input("请再次输入数字"))
    if guess_num> num:
        print("猜大了，第三次了")
        guess_num =int(input("请再次输入数字"))
        if guess_num != num:
            print("sorry，三次都猜错了")
        else:
            print("恭喜你，第三次猜对了")

    elif guess_num < num:
        print("猜小了，第三次了")
        guess_num =int(input("请再次输入数字"))
        if guess_num != num:
            print("sorry，三次都猜错了")
        else:
            print("恭喜你，第三次猜对了")
    else:
        print("恭喜你，第二次猜对了")
elif guess_num < num:
    print("猜小了，第二次了")
    guess_num =int(input("请再次输入数字"))
    if guess_num> num:
        print("猜大了，第三次了")
        guess_num =int(input("请再次输入数字"))
        if guess_num != num:
            print("sorry，三次都猜错了")
        else:
            print("恭喜你，第三次猜对了")

    elif guess_num < num:
        print("猜小了，第三次了")
        guess_num =int(input("请再次输入数字"))
        if guess_num != num:
            print("sorry，三次都猜错了")
        else:
            print("恭喜你，第三次猜对了")
    else:
        print("恭喜你，第二次猜对了")
else:
    print("恭喜你，第一次就猜对了")
    
                

