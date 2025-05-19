class student:
    def __init__(self,name,age,addr,i):
        self.name = name
        self.age = age
        self.addr = addr
        print(f"学生{i}信息录入完毕，信息为：学生姓名：{self.name},年龄：{self.age},地址：{self.addr}")

for i in range(1,11):
    print(f"当前录入{i}位学生信息，共需录入10位学生信息")
    stu =  student(input("name:"),int(input("age:")),input("address:"),i)
    