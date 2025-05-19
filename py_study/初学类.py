#设计类
class student:
    name = None
    gender = None
    age = None
    nationality = None
    native_place = None

#创建对象
stu_1 = student()

#填写对象
stu_1.name = "周杰论"
stu_1.gender = "男"
stu_1.age = 45
stu_1.nationality = "china"
stu_1.native_place = "中国台湾"

print(f"{stu_1.name}")

class clock:
    id = None
    price = None

    def ring(self):
        import winsound
        winsound.Beep(440,3000)

clock1 = clock()
clock1.ring()