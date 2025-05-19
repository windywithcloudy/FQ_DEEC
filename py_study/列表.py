# student_age = [21,25,21,23,22,20]
# student_age.append(31)
# print(f"{student_age}")
# new_add = [29,33,30]
# student_age.extend(new_add)
# print(f"{student_age}")
# a = student_age.pop(0)
# print(f"{a}")
# b = student_age.pop(8)
# print(f"{b}")
# c = student_age.index(31)
# print(f"{c}")

#元组
t1=('周杰伦',11,['football','music'])
a = t1.index(11)
b = t1[0]
del t1[2][1]
t1[2].append('coding')
print(f"{t1}")