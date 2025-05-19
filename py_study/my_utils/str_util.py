def str_reverse(s):
    new_str = s[::-1]
    return new_str

def substr(s,x,y):
    a = str(s)
    b=a.split(x)
    c =str(b).split(y)
    d = str(c).strip()
    return str(d)
