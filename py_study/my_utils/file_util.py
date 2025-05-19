def print_file_info(file_name):
    try :
        f = open(file_name,'r',encoding="UTF-8")
        print(f"{f.read()}")
    except Exception as e:
        print(f"{e}")
    finally:
        f.close()

def append_to_file(file_name,data):
    f = open(file_name,'a',encoding="UTF-8")
    f.write(data)
    f.flush()
    f.close()