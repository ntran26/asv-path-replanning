from timeit import default_timer as timer

hour = 0
minute = 0
second = 0

start = timer()
run = False
while run:
    time = timer() + 5460 - start

    hour = int(time//3600)
    time = int(time - 3600*hour)
    minute = int(time//60)
    second = int(time - 60*minute)

    # print(f"Time elapsed: {hour}: {minute}: {second}")
    
    if hour < 10:
        if minute < 10:
            if second < 10:
                print(f"Time elapsed: 0{hour} : 0{minute} : 0{second}")
            elif second >= 10 and minute < 10 and hour < 10:
                print(f"Time elapsed: 0{hour} : 0{minute} : {second}")
        elif minute >= 10:
            if second < 10:
                print(f"Time elapsed: 0{hour} : {minute} : 0{second}")
            elif second >= 10:
                print(f"Time elapsed: 0{hour} : {minute} : {second}")
    elif hour >= 10:
        if minute < 10:
            if second < 10:
                print(f"Time elapsed: 0{hour} : 0{minute} : 0{second}")
            elif second >= 10:
                print(f"Time elapsed: 0{hour} : 0{minute} : {second}")
        elif minute >= 10:
            if second < 10:
                print(f"Time elapsed: 0{hour} : {minute} : 0{second}")
            elif second >= 10:
                print(f"Time elapsed: {hour} : {minute} : {second}")


one = 1
two = 2
three = 3
four = 4
i = 0
while False:
    if i % 40 <= 10 and i % 40 != 0:
        print(one)
    elif i % 40 <= 20:
        print(two)
    elif i % 40 <= 30:
        print(three)
    else:
        print(four)
    i += 1

    



    


