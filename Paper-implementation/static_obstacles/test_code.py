from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

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

# Given point A
Ax, Ay = 50, 50

# Given points B for different cases
B_points = [
    (70, 70),  # Case 1
    (50, 70),  # Case 2
    (30, 70),  # Case 3
    (30, 50),  # Case 4
    (20, 20),  # Case 5
    (50, 20),  # Case 6
    (70, 20),  # Case 7
    (70, 50),  # Case 8
]

# Desired distance from A to C
d = 10

# Function to calculate point C given A and B with distance d from A
def calculate_point_C(Ax, Ay, Bx, By, d):
    # Calculate the distance between A and B
    D = np.sqrt((Bx - Ax)**2 + (By - Ay)**2)
    # Calculate direction ratios
    dir_x = (Bx - Ax) / D
    dir_y = (By - Ay) / D
    # Calculate point C
    Cx = Ax + d * dir_x
    Cy = Ay + d * dir_y
    return Cx, Cy

# Calculate all C points
C_points = [calculate_point_C(Ax, Ay, Bx, By, d) for Bx, By in B_points]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

for i, (Bx, By) in enumerate(B_points):
    Cx, Cy = C_points[i]
    plt.plot([Ax, Bx], [Ay, By], 'k--')  # Line from A to B
    plt.plot(Bx, By, 'ko')  # Point B
    plt.plot(Cx, Cy, 'ro')  # Point C

    # Annotate points B and C
    plt.text(Bx + 1, By + 1, f'B{i+1} ({Bx},{By})', fontsize=9)
    plt.text(Cx + 1, Cy + 1, f'C{i+1} ({Cx:.2f},{Cy:.2f})', fontsize=9)

plt.plot(Ax, Ay, 'bo', label='Point A (50,50)')
circle = plt.Circle((Ax, Ay), d, color=(1,0,0), fill=False)
ax.add_patch(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points A, B, and C with specified distances')
plt.grid(True)
plt.legend()
# plt.show()

random = np.random.choice([0,30,60,90,120,150,180])
print(random)




    


