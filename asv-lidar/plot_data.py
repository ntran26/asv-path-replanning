import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from images import BOAT_ICON

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def add_boat_icon(ax, x, y, heading, zoom=1.25):
    boat_img = Image.frombytes(BOAT_ICON["format"], BOAT_ICON["size"], BOAT_ICON["bytes"])
    rotated_img = boat_img.rotate(-heading, expand=True, resample=Image.BICUBIC)
    imgbox = OffsetImage(rotated_img, zoom=zoom)
    ab = AnnotationBbox(imgbox, (x, y), frameon=False)
    ax.add_artist(ab)

# List of dataset
datasets = [
    # ("PPO", "data/ppo_data_random_0.json", "purple", "dashdot"),
    # ("SAC", "data/sac_data_random_0.json", "blue", "solid"),
    ("$ \lambda $ = 0.5", "data/test_case_6/sac_0_5_data.json", "orange", "dashed"),
    ("$ \lambda $ = 0.6", "data/test_case_6/sac_0_6_data.json", "blue", "dotted"),
    ("$ \lambda $ = 0.7", "data/test_case_6/sac_0_7_data.json", "magenta", "dashdot"),
    ("$ \lambda $ = 0.8", "data/test_case_6/sac_0_8_data.json", "brown", (0, (3, 5, 1, 5, 1, 5))),
    ("$ \lambda $ = 0.9", "data/test_case_6/sac_0_9_data.json", "green", (0, (3, 1, 1, 1)))
]

# Use the first data file to set up the map
reference_data = load_data(datasets[0][1])
start = reference_data["start"]
goal = reference_data["goal"]
obstacles = reference_data["obstacles"]
path = reference_data["path"]

# Initalize plot
plt.figure(figsize=(6, 10))
ax = plt.gca()

# Plot reference path and static elements
plt.plot(*zip(*path), label="Reference Path", color="black", linestyle="solid")
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='red', label='Goal')
for obs in obstacles:
    poly = plt.Polygon(obs, color='red')
    ax.add_patch(poly)

# Plot each dataset
for label, filepath, color, style in datasets:
    try:
        data = load_data(filepath)
        path_data = data["asv_path"]
        heading = data.get("heading", 0)
        plt.plot(*zip(*path_data), label=label, color=color, linestyle=style, linewidth=2)

        final_x, final_y = path_data[-1]
        add_boat_icon(ax, final_x, final_y, heading)
    except Exception as e:
        print(f"Error loading {label} from {filepath}: {e}")

# Show plot
ax.invert_yaxis()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Test scenario 6")
plt.legend()
plt.xlim((0, 400))
plt.ylim((600, 0))
plt.grid(True)
plt.grid(color='#dddddd', alpha=0.7)
plt.savefig("Test scenario.png", dpi=300, bbox_inches='tight')
plt.show()

# plot data with pygame

# TEST_CASE = 1

# # define environment
# env = testEnv(render_mode="human")
# env.test_case = TEST_CASE

# path_surface = pygame.Surface((env.map_width, env.map_height))
# path_surface.fill((255,255,255))

# # for i in range(1, len(asv_path)):
# #     pygame.draw.circle(path_surface, (0, 0, 200), asv_path[i], 3)

# # Draw obstacles
# for obs in obstacles:
#     pygame.draw.polygon(path_surface, (200, 0, 0), obs)

# # Draw Path
# env.draw_dashed_line(path_surface,(0,200,0),(start[0],start[1]),(goal[0],goal[1]),width=5)
# pygame.draw.circle(path_surface,(100,0,0),(goal[0],goal[1]),5)

# # # Draw ship
# # display = pygame.display.set_mode(env.screen_size)
# # os_ = pygame.transform.rotozoom(env.icon,-asv_path[-1][1],2)
# # path_surface.blit(os_,os_.get_rect(center=(asv_path[-1][0],asv_path[-1][1])))
# # display.blit(path_surface,[0,0])
