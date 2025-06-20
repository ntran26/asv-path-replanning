import pygame
import matplotlib.pyplot as plt
from test_run import testEnv
import json
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from images import BOAT_ICON

# load data file
ppo = "data/ppo_data_random_0.json"
sac = "data/sac_data_random_0.json"
with open(ppo, "r") as f:
    ppo_data = json.load(f)
with open(sac, "r") as f:
    sac_data = json.load(f)

start = ppo_data["start"]
goal = ppo_data["goal"]
obstacles = ppo_data["obstacles"]
path = ppo_data["path"]
ppo_path = ppo_data["asv_path"]
sac_path = sac_data["asv_path"]
ppo_heading = ppo_data["heading"]
sac_heading = sac_data["heading"]

plt.figure(figsize=(6,10))

# Load the boat icon image from bytes
boat_img = Image.frombytes(BOAT_ICON["format"], BOAT_ICON["size"], BOAT_ICON["bytes"])
rotated_img = boat_img.rotate(-sac_heading, expand=True, resample=Image.BICUBIC)
imgbox = OffsetImage(rotated_img, zoom=1.5)
final_x, final_y = sac_path[-1]
ab1 = AnnotationBbox(imgbox, (final_x, final_y), frameon=False)

rotated_img = boat_img.rotate(-ppo_heading, expand=True, resample=Image.BICUBIC)
imgbox = OffsetImage(rotated_img, zoom=1.5)
final_x, final_y = ppo_path[-1]
ab2 = AnnotationBbox(imgbox, (final_x, final_y), frameon=False)

plt.plot(*zip(*ppo_path), label="PPO Path", color="purple", linestyle="dashdot")
plt.plot(*zip(*sac_path), label="SAC Path", color="blue", linestyle="solid")
plt.plot(*zip(*path), label="Path", color="green", alpha=0.5, linestyle="dotted")
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='red', label='Goal')
for obs in obstacles:
    poly = plt.Polygon(obs, color='red')
    plt.gca().add_patch(poly)

plt.gca().add_artist(ab1)
plt.gca().add_artist(ab2)

plt.gca().invert_yaxis()

plt.xlabel('X')
plt.ylabel('Y')
plt.title('ASV Path Visualization')
plt.legend()
# plt.axis('equal')
plt.xlim((0,400))
plt.ylim((600,0))
plt.grid(False)
plt.savefig("asv_plot.png", dpi=300, bbox_inches='tight')
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

