import math
import logging

import gymnasium as gym
# import gym
import numpy as np
try:
    from .reward_config import (
        CHECKPOINT_CROSS_TOLERANCE,
        CHECKPOINT_CROSS_TRIM,
        CHECKPOINT_REWARD,
        CHECKPOINT_TARGET_SAMPLES,
        CHECKPOINT_TARGET_TRIM,
        CHECKPOINT_VISIBILITY_WALL_MARGIN,
        COVERAGE_BLOCKED_WALL_PENALTY,
        COVERAGE_CELL_SIZE,
        COVERAGE_COLLISION_PENALTY_CAP,
        COVERAGE_COLLISION_PENALTY_RATE,
        COVERAGE_GOAL_SLOTS,
        COVERAGE_HOVER_PENALTY,
        COVERAGE_HOVER_SPEED_THRESHOLD,
        COVERAGE_PROGRESS_MARGIN,
        COVERAGE_PROGRESS_PENALTY,
        COVERAGE_PROGRESS_REWARD_CAP,
        COVERAGE_PROGRESS_REWARD_SCALE,
        ROBOT_WALL_CLEARANCE,
        ROOMS_SCALE,
        SENSOR_ORIGIN_WALL_BLOCK_MARGIN,
    )
    from .rooms_layout import load_rooms_layout
except ImportError:
    from reward_config import (
        CHECKPOINT_CROSS_TOLERANCE,
        CHECKPOINT_CROSS_TRIM,
        CHECKPOINT_REWARD,
        CHECKPOINT_TARGET_SAMPLES,
        CHECKPOINT_TARGET_TRIM,
        CHECKPOINT_VISIBILITY_WALL_MARGIN,
        COVERAGE_BLOCKED_WALL_PENALTY,
        COVERAGE_CELL_SIZE,
        COVERAGE_COLLISION_PENALTY_CAP,
        COVERAGE_COLLISION_PENALTY_RATE,
        COVERAGE_GOAL_SLOTS,
        COVERAGE_HOVER_PENALTY,
        COVERAGE_HOVER_SPEED_THRESHOLD,
        COVERAGE_PROGRESS_MARGIN,
        COVERAGE_PROGRESS_PENALTY,
        COVERAGE_PROGRESS_REWARD_CAP,
        COVERAGE_PROGRESS_REWARD_SCALE,
        ROBOT_WALL_CLEARANCE,
        ROOMS_SCALE,
        SENSOR_ORIGIN_WALL_BLOCK_MARGIN,
    )
    from rooms_layout import load_rooms_layout
WINDOW_WIDTH = int(1600 * ROOMS_SCALE)
WINDOW_HEIGHT = int(1000 * ROOMS_SCALE)
DISPLAY_SCALE = 0.45
DISPLAY_WIDTH = int(WINDOW_WIDTH * DISPLAY_SCALE)
DISPLAY_HEIGHT = int(WINDOW_HEIGHT * DISPLAY_SCALE)
ECHO_RAY_LENGTH = 1500 * ROOMS_SCALE
ECHO_MAX_DISTANCE = 500 * ROOMS_SCALE
GOAL_DISTANCE_NORM = 700 * ROOMS_SCALE
RENDER_RAY_LENGTH = 360 * ROOMS_SCALE
VELOCITY_NORM = 50 * ROOMS_SCALE
ROBOT_RENDER_SCALE = 0.75
ROOMS_SPAWN_POSES = np.array([
    [165, 520, 0.0],
    [360, 340, 0.0],
    [760, 520, 0.0],
    [1220, 500, np.pi],
    [470, 760, -np.pi / 2],
], dtype=float)

COLOR_BLACK = (0, 0, 0)
COLOR_CHECKPOINT = (220, 35, 20)
COLOR_VISITED = (35, 170, 70)
COLOR_RAY = (245, 205, 40)
COLOR_RAY_HIT = (250, 230, 75)
COLOR_CHECKPOINT_HIT = (255, 95, 45)
COLOR_THRUST = (35, 180, 80)
COLOR_REVERSE = (45, 120, 230)
COLOR_TURN = (245, 170, 35)
COLOR_VELOCITY = (50, 50, 50)
COLOR_PANEL_BG = (255, 255, 255)
COLOR_PANEL_BORDER = (70, 70, 70)
COLOR_TEXT = (19, 19, 41)

logger = logging.getLogger(__name__)

def distance_to_line_segment(x, y, x1, y1, x2, y2, d=1):
    # Calculate the distance between the point and the line segment
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    xx, yy = x1, y1
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    dist = math.sqrt(dx * dx + dy * dy)

    return dist < d


def point_to_line_segment_distance(x, y, x1, y1, x2, y2):
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    len_sq = C * C + D * D
    param = 0 if len_sq == 0 else max(0, min(1, (A * C + B * D) / len_sq))
    xx = x1 + param * C
    yy = y1 + param * D
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


def line_segment_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    if line_intersect_tolerant(x1, y1, x2, y2, x3, y3, x4, y4) is not None:
        return 0
    return min(
        point_to_line_segment_distance(x1, y1, x3, y3, x4, y4),
        point_to_line_segment_distance(x2, y2, x3, y3, x4, y4),
        point_to_line_segment_distance(x3, y3, x1, y1, x2, y2),
        point_to_line_segment_distance(x4, y4, x1, y1, x2, y2),
    )


def closest_point_on_line_segment(x, y, x1, y1, x2, y2):
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    len_sq = C * C + D * D
    param = 0 if len_sq == 0 else max(0, min(1, (A * C + B * D) / len_sq))
    return x1 + param * C, y1 + param * D


def line_segment_midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_on_line_segment(x1, y1, x2, y2, t):
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def point_line_segment_fraction(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return 0
    return ((px - x1) * dx + (py - y1) * dy) / len_sq


def signed_distance_to_line(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0
    return ((px - x1) * dy - (py - y1) * dx) / length


def wrap_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # returns a (x, y) tuple or None if there is no intersection
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return None
    if not (0 <= s <= 1 and 0 <= t <= 1):
        return None
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


def line_intersect_tolerant(x1, y1, x2, y2, x3, y3, x4, y4, eps=1e-6):
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(d) < eps:
        return None
    s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
    t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    if not (-eps <= s <= 1 + eps and -eps <= t <= 1 + eps):
        return None
    s = max(0, min(1, s))
    return x1 + s * (x2 - x1), y1 + s * (y2 - y1)


def line_intersect_front(x1, y1, x2, y2, x3, y3, x4, y4):
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return False
    if not (0 <= s and 0 <= t <= 1):
        return False
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


class Environment:
    def __init__(self, game):
        self.game = game
        self.L2_line1_array_source = np.array(
            [[1195., 986.], [1037., 987.], [817., 991.], [577., 995.], [348., 994.], [176., 987.], [69., 926.],
             [10., 800.], [12., 644.], [10., 504.], [38., 378.], [87., 328.], [251., 331.], [374., 375.], [443., 461.],
             [475., 567.], [502., 676.], [566., 758.], [656., 788.], [815., 791.], [990., 789.], [1112., 757.],
             [1162., 690.], [1168., 615.], [1142., 570.], [1098., 544.], [1017., 529.], [
                 942., 526.], [847., 526.], [751., 500.], [670., 459.], [578., 407.], [487., 362.], [376., 307.],
             [275., 274.], [143., 224.], [26., 91.], [59., 27.], [151., 9.], [329., 4.], [575., 4.], [838., 5.],
             [1018., 3.], [1241., 4.], [1375., 18.], [1466., 47.], [1525., 129.], [1552., 301.], [1578., 463.],
             [1585., 597.], [1591., 747.], [1584., 854.], [1529., 917.], [1395., 981.], [1195., 986.]])
        self.L2_line2_array_source = np.array(
            [[200., 685.], [209., 659.], [218., 652.], [240., 649.], [269., 661.], [291., 703.], [323., 758.],
             [392., 832.], [446., 865.], [576., 881.], [775., 881.], [958., 879.], [1130., 869.], [1243., 831.],
             [1269., 746.], [1290., 652.], [1301., 594.], [1300., 518.], [1296., 471.], [1282., 421.], [1252., 359.],
             [1215., 321.], [1152., 302.], [1087., 290.], [983., 286.], [
                 867., 282.], [769., 266.], [657., 246.], [591., 232.], [553., 205.], [561., 173.], [582., 161.],
             [689., 144.], [800., 136.], [953., 123.], [1185., 167.], [1249., 253.], [1290., 347.], [1311., 447.],
             [1328., 554.], [1330., 664.], [1312., 775.], [1265., 855.], [1185., 878.], [1082., 894.], [785., 894.],
             [443., 892.], [349., 855.], [258., 786.], [210., 744.], [200., 708.], [200., 685.]])
        self.L2_goals_array_source = np.array(
            [[627., 4., 636., 152.], [772., 5., 770., 138.], [941., 4., 925., 125.], [1149., 4., 1084., 148.],
             [1411., 30., 1212., 203.], [1536., 199., 1268., 296.], [1562., 362., 1306., 423.],
             [1581., 516., 1324., 529.], [1587., 656., 1329., 632.], [1589., 780., 1323., 709.],
             [1445., 957., 1282., 826.], [1173., 986., 1150., 883.], [968., 988., 960., 894.], [763., 992., 760., 894.],
             [579., 995., 582., 893.], [379., 994., 418., 882.], [128., 960., 290., 811.], [10., 780., 205., 725.],
             [12., 635., 203., 675.], [
                 29., 420., 232., 650.], [254., 332., 250., 653.], [435., 451., 282., 685.], [484., 603., 299., 718.],
             [516., 694., 353., 791.], [558., 748., 434., 857.], [621., 776., 583., 881.], [766., 790., 763., 881.],
             [902., 790., 908., 880.], [1046., 774., 1090., 871.], [1132., 730., 1250., 809.],
             [1166., 644., 1290., 652.], [1147., 579., 1300., 513.], [1055., 536., 1199., 316.],
             [890., 526., 959., 285.], [717., 483., 803., 272.], [587., 412., 681., 250.], [424., 331., 585., 228.],
             [136., 216., 554., 199.], [47., 50., 559., 182.], [396., 4., 572., 167.]])

        self.L5_line1_array_source = np.array([
            [100, 100, 1500, 100],     # top boundary
            [1500, 100, 1500, 900],    # right boundary
            [1500, 900, 100, 900],     # bottom boundary
            [100, 900, 100, 100],      # left boundary

            # square block
            [300, 300, 450, 300],
            [450, 300, 450, 450],
            [450, 450, 300, 450],
            [300, 450, 300, 300],

            # L-shape
            [1000, 250, 1200, 250],
            [1200, 250, 1200, 500],
            [1200, 500, 1100, 500],
            [1100, 500, 1100, 350],
            [1100, 350, 1000, 350],

            # corridor
            [600, 600, 1000, 600],
            [600, 650, 1000, 650],

            # vertical corridor walls
            [700, 100, 700, 300],
            [750, 100, 750, 300],
        ], dtype=float)

        self.L5_line2_array_source = np.zeros((0, 4))  # No second line needed

        self.L5_goals_array_source = np.array([
            # 1. Top-left wall to top-left corner of square
            [100, 100, 300, 300],

            [100, 372, 300, 375],

            [100, 540, 300, 450],

            [220, 900, 374, 450],

            [450, 900, 450, 450],

            [600, 900, 600, 650],

            [800, 900, 800, 650],

            [1250, 900, 1000, 650],

            [1200, 500, 1500, 500],

            [1200, 387, 1500, 300],

            [1200, 272, 1416, 100],

            [1154, 100, 1154, 250],

            [950, 100, 1015, 250],

            [750, 265, 1000, 350],

            [750, 300, 830, 600],

            [700, 300, 450, 700],

            [450, 377, 700, 175],

            [548, 100, 450, 300],

            [375, 100, 375, 300]
        ])

        self.rooms_line1_array_source, self.rooms_goals_array_source, self.rooms_layout_source_path = load_rooms_layout(
            self.game.rooms_layout_path
        )
        self.rooms_line2_array_source = np.zeros((0, 4))
        self.load_level()

    def load_level(self):
        line1, line2, goals = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 4))
        # default/playground
        if self.game.env_name in ['default', 'playground']:
            line1 = self.L5_line1_array_source.copy()
            line2 = self.L5_line2_array_source.copy()
            goals = self.L5_goals_array_source.copy()


        # level2 environment
        elif self.game.env_name == 'level2':
            line1 = self.L2_line1_array_source.copy()
            line2 = self.L2_line2_array_source.copy()
            goals = self.L2_goals_array_source.copy()

        elif self.game.env_name in ['rooms', '2d_checkpoint_exploration']:
            line1 = self.rooms_line1_array_source.copy()
            line2 = self.rooms_line2_array_source.copy()
            goals = self.rooms_goals_array_source.copy()
            line1 *= ROOMS_SCALE
            line2 *= ROOMS_SCALE
            goals *= ROOMS_SCALE

        # environment = empty: level with no boundaries
        elif self.game.env_name == 'empty':
            self.game.gui_draw_echo_points = False
            self.game.gui_draw_echo_vectors = False
            self.game.gui_draw_goal_all = False
            self.game.gui_draw_goal_next = False
            self.game.gui_draw_goal_points = False

        # environment = random: random level is generated
        elif self.game.env_name == 'random':
            # generate level and apply
            line1, line2, goals = self.generate_level_vectors_random(
                n_max=self.game.env_random_length)
            if self.game.camera_mode == 'fixed':
                print("When using env_name = 'random', the use of ",
                      "camera_mode = 'centered' is recommended.")


        if self.game.env_flipped:
            line1[:, 0] = -line1[:, 0] + WINDOW_WIDTH
            line2[:, 0] = -line2[:, 0] + WINDOW_WIDTH
            goals[:, [0, 2]] = -goals[:, [0, 2]] + WINDOW_WIDTH

        self.set_level_vectors(line1, line2, goals)
        self.n_goals = goals.shape[0]
        self.generate_collision_vectors(line1, line2)

    def move_env(self, d_x, d_y):
        # move the environment in fixed camera mode
        self.line1[:, 0] = self.line1[:, 0] - d_x
        self.line1[:, 1] = self.line1[:, 1] - d_y
        self.line2[:, 0] = self.line2[:, 0] - d_x
        self.line2[:, 1] = self.line2[:, 1] - d_y
        self.line1_list = self.line1.tolist()
        self.line2_list = self.line2.tolist()
        self.goals[:, [0, 2]] = self.goals[:, [0, 2]] - d_x
        self.goals[:, [1, 3]] = self.goals[:, [1, 3]] - d_y
        self.level_collision_vectors[:, [0, 2]] = self.level_collision_vectors[:, [0, 2]] - d_x
        self.level_collision_vectors[:, [1, 3]] = self.level_collision_vectors[:, [1, 3]] - d_y

    def set_level_vectors(self, line1, line2, goals, level_collision_vectors=None):
        # self.line1 = line1
        # self.line2 = line2
        # self.goals = goals
        # # list for pygame draw
        # self.line1_list = line1.tolist()
        # if level_collision_vectors:
        #     self.level_collision_vectors = level_collision_vectors
        self.line1 = line1
        self.line2 = line2
        self.goals = goals

        # Convert to lists of 2-point tuples for rendering
        self.line1_list = line1[:, [0, 1, 2, 3]].reshape(-1, 2, 2).tolist()
        self.line2_list = line2[:, [0, 1, 2, 3]].reshape(-1, 2, 2).tolist()

        if level_collision_vectors is not None:
            self.level_collision_vectors = level_collision_vectors

    def generate_collision_vectors(self, line1, line2):
        # for collision calculation, is numpy array
        # only call once to generate single line structe
        # n1, n2 = line1.shape[0], line2.shape[0]
        # line_combined = np.zeros((n1 + n2 - 2, 4))
        # line_combined[:n1 - 1, [0, 1]] = line1[:n1 - 1, [0, 1]]
        # line_combined[:n1 - 1, [2, 3]] = line1[1:n1, [0, 1]]
        # line_combined[n1 - 1:n1 + n2 - 2, [0, 1]] = line2[:n2 - 1, [0, 1]]
        # line_combined[n1 - 1:n1 + n2 - 2, [2, 3]] = line2[1:n2, [0, 1]]
        # self.level_collision_vectors = line_combined
        self.level_collision_vectors = np.concatenate([line1, line2], axis=0)

    def get_goal_line(self, level):
        return self.goals[level, :]

    def generate_level_vectors_random(self, n_max=50, steps_back=10):
        width = 100  # 10
        width_min = 40  # 10
        width_max = 150  # 25
        length_min = 60  # 30 # 5
        length_max = 150  # 20
        angle_mult = 0.5
        data = np.zeros((n_max, 7))
        data[0, 0] = 50  # first x is shifted
        counter = 1  # dont change first
        while counter < n_max:
            sign = +1 if np.random.rand() > 0.5 else -1
            for point in range(np.random.randint(low=3, high=10)):
                ang_new = data[counter - 1, 2] + sign * np.random.rand() * angle_mult
                if ang_new > np.pi:
                    ang_new -= 2 * np.pi
                if ang_new < -np.pi:
                    ang_new += 2 * np.pi
                data[counter, 2] = ang_new
                length = np.random.randint(low=length_min, high=length_max)
                x_old = data[counter - 1, 0]
                y_old = data[counter - 1, 1]
                x_new = x_old + length * np.cos(ang_new)
                y_new = y_old + length * np.sin(ang_new)
                data[counter, 0] = x_new
                data[counter, 1] = y_new
                for i in range(counter):
                    x3 = data[i, 0]
                    y3 = data[i, 0]
                    x4 = data[i + 1, 0]
                    y4 = data[i + 1, 0]
                    if line_intersect_front(x_old, y_old, x_new, y_new, x3, y3, x4, y4):
                        counter -= steps_back
                        counter = max(1, counter)
                        break
                # counter logic
                counter += 1
                if counter == n_max:
                    break
        # ─── CREATING LEFT AND RIGHT LINE ────────────────────────────────
        counter = 0
        while counter < n_max:
            sign = +1 if np.random.rand() > 0.4 else -1
            for point in range(np.random.randint(low=5, high=15)):
                width += sign * np.random.rand() * 5
                if width < width_min:
                    width = width_min
                    sign = +1
                if width > width_max:
                    width = width_max
                    sign = -1
                # width = max(5,min(20,width))
                data[counter, 3] = data[counter, 0] + width * np.cos(data[counter, 2] + 1 / 2 * np.pi)
                data[counter, 4] = data[counter, 1] + width * np.sin(data[counter, 2] + 1 / 2 * np.pi)
                data[counter, 5] = data[counter, 0] - width * np.cos(data[counter, 2] + 1 / 2 * np.pi)
                data[counter, 6] = data[counter, 1] - width * np.sin(data[counter, 2] + 1 / 2 * np.pi)
                counter += 1
                if counter == n_max:
                    break
        # todo: ─── IMPORTANT CALL AGAIN HERE if intersecting
        line1 = data[:, 3:5]
        line1[:, 0] = line1[:, 0] + WINDOW_WIDTH // 2
        line1[:, 1] = line1[:, 1] + WINDOW_HEIGHT // 2
        line2 = data[:, 5:7]
        line2[:, 0] = line2[:, 0] + WINDOW_WIDTH // 2
        line2[:, 1] = line2[:, 1] + WINDOW_HEIGHT // 2
        goals = np.zeros((n_max, 4))

        goals[:, [0, 1]] = line1.copy()
        goals[:, [2, 3]] = line2.copy()

        add_l1 = np.array([[-100 + WINDOW_WIDTH // 2, 0 + WINDOW_HEIGHT // 2],
                           [-50 + WINDOW_WIDTH // 2, 100 + WINDOW_HEIGHT // 2]])
        add_l2 = np.array([[-100 + WINDOW_WIDTH // 2, 0 + WINDOW_HEIGHT // 2],
                           [-50 + WINDOW_WIDTH // 2, -100 + WINDOW_HEIGHT // 2]])
        add_g = np.array([[-100 + WINDOW_WIDTH // 2, -100 + WINDOW_HEIGHT // 2,
                           -100 + WINDOW_WIDTH // 2, +100 + WINDOW_HEIGHT // 2],
                          [-.1 + WINDOW_WIDTH // 2, -100 + WINDOW_HEIGHT // 2,
                           -.11 + WINDOW_WIDTH // 2, +100 + WINDOW_HEIGHT // 2]])

        line1 = np.concatenate([add_l1, line1], axis=0)
        line2 = np.concatenate([add_l2, line2], axis=0)
        goals = np.concatenate([goals, add_g], axis=0)

        return line1, line2, goals


class Drone:
    VEL_MAX = 8 * ROOMS_SCALE
    DRAG = 0.96
    # ROT_VEL = 1.28
    # ROT_VEL = 0.64
    # ROT_VEL = 0.32
    ROT_VEL = 0.12
    TURN_VELOCITY_ALIGNMENT = 0.45
    # ROT_VEL = 0.04
    # ACCELERATION = 1.0
    ACCELERATION = 0.35 * ROOMS_SCALE
    # ACCELERATION = 0.3
    # ACCELERATION = 0.2
    # ACCELERATION = 0.05
    # ACCELERATION = 0.02
    # ACCELERATION = 0.01
    # ACCELERATION = 0.005
    N_ECHO = 7  # must be odd
    # N_ECHO = 15  # must be odd
    # N_ECHO = 31  # must be odd
    COVERAGE_REWARD = CHECKPOINT_REWARD

    def __init__(self, game, env):
        self.game = game
        self.env = env
        self.visible = True
        self.reset_game_state()

    def reset_game_state(self, x=300, y=200, ang=-np.pi, vel_x=0, vel_y=0, level=0):
        self.update_state(np.array([x, y, ang, vel_x, vel_y]))
        self.level = level
        self.level_previous = level
        # framecount_goal: since last goal
        self.framecount_goal = 0
        # framecount_total: since reset
        self.framecount_total = 0
        # reward: since last frame
        self.n_lap = 0
        self.reward_step = 0
        self.reward_total = 0
        self.done = False
        self.action = np.array([0, 0])
        self.action_state = 0
        self.last_accel_cmd = 0.0
        self.last_turn_cmd = 0.0
        self.collision_step = False
        self.coverage_collision_count = 0
        self.coverage_collision_count_since_checkpoint = 0
        self.coverage_collision_penalty_since_checkpoint = 0.0
        self.coverage_last_collision_penalty = 0.0
        self.goal_vector_last = None
        self.coverage_visited = np.zeros(self.env.n_goals, dtype=bool)
        self.coverage_count = 0
        self.coverage_count_previous = 0
        self.coverage_goal_features = np.zeros(COVERAGE_GOAL_SLOTS * 3, dtype=np.float32)
        self.coverage_progress_interp = -1.0
        self.coverage_stall_interp = -1.0
        self.coverage_target_point = np.array([self.x, self.y], dtype=float)
        self.coverage_target_visible = False
        self.coverage_target_distance = 0.0
        self.coverage_explored_cells = {
            (int(self.x // COVERAGE_CELL_SIZE), int(self.y // COVERAGE_CELL_SIZE))
        }
        self.coverage_hover_penalty_total = 0.0
        self.coverage_last_hover_penalty = 0.0
        self.coverage_collision_penalty_total = 0.0
        self.coverage_progress_reward_total = 0.0
        self.coverage_progress_penalty_total = 0.0
        self.coverage_last_progress_reward = 0.0
        self.coverage_last_progress_penalty = 0.0
        self.coverage_target_index_previous = None
        self.coverage_target_distance_previous = None
        self.update_echo_vectors()
        self.update_goal_vectors()
        self.check_collision_echo()

    def update_state(self, drone_state):
        self.x = drone_state[0]
        self.y = drone_state[1]
        self.ang = drone_state[2]
        self.vel_x = drone_state[3]
        self.vel_y = drone_state[4]

    def update_reward_continuous(self):
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.env.n_goals - 1:
            self.n_lap += 1
        if self.level == self.env.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1
        distance0 = np.sqrt((self.x - self.xi0) ** 2 + (self.y - self.yi0) ** 2)
        distance1 = np.sqrt((self.x - self.xi1) ** 2 + (self.y - self.yi1) ** 2)
        self.reward_total = self.n_lap * self.env.n_goals + self.level + 1 * (distance0 / (distance0 + distance1))
        self.reward_step = self.reward_total - reward_total_previous

    def update_reward_dynamic(self):  # dynamic
        if (self.level - self.level_previous == 1) or (self.level == 0 and self.level_previous == self.env.n_goals - 1):
            self.reward_step = max(1, (500 - self.framecount_goal)) / 500
            self.reward_total += self.reward_step
            self.framecount_goal = 0
        if (self.level - self.level_previous == -1) or (
                self.level == self.env.n_goals - 1 and self.level_previous == 0):
            self.reward_step = - 1
            self.reward_total += self.reward_step
            self.framecount_goal = 0

    def update_reward_static(self):  # static
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.env.n_goals - 1:
            self.n_lap += 1
        if self.level == self.env.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1
        self.reward_total = self.n_lap * self.env.n_goals + self.level
        self.reward_step = self.reward_total - reward_total_previous

    def update_reward_coverage(self):
        new_hits = self.coverage_count - self.coverage_count_previous
        self.reward_step = float(new_hits) * self.COVERAGE_REWARD
        if new_hits == 0 and self.coverage_last_hover_penalty > 0:
            self.reward_step -= self.coverage_last_hover_penalty
            self.coverage_hover_penalty_total += self.coverage_last_hover_penalty
        if new_hits == 0 and self.coverage_last_progress_reward > 0:
            self.reward_step += self.coverage_last_progress_reward
            self.coverage_progress_reward_total += self.coverage_last_progress_reward
        if new_hits == 0 and self.coverage_last_progress_penalty > 0:
            self.reward_step -= self.coverage_last_progress_penalty
            self.coverage_progress_penalty_total += self.coverage_last_progress_penalty
        if self.collision_step:
            self.reward_step -= self.coverage_last_collision_penalty
            self.coverage_collision_penalty_total += self.coverage_last_collision_penalty
        self.reward_total = (
            float(self.coverage_count) * self.COVERAGE_REWARD
            + self.coverage_progress_reward_total
            - self.coverage_hover_penalty_total
            - self.coverage_collision_penalty_total
            - self.coverage_progress_penalty_total
        )
        self.coverage_count_previous = self.coverage_count

    def update_coverage_exploration_state(self):
        self.coverage_last_hover_penalty = 0.0
        self.coverage_last_progress_reward = 0.0
        self.coverage_last_progress_penalty = 0.0
        if self.game.reward_mode != 'coverage':
            return
        cell = (int(self.x // COVERAGE_CELL_SIZE), int(self.y // COVERAGE_CELL_SIZE))
        new_cell = cell not in self.coverage_explored_cells
        if new_cell:
            self.coverage_explored_cells.add(cell)

        speed = np.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        if speed < COVERAGE_HOVER_SPEED_THRESHOLD:
            self.coverage_last_hover_penalty = COVERAGE_HOVER_PENALTY
        if self.collision_step:
            self.coverage_last_hover_penalty = max(self.coverage_last_hover_penalty, COVERAGE_BLOCKED_WALL_PENALTY)

        candidates = self.get_unvisited_goal_candidates()
        if not candidates:
            self.coverage_target_index_previous = None
            self.coverage_target_distance_previous = None
            return

        _, distance, target_index, _, visible, _, _ = candidates[0]
        if visible and self.coverage_target_index_previous == target_index and self.coverage_target_distance_previous is not None:
            progress = self.coverage_target_distance_previous - distance
            if progress > COVERAGE_PROGRESS_MARGIN:
                self.coverage_last_progress_reward = min(
                    COVERAGE_PROGRESS_REWARD_CAP,
                    (progress - COVERAGE_PROGRESS_MARGIN) * COVERAGE_PROGRESS_REWARD_SCALE,
                )
            elif not (new_cell and not self.collision_step):
                self.coverage_last_progress_penalty = COVERAGE_PROGRESS_PENALTY
        self.coverage_target_index_previous = target_index
        self.coverage_target_distance_previous = distance

    def get_unvisited_goal_candidates(self):
        unvisited = np.flatnonzero(~self.coverage_visited)
        candidates = []
        for goal_index in unvisited:
            goal = self.env.get_goal_line(goal_index)
            px, py, visible = self.get_checkpoint_target_point(goal)
            dx, dy = px - self.x, py - self.y
            distance = np.sqrt(dx ** 2 + dy ** 2)
            goal_ang = np.arctan2(-dy, dx)
            goal_ang_diff = wrap_angle(self.ang - goal_ang)
            candidates.append((not visible, distance, int(goal_index), goal_ang_diff, visible, px, py))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def get_checkpoint_target_point(self, goal):
        midpoint = line_segment_midpoint(*goal)
        visible_points = []
        for t in np.linspace(CHECKPOINT_TARGET_TRIM, 1 - CHECKPOINT_TARGET_TRIM, CHECKPOINT_TARGET_SAMPLES):
            px, py = point_on_line_segment(*goal, t)
            if self.is_goal_visible((px, py)):
                distance = np.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)
                center_distance = np.sqrt((midpoint[0] - px) ** 2 + (midpoint[1] - py) ** 2)
                visible_points.append((center_distance, distance, px, py))
        if visible_points:
            _, _, px, py = min(visible_points, key=lambda item: item[0])
            return px, py, True
        return midpoint[0], midpoint[1], False

    def update_goal_vectors(self):
        if self.game.reward_mode == 'coverage':
            candidates = self.get_unvisited_goal_candidates()
            if candidates:
                _, distance, target_index, _, visible, px, py = candidates[0]
                self.level = target_index
                self.coverage_target_point = np.array([px, py], dtype=float)
                self.coverage_target_visible = bool(visible)
                self.coverage_target_distance = float(distance)
            else:
                self.level = 0
                self.coverage_target_point = np.array([self.x, self.y], dtype=float)
                self.coverage_target_visible = False
                self.coverage_target_distance = 0.0
            self.goal_vector_next = self.env.get_goal_line(self.level)
            self.goal_vector_last = self.goal_vector_next
            return
        self.goal_vector_next = self.env.get_goal_line(self.level)
        self.goal_vector_last = self.env.get_goal_line(self.level - 1)

    def is_goal_visible(self, point):
        return self.is_point_visible(point, margin=CHECKPOINT_VISIBILITY_WALL_MARGIN)

    def is_point_visible(self, point, margin=3):
        for wall in self.env.level_collision_vectors:
            if point_to_line_segment_distance(self.x, self.y, *wall) <= SENSOR_ORIGIN_WALL_BLOCK_MARGIN:
                return False
        distance_to_goal = np.sqrt((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2)
        sight_line = [self.x, self.y, point[0], point[1]]
        for wall in self.env.level_collision_vectors:
            if line_segment_distance(*sight_line, *wall) >= margin:
                continue
            result = line_intersect_tolerant(*sight_line, *wall)
            if result is not None:
                distance_to_wall = np.sqrt((self.x - result[0]) ** 2 + (self.y - result[1]) ** 2)
                if distance_to_wall < distance_to_goal - margin:
                    return False
            else:
                return False
        return True

    def is_motion_to_point_clear(self, point, margin=3):
        start = (self.x_previous, self.y_previous)
        distance_to_point = np.sqrt((start[0] - point[0]) ** 2 + (start[1] - point[1]) ** 2)
        sight_line = [start[0], start[1], point[0], point[1]]
        for wall in self.env.level_collision_vectors:
            result = line_intersect_tolerant(*sight_line, *wall)
            if result is None:
                continue
            distance_to_wall = np.sqrt((start[0] - result[0]) ** 2 + (start[1] - result[1]) ** 2)
            if distance_to_wall < distance_to_point - margin:
                return False
        return True

    def checkpoint_crossing_point(self, goal):
        crossing = line_intersect_tolerant(*self.movement_vector, *goal)
        if crossing is not None:
            return crossing

        x0, y0, x1, y1 = self.movement_vector
        gx0, gy0, gx1, gy1 = goal
        d0 = signed_distance_to_line(x0, y0, gx0, gy0, gx1, gy1)
        d1 = signed_distance_to_line(x1, y1, gx0, gy0, gx1, gy1)
        if d0 * d1 > 0:
            return None
        if abs(d0) < 1e-6 and abs(d1) < 1e-6:
            return None

        ratio = abs(d0) / max(abs(d0) + abs(d1), 1e-6)
        px = x0 + ratio * (x1 - x0)
        py = y0 + ratio * (y1 - y0)
        fraction = point_line_segment_fraction(px, py, *goal)
        if not (CHECKPOINT_CROSS_TRIM <= fraction <= 1 - CHECKPOINT_CROSS_TRIM):
            return None
        distance = point_to_line_segment_distance(px, py, *goal)
        if distance > CHECKPOINT_CROSS_TOLERANCE:
            return None
        return closest_point_on_line_segment(px, py, *goal)

    def update_coverage_goal_features(self):
        features = np.zeros(COVERAGE_GOAL_SLOTS * 3, dtype=np.float32)
        features[1::3] = 1.0
        features[2::3] = -1.0
        candidates = self.get_unvisited_goal_candidates()[:1]
        for slot, (_, distance, _, goal_ang_diff, visible, _, _) in enumerate(candidates):
            base = slot * 3
            features[base] = np.interp(goal_ang_diff, [-np.pi, np.pi], [-1, 1])
            features[base + 1] = np.interp(
                min(distance, GOAL_DISTANCE_NORM),
                [0, GOAL_DISTANCE_NORM],
                [-1, 1],
            )
            features[base + 2] = 1.0 if visible else -1.0

        self.coverage_goal_features = features
        self.coverage_progress_interp = np.interp(self.coverage_count, [0, max(1, self.env.n_goals)], [-1, 1])
        self.coverage_stall_interp = np.interp(min(self.framecount_goal, 200), [0, 200], [-1, 1])

    def update_echo_vectors(self):
        n = self.N_ECHO
        if n % 2 == 0: n = max(n - 1, 3)  # make sure that n>=3 and odd
        n_sideangles = int((n - 1) / 2)  # 7 -> 3
        matrix = np.zeros((n, 4))
        matrix[:, 0] = int(self.x)
        matrix[:, 1] = int(self.y)
        # straight angle
        matrix[n_sideangles, 2] = int(self.x + ECHO_RAY_LENGTH * np.cos(self.ang))
        matrix[n_sideangles, 3] = int(self.y - ECHO_RAY_LENGTH * np.sin(self.ang))
        # angles from 90 deg to 0
        # ignore first angle
        angles = np.linspace(0, np.pi / 2, n_sideangles + 1)
        for i in range(n_sideangles):
            # first side
                matrix[i, 2] = int(self.x + ECHO_RAY_LENGTH * np.cos(self.ang + angles[i + 1]))  # x2
                matrix[i, 3] = int(self.y - ECHO_RAY_LENGTH * np.sin(self.ang + angles[i + 1]))  # y2
                # second side
                matrix[-(i + 1), 2] = int(self.x + ECHO_RAY_LENGTH * np.cos(self.ang - angles[i + 1]))  # x2
                matrix[-(i + 1), 3] = int(self.y - ECHO_RAY_LENGTH * np.sin(self.ang - angles[i + 1]))  # y2
        self.echo_vectors = matrix

    def rotate(self, rotate):  # input: action1
        heading_delta = self.ROT_VEL * rotate
        self.ang = self.ang + heading_delta
        # get angular in range of -pi,pi
        if self.ang > np.pi:
            self.ang = self.ang - 2 * np.pi
        if self.ang < -np.pi:
            self.ang = self.ang + 2 * np.pi
        if abs(heading_delta) > 1e-6:
            velocity_delta = heading_delta * self.TURN_VELOCITY_ALIGNMENT
            cos_delta = np.cos(velocity_delta)
            sin_delta = np.sin(velocity_delta)
            # Screen y is positive downward, so this rotates velocity in heading coordinates.
            vel_x = self.vel_x * cos_delta + self.vel_y * sin_delta
            vel_y = -self.vel_x * sin_delta + self.vel_y * cos_delta
            self.vel_x, self.vel_y = vel_x, vel_y

    def accelerate(self, accelerate):  # input: action0
        self.vel_x = self.DRAG * (self.vel_x + self.ACCELERATION * accelerate * np.cos(self.ang))
        self.vel_y = self.DRAG * (self.vel_y - self.ACCELERATION * accelerate * np.sin(self.ang))

        speed = np.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        if speed > self.VEL_MAX:
            scale = self.VEL_MAX / speed
            self.vel_x *= scale
            self.vel_y *= scale

    def update_observations(self):
        # ─── OBSERVATION 8: VELOCITY ─────────────────────────────────────
        vel = np.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        self.vel_interp = np.interp(vel, [0, VELOCITY_NORM], [-1, 1])

        # ─── OBSERVATION 9: VELOCITY ANGLE ───────────────────────────────
        # get angular difference
        vel_ang = np.arctan2(-self.vel_y, self.vel_x)
        vel_ang_diff = self.ang - vel_ang

        # set between -pi and pi
        if vel_ang_diff > np.pi:
            vel_ang_diff = vel_ang_diff - 2 * np.pi
        if vel_ang_diff < -np.pi:
            vel_ang_diff = vel_ang_diff + 2 * np.pi
        if self.vel_interp < 0.001 - 1:
            vel_ang_diff = 0

        # normalize
        self.vel_ang_diff_interp = np.interp(vel_ang_diff, [-np.pi, np.pi], [-1, 1])
        if self.game.reward_mode == 'coverage':
            self.update_coverage_goal_features()

        # ─── OBSERVATION 10: GOAL ANGLE ──────────────────────────────────
        def get_intersection_point(xp, yp, x1, y1, x2, y2):
            # check if line is vertical (infinite slope)
            if x1 == x2:
                return (x1, yp)
            if y1 == y2:
                y2 += 1e-2
            # slope: dy/dx
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            xi = (xp * (1 / a) + yp - b) * 1 / (a + 1 / a)
            yi = a * xi + b
            return xi, yi

        xp, yp = self.x, self.y

        if self.game.reward_mode == 'coverage':
            self.xi0, self.yi0 = self.coverage_target_point
            self.xi1, self.yi1 = self.coverage_target_point
        else:
            # from drone to next goal line direction
            goal_next = self.goal_vector_next
            goal_last = self.goal_vector_last

            x1, y1, x2, y2 = goal_last
            self.xi0, self.yi0 = get_intersection_point(xp, yp, x1, y1, x2, y2)

            x1, y1, x2, y2 = goal_next
            self.xi1, self.yi1 = get_intersection_point(xp, yp, x1, y1, x2, y2)

        dx, dy = self.xi1 - self.x, self.yi1 - self.y
        goal_ang = np.arctan2(-dy, dx)
        goal_ang_diff = self.ang - goal_ang

        if goal_ang_diff > np.pi:
            goal_ang_diff = goal_ang_diff - 2 * np.pi
        if goal_ang_diff < -np.pi:
            goal_ang_diff = goal_ang_diff + 2 * np.pi

        self.goal_ang_diff_interp = np.interp(goal_ang_diff, [-np.pi, np.pi], [-1, 1])

    def move(self, action):
        # first apply rotation!
        self.rotate(action[1])
        self.accelerate(action[0])

        # displacement
        d_x, d_y = self.vel_x, self.vel_y
        x_from, y_from = self.x, self.y
        self.x_previous, self.y_previous = x_from, y_from

        # ─── CENTERED MODE ───────────────────────────────────────────────
        if self.game.camera_mode == 'centered':
            self.env.move_env(d_x, d_y)
            self.movement_vector = [WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, WINDOW_WIDTH / 2 + d_x,
                                    WINDOW_HEIGHT / 2 + d_y]
        # ─── FIXED MODE ─────────────────────────────────────────────────
        if self.game.camera_mode == 'fixed':
            self.x = self.x + d_x
            self.y = self.y + d_y
            self.movement_vector = [x_from, y_from, self.x, self.y]

        # ─── KEEP ON SCREEN ──────────────────────────────────────────────
        # drone cannot leave fixed screen area
        if self.game.rule_keep_on_screen:
            if self.x > WINDOW_WIDTH:
                self.x = self.x - WINDOW_WIDTH
            elif self.x < 0:
                self.x = self.x + WINDOW_WIDTH
            if self.y > WINDOW_HEIGHT:
                self.y = self.y - WINDOW_HEIGHT
            elif self.y < 0:
                self.y = self.y + WINDOW_HEIGHT

    def check_collision_goal(self):
        if self.game.reward_mode == 'coverage':
            found_new_checkpoint = False
            for i, goal in enumerate(self.env.goals):
                if self.coverage_visited[i]:
                    continue
                crossing = self.checkpoint_crossing_point(goal)
                hit_goal = crossing is not None and self.is_motion_to_point_clear(crossing)
                if hit_goal:
                    self.coverage_visited[i] = True
                    self.coverage_count += 1
                    found_new_checkpoint = True
            if found_new_checkpoint:
                self.framecount_goal = 0
                self.coverage_collision_count_since_checkpoint = 0
                self.coverage_collision_penalty_since_checkpoint = 0.0
            self.update_goal_vectors()
            if self.coverage_count >= self.env.n_goals:
                self.game.set_done(reason="all_checkpoints")
            return

        result_last = line_intersect(*self.movement_vector, *self.goal_vector_last)
        result_next = line_intersect(*self.movement_vector, *self.goal_vector_next)
        if result_last is not None:
            self.level -= 1
            if self.level == -1:
                self.level = self.env.n_goals - 1
            self.update_goal_vectors()
        elif result_next is not None:
            self.level += 1
            if self.level == self.env.n_goals:
                self.level = 0
            if self.game.env == 'random':
                if self.level == self.game.env_random_length:
                    self.game.set_done()
            self.update_goal_vectors()

    def check_collision_env(self):
        self.collision_step = False
        self.coverage_last_collision_penalty = 0.0
        for line in self.env.level_collision_vectors:
            result = line_intersect(*self.movement_vector, *line)
            too_close = line_segment_distance(*self.movement_vector, *line) <= ROBOT_WALL_CLEARANCE
            if result is not None or too_close:
                if self.game.reward_mode == 'coverage' and not self.game.coverage_collision_ends_episode:
                    self.collision_step = True
                    self.coverage_collision_count += 1
                    self.coverage_collision_count_since_checkpoint += 1
                    next_penalty_since_checkpoint = COVERAGE_COLLISION_PENALTY_CAP * (
                        1 - (1 - COVERAGE_COLLISION_PENALTY_RATE) ** self.coverage_collision_count_since_checkpoint
                    )
                    self.coverage_last_collision_penalty = (
                        next_penalty_since_checkpoint - self.coverage_collision_penalty_since_checkpoint
                    )
                    self.coverage_collision_penalty_since_checkpoint = next_penalty_since_checkpoint
                    self.x, self.y = self.x_previous, self.y_previous
                    self.vel_x, self.vel_y = 0.0, 0.0
                    self.movement_vector = [self.x, self.y, self.x, self.y]
                    self.update_echo_vectors()
                    self.check_collision_echo()
                else:
                    self.game.set_done(reason="collision")
                break

    def check_collision_echo(self):
        # max_distance: Distance value maps to observation=1 if distance >= max_distance
        max_distance = ECHO_MAX_DISTANCE
        points = np.full((self.N_ECHO, 2), self.x)  # points for visualiziation
        points[:, 1] = self.y
        distances = np.full((self.N_ECHO), max_distance)  # distances for observation
        checkpoint_points = np.full((self.N_ECHO, 2), self.x)
        checkpoint_points[:, 1] = self.y
        checkpoint_indices = np.full((self.N_ECHO), -1, dtype=int)
        checkpoint_distances = np.full((self.N_ECHO), max_distance)
        if self.game.reward_mode == 'coverage':
            candidate_goal_indices = np.flatnonzero(~self.coverage_visited)
        else:
            candidate_goal_indices = np.arange(self.env.n_goals)

        n = self.env.level_collision_vectors.shape[0]
        for wall in self.env.level_collision_vectors:
            if point_to_line_segment_distance(self.x, self.y, *wall) <= SENSOR_ORIGIN_WALL_BLOCK_MARGIN:
                closest = closest_point_on_line_segment(self.x, self.y, *wall)
                points[:, :] = closest
                distances[:] = 0.0
                self.echo_collision_points = points
                self.echo_checkpoint_points = checkpoint_points
                self.echo_checkpoint_indices = checkpoint_indices
                self.echo_checkpoint_distances = checkpoint_distances
                self.echo_collision_distances_interp = np.interp(distances, [0, ECHO_MAX_DISTANCE], [-1, 1])
                return

        for i in range(self.N_ECHO):
            found = False
            line1 = self.echo_vectors[i, :]
            points_candidates = np.zeros((n, 2))
            distances_candidates = np.full((n), max_distance)
            for j, line2 in enumerate(self.env.level_collision_vectors):
                result = line_intersect(*line1, *line2)
                if result is not None:
                    found = True
                    points_candidates[j, :] = result
                    distances_candidates[j] = np.sqrt((self.x - result[0]) ** 2 + (self.y - result[1]) ** 2)
            if found:  # make sure one intersection is found
                argmin = np.argmin(distances_candidates)  # index of closest intersection 
                points[i, :] = points_candidates[argmin]
                distances[i] = distances_candidates[argmin]

            nearest_checkpoint = max_distance
            for goal_index in candidate_goal_indices:
                goal = self.env.goals[goal_index]
                result = line_intersect(*line1, *goal)
                if result is None:
                    continue
                distance = np.sqrt((self.x - result[0]) ** 2 + (self.y - result[1]) ** 2)
                goal_fraction = point_line_segment_fraction(result[0], result[1], *goal)
                if (
                    CHECKPOINT_TARGET_TRIM <= goal_fraction <= 1 - CHECKPOINT_TARGET_TRIM
                    and distance < distances[i] - CHECKPOINT_VISIBILITY_WALL_MARGIN
                    and self.is_point_visible(result, margin=CHECKPOINT_VISIBILITY_WALL_MARGIN)
                    and distance < nearest_checkpoint
                ):
                    nearest_checkpoint = distance
                    checkpoint_points[i, :] = result
                    checkpoint_indices[i] = int(goal_index)
            checkpoint_distances[i] = nearest_checkpoint

        self.echo_collision_points = points
        self.echo_checkpoint_points = checkpoint_points
        self.echo_checkpoint_indices = checkpoint_indices
        self.echo_checkpoint_distances = checkpoint_distances
        # ─── NORMALIZE DISTANCES ─────────────────────────────────────────
        # linear mapping from 0 to ECHO_MAX_DISTANCE into [-1, 1]
        # values always in range [-1,1]
        self.echo_collision_distances_interp = np.interp(distances, [0, ECHO_MAX_DISTANCE], [-1, 1])


class ExploreDrone(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_config={}):
        self.parse_env_config(env_config)
        self.win = None
        self.display_surface = None
        self.clock = None
        self.pygame_initialized = False
        self.display_initialized = False
        self.render_count = 0
        self._last_rgb_array = None
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32)
        self.env = Environment(self)
        self.drone = Drone(self, self.env)
        observation_size = self.drone.N_ECHO + 3
        if self.reward_mode == 'coverage':
            observation_size += COVERAGE_GOAL_SLOTS * 3 + 2
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(observation_size,),
            dtype=np.float32)
        self.spectator = None

        self.reset()
        # exit()

    def get_observation(self):
        distances = self.drone.echo_collision_distances_interp
        velocity = self.drone.vel_interp
        vel_ang_diff = self.drone.vel_ang_diff_interp
        goal_ang_diff = self.drone.goal_ang_diff_interp
        observation = np.concatenate((distances, np.array([velocity, vel_ang_diff, goal_ang_diff])))
        if self.reward_mode == 'coverage':
            coverage_state = np.array([
                self.drone.coverage_progress_interp,
                self.drone.coverage_stall_interp,
            ])
            observation = np.concatenate((observation, self.drone.coverage_goal_features, coverage_state))

        non_finite_mask = ~np.isfinite(observation)
        if non_finite_mask.any():
            logger.error(
                "Non-finite raw observation detected: observation=%s mask=%s. This shows instability in the training parameters!",
                observation,
                non_finite_mask.astype(np.int8),
            )

        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def parse_env_config(self, env_config):
        keyword_dict = {
            # these are all available keyboards and valid values respectively
            # the first value in the list is the default value
            'gui': [True, False],
            'camera_mode': ['fixed', 'centered'],
            'env_name': [
                'default',
                'empty',
                'level1',
                'level2',
                'random',
                'playground',
                'rooms',
                '2d_checkpoint_exploration',
            ],
            'env_random_length': [50, 'any', int],  # length of randomly generated environment
            'env_flipped': [False, True],  # activates normal environment, flipped
            'env_flipmode': [False, True],  # activates flip mode. Each reset() flips env
            'env_visible': [True, False],
            'reward_mode': ['dynamic', 'continuous', 'static', 'coverage'],  # choose reward mode
            'export_frames': [False, True],  # export rendered frames
            'export_states': [False, True],  # export every step
            'export_string': ['', 'any', str],  # string for export filename
            'export_highscore': [0, 'any', int],  # only export if highscore is beat
            'max_steps': [1000, 'any', int],
            'rule_collision': [True, False],
            'rule_max_steps': [True, False],
            'rule_keep_on_screen': [False, True],
            'coverage_collision_ends_episode': [False, True],
            'gui_echo_distances': [False, True],
            'gui_frames_remaining': [True, False],
            'gui_goal_ang': [False, True],
            'gui_level': [False, True],
            'gui_reward_total': [True, False],
            'gui_velocity': [False, True],
            'gui_draw_echo_points': [True, False],
            'gui_draw_echo_vectors': [True, False],
            'gui_draw_goal_all': [True, False],
            'gui_draw_goal_next': [True, False],
            'gui_draw_goal_points': [False, True],
            'render_mode': [None, 'human', 'rgb_array'],
            'render_every': [1, 'any', int],
            'render_fps': [30, 'any', int],
            'rooms_layout_path': ['', 'any', str],
            'spawn_mode': ['fixed', 'random'],
            'spawn_index': [0, 'any', int],
        }

        # ─── STEP 1 GET DEFAULT VALUE ────────────────────────────────────
        assign_dict = {}
        for keyword in keyword_dict:
            # asign default value form keyword_dict
            assign_dict[keyword] = keyword_dict[keyword][0]

        # ─── STEP 2 GET VALUE FROM env_config ─────────────────────────────
        for keyword in env_config:
            if keyword in keyword_dict:
                # possible keyword proceed with assigning
                if env_config[keyword] in keyword_dict[keyword]:
                    # valid value passed, assign
                    assign_dict[keyword] = env_config[keyword]
                elif 'any' in keyword_dict[keyword]:
                    # any value is allowed, assign if type matches
                    if isinstance(env_config[keyword], keyword_dict[keyword][2]):
                        assign_dict[keyword] = env_config[keyword]
                    else:
                        print('error: wrong type. type needs to be: ', keyword_dict[keyword][2])
                else:
                    print('given keyword exists, but given value is illegal')
            else:
                print('passed keyword does not exist: ', keyword)

        # ─── ASSIGN DEFAULT VALUES ───────────────────────────────────────
        self.camera_mode = assign_dict['camera_mode']
        self.env_name = assign_dict['env_name']
        self.env_random_length = assign_dict['env_random_length']
        self.env_flipped = assign_dict['env_flipped']
        self.env_flipmode = assign_dict['env_flipmode']
        self.env_visible = assign_dict['env_visible']
        self.reward_mode = assign_dict['reward_mode']
        self.export_frames = assign_dict['export_frames']
        self.export_states = assign_dict['export_states']
        self.export_string = assign_dict['export_string']
        self.export_highscore = assign_dict['export_highscore']
        self.max_steps = assign_dict['max_steps']
        self.rule_collision = assign_dict['rule_collision']
        self.rule_max_steps = assign_dict['rule_max_steps']
        self.rule_keep_on_screen = assign_dict['rule_keep_on_screen']
        self.coverage_collision_ends_episode = assign_dict['coverage_collision_ends_episode']
        self.gui = assign_dict['gui']
        self.gui_echo_distances = assign_dict['gui_echo_distances']
        self.gui_frames_remaining = assign_dict['gui_frames_remaining']
        self.gui_goal_ang = assign_dict['gui_goal_ang']
        self.gui_level = assign_dict['gui_level']
        self.gui_reward_total = assign_dict['gui_reward_total']
        self.gui_velocity = assign_dict['gui_velocity']
        self.gui_draw_echo_points = assign_dict['gui_draw_echo_points']
        self.gui_draw_echo_vectors = assign_dict['gui_draw_echo_vectors']
        self.gui_draw_goal_all = assign_dict['gui_draw_goal_all']
        self.gui_draw_goal_next = assign_dict['gui_draw_goal_next']
        self.gui_draw_goal_points = assign_dict['gui_draw_goal_points']
        self.render_mode = assign_dict['render_mode']
        if self.render_mode is None and self.gui and 'render_mode' not in env_config:
            self.render_mode = 'human'
        self.render_every = max(1, assign_dict['render_every'])
        self.render_fps = max(1, assign_dict['render_fps'])
        self.rooms_layout_path = assign_dict['rooms_layout_path']
        self.spawn_mode = assign_dict['spawn_mode']
        self.spawn_index = assign_dict['spawn_index']
        self.spawn_rng = np.random.default_rng()
        self.done_reason = "running"

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.spawn_rng = np.random.default_rng(seed)

        # ─── FLIP MIRROR ─────────────────────────────────────────────────
        if self.env_flipmode:
            if self.env_flipped:
                self.env_flipped = False
            else:
                self.env_flipped = True
            self.env.load_level()

        # if self.env == 'random' or self.camera_mode == 'centered':
        if self.camera_mode == 'centered':
            self.env.load_level()

        # ─── RESET EXPORT VARIALBES ──────────────────────────────────────
        # give unique session id for export
        self.session_id = str(int(np.random.rand(1) * 10 ** 6)).zfill(6)
        # dim0 : n_steps | dim1 : frame, x,y,ang,velx,vely
        self.statematrix = np.zeros((self.max_steps, 7))

        # ─── RESET drone ──────────────────────────────────────────────────
        self.done_reason = "running"
        self.reset_drone_state()
        # generate observation
        self.drone.update_observations()
        empty_dict = {}
        return self.get_observation(), empty_dict
        # return observation10, None

    def set_spectator_state(self, state, colors=[], frame=None):
        self.drone.visible = False
        self.spectator = state
        self.spectator_colorlist = colors
        if frame:
            self.drone.framecount_total = frame

    # def reset_drone_state(self, x=200, y=100, ang=1e-9, vel_x=0, vel_y=0, level=0):  # ang=1e-10
    def get_rooms_spawn_pose(self):
        poses = ROOMS_SPAWN_POSES.copy()
        poses[:, :2] *= ROOMS_SCALE
        if self.spawn_mode == 'random':
            index = int(self.spawn_rng.integers(0, len(poses)))
        else:
            index = self.spawn_index % len(poses)
        return poses[index]

    def reset_drone_state(self, x=300, y=200, ang=np.pi, vel_x=0, vel_y=0, level=0):  # ang=1e-10
        if self.env_name in ['rooms', '2d_checkpoint_exploration']:
            x, y, ang = self.get_rooms_spawn_pose()
            vel_x, vel_y = 0, 0
        elif self.env_name == 'random':
            x, y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        elif self.env_flipped:
            # mirror over y-axis
            x, ang, vel_x = -x + WINDOW_WIDTH, np.pi - ang, -vel_x
        # if camera_mode is centerd, the drone needs to go center too
        if self.camera_mode == 'centered':
            diff_x = WINDOW_WIDTH // 2 - x
            diff_y = WINDOW_HEIGHT // 2 - y
            # move environment
            if self.env_name in ['default', 'level1', 'level2']:
                self.env.move_env(-diff_x, -diff_y)
            # move player
            x, y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        return self.drone.reset_game_state(x, y, ang, vel_x, vel_y, level)

    def set_done(self, reason="done"):
        self.done_reason = reason
        self.drone.done = True
        if (self.export_states) and (self.drone.reward_total > self.export_highscore):
            import os
            # copy last state to remaining frames
            i = self.drone.framecount_total
            n_new = self.max_steps - i
            self.statematrix[i:, :] = np.repeat(self.statematrix[i - 1, :].reshape((1, 7)), n_new, axis=0)
            # mark at which frame agent is done
            self.statematrix[i:, 0] = 0

            # export
            filename = '_'.join([self.export_string,
                                 '-'.join([self.session_id, str(int(self.drone.reward_total)).zfill(4)])
                                 ])
            filenamepath = os.path.join('exported_states', filename)
            os.makedirs(os.path.dirname(filenamepath), exist_ok=True)
            np.save(filenamepath, self.statematrix)

    def step(self, action=None):
        # ─── NORMALIZE ACTION ────────────────────────────────────────────
        # action = [action_acc, action_turn]
        truncated = False
        if action is None:
            action = [0, 0]
        tmp_action = np.zeros(2)
        tmp_action[0] = max(min(action[0], 1), -1)
        tmp_action[1] = max(min(action[1], 1), -1)

        # set action_state for image
        self.drone.action = tmp_action.copy()
        self.drone.last_accel_cmd = float(tmp_action[0])
        self.drone.last_turn_cmd = float(tmp_action[1])
        if self.drone.action[0] < 0:
            self.drone.action_state = 2
        else:
            self.drone.action_state = 0
        if self.drone.action[0] > 0:
            self.drone.action_state = 1

        # ─── PERFORM STEP ───────────────────-─────────────────────────────
        if not self.drone.done:
            self.drone.move(tmp_action)
            # print(f"[STEP 0] action={action}, pos=({self.drone.x:.2f},{self.drone.y:.2f}), vel=({self.drone.vel_x:.3f},{self.drone.vel_y:.3f}), reward={self.drone.reward_step:.4f}")
            self.drone.update_echo_vectors()
            if self.rule_collision:
                self.drone.check_collision_goal()
                if self.done_reason == "all_checkpoints":
                    self.drone.collision_step = False
                    self.drone.coverage_last_collision_penalty = 0.0
                else:
                    self.drone.check_collision_echo()
                    self.drone.check_collision_env()
            if self.done_reason != "all_checkpoints":
                self.drone.update_coverage_exploration_state()
            else:
                self.drone.coverage_last_hover_penalty = 0.0
                self.drone.coverage_last_progress_penalty = 0.0
            self.drone.update_observations()

            # ─── EXPORT GAME STATE ───────────────────────────────────────────
            if self.export_states:
                i = self.drone.framecount_total
                # frame, x,y,ang,velx,vely
                self.statematrix[i, :] = [i, self.drone.x, self.drone.y, self.drone.ang,
                                          self.drone.vel_x, self.drone.vel_y, self.drone.action_state]

            self.drone.framecount_goal += 1
            self.drone.framecount_total += 1

            if self.reward_mode == 'static':
                self.drone.update_reward_static()
            elif self.reward_mode == 'dynamic':
                self.drone.update_reward_dynamic()
            elif self.reward_mode == 'coverage':
                self.drone.update_reward_coverage()
            else:
                # make default
                self.drone.update_reward_continuous()

            if self.rule_max_steps and not self.drone.done:
                if self.drone.framecount_total >= self.max_steps:
                    truncated = True
                    self.set_done(reason="time_limit")

        # ─── GET RETURN VARIABLES ────────────────────────────────────────
        reward = self.drone.reward_step
        done = self.drone.done and self.done_reason != "time_limit"
        info = {
            "x": self.drone.x,
            "y": self.drone.y,
            "ang": self.drone.ang,
            "done_reason": self.done_reason}
        if self.reward_mode == 'coverage':
            info.update({
                "visited_checkpoints": int(self.drone.coverage_count),
                "total_checkpoints": int(self.env.n_goals),
                "coverage_ratio": float(self.drone.coverage_count / max(1, self.env.n_goals)),
                "checkpoint_reward": float(self.drone.COVERAGE_REWARD),
                "max_reward": float(self.env.n_goals * self.drone.COVERAGE_REWARD),
                "hover_penalty": float(self.drone.coverage_hover_penalty_total),
                "last_hover_penalty": float(self.drone.coverage_last_hover_penalty),
                "progress_reward": float(self.drone.coverage_progress_reward_total),
                "last_progress_reward": float(self.drone.coverage_last_progress_reward),
                "collision": bool(self.drone.collision_step),
                "collision_count": int(self.drone.coverage_collision_count),
                "collision_count_since_checkpoint": int(self.drone.coverage_collision_count_since_checkpoint),
                "collision_penalty_cap": float(COVERAGE_COLLISION_PENALTY_CAP),
                "collision_penalty_since_checkpoint": float(self.drone.coverage_collision_penalty_since_checkpoint),
                "last_collision_penalty": float(self.drone.coverage_last_collision_penalty),
                "collision_penalty_total": float(self.drone.coverage_collision_penalty_total),
                "progress_penalty": float(self.drone.coverage_progress_penalty_total),
                "last_progress_penalty": float(self.drone.coverage_last_progress_penalty),
            })

        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.drone.reward_step = 0
        self.drone.level_previous = self.drone.level
        return self.get_observation(), reward, done, truncated,  info

    def render(self, mode=None):
        render_mode = self.render_mode if mode is None else mode
        if render_mode is None:
            return None
        if render_mode not in ('human', 'rgb_array'):
            raise ValueError("render mode must be None, 'human', or 'rgb_array'")

        self.render_count += 1
        if self.render_every > 1 and (self.render_count - 1) % self.render_every != 0:
            if render_mode == 'rgb_array':
                return self._last_rgb_array
            return None

        # initialize pygame only when rendering is requested
        import pygame
        import os
        from PIL import Image
        middle_echo_index = (self.drone.N_ECHO - 1) // 2

        def init_renderer(self, render_mode):
            if render_mode == 'human':
                pygame.init()
            else:
                pygame.font.init()
            self.pygame_initialized = True

            asset_dir = os.path.join(os.path.dirname(__file__), 'imgs')

            def load_robot_image(filename):
                image = pygame.transform.scale2x(pygame.image.load(os.path.join(asset_dir, filename)))
                width, height = image.get_size()
                return pygame.transform.smoothscale(
                    image,
                    (int(width * ROBOT_RENDER_SCALE), int(height * ROBOT_RENDER_SCALE)),
                )

            self.drone_IMG = [
                load_robot_image('tank_no_power.png'),
                load_robot_image('tank_power.png'),
                load_robot_image('tank_power_front.png'),
                load_robot_image('tank_black.png'),
            ]
            self.BG_IMG = pygame.transform.scale(
                pygame.image.load(os.path.join(asset_dir, 'white_bg.jpg')),
                (WINDOW_WIDTH, WINDOW_HEIGHT),
            )
            self.clock = pygame.time.Clock() if render_mode == 'human' else None
            if render_mode == 'human':
                pygame.display.set_caption("Exploring robot")
                self.display_surface = pygame.display.set_mode(
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                self.display_initialized = True
            else:
                self.display_surface = None
                self.display_initialized = False
            self.win = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

            if self.export_frames:
                self.image3d = np.ndarray(
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT, 3), np.uint8)

            self.gui_interface = []
            if self.gui_reward_total:
                self.gui_interface.append('reward_total')
            if self.gui_level:
                self.gui_interface.append('level')
            if self.gui_echo_distances:
                self.gui_interface.append('echo_distances')
            if self.gui_goal_ang:
                self.gui_interface.append('goal_ang')
            if self.gui_velocity:
                self.gui_interface.append('velocity')
            if self.gui_frames_remaining:
                self.gui_interface.append('frames_remaining')

        def draw_level():
            # pygame.draw.lines(self.win, (178, 190, 181), False, self.env.line1_list, 7)
            # pygame.draw.lines(self.win, (178, 190, 181), False, self.env.line2_list, 7)
            for wall in self.env.level_collision_vectors:
                pygame.draw.line(self.win, COLOR_BLACK, wall[:2], wall[2:], 7)

        def draw_goal_next():
            goal = tuple(self.env.goals[self.drone.level, :])
            # pygame.draw.lines(self.win, (60, 230, 255), False,
            #   (goal[0:2],goal[2:4]), 4)
            pygame.draw.lines(self.win, (2, 250, 155), False,
                              (goal[0:2], goal[2:4]), 7)

        def draw_goal_all():
            for i in range(self.env.goals.shape[0]):
                goal = tuple(self.env.goals[i, :])
                color = COLOR_CHECKPOINT
                if self.reward_mode == 'coverage' and self.drone.coverage_visited[i]:
                    color = COLOR_VISITED
                pygame.draw.lines(self.win, color, False, (goal[0:2], goal[2:4]), 7)

        def draw_motion_overlay():
            x, y = int(self.drone.x), int(self.drone.y)
            heading = self.drone.ang
            accel = self.drone.last_accel_cmd
            turn = self.drone.last_turn_cmd
            speed = np.sqrt(self.drone.vel_x ** 2 + self.drone.vel_y ** 2)
            pulse = 0.5 + 0.5 * np.sin(self.drone.framecount_total * 0.45)

            def draw_action_label(text, color, pos):
                font = pygame.font.Font(pygame.font.match_font('consolas'), 18)
                label = font.render(text, True, color)
                rect = label.get_rect(center=pos)
                bg = rect.inflate(10, 6)
                bg.clamp_ip(self.win.get_rect())
                rect.center = bg.center
                pygame.draw.rect(self.win, COLOR_PANEL_BG, bg)
                pygame.draw.rect(self.win, color, bg, 2)
                self.win.blit(label, rect)

            def draw_arrow(start, end, color, width):
                pygame.draw.line(self.win, color, start, end, width)
                angle = np.arctan2(start[1] - end[1], end[0] - start[0])
                head_len = 14 + int(5 * pulse)
                for offset in (0.65, -0.65):
                    head = (
                        int(end[0] - head_len * np.cos(angle + offset)),
                        int(end[1] + head_len * np.sin(angle + offset)),
                    )
                    pygame.draw.line(self.win, color, end, head, max(2, width - 2))

            # Velocity vector shows actual motion, independent from the commanded action.
            if speed > 0.1:
                vx_end = (int(x + self.drone.vel_x * 8), int(y + self.drone.vel_y * 8))
                draw_arrow((x, y), vx_end, COLOR_VELOCITY, 4)
                pygame.draw.circle(self.win, COLOR_VELOCITY, vx_end, 5)

            # Forward and reverse command indicator.
            if abs(accel) > 0.05:
                color = COLOR_THRUST if accel > 0 else COLOR_REVERSE
                label = "ACCEL" if accel > 0 else "REVERSE"
                forward = np.array([np.cos(heading), -np.sin(heading)])
                command_dir = forward if accel > 0 else -forward
                length = 58 + 28 * abs(accel) + 12 * pulse
                start_vec = np.array([x, y], dtype=float) + command_dir * 22
                end_vec = np.array([x, y], dtype=float) + command_dir * length
                start = (int(start_vec[0]), int(start_vec[1]))
                end = (int(end_vec[0]), int(end_vec[1]))
                draw_arrow(start, end, color, 8)
                pygame.draw.circle(self.win, color, end, 9)
                draw_action_label(label, color, (end[0], end[1]))

            # Turning command indicator.
            if abs(turn) > 0.05:
                radius = 48 + int(8 * pulse)
                rect = pygame.Rect(x - radius, y - radius, 2 * radius, 2 * radius)
                if turn > 0:
                    start_ang, end_ang = heading - 0.2, heading + 1.0
                    label = "TURN R"
                else:
                    start_ang, end_ang = heading - 1.0, heading + 0.2
                    label = "TURN L"
                pygame.draw.arc(self.win, COLOR_TURN, rect, start_ang, end_ang, 5)
                tip_ang = end_ang if turn > 0 else start_ang
                tip = (
                    int(x + radius * np.cos(tip_ang)),
                    int(y - radius * np.sin(tip_ang)),
                )
                pygame.draw.circle(self.win, COLOR_TURN, tip, 7)
                draw_action_label(label, COLOR_TURN, (tip[0], tip[1] + 28))

        def draw_drone():
            self.drone.img = self.drone_IMG[self.drone.action_state]
            # pygame.transform.rotate takes angle in degree
            rotated_image = pygame.transform.rotate(
                self.drone.img, self.drone.ang / np.pi * 180)
            new_rect = rotated_image.get_rect(center=self.drone.img.get_rect(
                center=(self.drone.x, self.drone.y)).center)
            self.win.blit(rotated_image, new_rect.topleft)

        def draw_spectators():
            if not self.drone.visible and self.spectator is not None:
                for i, row in enumerate(self.spectator):
                    framecount_total, x, y, ang, vel_x, vel_y, action_state = row
                    image = self.drone_IMG[int(action_state)]
                    # pygame.transform.rotate takes angle in degree
                    rotated_image = pygame.transform.rotate(
                        image, ang / np.pi * 180)
                    new_rect = rotated_image.get_rect(center=image.get_rect(
                        center=(x, y)).center)
                    self.win.blit(rotated_image, new_rect.topleft)
                    # color marker
                    if self.spectator_colorlist:
                        color = self.spectator_colorlist[i]
                        pygame.draw.circle(self.win, color, (int(x), int(y)), 10)

        def draw_goal_intersection_points():
            pygame.draw.circle(self.win, (250, 0, 250), (int(self.drone.xi0), int(self.drone.yi0)), 6)
            pygame.draw.circle(self.win, (250, 0, 250), (int(self.drone.xi1), int(self.drone.yi1)), 6)

        def draw_echo_vector():
            n = self.drone.N_ECHO
            visual_ray_length = RENDER_RAY_LENGTH
            for i, vector in enumerate(self.drone.echo_vectors):
                start = vector[0:2].astype(float)
                ray_end = vector[2:4].astype(float)
                direction = ray_end - start
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.array([1.0, 0.0])

                end = start + direction * visual_ray_length
                end_distance = visual_ray_length
                if len(self.drone.echo_collision_points) == n:
                    hit = self.drone.echo_collision_points[i].astype(float)
                    hit_distance = np.linalg.norm(hit - start)
                    valid_hit = hit_distance > 2 and not np.allclose(hit, [0, 0])
                    if valid_hit and hit_distance <= visual_ray_length:
                        end = hit
                        end_distance = hit_distance

                if hasattr(self.drone, 'echo_checkpoint_points') and len(self.drone.echo_checkpoint_points) == n:
                    checkpoint = self.drone.echo_checkpoint_points[i].astype(float)
                    checkpoint_distance = np.linalg.norm(checkpoint - start)
                    valid_checkpoint = (
                        self.drone.echo_checkpoint_indices[i] >= 0
                        and checkpoint_distance > 2
                        and checkpoint_distance <= end_distance + 1e-6
                    )
                    if valid_checkpoint:
                        end = checkpoint

                width = 5 if i == middle_echo_index else 3
                pygame.draw.line(self.win, COLOR_RAY, start, end, width)

        def draw_echo_collision_points():
            start = np.array([self.drone.x, self.drone.y], dtype=float)
            for point in self.drone.echo_collision_points:
                point = point.astype(float)
                hit_distance = np.linalg.norm(point - start)
                if hit_distance <= 2 or hit_distance > RENDER_RAY_LENGTH or np.allclose(point, [0, 0]):
                    continue
                pygame.draw.circle(self.win, COLOR_RAY_HIT, (int(point[0]), int(point[1])), 7)
                pygame.draw.circle(self.win, COLOR_BLACK, (int(point[0]), int(point[1])), 7, 2)

        def draw_echo_checkpoint_points():
            if not hasattr(self.drone, 'echo_checkpoint_points'):
                return
            start = np.array([self.drone.x, self.drone.y], dtype=float)
            seen = set()
            for point, goal_index in zip(self.drone.echo_checkpoint_points, self.drone.echo_checkpoint_indices):
                if goal_index < 0 or goal_index in seen:
                    continue
                point = point.astype(float)
                hit_distance = np.linalg.norm(point - start)
                if hit_distance <= 2 or hit_distance > RENDER_RAY_LENGTH or np.allclose(point, [0, 0]):
                    continue
                seen.add(int(goal_index))
                pygame.draw.circle(self.win, COLOR_CHECKPOINT_HIT, (int(point[0]), int(point[1])), 11)
                pygame.draw.circle(self.win, COLOR_BLACK, (int(point[0]), int(point[1])), 11, 2)

        def draw_text(surface, text=None, size=30, x=0, y=0,
                      font_name=pygame.font.match_font('consolas'),
                      position='topleft'):
            font = pygame.font.Font(font_name, size)
            text_surface = font.render(text, True, COLOR_TEXT)
            text_rect = text_surface.get_rect()
            if position == 'topleft':
                text_rect.topleft = (x, y)
            if position == 'topright':
                text_rect.topright = (x, y)
            surface.blit(text_surface, text_rect)

        def draw_legend():
            items = [
                ("wall", COLOR_BLACK, "line"),
                ("checkpoint", COLOR_CHECKPOINT, "line"),
                ("visited", COLOR_VISITED, "line"),
                ("sensor ray", COLOR_RAY, "line"),
                ("ray hit", COLOR_RAY_HIT, "dot"),
                ("checkpoint hit", COLOR_CHECKPOINT_HIT, "dot"),
                ("accel", COLOR_THRUST, "arrow"),
                ("reverse", COLOR_REVERSE, "arrow"),
                ("turn", COLOR_TURN, "arc"),
                ("velocity", COLOR_VELOCITY, "arrow"),
            ]
            row_h = 24
            pad = 12
            width = 240
            height = pad * 2 + row_h * (len(items) + 1)
            x0 = WINDOW_WIDTH - width - 18
            y0 = 64
            panel = pygame.Surface((width, height), pygame.SRCALPHA)
            panel.fill((255, 255, 255, 245))
            self.win.blit(panel, (x0, y0))
            pygame.draw.rect(self.win, COLOR_PANEL_BORDER, (x0, y0, width, height), 2)
            draw_text(self.win, text="legend", size=18, x=x0 + pad, y=y0 + 8)
            y = y0 + pad + row_h
            for label, color, kind in items:
                sx = x0 + pad
                sy = y + row_h // 2
                if kind == "line":
                    pygame.draw.line(self.win, color, (sx, sy), (sx + 42, sy), 5)
                elif kind == "dot":
                    pygame.draw.circle(self.win, color, (sx + 20, sy), 7)
                    pygame.draw.circle(self.win, COLOR_BLACK, (sx + 20, sy), 7, 2)
                elif kind == "arc":
                    pygame.draw.arc(self.win, color, pygame.Rect(sx + 4, sy - 12, 28, 24), -0.6, 1.2, 4)
                    pygame.draw.circle(self.win, color, (sx + 31, sy - 3), 5)
                else:
                    pygame.draw.line(self.win, color, (sx, sy), (sx + 42, sy), 5)
                    pygame.draw.line(self.win, color, (sx + 42, sy), (sx + 30, sy - 8), 4)
                    pygame.draw.line(self.win, color, (sx + 42, sy), (sx + 30, sy + 8), 4)
                draw_text(self.win, text=label, size=16, x=x0 + 66, y=y + 2)
                y += row_h

        def get_gui_value(value: str):
            if value == 'reward_total':
                return str(round(self.drone.reward_total, 2))
            elif value == 'level':
                if self.reward_mode == 'coverage':
                    return f"{self.drone.coverage_count}/{self.env.n_goals}"
                return str(self.drone.level)
            elif value == 'echo_distances':
                return str(round(self.drone.echo_collision_distances_interp[middle_echo_index], 2))
            elif value == 'velocity':
                return str(round(np.sqrt(self.drone.vel_x ** 2 + self.drone.vel_y ** 2), 2))
            elif value == 'goal_ang':
                return str(round(self.drone.goal_ang_diff_interp, 2))
            elif value == 'frames_remaining':
                return str(self.max_steps - self.drone.framecount_total)
                # return str(self.drone.framecount_total)
            else:
                return 'value not found'

        # ─── INIT RENDERER ───────────────────────────────────────────────
        if self.win is None:
            init_renderer(self, render_mode)
        elif render_mode == 'human' and not self.display_initialized:
            pygame.init()
            pygame.display.set_caption("Exploring robot")
            self.display_surface = pygame.display.set_mode(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            self.clock = pygame.time.Clock()
            self.display_initialized = True

        # ─── RECURING RENDERING ──────────────────────────────────────────
        self.win.blit(self.BG_IMG, (0, 0))
        if self.env_visible:
            draw_level()
        if self.gui_draw_goal_all:
            draw_goal_all()
        if self.gui_draw_goal_next and self.reward_mode != 'coverage':
            draw_goal_next()
        if self.gui_draw_echo_vectors:
            draw_echo_vector()
        if self.gui_draw_echo_points:
            draw_echo_collision_points()
            draw_echo_checkpoint_points()
        if self.gui_draw_goal_points:
            draw_goal_intersection_points()
        if self.drone.visible:
            draw_motion_overlay()
            draw_drone()
        draw_spectators()

        # ─── INTERFACE ───────────────────────────────────────────────────
        if self.gui:
            gui_n = len(self.gui_interface)
            gui_x_pad = 10
            if gui_n == 1:
                gui_x_list = [WINDOW_WIDTH - gui_x_pad]
            else:
                gui_x_list = np.linspace(0 + gui_x_pad, WINDOW_WIDTH - gui_x_pad, gui_n)
            for i in range(gui_n):
                key = self.gui_interface[i]
                pos = 'topright' if (i == gui_n - 1) else 'topleft'
                # draw key
                draw_text(self.win, text=key,
                          size=15, x=gui_x_list[i], y=8, position=pos)
                # draw value
                draw_text(self.win, text=get_gui_value(key),
                          size=30, x=gui_x_list[i], y=20, position=pos)
            draw_legend()

        # ─── RENDER GAME ─────────────────────────────────────────────────
        frame_surface = self.win
        if DISPLAY_SCALE != 1.0:
            frame_surface = pygame.transform.smoothscale(
                self.win,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            )

        if render_mode == 'human':
            pygame.event.pump()
            if DISPLAY_SCALE == 1.0:
                self.display_surface.blit(self.win, (0, 0))
            else:
                self.display_surface.blit(frame_surface, (0, 0))
            pygame.display.update()
            self.clock.tick(self.render_fps)

        if render_mode == 'rgb_array' or self.export_frames:
            frame_array = pygame.surfarray.array3d(frame_surface)
            self._last_rgb_array = np.transpose(frame_array, axes=[1, 0, 2])

        # ─── EXPORT GAME FRAMES ──────────────────────────────────────────
        if self.export_frames:
            im = Image.fromarray(self._last_rgb_array)  # monochromatic image
            imrgb = im.convert('RGB')  # color image

            filename = ''.join([
                self.export_string,
                self.session_id,
                '-frame-',
                str(self.drone.framecount_total).zfill(5),
                '.jpg'])
            filenamepath = os.path.join('exported_frames', filename)
            os.makedirs(os.path.dirname(filenamepath), exist_ok=True)
            imrgb.save(filenamepath)

        if render_mode == 'rgb_array':
            return self._last_rgb_array
        return None

    def close(self):
        if self.pygame_initialized:
            import pygame
            pygame.quit()
        self.win = None
        self.display_surface = None
        self.clock = None
        self.pygame_initialized = False
        self.display_initialized = False
        self._last_rgb_array = None

    def get_drone_state(self):
        return np.array([
            self.drone.x,
            self.drone.y,
            self.drone.ang,
            self.drone.vel_x,
            self.drone.vel_y,
        ])

    def update_drone_state(self, drone_state):
        self.drone.update_state(drone_state)

    def update_interface_vars(self, action_next):
        self.action_next = action_next
