from controller import Robot, DistanceSensor, InertialUnit
from dstar_lite_core import dstar_lite
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())
sensor = robot.getDevice("so4")
sensor.enable(timestep)

# IMU 
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)
start_time = robot.getTime()


# Motors
back_left = robot.getDevice("back left wheel")
back_right = robot.getDevice("back right wheel")
front_left = robot.getDevice("front left wheel")
front_right = robot.getDevice("front right wheel")

motors = [back_left, back_right, front_left, front_right]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

def object_detection(duration=2.0):
    readings = []
    start = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - start >= duration:
            break
        readings.append(sensor.getValue())

    if readings:
        readings.sort()
        mid = len(readings) // 2
        median = (readings[mid] + readings[mid - 1]) / 2 if len(readings) % 2 == 0 else readings[mid]
        median = round(median, 3)
        if median >= 850:
            print(f"Obstacle detected. Median distance: {median} → Update grid now.")
            return True
        else:
            print(f"Green light. Median distance: {median}")
    return False

def update_grid(row, col):
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        grid[row][col] = 9
        print(f"🧱 Grid updated at ({row}, {col}) → now marked as obstacle.")
        print("\n📍 Current Grid State:")
        for r in grid:
            print(" ".join(str(cell) for cell in r))
# --- GPS ---
gps = robot.getDevice("gps")
gps.enable(timestep)

def move_forward(target, speed=5, tol=0.1, slow_down_radius=0.5):
    tx, ty = target

    while robot.step(timestep) != -1:
        x, y, z = gps.getValues()
        dist = math.hypot(tx - x, ty - y)
        print(f"→ GPS pos=(x={x:.3f}, y={y:.3f}, z={z:.3f}), dist to target={dist:.3f}")

        if dist <= tol:
            break

        # Slow down as we approach the target
        if dist < slow_down_radius:
            adjusted_speed = speed * (dist / slow_down_radius)
        else:
            adjusted_speed = speed

        # Apply speed to all motors
        for m in motors:
            m.setVelocity(adjusted_speed)

    # Stop motors
    for m in motors:
        m.setVelocity(0.0)
    print(f"✅ Reached target within {tol} m → final pos=(x={x:.3f}, y={y:.3f}, z={z:.3f})")

    # stop
    for m in motors:
        m.setVelocity(0.0)
    print(f"✅ Reached target within {tol} m → final pos=(x={x:.3f}, y={y:.3f}, z={z:.3f})")

def get_yaw():
    """Return yaw from IMU in [0,360)."""
    r, p, y = imu.getRollPitchYaw()
    return (math.degrees(y) + 360) % 360

def shortest_diff(target, current):
    """
    Signed error from current→target in [-180,180).
    Positive means rotate CCW, negative means CW.
    """
    d = (target - current + 180) % 360 - 180
    return d

def _rotate_by(delta_deg, speed_rad=1.0, tol_deg=0.35):
    """Rotate in place by delta_deg (signed), stops when within tol_deg."""
    start = get_yaw()
    target = (start + delta_deg) % 360
    while robot.step(timestep) != -1:
        yaw = get_yaw()
        err = shortest_diff(target, yaw)
        # print(f" rotating… yaw={yaw:.1f}°, err={err:.1f}")
        if abs(err) <= tol_deg:
            break
        # decide direction
        direction = 1 if err > 0 else -1
        # even indices = left wheels, odd = right wheels
        for i, m in enumerate(motors):
            v = -speed_rad if i % 2 == 0 else speed_rad
            m.setVelocity(direction * v)
    # stop immediately
    for m in motors:
        m.setVelocity(0.0)

def rotate_45(direction):
    """Rotate exactly ±45° using IMU."""
    d = 45 if direction == 'right' else -45
    _rotate_by(d)

def rotate_90(direction):
    """Rotate exactly ±90° using IMU."""
    d = 90 if direction == 'right' else -90
    _rotate_by(d)

def rotate_180():
    """Rotate exactly 180° using IMU (always CCW)."""
    _rotate_by(180)
        
        
def normalize_angle(angle):
    return angle % 360

def rotate_to(target_angle):
    global current_angle, rotation_count, total_rotation_degrees

    angle_diff = (target_angle - current_angle) % 360
    print(f'target_angle: {target_angle}')
    print(f'angle_diff: {angle_diff}')

    if angle_diff == 45:
        rotate_45("right")
        rotation_count += 1
        total_rotation_degrees += 45
    elif angle_diff == 90:
        rotate_90("right")
        rotation_count += 1
        total_rotation_degrees += 90
    elif angle_diff == 135:
        rotate_90("right")
        rotate_45("right")
        rotation_count += 2
        total_rotation_degrees += 135
    elif angle_diff == 180:
        rotate_180()
        rotation_count += 1
        total_rotation_degrees += 180
    elif angle_diff == 225:
        rotate_90("left")
        rotate_45("left")
        rotation_count += 2
        total_rotation_degrees += 225
    elif angle_diff == 270:
        rotate_90("left")
        rotation_count += 1
        total_rotation_degrees += 90
    elif angle_diff == 315:
        rotate_45("left")
        rotation_count += 1
        total_rotation_degrees += 45

    current_angle = normalize_angle(target_angle)
    print(f"↪️ Facing: {current_angle}° after rotation.")
    
# Grid

# Flipped
grid = [
    [1, 2, 1, 1, 1],
    [9, 2, 9, 1, 9],
    [2, 2, 1, 2, 2],
    [1, 9, 1, 9, 2],
    [1, 2, 9, 1, 1]
]

start = (0, 0)
goal = (4, 4)


DIRECTION_TO_ANGLE = {
    (0, 1): 0, (1, 1): 45, (1, 0): 90, (1, -1): 135,
    (0, -1): 180, (-1, 0): 270, (-1, -1): 225, (-1, 1): 315
}

rotation_count = total_rotation_degrees = move_count = total_distance_moved = 0
current_angle = 0
current = start
replan_count = 0
dynamic_obstacle_count = 0
total_path_cost = 0
i = 1

# … up to your main loop

# … up in your loop 
while current != goal:
    path, nodes_expanded = dstar_lite(grid, current, goal)
    print("current path:", path, "nodes expanded:", nodes_expanded)
    if len(path) < 2:
        print("No valid path found. Stopping.")
        break

    for next_cell in path[1:]:
        # rotate in place to face the next cell
        dx, dy = next_cell[0] - current[0], next_cell[1] - current[1]
        direction = (dx, dy)
        target_angle = DIRECTION_TO_ANGLE.get(direction)
        if target_angle is None:
            print(f"Unknown direction {direction}, skipping")
            continue
        rotate_to(target_angle)

        # obstacle?
        if object_detection():
            update_grid(next_cell[0], next_cell[1])
            replan_count += 1
            dynamic_obstacle_count += 1
            # force A* to rerun from the same `current`
            break

        # convert grid coords (row, col) to world-space (x, y):
        world_x, world_y = next_cell[1], next_cell[0]
        move_forward((world_x, world_y))

        # bookkeeping
        move_count += 1
        total_distance_moved += math.hypot(dx, dy)
        total_path_cost += grid[next_cell[0]][next_cell[1]]
        current = next_cell

    else:
        # only executed if we didn’t break out for replanning
        break



path_length = len(path)
end_time = robot.getTime()
total_time = end_time - start_time

average_step_cost = total_path_cost / path_length if path_length > 0 else 0
average_move_distance = total_distance_moved / move_count if move_count > 0 else 0

print("\n📊 METRICS REPORT")
print(f"🕒 Total simulation time      : {total_time:.2f} seconds")
print(f"🔄 Total replans triggered    : {replan_count}")
print(f"🧱 Dynamic obstacles detected : {dynamic_obstacle_count}")
print(f"🧠 Nodes expanded (last A*)   : {nodes_expanded}")
print(f"↩️ Rotations made             : {rotation_count}")
print(f"🧭 Total degrees rotated      : {total_rotation_degrees}°")
print(f"🚶 Moves made                 : {move_count}")
print(f"📏 Total distance moved       : {total_distance_moved:.3f} meters")
print(f"🧭 Path length (steps)        : {path_length}")
print(f"💰 Total path cost            : {total_path_cost:.2f}")
print(f"📊 Average step cost          : {average_step_cost:.2f}")
print(f"📐 Average move distance      : {average_move_distance:.2f} meters")
