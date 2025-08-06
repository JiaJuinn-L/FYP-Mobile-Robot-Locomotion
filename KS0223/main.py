import RPi.GPIO as GPIO
import time
import math
from astar_core import astar

# === GPIO Setup ===
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Motor Pins
L_IN1, L_IN2, L_PWM1 = 20, 21, 0   # Upper Left
L_IN3, L_IN4, L_PWM2 = 22, 23, 1   # Lower Left
R_IN1, R_IN2, R_PWM1 = 24, 25, 12  # Upper Right
R_IN3, R_IN4, R_PWM2 = 26, 27, 13  # Lower Right

MOTOR_PINS = [L_IN1, L_IN2, L_IN3, L_IN4, R_IN1, R_IN2, R_IN3, R_IN4]
for pin in MOTOR_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# PWM setup
GPIO.setup(L_PWM1, GPIO.OUT)
GPIO.setup(L_PWM2, GPIO.OUT)
GPIO.setup(R_PWM1, GPIO.OUT)
GPIO.setup(R_PWM2, GPIO.OUT)

pwm_L1 = GPIO.PWM(L_PWM1, 100)
pwm_L2 = GPIO.PWM(L_PWM2, 100)
pwm_R1 = GPIO.PWM(R_PWM1, 100)
pwm_R2 = GPIO.PWM(R_PWM2, 100)

for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
    pwm.start(0)

# Ultrasonic Sensor Pins
TRIG = 14  # GPIO 14
ECHO = 4   # GPIO 4

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# === Helper Functions ===
def stop_all():
    for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
        pwm.ChangeDutyCycle(0)
    for pin in MOTOR_PINS:
        GPIO.output(pin, GPIO.LOW)

def move_forward(duration):
    #Set Motors 
    GPIO.output(L_IN1, GPIO.LOW); GPIO.output(L_IN2, GPIO.HIGH)
    GPIO.output(L_IN3, GPIO.HIGH); GPIO.output(L_IN4, GPIO.LOW)
    GPIO.output(R_IN1, GPIO.HIGH); GPIO.output(R_IN2, GPIO.LOW)
    GPIO.output(R_IN3, GPIO.LOW); GPIO.output(R_IN4, GPIO.HIGH)
    
    #Set Speed
    for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
        pwm.ChangeDutyCycle(27)
    
    time.sleep(duration)
    stop_all()
    time.sleep(0.3)

    
def rotate_90(direction):
    if direction == 'left':
        # Left-Sided Wheels Turn Backwards
        GPIO.output(L_IN1, GPIO.HIGH)
        GPIO.output(L_IN2, GPIO.LOW)
        GPIO.output(L_IN3, GPIO.LOW)
        GPIO.output(L_IN4, GPIO.HIGH)
        
        # Right-Sided Wheels Turn Backwards
        GPIO.output(R_IN1, GPIO.HIGH)
        GPIO.output(R_IN2, GPIO.LOW)
        GPIO.output(R_IN3, GPIO.LOW)
        GPIO.output(R_IN4, GPIO.HIGH)
        
        for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
            pwm.ChangeDutyCycle(50)
        time.sleep(.6)
        
    elif direction == 'right':
        # Right-Sided Wheels Turn Backwards
        GPIO.output(L_IN1, GPIO.LOW)
        GPIO.output(L_IN2, GPIO.HIGH)
        GPIO.output(L_IN3, GPIO.HIGH)
        GPIO.output(L_IN4, GPIO.LOW)
        
        # Left-Sided Wheels Turn Backwards
        GPIO.output(R_IN1, GPIO.LOW)
        GPIO.output(R_IN2, GPIO.HIGH)
        GPIO.output(R_IN3, GPIO.HIGH)
        GPIO.output(R_IN4, GPIO.LOW)
        
        for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
            pwm.ChangeDutyCycle(50)
        time.sleep(.7)
    
    else:
        print("Invalid Selection of Direction")
    
    
    
    stop_all()
    time.sleep(1)

def rotate_180():
    # Left-Sided Wheels Turn Backwards
    GPIO.output(L_IN1, GPIO.HIGH)
    GPIO.output(L_IN2, GPIO.LOW)
    GPIO.output(L_IN3, GPIO.LOW)
    GPIO.output(L_IN4, GPIO.HIGH)
        
    # Right-Sided Wheels Turn Backwards
    GPIO.output(R_IN1, GPIO.HIGH)
    GPIO.output(R_IN2, GPIO.LOW)
    GPIO.output(R_IN3, GPIO.LOW)
    GPIO.output(R_IN4, GPIO.HIGH)
        
    for pwm in [pwm_L1, pwm_L2, pwm_R1, pwm_R2]:
        pwm.ChangeDutyCycle(50)
    time.sleep(1.2)
    stop_all()
    time.sleep(0.3)
    
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.01)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > 0.04:
            break

    duration = pulse_end - pulse_start
    distance = duration * 17150
    return round(distance, 2)

def object_detected(threshold=15):
    d = get_distance()
    print(f"Ultrasonic distance: {d} cm")
    return d < threshold

# === A* and Grid Setup ===
grid = [[1]*10 for _ in range(10)]
grid[0][2] = grid[0][6] = 9
start = (0, 0)
goal = (9, 9)
DIRECTION_TO_ANGLE = {
    (0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270
}

current_angle = 0

def rotate_to(target_angle):
    global current_angle
    diff = (target_angle - current_angle) % 360
    
    if diff == 0:
        return
    elif diff == 90:
        rotate_90("right")
    elif diff == 270:
        rotate_90("left")
    elif diff == 180:
        rotate_180()
    else:
        print(f"Unsupported RotationL {diff}")
        
    current_angle = target_angle

# grid = [
#     [1,1,1,1,1],
#     [1,9,9,9,1],
#     [1,2,1,2,1],
#     [1,2,9,2,1],
#     [1,1,1,1,1]
#         ]
# staty = (0,0)
# goal = (4,4)

grid = [
    [1,1,1],
    [1,9,1],
    [1,2,1]
    ]
start = (0,0)
goal = (2,2)
# === Main Navigation ===
current = start
while current != goal:
    path, _ = astar(grid, current, goal)
    if not path or len(path) < 2:
        print("No path found.")
        break

    for next_cell in path[1:]:
        dx, dy = next_cell[0] - current[0], next_cell[1] - current[1]
        if (dx, dy) not in DIRECTION_TO_ANGLE:
            continue
        rotate_to(DIRECTION_TO_ANGLE[(dx, dy)])

        if object_detected():
            print(f"Obstacle at {next_cell}, updating grid.")
            grid[next_cell[0]][next_cell[1]] = 9
            break  # replan

        distance = math.sqrt(dx**2 + dy**2)
        move_forward(distance * 1.5)  # scale for 20cm cells
        current = next_cell
    
print("Simulation End")
# Cleanup
stop_all()
GPIO.cleanup()
