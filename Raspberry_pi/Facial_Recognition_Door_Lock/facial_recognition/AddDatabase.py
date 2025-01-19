import os
import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from imutils import paths
import face_recognition
import pickle
import RPi.GPIO as GPIO
from smbus2 import SMBus

# Ultrasonic Sensor Pins
TRIG = 18
ECHO = 24

# Push Button Pin
BUTTON = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull-up resistor for the button

# I2C LCD Configuration
I2C_ADDR = 0x27  # Adjust based on your LCD's address
LCD_WIDTH = 16   # 16 characters per line
LCD_CHR = True
LCD_CMD = False
LCD_LINE_1 = 0x80  # LCD memory location for the 1st line
LCD_LINE_2 = 0xC0  # LCD memory location for the 2nd line

# Initialize I2C bus
bus = SMBus(1)

# LCD Functions
def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.0005)

def lcd_byte(bits, mode):
    bits_high = mode | (bits & 0xF0) | 0x08
    bits_low = mode | ((bits << 4) & 0xF0) | 0x08
    bus.write_byte(I2C_ADDR, bits_high)
    lcd_toggle_enable(bits_high)
    bus.write_byte(I2C_ADDR, bits_low)
    lcd_toggle_enable(bits_low)

def lcd_toggle_enable(bits):
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, bits | 0x04)
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, bits & ~0x04)
    time.sleep(0.0005)

def lcd_display(message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(line, LCD_CMD)
    for char in message:
        lcd_byte(ord(char), LCD_CHR)

# Ultrasonic Sensor Distance Measurement
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()

    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Distance in cm
    return distance

# Create Directory for New Person
def get_next_person_directory():
    base_dir = "dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("Person")]
    next_person_id = len(existing_dirs) + 1
    person_dir = os.path.join(base_dir, f"Person{next_person_id}")
    os.makedirs(person_dir)
    return person_dir

# Capture Faces
def capture_faces(person_dir):
    cam = PiCamera()
    cam.resolution = (512, 304)
    cam.framerate = 10
    rawCapture = PiRGBArray(cam, size=(512, 304))

    img_counter = 0
    lcd_display("Press button", LCD_LINE_1)
    lcd_display("to start...", LCD_LINE_2)

    try:
        while True:
            if GPIO.input(BUTTON) == GPIO.LOW:  # Button is pressed
                lcd_display("Waiting for", LCD_LINE_1)
                lcd_display("person...", LCD_LINE_2)

                while GPIO.input(BUTTON) == GPIO.LOW:  # Button remains pressed
                    distance = measure_distance()
                    if distance < 100:  # Person detected within 100 cm
                        lcd_display("Person detected", LCD_LINE_1)
                        lcd_display(f"Dist: {distance:.1f} cm", LCD_LINE_2)
                        print(f"[INFO] Person detected at {distance:.2f} cm")

                        for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                            image = frame.array
                            img_name = os.path.join(person_dir, f"image_{img_counter}.jpg")
                            cv2.imwrite(img_name, image)
                            print(f"[INFO] {img_name} written!")
                            img_counter += 1
                            rawCapture.truncate(0)

                            lcd_display(f"Captured {img_counter}", LCD_LINE_1)
                            lcd_display(f"images.", LCD_LINE_2)

                            if img_counter >= 10 or GPIO.input(BUTTON) == GPIO.HIGH:
                                break

                    else:
                        lcd_display("No person in", LCD_LINE_1)
                        lcd_display("range.", LCD_LINE_2)
                        time.sleep(0.5)

                if img_counter >= 10:
                    break

    finally:
        cam.close()

# Train Model
def train_model():
    lcd_display("Processing", LCD_LINE_1)
    lcd_display("faces...", LCD_LINE_2)
    print("[INFO] Start processing faces...")

    imagePaths = list(paths.list_images("dataset"))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))

    lcd_display("Training done!", LCD_LINE_1)
    lcd_display("", LCD_LINE_2)

# Main Execution
if __name__ == "__main__":
    try:
        lcd_init()
        lcd_display("System starting", LCD_LINE_1)
        lcd_display("", LCD_LINE_2)
        time.sleep(2)

        person_dir = get_next_person_directory()
        capture_faces(person_dir)
        train_model()
    finally:
        lcd_display("System halted", LCD_LINE_1)
        lcd_display("", LCD_LINE_2)
        GPIO.cleanup()
