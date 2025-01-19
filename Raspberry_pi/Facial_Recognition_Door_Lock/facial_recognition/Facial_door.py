import os
import time
import pickle
import cv2
import imutils
import face_recognition
from imutils.video import VideoStream, FPS
import RPi.GPIO as GPIO

# GPIO setup
SERVO_PIN = 18  # Servo Motor
BUZZER_PIN = 21  # Buzzer
RED_LED_PIN = 17  # Red LED
GREEN_LED_PIN = 9  # Green LED

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Set up PWM for the servo motor
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
pwm.start(0)  # Initialize PWM with 0 duty cycle

def set_servo_angle(angle):
    """Sets the servo to the specified angle."""
    duty = 2 + (angle / 18)  # Duty cycle calculation for the angle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

def buzz_alarm():
    """Sounds the buzzer for a short duration."""
    GPIO.output(BUZZER_PIN, True)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, False)

def blink_led(pin, times, delay=0.2):
    """Blinks an LED a specified number of times."""
    for _ in range(times):
        GPIO.output(pin, True)
        time.sleep(delay)
        GPIO.output(pin, False)
        time.sleep(delay)

# Load encodings and known names
ENCODINGS_PATH = "encodings.pickle"
print("[INFO] Loading encodings and face detector...")
data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

# Initialize the video stream
print("[INFO] Starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# State variables
door_open = False
last_face_time = time.time()

# Main loop
try:
    while True:
        # Read a frame from the video stream
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # Detect faces in the frame
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        face_detected = False
        recognized = False

        # Compare detected faces with known encodings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Count votes for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Get the name with the highest count
                name = max(counts, key=counts.get)
                recognized = True

            names.append(name)

            # Open the door and blink green LED for recognized faces
            if recognized and not door_open:
                print(f"[INFO] {name} detected. Opening the door.")
                set_servo_angle(90)  # Open the door
                door_open = True
                blink_led(GREEN_LED_PIN, 2)  # Blink green LED twice

            # Sound the alarm and blink red LED for unknown faces
            if name == "Unknown":
                print("[WARNING] Unknown face detected. Triggering alarm.")
                buzz_alarm()
                blink_led(RED_LED_PIN, 3)  # Blink red LED three times

        # Draw bounding boxes and names on the frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2 if name != "Unknown" else (0, 0, 255))
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        # Close the door if no recognized face is detected for a certain period
        if not recognized:
            if door_open and time.time() - last_face_time > 3:  # 3-second timeout
                print("[INFO] No recognized face detected. Closing the door.")
                set_servo_angle(0)  # Close the door
                door_open = False
        else:
            last_face_time = time.time()  # Reset the timer if a face is recognized

        # Display the frame
        cv2.imshow("Facial Recognition System", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit the loop if 'q' is pressed
        if key == ord("q"):
            break

        # Update FPS counter
        fps.update()

except KeyboardInterrupt:
    print("\n[INFO] Exiting program...")

finally:
    # Clean up resources
    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
    cv2.destroyAllWindows()
    vs.stop()
    pwm.stop()
    GPIO.cleanup()
