import cv2
import mediapipe as mp
import math
import time
import subprocess
import sys


# =========================================================
# MEDIAPIPE FACE LANDMARKER SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(options)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye):
    """
    Eye Aspect Ratio calculate karta hai.
    """

    vertical_1 = distance(
        eye[1],
        eye[5]
    )

    vertical_2 = distance(
        eye[2],
        eye[4]
    )

    horizontal = distance(
        eye[0],
        eye[3]
    )

    if horizontal == 0:
        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear


def get_point(face_landmarks, index, width, height):

    landmark = face_landmarks[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)


# =========================================================
# EYE LANDMARK INDEXES
# =========================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# =========================================================
# HEAD DIRECTION LANDMARKS
# =========================================================

NOSE = 1

LEFT_EAR = 234

RIGHT_EAR = 454

CHIN = 152


# =========================================================
# SETTINGS
# =========================================================

EAR_THRESHOLD = 0.22

DROWSY_TIME = 2.0

ALARM_COOLDOWN = 3.0


# =========================================================
# VARIABLES
# =========================================================

eyes_closed = False

blink_count = 0

eye_close_start = None

last_alarm_time = 0

focus_score = 100.0

last_focus_time = time.time()

current_status = "STARTING..."

head_direction = "UNKNOWN"


# =========================================================
# ALARM FUNCTION
# =========================================================

def start_alarm():

    try:

        subprocess.Popen(
            [
                sys.executable,
                "alarm.py"
            ],
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    except Exception as error:

        print(
            "Alarm start nahi hua:",
            error
        )


# =========================================================
# HEAD DIRECTION
# =========================================================

def get_head_direction(
    face_landmarks,
    width,
    height
):

    nose = get_point(
        face_landmarks,
        NOSE,
        width,
        height
    )

    left_ear = get_point(
        face_landmarks,
        LEFT_EAR,
        width,
        height
    )

    right_ear = get_point(
        face_landmarks,
        RIGHT_EAR,
        width,
        height
    )

    chin = get_point(
        face_landmarks,
        CHIN,
        width,
        height
    )


    # Face center

    face_center_x = (
        left_ear[0] +
        right_ear[0]
    ) / 2

    face_center_y = (
        left_ear[1] +
        right_ear[1]
    ) / 2


    # Face width

    face_width = abs(
        right_ear[0] -
        left_ear[0]
    )

    if face_width == 0:

        return "UNKNOWN"


    # Horizontal position

    horizontal_ratio = (
        nose[0] -
        face_center_x
    ) / face_width


    # Face height

    face_height = abs(
        chin[1] -
        face_center_y
    )

    if face_height == 0:

        return "UNKNOWN"


    # Vertical position

    vertical_ratio = (
        nose[1] -
        face_center_y
    ) / face_height


    # Direction detection

    if horizontal_ratio < -0.10:

        return "LOOKING LEFT"

    elif horizontal_ratio > 0.10:

        return "LOOKING RIGHT"

    elif vertical_ratio > 0.20:

        return "LOOKING DOWN"

    elif vertical_ratio < -0.20:

        return "LOOKING UP"

    else:

        return "LOOKING FORWARD"


# =========================================================
# DRAW TEXT HELPER
# =========================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # -----------------------------------------------------
    # CAMERA FRAME
    # -----------------------------------------------------

    success, frame = camera.read()

    if not success:

        print("Camera frame nahi mil raha.")

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    current_time = time.time()

    elapsed = (
        current_time -
        last_focus_time
    )

    last_focus_time = current_time


    # -----------------------------------------------------
    # RGB CONVERSION
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # MEDIAPIPE IMAGE
    # -----------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # FACE LANDMARK DETECTION
    # -----------------------------------------------------

    result = landmarker.detect(
        mp_image
    )


    # =====================================================
    # DASHBOARD BACKGROUND
    # =====================================================

    # Camera ke upar transparent-style dark dashboard
    # banane ke liye overlay use kar rahe hain.

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 350),
        (20, 20, 20),
        -1
    )


    # Transparency

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )


    # =====================================================
    # FACE FOUND
    # =====================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]


        # -------------------------------------------------
        # EYE POINTS
        # -------------------------------------------------

        left_eye_points = []

        right_eye_points = []


        for index in LEFT_EYE:

            point = get_point(
                face_landmarks,
                index,
                width,
                height
            )

            left_eye_points.append(
                point
            )


        for index in RIGHT_EYE:

            point = get_point(
                face_landmarks,
                index,
                width,
                height
            )

            right_eye_points.append(
                point
            )


        # -------------------------------------------------
        # EAR
        # -------------------------------------------------

        left_ear_value = calculate_ear(
            left_eye_points
        )

        right_ear_value = calculate_ear(
            right_eye_points
        )


        ear_value = (
            left_ear_value +
            right_ear_value
        ) / 2


        # -------------------------------------------------
        # DRAW EYE POINTS
        # -------------------------------------------------

        for point in left_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1
            )


        for point in right_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1
            )


        # -------------------------------------------------
        # HEAD DIRECTION
        # -------------------------------------------------

        head_direction = get_head_direction(
            face_landmarks,
            width,
            height
        )


        # =================================================
        # EYES CLOSED
        # =================================================

        if ear_value < EAR_THRESHOLD:

            if not eyes_closed:

                eyes_closed = True

                eye_close_start = current_time


            closed_duration = (
                current_time -
                eye_close_start
            )


            # ---------------------------------------------
            # FOCUS DECREASE
            # ---------------------------------------------

            focus_score -= (
                3.0 *
                elapsed
            )


            # ---------------------------------------------
            # DROWSINESS
            # ---------------------------------------------

            if closed_duration >= DROWSY_TIME:

                current_status = "DROWSY"


                # Alarm cooldown

                if (
                    current_time -
                    last_alarm_time
                    >= ALARM_COOLDOWN
                ):

                    start_alarm()

                    last_alarm_time = (
                        current_time
                    )


            else:

                current_status = "EYES CLOSED"


        # =================================================
        # EYES OPEN
        # =================================================

        else:

            if eyes_closed:

                blink_count += 1


            eyes_closed = False

            eye_close_start = None


            # ---------------------------------------------
            # FOCUS RECOVERY
            # ---------------------------------------------

            focus_score += (
                1.5 *
                elapsed
            )


            current_status = "EYES OPEN"


        # =================================================
        # HEAD DIRECTION EFFECT
        # =================================================

        if head_direction == "LOOKING FORWARD":

            focus_score += (
                1.5 *
                elapsed
            )


        elif head_direction == "LOOKING LEFT":

            focus_score -= (
                2.0 *
                elapsed
            )


        elif head_direction == "LOOKING RIGHT":

            focus_score -= (
                2.0 *
                elapsed
            )


        elif head_direction == "LOOKING DOWN":

            focus_score -= (
                2.5 *
                elapsed
            )


        elif head_direction == "LOOKING UP":

            focus_score -= (
                1.5 *
                elapsed
            )


        # =================================================
        # LIMIT FOCUS SCORE
        # =================================================

        focus_score = max(
            0,
            min(
                100,
                focus_score
            )
        )


        # =================================================
        # FINAL FOCUS STATUS
        # =================================================

        if current_status == "DROWSY":

            focus_status = "DROWSY"

        elif focus_score >= 80:

            focus_status = "FOCUSED"

        elif focus_score >= 50:

            focus_status = "DISTRACTED"

        else:

            focus_status = "LOW FOCUS"


        # =================================================
        # FACE CENTER POINT
        # =================================================

        nose_point = get_point(
            face_landmarks,
            NOSE,
            width,
            height
        )


        cv2.circle(
            frame,
            nose_point,
            5,
            (0, 255, 255),
            -1
        )


        # =================================================
        # DASHBOARD TEXT
        # =================================================

        draw_text(
            frame,
            "AI SMART STUDENT ASSISTANT",
            (25, 40),
            0.9,
            2
        )


        draw_text(
            frame,
            f"EAR: {ear_value:.3f}",
            (25, 85),
            0.65,
            2
        )


        draw_text(
            frame,
            f"BLINKS: {blink_count}",
            (25, 125),
            0.65,
            2
        )


        # -------------------------------------------------
        # CLOSED TIME
        # -------------------------------------------------

        if eyes_closed:

            closed_time = (
                current_time -
                eye_close_start
            )

        else:

            closed_time = 0


        draw_text(
            frame,
            f"EYES CLOSED: {closed_time:.1f}s",
            (25, 165),
            0.65,
            2
        )


        draw_text(
            frame,
            f"HEAD: {head_direction}",
            (25, 205),
            0.65,
            2
        )


        draw_text(
            frame,
            f"FOCUS SCORE: {focus_score:.0f}%",
            (25, 245),
            0.65,
            2
        )


        draw_text(
            frame,
            f"STATUS: {focus_status}",
            (25, 285),
            0.65,
            2
        )


        # =================================================
        # FOCUS PROGRESS BAR
        # =================================================

        bar_x = 330

        bar_y = 220

        bar_width = 400

        bar_height = 28


        # Background

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (70, 70, 70),
            -1
        )


        # Filled width

        filled_width = int(
            bar_width *
            (focus_score / 100)
        )


        # Bar color

        if focus_score >= 80:

            bar_color = (
                0,
                200,
                0
            )

        elif focus_score >= 50:

            bar_color = (
                0,
                200,
                255
            )

        else:

            bar_color = (
                0,
                0,
                255
            )


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + filled_width,
                bar_y + bar_height
            ),
            bar_color,
            -1
        )


        # Bar outline

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (255, 255, 255),
            2
        )


        draw_text(
            frame,
            "FOCUS",
            (
                bar_x,
                bar_y - 10
            ),
            0.6,
            2
        )


        # =================================================
        # STATUS BOX
        # =================================================

        status_x = 780

        status_y = 215

        status_width = 350

        status_height = 65


        if focus_status == "FOCUSED":

            status_color = (
                0,
                170,
                0
            )

        elif focus_status == "DISTRACTED":

            status_color = (
                0,
                170,
                255
            )

        elif focus_status == "DROWSY":

            status_color = (
                0,
                0,
                255
            )

        else:

            status_color = (
                0,
                0,
                200
            )


        cv2.rectangle(
            frame,
            (
                status_x,
                status_y
            ),
            (
                status_x + status_width,
                status_y + status_height
            ),
            status_color,
            -1
        )


        cv2.rectangle(
            frame,
            (
                status_x,
                status_y
            ),
            (
                status_x + status_width,
                status_y + status_height
            ),
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            focus_status,
            (
                status_x + 25,
                status_y + 42
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # =====================================================
    # FACE NOT FOUND
    # =====================================================

    else:

        current_status = "FACE NOT DETECTED"

        head_direction = "UNKNOWN"


        # Focus decrease

        focus_score -= (
            3.0 *
            elapsed
        )


        focus_score = max(
            0,
            min(
                100,
                focus_score
            )
        )


        # Dashboard

        draw_text(
            frame,
            "AI SMART STUDENT ASSISTANT",
            (25, 40),
            0.9,
            2
        )


        draw_text(
            frame,
            "FACE NOT DETECTED",
            (25, 100),
            0.9,
            2
        )


        draw_text(
            frame,
            f"FOCUS SCORE: {focus_score:.0f}%",
            (25, 150),
            0.7,
            2
        )


        draw_text(
            frame,
            "Please look at the camera",
            (25, 195),
            0.65,
            2
        )


        # ---------------------------------------------
        # Empty progress bar
        # ---------------------------------------------

        bar_x = 25

        bar_y = 235

        bar_width = 500

        bar_height = 28


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (70, 70, 70),
            -1
        )


        filled_width = int(
            bar_width *
            (focus_score / 100)
        )


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + filled_width,
                bar_y + bar_height
            ),
            (0, 0, 255),
            -1
        )


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (255, 255, 255),
            2
        )


    # =====================================================
    # BOTTOM INFORMATION
    # =====================================================

    cv2.rectangle(
        frame,
        (
            0,
            height - 45
        ),
        (
            width,
            height
        ),
        (15, 15, 15),
        -1
    )


    draw_text(
        frame,
        "Q = EXIT",
        (
            20,
            height - 15
        ),
        0.55,
        1
    )


    draw_text(
        frame,
        "AI Monitoring Active",
        (
            width - 230,
            height - 15
        ),
        0.55,
        1
    )


    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "AI Smart Student Assistant (made by Suraj)",
        frame
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()