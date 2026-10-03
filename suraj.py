import cv2
import mediapipe as mp
import math
import time
import os
import numpy as np
from ultralytics import YOLO
import pygame


# =========================================================
# AI SMART STUDENT ASSISTANT
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(BASE_DIR, "sounds")


# =========================================================
# SOUNDS
# =========================================================

DROWSY_SOUND = os.path.join(
    SOUND_DIR,
    "emircanalp-beep-125033.mp3"
)

PHONE_SOUND = os.path.join(
    SOUND_DIR,
    "Phone.mp3"
)

LOW_FOCUS_SOUND = os.path.join(
    SOUND_DIR,
    "aisa-mat-karo.mp3"
)

DISTRACTED_SOUND = os.path.join(
    SOUND_DIR,
    "dekho.mp3"
)

WAKE_SOUND = os.path.join(
    SOUND_DIR,
    "uth-ja.mp3"
)

WELCOME_SOUND = os.path.join(
    SOUND_DIR,
    "wel.mp3"
)

HEART_SOUND = os.path.join(
    SOUND_DIR,
    "tum-hi-ho.mp3"
)


# =========================================================
# PYGAME
# =========================================================

pygame.mixer.init()


def load_sound(path):

    if not os.path.exists(path):
        print("[WARNING] Sound not found:", path)
        return None

    try:
        return pygame.mixer.Sound(path)

    except Exception as error:
        print("[WARNING] Sound load error:", path)
        print(error)
        return None


sounds = {
    "drowsy": load_sound(DROWSY_SOUND),
    "phone": load_sound(PHONE_SOUND),
    "focus": load_sound(LOW_FOCUS_SOUND),
    "distracted": load_sound(DISTRACTED_SOUND),
    "wake": load_sound(WAKE_SOUND),
    "welcome": load_sound(WELCOME_SOUND),
    "heart": load_sound(HEART_SOUND),
}


# =========================================================
# AUDIO CHANNELS
# =========================================================

ch_drowsy = pygame.mixer.Channel(0)
ch_phone = pygame.mixer.Channel(1)
ch_focus = pygame.mixer.Channel(2)
ch_distracted = pygame.mixer.Channel(3)
ch_wake = pygame.mixer.Channel(4)
ch_welcome = pygame.mixer.Channel(5)
ch_heart = pygame.mixer.Channel(6)


def play_once(channel, name):

    sound = sounds.get(name)

    if sound is None:
        return

    channel.play(sound)


def play_loop(channel, name):

    sound = sounds.get(name)

    if sound is None:
        return

    if not channel.get_busy():
        channel.play(sound, loops=-1)


def stop_sound(channel):

    if channel.get_busy():
        channel.stop()


# =========================================================
# MEDIAPIPE
# =========================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# FACE LANDMARKER
# =========================================================

FACE_MODEL = os.path.join(
    BASE_DIR,
    "face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

face_landmarker = (
    vision.FaceLandmarker.create_from_options(
        face_options
    )
)


# =========================================================
# HAND LANDMARKER
# =========================================================

HAND_MODEL = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

hand_landmarker = (
    vision.HandLandmarker.create_from_options(
        hand_options
    )
)


# =========================================================
# YOLO
# =========================================================

YOLO_MODEL = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)

model = YOLO(YOLO_MODEL)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print("ERROR: Camera open nahi hua.")

    pygame.mixer.quit()

    raise SystemExit


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = "AI Smart Student Assistant - Made by Suraj"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# =========================================================
# EYE
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
# HEAD
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


# =========================================================
# PHONE SETTINGS
# =========================================================
#
# Pehle:
# confidence = 0.40
# every 5 frames
# imgsz = 416
#
# Ab phone ko zyada reliably detect karne ke liye:
#

PHONE_CONFIDENCE = 0.25

YOLO_INTERVAL = 2

YOLO_IMAGE_SIZE = 640

PHONE_CLASS_ID = 67


# =========================================================
# WELCOME
# =========================================================

WELCOME_DELAY = 2.0

WELCOME_DURATION = 4.0

welcome_pending = False

welcome_click_time = 0

welcome_playing = False

welcome_start_time = 0


# =========================================================
# HEART
# =========================================================

heart_active = False

heart_x = 0

heart_y = 0

heart_start_time = 0


# =========================================================
# GENERAL STATES
# =========================================================

frame_count = 0

blink_count = 0

eyes_closed_start = None

drowsy_triggered = False

drowsy_finished = False

wake_started = False

phone_detected = False

phone_boxes = []

head_direction = "CENTER"

focus_score = 100


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global welcome_pending
    global welcome_click_time

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # -----------------------------------------------------
    # MADE BY SURAJ CLICK AREA
    # -----------------------------------------------------

    if (
        450 <= x <= 590
        and
        25 <= y <= 65
    ):

        if welcome_pending:
            return

        if welcome_playing:
            return

        welcome_pending = True

        welcome_click_time = time.time()

        print(
            "[WELCOME] Clicked - 2 sec wait..."
        )


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# =========================================================
# DISTANCE
# =========================================================

def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


# =========================================================
# ANGLE
# =========================================================

def calculate_angle(a, b, c):

    ba = (
        a[0] - b[0],
        a[1] - b[1]
    )

    bc = (
        c[0] - b[0],
        c[1] - b[1]
    )

    dot = (
        ba[0] * bc[0]
        +
        ba[1] * bc[1]
    )

    mag1 = math.sqrt(
        ba[0] ** 2
        +
        ba[1] ** 2
    )

    mag2 = math.sqrt(
        bc[0] ** 2
        +
        bc[1] ** 2
    )

    if mag1 == 0 or mag2 == 0:
        return 0

    value = dot / (mag1 * mag2)

    value = max(
        -1,
        min(1, value)
    )

    return math.degrees(
        math.acos(value)
    )


# =========================================================
# EAR
# =========================================================

def calculate_ear(points):

    vertical_1 = distance(
        points[1],
        points[5]
    )

    vertical_2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
    )

    if horizontal == 0:
        return 0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2 * horizontal
    )


# =========================================================
# LANDMARK TO PIXEL
# =========================================================

def landmark_to_pixel(
    landmark,
    width,
    height
):

    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


# =========================================================
# TEXT
# =========================================================

def put_text(
    frame,
    text,
    position,
    size=0.5,
    color=(255, 255, 255),
    thickness=1
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# FINGER STATUS
# =========================================================

def get_finger_status(points):

    wrist = points[0]

    fingers = {}

    index_angle = calculate_angle(
        points[6],
        points[7],
        points[8]
    )

    fingers["INDEX"] = (
        index_angle > 150
        and
        distance(
            points[8],
            wrist
        )
        >
        distance(
            points[6],
            wrist
        )
    )

    middle_angle = calculate_angle(
        points[10],
        points[11],
        points[12]
    )

    fingers["MIDDLE"] = (
        middle_angle > 150
        and
        distance(
            points[12],
            wrist
        )
        >
        distance(
            points[10],
            wrist
        )
    )

    ring_angle = calculate_angle(
        points[14],
        points[15],
        points[16]
    )

    fingers["RING"] = (
        ring_angle > 150
        and
        distance(
            points[16],
            wrist
        )
        >
        distance(
            points[14],
            wrist
        )
    )

    pinky_angle = calculate_angle(
        points[18],
        points[19],
        points[20]
    )

    fingers["PINKY"] = (
        pinky_angle > 150
        and
        distance(
            points[20],
            wrist
        )
        >
        distance(
            points[18],
            wrist
        )
    )

    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    fingers["THUMB"] = (
        thumb_angle > 145
        and
        distance(
            points[4],
            wrist
        )
        >
        distance(
            points[3],
            wrist
        )
    )

    return fingers


# =========================================================
# FINGER LENGTH
# =========================================================

def get_finger_lengths(points):

    return {

        "THUMB":
        distance(
            points[1],
            points[4]
        ),

        "INDEX":
        distance(
            points[5],
            points[8]
        ),

        "MIDDLE":
        distance(
            points[9],
            points[12]
        ),

        "RING":
        distance(
            points[13],
            points[16]
        ),

        "PINKY":
        distance(
            points[17],
            points[20]
        )
    }


# =========================================================
# HEART POINTS
# =========================================================

def create_heart_points(
    center_x,
    center_y,
    scale
):

    points = []

    for degree in range(
        0,
        361,
        2
    ):

        t = math.radians(degree)

        x = (
            16 *
            math.sin(t) ** 3
        )

        y = (
            13 * math.cos(t)
            -
            5 * math.cos(2 * t)
            -
            2 * math.cos(3 * t)
            -
            math.cos(4 * t)
        )

        px = int(
            center_x +
            x * scale
        )

        py = int(
            center_y -
            y * scale
        )

        points.append(
            [px, py]
        )

    return np.asarray(
        points,
        dtype=np.int32
    )


# =========================================================
# BEAUTIFUL HEART
# =========================================================

def draw_beautiful_heart(
    frame,
    center_x,
    center_y,
    elapsed
):

    cycle = elapsed % 1.15

    if cycle < 0.12:

        p = cycle / 0.12

        pulse = (
            math.sin(
                p * math.pi
            )
            *
            0.115
        )

    elif cycle < 0.28:

        p = (
            cycle - 0.12
        ) / 0.16

        pulse = (
            0.115 *
            (1 - p)
        )

    elif cycle < 0.38:

        p = (
            cycle - 0.28
        ) / 0.10

        pulse = (
            math.sin(
                p * math.pi
            )
            *
            0.065
        )

    elif cycle < 0.55:

        p = (
            cycle - 0.38
        ) / 0.17

        pulse = (
            0.065 *
            (1 - p)
        )

    else:

        pulse = 0


    pulse_scale = 1 + pulse

    outer_scale = (
        2.45 *
        pulse_scale
    )

    inner_scale = (
        1.92 *
        pulse_scale
    )


    # =====================================================
    # GLOW
    # =====================================================

    glow = np.zeros_like(frame)

    glow_points = create_heart_points(
        center_x,
        center_y,
        outer_scale * 1.08
    )

    cv2.fillPoly(
        glow,
        [glow_points],
        (70, 60, 255)
    )

    glow = cv2.GaussianBlur(
        glow,
        (0, 0),
        15
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.94,
        glow,
        0.06,
        0
    )


    # =====================================================
    # HEART
    # =====================================================

    heart = np.zeros_like(frame)


    outer = create_heart_points(
        center_x,
        center_y,
        outer_scale
    )

    cv2.fillPoly(
        heart,
        [outer],
        (245, 205, 220)
    )

    cv2.polylines(
        heart,
        [outer],
        True,
        (255, 225, 238),
        2,
        cv2.LINE_AA
    )


    outer_border = create_heart_points(
        center_x,
        center_y + 1,
        outer_scale * 0.91
    )

    cv2.polylines(
        heart,
        [outer_border],
        True,
        (225, 125, 165),
        2,
        cv2.LINE_AA
    )


    inner = create_heart_points(
        center_x,
        center_y + 3,
        inner_scale
    )


    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [inner],
        255
    )


    # =====================================================
    # GRADIENT
    # =====================================================

    gradient = np.zeros_like(frame)

    height = frame.shape[0]

    top_y = max(
        0,
        center_y -
        int(15 * inner_scale)
    )

    bottom_y = min(
        height - 1,
        center_y +
        int(15 * inner_scale)
    )


    top_color = np.array(
        [255, 145, 190],
        dtype=np.float32
    )

    bottom_color = np.array(
        [205, 35, 105],
        dtype=np.float32
    )


    if bottom_y > top_y:

        for y in range(
            top_y,
            bottom_y + 1
        ):

            ratio = (
                y - top_y
            ) / (
                bottom_y - top_y
            )

            ratio = max(
                0,
                min(1, ratio)
            )

            color = (
                top_color * (1 - ratio)
                +
                bottom_color * ratio
            )

            gradient[y, :, :] = (
                color[::-1]
            )


    gradient_part = cv2.bitwise_and(
        gradient,
        gradient,
        mask=mask
    )

    heart = cv2.add(
        heart,
        gradient_part
    )


    cv2.polylines(
        heart,
        [inner],
        True,
        (205, 55, 115),
        2,
        cv2.LINE_AA
    )


    # =====================================================
    # GLOSS
    # =====================================================

    cv2.ellipse(
        heart,
        (
            int(
                center_x -
                inner_scale * 5.5
            ),
            int(
                center_y -
                inner_scale * 5.2
            )
        ),
        (
            max(
                3,
                int(
                    inner_scale * 2.2
                )
            ),
            max(
                5,
                int(
                    inner_scale * 3.8
                )
            )
        ),
        -35,
        0,
        360,
        (255, 245, 250),
        -1,
        cv2.LINE_AA
    )


    cv2.ellipse(
        heart,
        (
            int(
                center_x -
                inner_scale * 2.5
            ),
            int(
                center_y -
                inner_scale * 8
            )
        ),
        (
            max(
                2,
                int(
                    inner_scale * 0.9
                )
            ),
            max(
                2,
                int(
                    inner_scale * 1.4
                )
            )
        ),
        -25,
        0,
        360,
        (255, 220, 235),
        -1,
        cv2.LINE_AA
    )


    # =====================================================
    # ONLY HEART PIXELS
    # BLACK BACKGROUND NEVER GOES ON CAMERA
    # =====================================================

    heart_mask = cv2.cvtColor(
        heart,
        cv2.COLOR_BGR2GRAY
    )

    _, heart_mask = cv2.threshold(
        heart_mask,
        5,
        255,
        cv2.THRESH_BINARY
    )

    heart_mask = cv2.GaussianBlur(
        heart_mask,
        (3, 3),
        0
    )

    alpha = (
        heart_mask.astype(
            np.float32
        )
        /
        255.0
    )

    alpha = alpha[:, :, None]


    frame_float = frame.astype(
        np.float32
    )

    heart_float = heart.astype(
        np.float32
    )


    result = (
        frame_float * (1 - alpha)
        +
        heart_float * alpha
    )


    frame[:] = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# =========================================================
# STRICT CHEST DETECTION
# =========================================================

def is_hand_on_chest(
    points,
    face_points
):

    if len(points) < 21:
        return False, 0, 0

    if len(face_points) == 0:
        return False, 0, 0


    chin = face_points[CHIN]

    left_ear = face_points[LEFT_EAR]

    right_ear = face_points[RIGHT_EAR]


    face_center_x = int(
        (
            left_ear[0]
            +
            right_ear[0]
        ) / 2
    )


    face_width = abs(
        right_ear[0]
        -
        left_ear[0]
    )


    if face_width < 40:
        return False, 0, 0


    # =====================================================
    # PALM CENTER
    # =====================================================

    palm_x = int(
        (
            points[0][0]
            +
            points[5][0]
            +
            points[9][0]
            +
            points[13][0]
            +
            points[17][0]
        ) / 5
    )


    palm_y = int(
        (
            points[0][1]
            +
            points[5][1]
            +
            points[9][1]
            +
            points[13][1]
            +
            points[17][1]
        ) / 5
    )


    wrist_y = points[0][1]


    palm_width = distance(
        points[5],
        points[17]
    )


    if palm_width < 20:
        return False, 0, 0


    # =====================================================
    # CHEST AREA
    # =====================================================

    chest_top = int(
        chin[1]
        +
        face_width * 0.20
    )

    chest_bottom = int(
        chin[1]
        +
        face_width * 1.65
    )

    chest_left = int(
        face_center_x
        -
        face_width * 1.25
    )

    chest_right = int(
        face_center_x
        +
        face_width * 1.25
    )


    if not (
        chest_left
        <=
        palm_x
        <=
        chest_right
    ):

        return False, 0, 0


    if not (
        chest_top
        <=
        palm_y
        <=
        chest_bottom
    ):

        return False, 0, 0


    if wrist_y <= chin[1]:
        return False, 0, 0


    # =====================================================
    # FINGERTIPS
    # =====================================================

    tips = [
        points[4],
        points[8],
        points[12],
        points[16],
        points[20]
    ]


    inside = 0


    for x, y in tips:

        if (
            chest_left <= x <= chest_right
            and
            chest_top <= y <= chest_bottom
        ):

            inside += 1


    if inside < 2:
        return False, 0, 0


    heart_x = palm_x

    heart_y = (
        palm_y
        -
        int(
            palm_width * 0.10
        )
    )


    return True, heart_x, heart_y


# =========================================================
# MAIN LOOP
# =========================================================

try:

    print()
    print(
        "======================================"
    )
    print(
        " AI SMART STUDENT ASSISTANT"
    )
    print(
        "======================================"
    )
    print(
        "Made by Suraj -> click"
    )
    print(
        "Q / ESC -> Exit"
    )
    print()


    while True:

        # =================================================
        # CAMERA
        # =================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "Camera frame nahi mila."
            )

            break


        frame = cv2.flip(
            frame,
            1
        )


        height, width = (
            frame.shape[:2]
        )


        frame_count += 1

        now = time.time()


        # =================================================
        # WELCOME
        # =================================================

        if (
            welcome_pending
            and
            not welcome_playing
            and
            now -
            welcome_click_time
            >=
            WELCOME_DELAY
        ):

            sound = sounds.get(
                "welcome"
            )


            if sound is not None:

                ch_welcome.play(
                    sound
                )

                welcome_playing = True

                welcome_start_time = now

                print(
                    "[WELCOME] Playing..."
                )

            else:

                welcome_pending = False


        # =================================================
        # WELCOME STOP
        # =================================================

        if (
            welcome_playing
            and
            now -
            welcome_start_time
            >=
            WELCOME_DURATION
        ):

            stop_sound(
                ch_welcome
            )

            welcome_playing = False

            welcome_pending = False

            welcome_start_time = 0

            print(
                "[WELCOME] Stopped."
            )


        # =================================================
        # RGB
        # =================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # =================================================
        # FACE
        # =================================================

        face_result = (
            face_landmarker.detect(
                mp_image
            )
        )


        face_detected = False

        face_points = []

        current_ear = 0

        eyes_closed = False

        head_direction = "NO FACE"


        if face_result.face_landmarks:

            face_detected = True

            landmarks = (
                face_result.face_landmarks[0]
            )


            for landmark in landmarks:

                point = (
                    landmark_to_pixel(
                        landmark,
                        width,
                        height
                    )
                )

                face_points.append(
                    point
                )


                cv2.circle(
                    frame,
                    point,
                    1,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # EYES
            # =================================================

            left_eye = [
                face_points[i]
                for i in LEFT_EYE
            ]

            right_eye = [
                face_points[i]
                for i in RIGHT_EYE
            ]


            current_ear = (
                calculate_ear(left_eye)
                +
                calculate_ear(right_eye)
            ) / 2


            for point in left_eye:

                cv2.circle(
                    frame,
                    point,
                    3,
                    (0, 255, 0),
                    -1
                )


            for point in right_eye:

                cv2.circle(
                    frame,
                    point,
                    3,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # EYE CLOSED
            # =================================================

            eyes_closed = (
                current_ear
                <
                EAR_THRESHOLD
            )


            if eyes_closed:

                if eyes_closed_start is None:

                    eyes_closed_start = (
                        time.time()
                    )


                closed_time = (
                    time.time()
                    -
                    eyes_closed_start
                )


                # =================================================
                # DROWSY
                # =================================================

                if (
                    closed_time
                    >=
                    DROWSY_TIME
                ):

                    if not drowsy_triggered:

                        drowsy_triggered = True

                        drowsy_finished = False

                        wake_started = False

                        play_once(
                            ch_drowsy,
                            "drowsy"
                        )


                    if (
                        not drowsy_finished
                        and
                        not ch_drowsy.get_busy()
                    ):

                        drowsy_finished = True


                    if (
                        drowsy_finished
                        and
                        not wake_started
                    ):

                        play_loop(
                            ch_wake,
                            "wake"
                        )

                        wake_started = True


            else:

                if (
                    eyes_closed_start
                    is not None
                ):

                    closed_time = (
                        time.time()
                        -
                        eyes_closed_start
                    )


                    if (
                        0.08
                        <
                        closed_time
                        <
                        DROWSY_TIME
                    ):

                        blink_count += 1


                stop_sound(
                    ch_drowsy
                )

                stop_sound(
                    ch_wake
                )

                eyes_closed_start = None

                drowsy_triggered = False

                drowsy_finished = False

                wake_started = False


            # =================================================
            # HEAD
            # =================================================

            nose = face_points[NOSE]

            left_side = face_points[
                LEFT_EAR
            ]

            right_side = face_points[
                RIGHT_EAR
            ]


            face_center = (
                left_side[0]
                +
                right_side[0]
            ) / 2


            face_width = abs(
                right_side[0]
                -
                left_side[0]
            )


            if face_width > 0:

                ratio = (
                    nose[0]
                    -
                    face_center
                ) / face_width

            else:

                ratio = 0


            if ratio < -0.13:

                head_direction = "LEFT"

            elif ratio > 0.13:

                head_direction = "RIGHT"

            else:

                head_direction = "CENTER"


        else:

            stop_sound(
                ch_drowsy
            )

            stop_sound(
                ch_wake
            )

            stop_sound(
                ch_distracted
            )

            eyes_closed_start = None

            drowsy_triggered = False

            drowsy_finished = False

            wake_started = False


        # =================================================
        # HEAD SOUND
        # =================================================

        if head_direction in (
            "LEFT",
            "RIGHT"
        ):

            play_loop(
                ch_distracted,
                "distracted"
            )

        else:

            stop_sound(
                ch_distracted
            )


        # =================================================
        # =================================================
        # YOLO PHONE DETECTION
        # =================================================
        #
        # PHONE DETECTION PEHLE KAR RAHE HAIN
        # TAAKI PHONE KO HEART SE PRIORITY MILE.
        #
        # =================================================
        # =================================================

        if (
            frame_count
            %
            YOLO_INTERVAL
            ==
            0
        ):

            phone_boxes = []

            try:

                results = model.predict(
                    frame,
                    imgsz=YOLO_IMAGE_SIZE,
                    conf=PHONE_CONFIDENCE,
                    iou=0.45,
                    max_det=10,
                    verbose=False
                )


                for result in results:

                    if result.boxes is None:
                        continue


                    for box in result.boxes:

                        cls_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )


                        # =================================================
                        # ONLY CELL PHONE
                        # =================================================

                        if (
                            cls_id
                            ==
                            PHONE_CLASS_ID
                            and
                            confidence
                            >=
                            PHONE_CONFIDENCE
                        ):

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0]
                            )


                            # Make sure box is valid

                            if (
                                x2 > x1
                                and
                                y2 > y1
                            ):

                                phone_boxes.append(
                                    (
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        confidence
                                    )
                                )


            except Exception as error:

                print(
                    "[YOLO ERROR]",
                    error
                )


        # =================================================
        # PHONE STATE
        # =================================================

        phone_detected = (
            len(phone_boxes) > 0
        )


        # =================================================
        # PHONE SOUND
        # =================================================

        if phone_detected:

            # Phone has priority

            play_loop(
                ch_phone,
                "phone"
            )

        else:

            stop_sound(
                ch_phone
            )


        # =================================================
        # PHONE BOX
        # =================================================

        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in phone_boxes:


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )


            put_text(
                frame,
                f"PHONE {confidence:.0%}",
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
                0.5,
                (0, 165, 255),
                2
            )


        # =================================================
        # HAND DETECTION
        # =================================================

        hand_result = (
            hand_landmarker.detect(
                mp_image
            )
        )


        detected_hands = []

        current_heart = False

        new_heart_x = 0

        new_heart_y = 0


        if hand_result.hand_landmarks:

            for hand_landmarks in (
                hand_result.hand_landmarks
            ):

                points = []


                for landmark in hand_landmarks:

                    points.append(
                        landmark_to_pixel(
                            landmark,
                            width,
                            height
                        )
                    )


                detected_hands.append(
                    points
                )


                # =================================================
                # HAND POINTS
                # =================================================

                for point in points:

                    cv2.circle(
                        frame,
                        point,
                        3,
                        (0, 255, 0),
                        -1
                    )


                # =================================================
                # HAND CONNECTIONS
                # =================================================

                connections = [

                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),

                    (0, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),

                    (0, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),

                    (0, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),

                    (0, 17),
                    (17, 18),
                    (18, 19),
                    (19, 20),

                    (5, 9),
                    (9, 13),
                    (13, 17)

                ]


                for a, b in connections:

                    cv2.line(
                        frame,
                        points[a],
                        points[b],
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )


                # =================================================
                # HEART DETECTION
                # =================================================
                #
                # IMPORTANT:
                #
                # PHONE DETECTED HAI TOH HEART ALLOW NAHI.
                #
                # =================================================

                if not phone_detected:

                    on_chest, hx, hy = (
                        is_hand_on_chest(
                            points,
                            face_points
                        )
                    )


                    if on_chest:

                        current_heart = True

                        new_heart_x = hx

                        new_heart_y = hy


        # =================================================
        # =================================================
        # HEART / PHONE MUTUAL EXCLUSION
        # =================================================
        # =================================================

        if phone_detected:

            # -----------------------------------------------
            # PHONE ALWAYS WINS
            # -----------------------------------------------

            current_heart = False

            heart_active = False

            heart_x = 0

            heart_y = 0

            stop_sound(
                ch_heart
            )


        # =================================================
        # HEART STATE
        # =================================================

        if current_heart and not phone_detected:

            if not heart_active:

                heart_active = True

                heart_start_time = (
                    time.time()
                )

                heart_x = (
                    new_heart_x
                )

                heart_y = (
                    new_heart_y
                )


                play_loop(
                    ch_heart,
                    "heart"
                )


            else:

                heart_x = int(
                    heart_x * 0.82
                    +
                    new_heart_x * 0.18
                )


                heart_y = int(
                    heart_y * 0.82
                    +
                    new_heart_y * 0.18
                )


        else:

            heart_active = False

            heart_x = 0

            heart_y = 0

            stop_sound(
                ch_heart
            )


        # =================================================
        # DRAW HEART
        # =================================================

        if (
            heart_active
            and
            not phone_detected
        ):

            elapsed = (
                time.time()
                -
                heart_start_time
            )


            draw_beautiful_heart(
                frame,
                heart_x,
                heart_y,
                elapsed
            )


        # =================================================
        # FOCUS
        # =================================================

        focus_score = 100


        if eyes_closed:

            focus_score -= 30


        if phone_detected:

            focus_score -= 35


        if head_direction in (
            "LEFT",
            "RIGHT"
        ):

            focus_score -= 20


        if not face_detected:

            focus_score -= 20


        focus_score = max(
            0,
            min(
                100,
                focus_score
            )
        )


        # =================================================
        # LOW FOCUS SOUND
        # =================================================

        if focus_score < 50:

            play_loop(
                ch_focus,
                "focus"
            )

        else:

            stop_sound(
                ch_focus
            )


        # =================================================
        # RIGHT PANEL
        # =================================================

        panel_x = 450


        overlay = frame.copy()


        cv2.rectangle(
            overlay,
            (panel_x, 0),
            (width, height),
            (12, 18, 25),
            -1
        )


        frame = cv2.addWeighted(
            overlay,
            0.90,
            frame,
            0.10,
            0
        )


        # =================================================
        # HEADER
        # =================================================

        put_text(
            frame,
            "AI MONITOR",
            (465, 28),
            0.60,
            (0, 255, 0),
            2
        )


        put_text(
            frame,
            "Made by Suraj",
            (465, 47),
            0.35,
            (0, 255, 255),
            1
        )


        put_text(
            frame,
            "● LIVE",
            (565, 28),
            0.42,
            (0, 255, 0),
            1
        )


        # =================================================
        # FACE
        # =================================================

        put_text(
            frame,
            "FACE",
            (465, 88),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            (
                "DETECTED"
                if face_detected
                else
                "NOT FOUND"
            ),
            (525, 88),
            0.45,
            (
                (0, 255, 0)
                if face_detected
                else
                (0, 0, 255)
            ),
            1
        )


        # =================================================
        # EAR
        # =================================================

        put_text(
            frame,
            "EAR",
            (465, 113),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            f"{current_ear:.2f}",
            (525, 113),
            0.45,
            (255, 255, 255),
            1
        )


        # =================================================
        # BLINKS
        # =================================================

        put_text(
            frame,
            "BLINKS",
            (465, 138),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            str(blink_count),
            (525, 138),
            0.45,
            (255, 255, 255),
            1
        )


        # =================================================
        # HEAD
        # =================================================

        put_text(
            frame,
            "HEAD",
            (465, 163),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            head_direction,
            (525, 163),
            0.45,
            (
                (0, 255, 0)
                if head_direction == "CENTER"
                else
                (0, 165, 255)
            ),
            1
        )


        # =================================================
        # FOCUS
        # =================================================

        put_text(
            frame,
            "FOCUS",
            (465, 193),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            f"{focus_score}%",
            (525, 193),
            0.45,
            (
                (0, 255, 0)
                if focus_score >= 70
                else
                (0, 165, 255)
            ),
            1
        )


        # =================================================
        # FOCUS BAR
        # =================================================

        cv2.rectangle(
            frame,
            (465, 205),
            (620, 213),
            (70, 70, 70),
            -1
        )


        filled = int(
            155 *
            focus_score /
            100
        )


        cv2.rectangle(
            frame,
            (465, 205),
            (465 + filled, 213),
            (0, 255, 0),
            -1
        )


        # =================================================
        # PHONE
        # =================================================

        put_text(
            frame,
            "PHONE",
            (465, 237),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            (
                "DETECTED"
                if phone_detected
                else
                "NONE"
            ),
            (525, 237),
            0.45,
            (
                (0, 165, 255)
                if phone_detected
                else
                (0, 255, 0)
            ),
            1
        )


        # =================================================
        # HANDS
        # =================================================

        put_text(
            frame,
            "HANDS",
            (465, 267),
            0.45,
            (180, 180, 180),
            1
        )


        put_text(
            frame,
            str(len(detected_hands)),
            (525, 267),
            0.45,
            (255, 255, 255),
            1
        )


        # =================================================
        # FINGER DETAILS
        # =================================================

        panel_y = 297


        for hand_index, points in enumerate(
            detected_hands
        ):

            status = get_finger_status(
                points
            )


            lengths = get_finger_lengths(
                points
            )


            open_fingers = [

                name

                for name, state
                in status.items()

                if state

            ]


            put_text(
                frame,
                f"HAND {hand_index + 1}",
                (465, panel_y),
                0.42,
                (0, 255, 0),
                1
            )


            panel_y += 18


            if not open_fingers:

                put_text(
                    frame,
                    "No open fingers",
                    (465, panel_y),
                    0.38,
                    (150, 150, 150),
                    1
                )

                panel_y += 18


            else:

                for finger in open_fingers:

                    put_text(
                        frame,
                        (
                            f"{finger}: "
                            f"{lengths[finger]:.0f}px"
                        ),
                        (465, panel_y),
                        0.38,
                        (220, 255, 220),
                        1
                    )

                    panel_y += 17


            panel_y += 5


            if panel_y > 465:
                break


        # =================================================
        # EYE STATUS
        # =================================================

        if eyes_closed:

            if eyes_closed_start is not None:

                closed_time = (
                    time.time()
                    -
                    eyes_closed_start
                )


                if (
                    closed_time
                    >=
                    DROWSY_TIME
                ):

                    put_text(
                        frame,
                        "DROWSY!",
                        (20, 40),
                        0.80,
                        (0, 0, 255),
                        2
                    )

                else:

                    put_text(
                        frame,
                        "EYE CLOSED",
                        (20, 40),
                        0.70,
                        (0, 165, 255),
                        2
                    )


        # =================================================
        # PHONE WARNING
        # =================================================

        if phone_detected:

            put_text(
                frame,
                "PHONE DETECTED",
                (20, height - 20),
                0.60,
                (0, 165, 255),
                2
            )


        # =================================================
        # HEART STATUS
        # =================================================

        if (
            heart_active
            and
            not phone_detected
        ):

            put_text(
                frame,
                "HAND ON HEART",
                (20, height - 48),
                0.50,
                (0, 80, 255),
                2
            )


        # =================================================
        # SHOW
        # =================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # =================================================
        # KEY
        # =================================================

        key = cv2.waitKeyEx(1)


        if key == 27:
            break

        if key == ord("q"):
            break

        if key == ord("Q"):
            break


# =========================================================
# ERROR
# =========================================================

except Exception as error:

    print()
    print(
        "======================================"
    )
    print(
        "PROGRAM ERROR"
    )
    print(
        "======================================"
    )
    print(error)
    print()


# =========================================================
# CLEANUP
# =========================================================

finally:

    print(
        "Closing camera..."
    )


    stop_sound(ch_drowsy)
    stop_sound(ch_phone)
    stop_sound(ch_focus)
    stop_sound(ch_distracted)
    stop_sound(ch_wake)
    stop_sound(ch_welcome)
    stop_sound(ch_heart)


    cap.release()


    cv2.destroyAllWindows()

    cv2.waitKey(1)


    try:
        face_landmarker.close()
    except:
        pass


    try:
        hand_landmarker.close()
    except:
        pass


    try:
        pygame.mixer.quit()
    except:
        pass


    print(
        "AI Smart Student Assistant stopped."
    )