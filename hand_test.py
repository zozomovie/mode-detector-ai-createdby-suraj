import cv2
import mediapipe as mp
import math


# ============================================================
#                    MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)


landmarker = HandLandmarker.create_from_options(options)


# ============================================================
#                    CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera open nahi hua!")

    exit()


# ============================================================
#                    DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
#                    FINGER STATUS
# ============================================================

def get_finger_status(points):

    wrist = points[0]

    fingers = {}


    # ========================================================
    # INDEX FINGER
    # ========================================================

    index_open = (
        distance(points[8], wrist)
        >
        distance(points[6], wrist)
    )

    fingers["INDEX"] = index_open


    # ========================================================
    # MIDDLE FINGER
    # ========================================================

    middle_open = (
        distance(points[12], wrist)
        >
        distance(points[10], wrist)
    )

    fingers["MIDDLE"] = middle_open


    # ========================================================
    # RING FINGER
    # ========================================================

    ring_open = (
        distance(points[16], wrist)
        >
        distance(points[14], wrist)
    )

    fingers["RING"] = ring_open


    # ========================================================
    # PINKY
    # ========================================================

    pinky_open = (
        distance(points[20], wrist)
        >
        distance(points[18], wrist)
    )

    fingers["PINKY"] = pinky_open


    # ========================================================
    # THUMB
    # ========================================================

    thumb_tip_to_index = distance(
        points[4],
        points[5]
    )

    thumb_ip_to_index = distance(
        points[3],
        points[5]
    )


    thumb_open = (
        thumb_tip_to_index
        >
        thumb_ip_to_index * 1.25
    )


    fingers["THUMB"] = thumb_open


    return fingers


# ============================================================
#                    FINGER LENGTH
# ============================================================

def get_finger_lengths(points):

    lengths = {}


    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    lengths["THUMB"] = (
        distance(points[1], points[2])
        +
        distance(points[2], points[3])
        +
        distance(points[3], points[4])
    )


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    lengths["INDEX"] = (
        distance(points[5], points[6])
        +
        distance(points[6], points[7])
        +
        distance(points[7], points[8])
    )


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    lengths["MIDDLE"] = (
        distance(points[9], points[10])
        +
        distance(points[10], points[11])
        +
        distance(points[11], points[12])
    )


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    lengths["RING"] = (
        distance(points[13], points[14])
        +
        distance(points[14], points[15])
        +
        distance(points[15], points[16])
    )


    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    lengths["PINKY"] = (
        distance(points[17], points[18])
        +
        distance(points[18], points[19])
        +
        distance(points[19], points[20])
    )


    return lengths


# ============================================================
#                    MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Camera frame nahi mila!")

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Hand detection
    # --------------------------------------------------------

    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    #                    HAND FOUND
    # ========================================================

    if result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):


            # ------------------------------------------------
            # Convert landmarks to pixels
            # ------------------------------------------------

            points = []


            for landmark in hand_landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                points.append(
                    (x, y)
                )


            # ------------------------------------------------
            # Finger status
            # ------------------------------------------------

            finger_status = get_finger_status(
                points
            )


            # ------------------------------------------------
            # Finger lengths
            # ------------------------------------------------

            finger_lengths = get_finger_lengths(
                points
            )


            # ------------------------------------------------
            # Get only OPEN fingers
            # ------------------------------------------------

            open_fingers = [

                name

                for name, is_open
                in finger_status.items()

                if is_open
            ]


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # HAND CONNECTIONS
            # =================================================

            connections = [

                # Thumb
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                # Index
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                # Middle
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                # Ring
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                # Pinky
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                # Palm
                (5, 9),
                (9, 13),
                (13, 17)
            ]


            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )


            # =================================================
            # DISPLAY POSITION
            # =================================================

            x = 20

            y = 40


            if hand_index == 1:

                x = width // 2

                y = 40


            # =================================================
            # HAND NUMBER
            # =================================================

            cv2.putText(
                frame,
                f"HAND {hand_index + 1}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


            # =================================================
            # FINGER COUNT
            # =================================================

            cv2.putText(
                frame,
                f"FINGERS: {len(open_fingers)}",
                (x, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # =================================================
            # SHOW ONLY OPEN FINGERS
            # =================================================

            display_y = y + 65


            if open_fingers:

                for finger_name in open_fingers:

                    length = finger_lengths[
                        finger_name
                    ]


                    text = (
                        f"{finger_name}: "
                        f"{length:.0f} px"
                    )


                    cv2.putText(
                        frame,
                        text,
                        (x, display_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 255),
                        2
                    )


                    display_y += 30


            else:

                cv2.putText(
                    frame,
                    "NO FINGER OPEN",
                    (x, display_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 0, 255),
                    2
                )


    # ========================================================
    #                    NO HAND
    # ========================================================

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2
        )


    # ========================================================
    #                    SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Finger Length Measurement",
        frame
    )


    # ========================================================
    #                    EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
#                    CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Finger measurement stopped.")