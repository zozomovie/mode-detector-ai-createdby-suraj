import cv2
import mediapipe as mp
import math


# ============================================================
#                    SETTINGS
# ============================================================

WINDOW_NAME = "Finger Measurement + Calibration"

REFERENCE_LENGTH_INCHES = 1.0

pixels_per_inch = None

calibration_points = []

calibration_mode = False


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


landmarker = HandLandmarker.create_from_options(
    options
)


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


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    fingers["INDEX"] = (
        distance(points[8], wrist)
        >
        distance(points[6], wrist)
    )


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    fingers["MIDDLE"] = (
        distance(points[12], wrist)
        >
        distance(points[10], wrist)
    )


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    fingers["RING"] = (
        distance(points[16], wrist)
        >
        distance(points[14], wrist)
    )


    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    fingers["PINKY"] = (
        distance(points[20], wrist)
        >
        distance(points[18], wrist)
    )


    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip_to_index = distance(
        points[4],
        points[5]
    )

    thumb_ip_to_index = distance(
        points[3],
        points[5]
    )


    fingers["THUMB"] = (
        thumb_tip_to_index
        >
        thumb_ip_to_index * 1.25
    )


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
#                    MOUSE CLICK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global calibration_points
    global calibration_mode
    global pixels_per_inch


    # Only accept clicks during calibration

    if not calibration_mode:

        return


    if event == cv2.EVENT_LBUTTONDOWN:

        calibration_points.append(
            (x, y)
        )


        print(
            f"Calibration point {len(calibration_points)}:",
            x,
            y
        )


        # ----------------------------------------------------
        # Two points selected
        # ----------------------------------------------------

        if len(calibration_points) == 2:

            p1 = calibration_points[0]

            p2 = calibration_points[1]


            pixel_distance = distance(
                p1,
                p2
            )


            if pixel_distance > 0:

                pixels_per_inch = (
                    pixel_distance /
                    REFERENCE_LENGTH_INCHES
                )


                print()
                print("================================")
                print("CALIBRATION COMPLETE")
                print("================================")

                print(
                    f"Reference: "
                    f"{REFERENCE_LENGTH_INCHES:.2f} inch"
                )

                print(
                    f"Reference pixels: "
                    f"{pixel_distance:.2f}"
                )

                print(
                    f"Pixels per inch: "
                    f"{pixels_per_inch:.2f}"
                )

                print(
                    "================================"
                )

                print()


            calibration_mode = False


# ============================================================
#                    WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
#                    MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "Camera frame nahi mila!"
        )

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    # ========================================================
    #                    MEDIAPIPE
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    #                    HAND DETECTION
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
            # Open fingers only
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

            if hand_index == 0:

                text_x = 20

            else:

                text_x = width // 2


            text_y = 40


            # =================================================
            # HAND
            # =================================================

            cv2.putText(
                frame,
                f"HAND {hand_index + 1}",
                (text_x, text_y),
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
                (
                    text_x,
                    text_y + 32
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # =================================================
            # FINGER LENGTHS
            # =================================================

            display_y = text_y + 65


            if open_fingers:

                for finger_name in open_fingers:

                    px_length = finger_lengths[
                        finger_name
                    ]


                    # ------------------------------------------------
                    # Convert PX → INCH
                    # ------------------------------------------------

                    if pixels_per_inch is not None:

                        inch_length = (
                            px_length /
                            pixels_per_inch
                        )


                        cm_length = (
                            inch_length *
                            2.54
                        )


                        text = (
                            f"{finger_name}: "
                            f"{inch_length:.2f} in "
                            f"({cm_length:.1f} cm)"
                        )

                    else:

                        text = (
                            f"{finger_name}: "
                            f"{px_length:.0f} px"
                        )


                    cv2.putText(
                        frame,
                        text,
                        (
                            text_x,
                            display_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2
                    )


                    display_y += 30


            else:

                cv2.putText(
                    frame,
                    "NO FINGER OPEN",
                    (
                        text_x,
                        display_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2
                )


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
    #                    CALIBRATION DRAWING
    # ========================================================

    if calibration_mode:

        cv2.putText(
            frame,
            "CALIBRATION MODE",
            (20, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Click BOTH ends of {REFERENCE_LENGTH_INCHES:.2f} inch reference",
            (20, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Points: {len(calibration_points)}/2",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


    # ========================================================
    #                    DRAW CALIBRATION LINE
    # ========================================================

    if len(calibration_points) >= 1:

        p1 = calibration_points[0]


        cv2.circle(
            frame,
            p1,
            7,
            (0, 0, 255),
            -1
        )


    if len(calibration_points) == 2:

        p1 = calibration_points[0]

        p2 = calibration_points[1]


        cv2.circle(
            frame,
            p2,
            7,
            (0, 0, 255),
            -1
        )


        cv2.line(
            frame,
            p1,
            p2,
            (0, 0, 255),
            3
        )


        pixel_distance = distance(
            p1,
            p2
        )


        cv2.putText(
            frame,
            f"Reference: {pixel_distance:.0f} px",
            (
                p1[0],
                p1[1] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2
        )


    # ========================================================
    #                    CALIBRATION STATUS
    # ========================================================

    if pixels_per_inch is not None:

        cv2.putText(
            frame,
            f"CALIBRATED: {pixels_per_inch:.1f} px/in",
            (
                20,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Press C = Calibrate",
            (
                20,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    #                    SHOW WINDOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    #                    KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # C = Calibration
    # --------------------------------------------------------

    if key == ord("c"):

        calibration_points = []

        pixels_per_inch = None

        calibration_mode = True


        print()
        print("================================")
        print("CALIBRATION STARTED")
        print("================================")

        print(
            f"Reference length: "
            f"{REFERENCE_LENGTH_INCHES} inch"
        )

        print(
            "Camera window me reference object ke"
        )

        print(
            "DONO ENDS par mouse LEFT CLICK karo."
        )

        print("================================")
        print()


    # --------------------------------------------------------
    # R = Reset calibration
    # --------------------------------------------------------

    elif key == ord("r"):

        calibration_points = []

        pixels_per_inch = None

        calibration_mode = False


        print(
            "Calibration reset ho gayi."
        )


    # --------------------------------------------------------
    # Q = Quit
    # --------------------------------------------------------

    elif key == ord("q"):

        break


# ============================================================
#                    CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print(
    "Calibration test stopped."
)