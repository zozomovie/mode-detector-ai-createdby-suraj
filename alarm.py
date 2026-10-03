import winsound

# Windows ki alarm sound
ALARM_FILE = r"C:\Windows\Media\Alarm01.wav"

# Alarm play karo
winsound.PlaySound(
    ALARM_FILE,
    winsound.SND_FILENAME
)