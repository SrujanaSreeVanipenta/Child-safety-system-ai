from twilio.rest import Client
from datetime import datetime
import pytz
import time
import os
from dotenv import load_dotenv

load_dotenv()

alert_active = False

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

twilio_whatsapp_number = os.getenv("TWILIO_NUMBER")
parent_whatsapp_number = os.getenv("PARENT_NUMBER")

if not account_sid or not auth_token:
    raise ValueError("Twilio credentials not set properly")
    
client = Client(account_sid, auth_token)


# ---------------- Send WhatsApp Alert ----------------
def send_whatsapp_alert(title, message):

    client.messages.create(
        body=message,
        from_=twilio_whatsapp_number,
        to=parent_whatsapp_number
    )


# ---------------- Alert + Reminders ----------------
def send_voice_alert(emotion, speech, location):
    
    global alert_active

    if alert_active:
        return

    alert_active = True

    india = pytz.timezone("Asia/Kolkata")
    now = datetime.now(india)

    date = now.strftime("%d-%m-%Y")
    time_now = now.strftime("%H:%M:%S")

    alert_message = f"""🚨CHILD DISTRESS ALERT 🚨

Emotion: {emotion}
Speech: "{speech}"
Location: {location}
Date and time: {date}, {time_now}

Immediate attention required!
"""

    send_whatsapp_alert("Voice Distress", alert_message)

    # Reminder alerts
    for i in range(2):

        time.sleep(10)

        reminder_time = datetime.now(india)
        date = reminder_time.strftime("%d-%m-%Y")
        time_now = reminder_time.strftime("%H:%M:%S")

        reminder_message = f"""🚨CHILD DISTRESS ALERT 🚨

Date and time: {date}, {time_now}
Action: Reminder {i+1}
Speech: "Distress still active. Please respond immediately."

Immediate attention required!
"""

        send_whatsapp_alert(f"Reminder {i+1}", reminder_message)
    
    alert_active = False

# ---------------- MANUAL SOS ALERT ----------------
def send_manual_sos(location):
    
    global alert_active

    if alert_active:
        return

    alert_active = True

    india = pytz.timezone("Asia/Kolkata")
    now = datetime.now(india)

    date = now.strftime("%d-%m-%Y")
    time_now = now.strftime("%H:%M:%S")

    sos_message = f"""🚨CHILD DISTRESS ALERT 🚨

Emotion: N/A
Action: Manual SOS
Speech: "Manual emergency triggered by User"
Location: {location}
Date and time: {date}, {time_now}

Immediate attention required!
"""

    send_whatsapp_alert("Manual SOS", sos_message)

    # Reminder alerts
    for i in range(2):

        time.sleep(10)

        reminder_time = datetime.now(india)
        date = reminder_time.strftime("%d-%m-%Y")
        time_now = reminder_time.strftime("%H:%M:%S")

        reminder_message = f"""🚨CHILD DISTRESS ALERT 🚨

Date and time: {date}, {time_now}
Action: Reminder {i+1}
Speech: "Distress still active. Please respond immediately."

Immediate attention required!
"""

        send_whatsapp_alert(f"Reminder {i+1}", reminder_message)

    alert_active = False