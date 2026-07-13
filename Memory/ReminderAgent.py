from google import genai
from google.genai import types
from memory_manager import Neo4j
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
import uuid

class ReminderAgent:
    def __init__(self):
        client = genai.Client(api_key="")
        self.chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(response_mime_type="text/plain", system_instruction="""You are Jarvis, a voice-based AI assistant designed to interact with the user through natural conversation. Address the user as sir. Follow these instructions:
                                                                                        1) You are speaking, not typing. User is listening, not reading. So avoid reading out code or overly long answers. Summarise if needed or speak some ideas and ask user if you should continue the list.
                                                                                        2) Always keep the answers being spoken out loud short (just a few sentences). Details of the answer show on screen or ask user if you should elaborate.   
                                                                                        3) Use the following tags:
                                                                                            a) <speak></speak> - To speak something out loud. Kepp it concise, summarised, and short. Frequency - Always. Every response should have this tag because You are voice-based assistant.
                                                                                            b) <screen></screen> - To show the long form of the answer or anything on screen. Frequency - Only when needed. 
                                                                                            c) <type></type> - Type the text on screen at current cursor position. Frequency - Only when User asks.
                                                                                            **IMPORTANT: Every part of the response should be covered in a tag**
                                                                                        4) When User says vague things like "What does this mean?" or "Analyze this", Check the on-screen text using the provided tool.
                                                                                        5) Be proactive. You are friend. Get to know your user more. Ask questions, clarifications if needed. You can also ask questions just to get to know the user more."""))
        self.db = Neo4j()
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self._restore_reminders()

    def _reminder_callback(self, reminder_id, repeat):
        print(f"🔔 Reminder triggered: {reminder_id}")
        if repeat == "once":
            self.db.save_node("Reminder", {
                "name": reminder_id,
                "status": "triggered"
            })
        else:
            # Don't change status for recurring; leave it as 'scheduled'
            pass

    def set_reminder(self, message: str, time: str, repeat: str = "once") -> str:
        """
        Schedule a one-time or repeating reminder.

        Args:
            message: Reminder message
            time: ISO format datetime string
            repeat: 'once', 'daily', 'weekly', or 'monthly'

        Returns:
            Unique reminder ID
        """
        try:
            dt = datetime.fromisoformat(time)
        except Exception:
            raise ValueError("Invalid ISO datetime format")

        reminder_id = str(uuid.uuid4())

        self.db.save_node("Reminder", {
            "name": reminder_id,
            "message": message,
            "time": time,
            "repeat": repeat,
            "status": "scheduled"
        })

        self._schedule_job(reminder_id, dt, repeat)
        return reminder_id

    def _schedule_job(self, reminder_id, dt, repeat):
        if repeat == "once":
            self.scheduler.add_job(
                self._reminder_callback,
                'date',
                run_date=dt,
                args=[reminder_id, repeat],
                id=reminder_id
            )
        else:
            interval = {
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
                "monthly": timedelta(days=30)  # Simplified monthly
            }.get(repeat)

            if not interval:
                print(f"⚠️ Invalid repeat value for {reminder_id}")
                return

            self.scheduler.add_job(
                self._reminder_callback,
                'interval',
                start_date=dt,
                weeks=1 if repeat == "weekly" else 0,
                days=1 if repeat == "daily" else (30 if repeat == "monthly" else 0),
                args=[reminder_id, repeat],
                id=reminder_id
            )

    def get_reminders(self, status: str = None, contains: str = None,
                      after: str = None, before: str = None):
        all_reminders = self.db.search_nodes_by_property("Reminder", "status", status) if status else self.db.search_node("Reminder")

        try:
            parsed = []
            for _, node in eval(all_reminders):
                if "message" not in node or "time" not in node:
                    continue
                if contains and contains.lower() not in node["message"].lower():
                    continue

                node_time = datetime.fromisoformat(node["time"])
                if after:
                    if node_time < datetime.fromisoformat(after):
                        continue
                if before:
                    if node_time > datetime.fromisoformat(before):
                        continue

                parsed.append({
                    "id": node["name"],
                    "message": node["message"],
                    "time": node["time"],
                    "repeat": node.get("repeat", "once"),
                    "status": node.get("status", "unknown")
                })
            return parsed
        except Exception as e:
            print("❌ Error parsing reminders:", e)
            return []

    def delete_reminder(self, reminder_id: str):
        try:
            self.scheduler.remove_job(reminder_id)
        except JobLookupError:
            pass
        self.db.run_query("MATCH (n:Reminder {name: $name}) DETACH DELETE n", {"name": reminder_id})

    def _restore_reminders(self):
        try:
            all = eval(self.db.search_nodes_by_property("Reminder", "status", "scheduled"))
        except Exception as e:
            print("⚠️ Failed to load reminders from Neo4j:", e)
            return

        for _, node in all:
            try:
                reminder_time = datetime.fromisoformat(node["time"])
                repeat = node.get("repeat", "once")
                now = datetime.now()

                if repeat == "once":
                    if reminder_time > now:
                        self._schedule_job(node["name"], reminder_time, repeat)
                    else:
                        self.db.save_node("Reminder", {
                            "name": node["name"],
                            "status": "missed"
                        })
                else:
                    # For recurring jobs, resume from now
                    self._schedule_job(node["name"], max(now, reminder_time), repeat)

            except Exception as e:
                print(f"⚠️ Failed to restore reminder {node.get('name')}: {e}")
