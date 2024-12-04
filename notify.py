import datetime


class Notify:
    def __init__(self, tb, message_id):
        self.tb = tb
        self.chat_id = message_id.chat.id

    def initial_notify(self):
        return self.tb.send_message(self.chat_id, "Training")

    def notify(self, message):
        return self.tb.send_message(self.chat_id, message)

    def notify_image(self, photo, message):
        return self.tb.send_photo(self.chat_id, photo, caption=message)