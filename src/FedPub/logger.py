from datetime import datetime


class Logger:
    def __init__(self, args, c_id, is_server=False):
        self.args = args
        self.is_server = is_server
        self.c_id = c_id

    def print(self, message):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        msg = f"[{now}]"
        msg += f"[{self.args.model}]"
        msg += f"[server]" if self.is_server else f"[c:{self.c_id}]"
        msg += f" {message}\n"
        print(msg)
