from flask import Flask
from jsonrpc.backend.flask import JSONRPCAPI


class WebAPIService(Flask):
    def __init__(self):
        super(WebAPIService, self
              ).__init__(self.__class__.__name__,
                         static_folder="./static/")
        self.api = JSONRPCAPI()
        self.add_url_rule("/api", view_func=self.api.as_view(), methods=["POST"])

