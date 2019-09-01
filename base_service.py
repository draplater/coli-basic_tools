try:
    from flask import Flask
    from jsonrpc.backend.flask import JSONRPCAPI


    class WebAPIService(Flask):
        def __init__(self, name=None):
            if name is None:
                name = self.__class__.__name__
            super(WebAPIService, self
                  ).__init__(name, static_folder="./static/")
            self.api = JSONRPCAPI()
            self.add_url_rule("/api", view_func=self.api.as_view(), methods=["POST"])
except ImportError:
    class WebAPIService(object):
        def __init__(self, *args, **kwargs):
            raise Exception("requires flask and jsonrpc")
