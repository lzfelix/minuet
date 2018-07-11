"""
Minimal example of how to expose a Minuet model for prediction on a server.
Simple send to localhost:8090/ a POST request in the following format:
{
    "sentences": ["apple inc is selling apples in the bay area"]
}
"""


from minuet import Minuet

from tornado import escape
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.web import Application
from tornado.web import RequestHandler

import logging
logging.basicConfig(level=logging.DEBUG)

PORT=8090
NER_PATH = '../models/small_ner/'

class DefaultHandler(RequestHandler):

    def initialize(self, **kwargs):
        self.model = kwargs['model']

    def post(self):
        body = escape.json_decode(self.request.body)
        sentences = body.get('sentences', None)

        if not isinstance(sentences, list):
            raise RuntimeError('sentences should be a list')

        try:
            sentences = [s.split() for s in sentences]
            labels = model.decode_predictions(model.predict(sentences))
            result = [[f'{w}/{l}' for w, l in zip(se, la)]
                        for se, la in zip(sentences, labels)]

            self.write(dict(result=result))
        except:
            raise RuntimeError('Failed to perform prediction.')


class Server(Application):
    def __init__(self, model):
        handlers = [(r'/', DefaultHandler, dict(model=model))]
        Application.__init__(self, handlers, debug=True, autoreload=True)


if __name__ == '__main__':
    logging.debug('Loading Minuet model.')
    model = Minuet.load(NER_PATH)
    logging.debug('Starting tornado server.')

    try:
        logging.info('Serving on port {}'.format(PORT))
        app = HTTPServer(Server(model))
        app.listen(PORT)
        IOLoop.instance().start()
    except KeyboardInterrupt:
        exit()
