import json, time
from   threading import Thread
from   websocket import create_connection

class WebsocketClient():
    def __init__(self, ws_url="wss://ws-feed.gdax.com", product_id="BTC-USD", channel=None):
        if ws_url[-1] == "/":
            self.url = ws_url[:-1]
        else:
            self.url = ws_url
        self.product_id = product_id
        self.channel = channel
        self.lastMsg = None
        self.stop = False
        self.thread = Thread(target=self.setup)
        self.thread.start()

    def setup(self):
        self.open()
        self.ws = create_connection(self.url)
        product = "product_ids"
        subParams = json.dumps({"type": "subscribe", product: [self.product_id]})
        if self.channel:
            subParams = json.dumps({"type": "subscribe", "channels": [{"name":self.channel, product:[self.product_id]}] })
        self.ws.send(subParams)
        self.listen()

    def open(self):
        print("-- Subscribed! --")

    def listen(self):
        while not self.stop:
            try:
                msg = json.loads(self.ws.recv())
            except Exception as e:
                print(e)
                break
            else:
                self.message(msg)

    def message(self, msg):
        if 'side' in msg:
            self.lastMsg = msg
            
        #print(msg)

    def getLastMessage(self):
        return self.lastMsg

    def close(self):
        self.ws.close()
        self.closed()

    def closed(self):
        print("Socket Closed")

if __name__ == "__main__":
    newWS = WebsocketClient() # Runs in a separate thread
    try:
      while True:
        time.sleep(0.1)
    except KeyboardInterrupt:
      newWS.stop = True
      newWS.thread.join()
    newWS.close()