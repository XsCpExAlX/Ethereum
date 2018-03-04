import json, time
from   threading import Thread
from   websocket import create_connection
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
 
pnconfig = PNConfiguration()
 
pnconfig.subscribe_key = 'sub-c-52a9ab50-291b-11e5-baaa-0619f8945a4f'
pnconfig.publish_key = 'demo'
 
self.pubnub = PubNub(pnconfig)
"""
class WebsocketClient():
    def __init__(self, ws_url="wss://ws-feed.gdax.com", product_id="BTC-USD"):
        if ws_url[-1] == "/":
            self.url = ws_url[:-1]
        else:
            self.url = ws_url
        self.stop = False
        self.product_id = product_id
        self.thread = Thread(target=self.setup)
        self.thread.start()
        """
class MySubscribeCallback(SubscribeCallback):
    def presence(self, pubnub, presence):
        pass  # handle incoming presence data
 
    def status(self, pubnub, status):
        if status.category == PNStatusCategory.PNUnexpectedDisconnectCategory:
            pass  # This event happens when radio / connectivity is lost
 
        elif status.category == PNStatusCategory.PNConnectedCategory:
            # Connect event. You can do stuff like publish, and know you'll get it.
            # Or just use the connected event to confirm you are subscribed for
            # UI / internal notifications, etc
            pubnub.publish().channel("lightning_executions_BTC_USD").message("hello!!").async(my_publish_callback)
        elif status.category == PNStatusCategory.PNReconnectedCategory:
            pass
            # Happens as part of our regular operation. This event happens when
            # radio / connectivity is lost, then regained.
        elif status.category == PNStatusCategory.PNDecryptionErrorCategory:
            pass
            # Handle message decryption error. Probably client configured to
            # encrypt messages and on live data feed it received plain text.
 
    def message(self, pubnub, message):
        pass  # Handle new message stored in message.message
 
 
pubnub.add_listener(MySubscribeCallback())
pubnub.subscribe().channels('awesomeChannel').execute()
"""
    def setup(self):
        self.open()
        self.ws = create_connection(self.url)
        if type(self.product_id) is list:
            #product_ids - plural for multiple products
            subParams = json.dumps({"type": "subscribe", "product_ids": self.product_id})
        else:
            subParams = json.dumps({"type": "subscribe", "product_id": self.product_id})
        self.ws.send(subParams)
        self.listen()

    def open(self):
        print("-- Subscribed! --")

    def listen(self):
        while not self.stop:
            try:
                msg = json.loads(self.ws.recv())
            except Exception as e:
                print e
                break
            else:
                self.message(msg)

    def message(self, msg):
        print(msg)

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
"""