from sandbox_api import sandbox_api_key, sandbox_secret_key, sandbox_passphrase
from real_api import real_api_key, real_secret_key, real_passphrase
from authenticated_client import GDAXRequestAuth


# Sandbox api base
API_BASE = 'https://api-public.sandbox.gdax.com'
# # Real api base
# API_BASE = 'https://api.gdax.com'


API_KEY = sandbox_api_key
SECRET_KEY = sandbox_secret_key
API_PASS = sandbox_passphrase
client = GDAXRequestAuth(API_KEY, SECRET_KEY, API_PASS)


# Place an order
order1 = client.buy_limit('BTC-USD', 3000, 0.02, 'GTC')
print(order1)

order2 = client.sell_limit('BTC-USD', 5000, 0.01, 'GTC')
print(order2)

order3 = client.buy_market('BTC-USD', 0.5)
print(order3)

order4 = client.sell_market('BTC-USD', 0.3)
print(order4)