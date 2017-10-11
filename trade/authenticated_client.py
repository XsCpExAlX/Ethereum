import hmac, hashlib, time, base64
import requests
import json
from requests.auth import AuthBase


# Sandbox api base
API_BASE = 'https://api-public.sandbox.gdax.com'
# # Real api base
# API_BASE = 'https://api.gdax.com'


class GDAXRequestAuth(AuthBase):

    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase


    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + request.path_url + (request.body or '')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest())
        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request

    def get_account(self, account_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + '/accounts/' + account_id, auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_accounts(self):
        return self.get_account('')

    def get_account_history(self, account_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        result = []
        r = requests.get(API_BASE + '/accounts/{}/ledger'.format(account_id), auth=auth, timeout=30)
        # r.raise_for_status()
        result.append(r.json())
        if "cb-after" in r.headers:
            self.history_pagination(account_id, result, r.headers["cb-after"])
        return result

    def history_pagination(self, account_id, result, after):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + '/accounts/{}/ledger?after={}'.format(account_id, str(after)), auth=auth, timeout=30)
        # r.raise_for_status()
        if r.json():
            result.append(r.json())
        if "cb-after" in r.headers:
            self.history_pagination(account_id, result, r.headers["cb-after"])
        return result

    def get_account_holds(self, account_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        result = []
        r = requests.get(API_BASE + '/accounts/{}/holds'.format(account_id), auth=auth, timeout=30)
        # r.raise_for_status()
        result.append(r.json())
        if "cb-after" in r.headers:
            self.holds_pagination(account_id, result, r.headers["cb-after"])
        return result

    def holds_pagination(self, account_id, result, after):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + '/accounts/{}/holds?after={}'.format(account_id, str(after)), auth=auth, timeout=30)
        # r.raise_for_status()
        if r.json():
            result.append(r.json())
        if "cb-after" in r.headers:
            self.holds_pagination(account_id, result, r.headers["cb-after"])
        return result


    def buy_market(self, product_id, size):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        order_data = {
            'type': 'market',
            'side': 'buy',
            'product_id': product_id,
            'size': size
        }
        r = requests.post(API_BASE + '/orders', data=json.dumps(order_data), auth=auth, timeout=30)
        return r.json()


    def sell_market(self, product_id, size):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        order_data = {
            'type': 'market',
            'side': 'sell',
            'product_id': product_id,
            'size': size
        }
        r = requests.post(API_BASE + '/orders', data=json.dumps(order_data), auth=auth, timeout=30)
        return r.json()


    def buy_limit(self, product_id, price, size, time_in_force='GTC', cancel_after=None, post_only=None):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        order_data = {
            'type': 'limit',
            'side': 'buy',
            'product_id': product_id,
            'price': price,
            'size': size,
            'time_in_force': time_in_force
        }
        if 'time_in_force' is 'GTT':
            order_data['cancel_after'] = cancel_after
        if 'time_in_force' not in ['IOC', 'FOK']:
            order_data['post_only'] = post_only
        r = requests.post(API_BASE + '/orders', data=json.dumps(order_data), auth=auth, timeout=30)
        if r.status_code is not 200:
            raise Exception('Invalid GDAX Status Code: %d' % r.status_code)
        return r.json()


    def sell_limit(self, product_id, price, size, time_in_force='GTC', cancel_after=None, post_only=None):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        order_data = {
            'type': 'limit',
            'side': 'sell',
            'product_id': product_id,
            'price': price,
            'size': size,
            'time_in_force': time_in_force
        }
        if 'time_in_force' is 'GTT':
            order_data['cancel_after'] = cancel_after
        if 'time_in_force' not in ['IOC', 'FOK']:
            order_data['post_only'] = post_only
        r = requests.post(API_BASE + '/orders', data=json.dumps(order_data), auth=auth, timeout=30)
        if r.status_code is not 200:
            raise Exception('Invalid GDAX Status Code: %d' % r.status_code)
        return r.json()


    def order_status(self, order_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        order_url = API_BASE + '/orders/' + order_id
        r = requests.post(API_BASE + '/orders', auth=auth, timeout=30)
        if r.status_code is not 200:
            raise Exception('Invalid GDAX Status Code: %d' % r.status_code)
        return r.json()

    def cancel_order(self, order_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.delete(API_BASE + '/orders/' + order_id, auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def cancel_all(self, product_id=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        url = API_BASE + '/orders/'
        if product_id:
            url += "?product_id={}&".format(str(product_id))
        r = requests.delete(url, auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_order(self, order_id):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + '/orders/' + order_id, auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_orders(self, product_id=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        result = []
        url = API_BASE + '/orders/'
        if product_id:
            url += "?product_id={}&".format(product_id)
        r = requests.get(url, auth=auth, timeout=30)
        # r.raise_for_status()
        result.append(r.json())
        if 'cb-after' in r.headers:
            self.paginate_orders(product_id, result, r.headers['cb-after'])
        return result

    def paginate_orders(self, product_id, result, after):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        url = API_BASE + '/orders?after={}&'.format(str(after))
        if product_id:
            url += "product_id={}&".format(product_id)
        r = requests.get(url, auth=auth, timeout=30)
        # r.raise_for_status()
        if r.json():
            result.append(r.json())
        if 'cb-after' in r.headers:
            self.paginate_orders(product_id, result, r.headers['cb-after'])
        return result

    def get_fills(self, order_id='', product_id='', before='', after='', limit=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        result = []
        url = API_BASE + '/fills?'
        if order_id:
            url += "order_id={}&".format(str(order_id))
        if product_id:
            url += "product_id={}&".format(product_id)
        if before:
            url += "before={}&".format(str(before))
        if after:
            url += "after={}&".format(str(after))
        if limit:
            url += "limit={}&".format(str(limit))
        r = requests.get(url, auth=auth, timeout=30)
        # r.raise_for_status()
        result.append(r.json())
        if 'cb-after' in r.headers and limit is not len(r.json()):
            return self.paginate_fills(result, r.headers['cb-after'], order_id=order_id, product_id=product_id)
        return result

    def paginate_fills(self, result, after, order_id='', product_id=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        url = API_BASE + '/fills?after={}&'.format(str(after))
        if order_id:
            url += "order_id={}&".format(str(order_id))
        if product_id:
            url += "product_id={}&".format(product_id)
        r = requests.get(url, auth=auth, timeout=30)
        # r.raise_for_status()
        if r.json():
            result.append(r.json())
        if 'cb-after' in r.headers:
            return self.paginate_fills(result, r.headers['cb-after'], order_id=order_id, product_id=product_id)
        return result

    def get_fundings(self, result='', status='', after=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        if not result:
            result = []
        url = API_BASE + '/funding?'
        if status:
            url += "status={}&".format(str(status))
        if after:
            url += 'after={}&'.format(str(after))
        r = requests.get(url, auth=auth, timeout=30)
        # r.raise_for_status()
        result.append(r.json())
        if 'cb-after' in r.headers:
            return self.get_fundings(result, status=status, after=r.headers['cb-after'])
        return result

    def repay_funding(self, amount='', currency=''):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency  # example: USD
        }
        r = requests.post(API_BASE + "/funding/repay", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def margin_transfer(self, margin_profile_id="", transfer_type="", currency="", amount=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "margin_profile_id": margin_profile_id,
            "type": transfer_type,
            "currency": currency,  # example: USD
            "amount": amount
        }
        r = requests.post(API_BASE + "/profiles/margin-transfer", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_position(self):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + "/position", auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def close_position(self, repay_only=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "repay_only": repay_only or False
        }
        r = requests.post(API_BASE + "/position/close", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def deposit(self, amount="", currency="", payment_method_id=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency,
            "payment_method_id": payment_method_id
        }
        r = requests.post(API_BASE + "/deposits/payment-method", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def coinbase_deposit(self, amount="", currency="", coinbase_account_id=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency,
            "coinbase_account_id": coinbase_account_id
        }
        r = requests.post(API_BASE + "/deposits/coinbase-account", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def withdraw(self, amount="", currency="", payment_method_id=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency,
            "payment_method_id": payment_method_id
        }
        r = requests.post(API_BASE + "/withdrawals/payment-method", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def coinbase_withdraw(self, amount="", currency="", coinbase_account_id=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency,
            "coinbase_account_id": coinbase_account_id
        }
        r = requests.post(API_BASE + "/withdrawals/coinbase", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def crypto_withdraw(self, amount="", currency="", crypto_address=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "amount": amount,
            "currency": currency,
            "crypto_address": crypto_address
        }
        r = requests.post(API_BASE + "/withdrawals/crypto", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_payment_methods(self):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + "/payment-methods", auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_coinbase_accounts(self):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + "/coinbase-accounts", auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def create_report(self, report_type="", start_date="", end_date="", product_id="", account_id="", report_format="",
                      email=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        payload = {
            "type": report_type,
            "start_date": start_date,
            "end_date": end_date,
            "product_id": product_id,
            "account_id": account_id,
            "format": report_format,
            "email": email
        }
        r = requests.post(API_BASE + "/reports", data=json.dumps(payload), auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_report(self, report_id=""):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + "/reports/" + report_id, auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()

    def get_trailing_volume(self):
        auth = GDAXRequestAuth(self.api_key, self.secret_key, self.passphrase)
        r = requests.get(API_BASE + "/users/self/trailing-volume", auth=auth, timeout=30)
        # r.raise_for_status()
        return r.json()