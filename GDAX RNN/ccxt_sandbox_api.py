import ccxt


sandbox_api_key = '7b34e0d36ea3726d856fd45d26343150'
sandbox_secret_key = 'pCU/ks3QzKWTs4WV+PbSMddcYgpOPLB6p8KcHXIFdoUp/Kz4FvTKxUz19wB336zaKYL4J2hCb6HOsGm+DWykcg=='
sandbox_passphrase = 'k9328vg3eh'


gdax = ccxt.gdax()
gdax.apiKey = sandbox_api_key
gdax.secret = sandbox_secret_key
gdax.password = sandbox_passphrase