from web3 import Web3
from eth_account import Account
import secrets

# Generate a random private key
private_key = "0x" + secrets.token_hex(32)

# Create account from private key
account = Account.from_key(private_key)
address = account.address

print(f"Oracle Address: {address}")
print(f"Oracle Private Key: {private_key}")
print("IMPORTANT: Save this private key securely. If lost, you cannot recover it!")