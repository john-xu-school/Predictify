# @version ^0.3.9

# PredictifyContract: A smart contract for NewsOracle DApp
# This contract allows users to make predictions about future events
# and verify them based on evidence

from vyper.interfaces import ERC20

struct Prediction:
    predictor: address
    text: String[1000]  # Prediction text (title + details)
    keywords: String[200]  # Keywords for verification
    expiry_date: uint256  # Timestamp when the prediction expires
    stake_amount: uint256  # Amount staked on the prediction
    is_verified: bool  # Whether the prediction has been verified
    is_correct: bool  # Whether the prediction was correct
    evidence: String[1000]  # Evidence of verification
    is_claimed: bool  # Whether the reward has been claimed

# State variables
owner: public(address)
oracle_addresses: public(HashMap[address, bool])  # Addresses authorized to verify predictions
prediction_count: public(uint256)
predictions: public(HashMap[uint256, Prediction])
user_predictions: public(HashMap[address, DynArray[uint256, 100]])  # Maps user address to their prediction IDs
fee_percentage: public(uint256)  # Fee percentage (out of 100)

# Events
event PredictionCreated:
    prediction_id: indexed(uint256)
    predictor: indexed(address)
    text: String[1000]
    expiry_date: uint256
    stake_amount: uint256

event PredictionVerified:
    prediction_id: indexed(uint256)
    is_correct: bool
    evidence: String[1000]
    
event RewardClaimed:
    prediction_id: indexed(uint256)
    predictor: indexed(address)
    amount: uint256

@external
def __init__():
    """
    Initialize the contract.
    """
    self.owner = msg.sender
    self.oracle_addresses[msg.sender] = True
    self.prediction_count = 0
    self.fee_percentage = 5  # 5% fee

@external
@payable
def create_prediction(prediction_text: String[1000], keywords: String[200], expiry_date: uint256) -> uint256:
    """
    Create a new prediction.
    """
    # Ensure stake amount > 0
    assert msg.value >= 0, "Stake amount must be greater than 0"
    
    # Ensure expiry date is in the future
    assert expiry_date > block.timestamp, "Expiry date must be in the future"
    
    # Create prediction
    prediction_id: uint256 = self.prediction_count + 1
    self.prediction_count = prediction_id
    
    # Store prediction details
    self.predictions[prediction_id] = Prediction({
        predictor: msg.sender,
        text: prediction_text,
        keywords: keywords,
        expiry_date: expiry_date,
        stake_amount: 0,
        is_verified: False,
        is_correct: False,
        evidence: "",
        is_claimed: False
    })
    
    # Add prediction ID to user's list
    user_preds: DynArray[uint256, 100] = self.user_predictions[msg.sender]
    user_preds.append(prediction_id)
    self.user_predictions[msg.sender] = user_preds
    
    # Emit event
    log PredictionCreated(prediction_id, msg.sender, prediction_text, expiry_date, msg.value)
    
    return prediction_id

@external
def verify_prediction(prediction_id: uint256, is_correct: bool, evidence: String[1000]):
    """
    Verify a prediction.
    """
    # Ensure caller is an oracle
    assert self.oracle_addresses[msg.sender], "Caller is not an oracle"
    
    # Ensure prediction exists
    assert prediction_id <= self.prediction_count, "Prediction does not exist"
    
    # Get prediction
    prediction: Prediction = self.predictions[prediction_id]
    
    # Ensure prediction is not already verified
    assert not prediction.is_verified, "Prediction already verified"
    
    # Ensure prediction has expired
    assert block.timestamp > prediction.expiry_date, "Prediction has not expired yet"
    
    # Update prediction
    prediction.is_verified = True
    prediction.is_correct = is_correct
    prediction.evidence = evidence
    self.predictions[prediction_id] = prediction
    
    # Emit event
    log PredictionVerified(prediction_id, is_correct, evidence)

@external
def claim_reward(prediction_id: uint256):
    """
    Claim reward for a correct prediction.
    """
    # Ensure prediction exists
    assert prediction_id <= self.prediction_count, "Prediction does not exist"
    
    # Get prediction
    prediction: Prediction = self.predictions[prediction_id]
    
    # Ensure caller is the predictor
    assert msg.sender == prediction.predictor, "Only predictor can claim reward"
    
    # Ensure prediction is verified
    assert prediction.is_verified, "Prediction not verified yet"
    
    # Ensure prediction is correct
    assert prediction.is_correct, "Prediction was incorrect"
    
    # Ensure reward has not been claimed
    assert not prediction.is_claimed, "Reward already claimed"
    
    # Mark as claimed
    prediction.is_claimed = True
    self.predictions[prediction_id] = prediction
    
    # Calculate reward
    # For correct predictions, user gets their stake back plus fees from incorrect predictions
    reward: uint256 = prediction.stake_amount
    
    # Send reward
    send(prediction.predictor, reward)
    
    # Emit event
    log RewardClaimed(prediction_id, prediction.predictor, reward)

@view
@external
def get_prediction_details(prediction_id: uint256) -> Prediction:
    """
    Get details of a prediction.
    """
    # Ensure prediction exists
    assert prediction_id <= self.prediction_count, "Prediction does not exist"
    
    return self.predictions[prediction_id]

@view
@external
def get_user_predictions(user: address) -> DynArray[uint256, 100]:
    """
    Get all prediction IDs for a user.
    """
    return self.user_predictions[user]

@external
def add_oracle(oracle_address: address):
    """
    Add a new oracle address.
    """
    # Ensure caller is owner
    assert msg.sender == self.owner, "Only owner can add oracles"
    
    # Add oracle
    self.oracle_addresses[oracle_address] = True

@external
def remove_oracle(oracle_address: address):
    """
    Remove an oracle address.
    """
    # Ensure caller is owner
    assert msg.sender == self.owner, "Only owner can remove oracles"
    
    # Remove oracle
    self.oracle_addresses[oracle_address] = False

@external
def set_fee_percentage(new_percentage: uint256):
    """
    Set the fee percentage.    
    """
    # Ensure caller is owner
    assert msg.sender == self.owner, "Only owner can set fee percentage"
    
    # Ensure percentage is valid
    assert new_percentage <= 20, "Fee percentage cannot exceed 20%"
    
    # Set fee percentage
    self.fee_percentage = new_percentage

@external
def distribute_rewards_to_pool(pool_address: address):
    """
    Distribute accumulated fees to reward pool.
    """
    # Ensure caller is owner
    assert msg.sender == self.owner, "Only owner can distribute rewards"
    
    # Get contract balance
    balance: uint256 = self.balance
    
    # Ensure there are funds to distribute
    assert balance > 0, "No funds to distribute"
    
    # Send funds to pool
    send(pool_address, balance)

@external
@payable
def add_to_reward_pool():
    """
    Add funds to the reward pool.
    """
    # Nothing to do, funds are automatically added to contract balance
    pass

@external
def withdraw_unclaimed_funds(amount: uint256):
    """
    Withdraw unclaimed funds.    
    """
    # Ensure caller is owner
    assert msg.sender == self.owner, "Only owner can withdraw funds"
    
    # Ensure amount is valid
    assert amount <= self.balance, "Insufficient balance"
    
    # Send funds to owner
    send(self.owner, amount)

@external
@view
def get_contract_balance() -> uint256:
    """
    Get the contract balance.
    
    """
    return self.balance