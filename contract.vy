# @version 0.3.7

# Interfaces
from vyper.interfaces import ERC20

# Events
event PredictionCreated:
    prediction_id: uint256
    predictor: address
    text: String[500]
    keywords: String[100]
    expiry_timestamp: uint256
    stake_amount: uint256

event PredictionVerified:
    prediction_id: uint256
    is_correct: bool
    evidence: String[200]

event RewardClaimed:
    prediction_id: uint256
    predictor: address
    amount: uint256

# Structs
struct Prediction:
    predictor: address
    text: String[500]
    keywords: String[100]
    expiry_timestamp: uint256
    stake_amount: uint256
    is_verified: bool
    is_correct: bool
    evidence: String[200]
    is_claimed: bool

# Storage variables
prediction_count: public(uint256)
predictions: public(HashMap[uint256, Prediction])
user_predictions: public(HashMap[address, DynArray[uint256, 100]])
oracle_address: public(address)
verification_fee_percentage: public(uint256)  # In basis points (e.g., 500 = 5%)
owner: public(address)

# Constants
REWARD_MULTIPLIER: constant(uint256) = 150  # 150% of stake (get back 1.5x stake)
BASIS_POINTS: constant(uint256) = 10000
MAX_PREDICTIONS: constant(uint256) = 1000  # Maximum number of predictions allowed

@external
def __init__():
    """
    Initialize contract with the deployer as owner and oracle
    """
    self.owner = msg.sender
    self.oracle_address = msg.sender
    self.prediction_count = 0
    self.verification_fee_percentage = 500  # 5% fee by default

@external
@payable
def create_prediction(text: String[500], keywords: String[100], expiry_timestamp: uint256) -> uint256:
    """
    Create a new prediction with a stake
    
    @param text The prediction text
    @param keywords Keywords related to the prediction
    @param expiry_timestamp Unix timestamp when the prediction expires
    @return The prediction ID
    """
    # Validate inputs
    assert len(text) > 0, "Prediction text cannot be empty"
    assert len(keywords) > 0, "Keywords cannot be empty"
    assert expiry_timestamp > block.timestamp, "Expiry must be in the future"
    assert msg.value > 0, "Stake amount must be greater than 0"
    
    # Create new prediction ID
    prediction_id: uint256 = self.prediction_count + 1
    self.prediction_count = prediction_id
    
    # Store prediction
    self.predictions[prediction_id] = Prediction({
        predictor: msg.sender,
        text: text,
        keywords: keywords,
        expiry_timestamp: expiry_timestamp,
        stake_amount: msg.value,
        is_verified: False,
        is_correct: False,
        evidence: "",
        is_claimed: False
    })
    
    # Add to user's predictions
    user_predictions_list: DynArray[uint256, 100] = self.user_predictions[msg.sender]
    user_predictions_list.append(prediction_id)
    self.user_predictions[msg.sender] = user_predictions_list
    
    # Emit event
    log PredictionCreated(prediction_id, msg.sender, text, keywords, expiry_timestamp, msg.value)
    
    return prediction_id

@external
def verify_prediction(prediction_id: uint256, is_correct: bool, evidence: String[200]):
    """
    Verify a prediction as correct or incorrect
    
    @param prediction_id The ID of the prediction to verify
    @param is_correct Whether the prediction is correct
    @param evidence Evidence supporting the verification
    """
    # Only oracle can verify predictions
    assert msg.sender == self.oracle_address, "Only oracle can verify predictions"
    
    # Check prediction exists
    assert prediction_id > 0 and prediction_id <= self.prediction_count, "Invalid prediction ID"
    
    prediction: Prediction = self.predictions[prediction_id]
    
    # Check prediction is not already verified
    assert not prediction.is_verified, "Prediction already verified"
    
    # Check prediction has expired
    assert block.timestamp >= prediction.expiry_timestamp, "Prediction has not expired yet"
    
    # Update prediction
    self.predictions[prediction_id].is_verified = True
    self.predictions[prediction_id].is_correct = is_correct
    self.predictions[prediction_id].evidence = evidence
    
    # Emit event
    log PredictionVerified(prediction_id, is_correct, evidence)

@external
def claim_reward(prediction_id: uint256):
    """
    Claim reward for a correct prediction
    
    @param prediction_id The ID of the prediction to claim reward for
    """
    # Check prediction exists
    assert prediction_id > 0 and prediction_id <= self.prediction_count, "Invalid prediction ID"
    
    prediction: Prediction = self.predictions[prediction_id]
    
    # Check prediction belongs to caller
    assert prediction.predictor == msg.sender, "Not your prediction"
    
    # Check prediction is verified as correct
    assert prediction.is_verified, "Prediction not verified yet"
    assert prediction.is_correct, "Prediction was incorrect"
    
    # Check reward not already claimed
    assert not prediction.is_claimed, "Reward already claimed"
    
    # Mark as claimed
    self.predictions[prediction_id].is_claimed = True
    
    # Calculate reward amount
    reward_amount: uint256 = prediction.stake_amount * REWARD_MULTIPLIER / 100
    
    # Transfer reward
    send(msg.sender, reward_amount)
    
    # Emit event
    log RewardClaimed(prediction_id, msg.sender, reward_amount)

@external
@view
def get_prediction_details(prediction_id: uint256) -> (address, String[500], String[100], uint256, uint256, bool, bool, String[200], bool):
    """
    Get all details about a prediction
    
    @param prediction_id The ID of the prediction
    @return All prediction details
    """
    assert prediction_id > 0 and prediction_id <= self.prediction_count, "Invalid prediction ID"
    
    prediction: Prediction = self.predictions[prediction_id]
    
    return (
        prediction.predictor,
        prediction.text,
        prediction.keywords,
        prediction.expiry_timestamp,
        prediction.stake_amount,
        prediction.is_verified,
        prediction.is_correct,
        prediction.evidence,
        prediction.is_claimed
    )

@external
@view
def get_user_predictions(user: address) -> DynArray[uint256, 100]:
    """
    Get all prediction IDs created by a user
    
    @param user The user address
    @return Array of prediction IDs
    """
    return self.user_predictions[user]

@external
def change_oracle(new_oracle: address):
    """
    Change the oracle address
    
    @param new_oracle The new oracle address
    """
    assert msg.sender == self.owner, "Only owner can change oracle"
    assert new_oracle != empty(address), "Invalid oracle address"
    
    self.oracle_address = new_oracle

@external
def change_verification_fee(new_fee_percentage: uint256):
    """
    Change the verification fee percentage
    
    @param new_fee_percentage The new fee percentage in basis points
    """
    assert msg.sender == self.owner, "Only owner can change fee"
    assert new_fee_percentage <= 2000, "Fee cannot exceed 20%"
    
    self.verification_fee_percentage = new_fee_percentage

@external
def withdraw_fees():
    """
    Withdraw accumulated fees to owner
    """
    assert msg.sender == self.owner, "Only owner can withdraw fees"
    
    # Calculate amount that can be withdrawn (contract balance minus staked funds)
    staked_amount: uint256 = 0
    
    # Calculate total staked amount for active predictions
    for i in range(MAX_PREDICTIONS):
        if i >= self.prediction_count:
            break
        prediction: Prediction = self.predictions[i + 1]  # +1 because prediction IDs start at 1
        # Add stake amounts for predictions that are not verified or verified correct but not claimed
        if (not prediction.is_verified) or (prediction.is_correct and not prediction.is_claimed):
            staked_amount += prediction.stake_amount
    
    withdrawable_amount: uint256 = self.balance - staked_amount
    assert withdrawable_amount > 0, "No fees to withdraw"
    
    # Transfer withdrawable amount to owner
    send(self.owner, withdrawable_amount)

@external
@payable
def __default__():
    """
    Fallback function to accept ETH
    """
    pass