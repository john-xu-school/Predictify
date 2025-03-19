# @version ^0.3.7

# Events
event PredictionCreated:
    prediction_id: uint256
    predictor: address
    prediction_text: String[500]
    keywords: String[200]
    expiry_date: uint256
    stake_amount: uint256

event PredictionVerified:
    prediction_id: uint256
    result: bool
    evidence: String[500]

event RewardClaimed:
    prediction_id: uint256
    predictor: address
    amount: uint256

# Structs
struct Prediction:
    predictor: address
    prediction_text: String[500]
    keywords: String[200]
    expiry_date: uint256
    stake_amount: uint256
    is_verified: bool
    is_correct: bool
    evidence: String[500]
    is_claimed: bool

# State variables
owner: public(address)
oracle: public(address)
prediction_count: public(uint256)
predictions: public(HashMap[uint256, Prediction])
user_predictions: public(HashMap[address, DynArray[uint256, 100]])
service_fee: public(uint256)  # in basis points (1/100 of a percent)
min_stake: public(uint256)

# Constructor
@external
def __init__():
    self.owner = msg.sender
    self.oracle = msg.sender  # Initially, owner is the oracle
    self.prediction_count = 0
    self.service_fee = 200  # 2% fee
    self.min_stake = 10**16  # 0.01 ETH

# Modifiers (implemented as functions in Vyper)
@internal
def _only_owner():
    assert msg.sender == self.owner, "Only owner can call this function"

@internal
def _only_oracle():
    assert msg.sender == self.oracle, "Only oracle can call this function"

# Core functions
@external
@payable
def create_prediction(prediction_text: String[500], keywords: String[200], expiry_date: uint256) -> uint256:
    """
    Create a new prediction by staking ETH
    Returns the prediction ID
    """
    assert msg.value >= self.min_stake, "Stake amount too low"
    assert expiry_date > block.timestamp, "Expiry date must be in the future"
    
    # Create prediction
    prediction_id: uint256 = self.prediction_count
    self.predictions[prediction_id] = Prediction({
        predictor: msg.sender,
        prediction_text: prediction_text,
        keywords: keywords,
        expiry_date: expiry_date,
        stake_amount: msg.value,
        is_verified: False,
        is_correct: False,
        evidence: "",
        is_claimed: False
    })
    
    # Add to user's predictions
    if len(self.user_predictions[msg.sender]) == 0:
        self.user_predictions[msg.sender] = [prediction_id]
    else:
        self.user_predictions[msg.sender].append(prediction_id)
    
    # Increment counter
    self.prediction_count += 1
    
    log PredictionCreated(prediction_id, msg.sender, prediction_text, keywords, expiry_date, msg.value)
    return prediction_id

@external
def verify_prediction(prediction_id: uint256, is_correct: bool, evidence: String[500]) -> bool:
    """
    Verify a prediction result, can only be called by the oracle
    Returns success status
    """
    self._only_oracle()
    
    prediction: Prediction = self.predictions[prediction_id]
    assert prediction.predictor != empty(address), "Prediction does not exist"
    assert not prediction.is_verified, "Prediction already verified"
    assert block.timestamp >= prediction.expiry_date, "Prediction not yet expired"
    
    self.predictions[prediction_id].is_verified = True
    self.predictions[prediction_id].is_correct = is_correct
    self.predictions[prediction_id].evidence = evidence
    
    log PredictionVerified(prediction_id, is_correct, evidence)
    return True

@external
def claim_reward(prediction_id: uint256) -> bool:
    """
    Claim rewards for a correct prediction
    Returns success status
    """
    prediction: Prediction = self.predictions[prediction_id]
    assert prediction.predictor == msg.sender, "Only predictor can claim"
    assert prediction.is_verified, "Prediction not yet verified"
    assert prediction.is_correct, "Prediction was incorrect"
    assert not prediction.is_claimed, "Reward already claimed"
    
    self.predictions[prediction_id].is_claimed = True
    
    # Calculate reward (original stake minus service fee)
    fee_amount: uint256 = prediction.stake_amount * self.service_fee / 10000
    reward_amount: uint256 = prediction.stake_amount - fee_amount
    
    # Transfer reward
    send(msg.sender, reward_amount)
    send(self.owner, fee_amount)
    
    log RewardClaimed(prediction_id, msg.sender, reward_amount)
    return True

# Admin functions
@external
def set_oracle(new_oracle: address) -> bool:
    """
    Set a new oracle address
    Returns success status
    """
    self._only_owner()
    self.oracle = new_oracle
    return True

@external
def set_service_fee(new_fee: uint256) -> bool:
    """
    Set a new service fee (in basis points)
    Returns success status
    """
    self._only_owner()
    assert new_fee <= 1000, "Fee cannot exceed 10%"
    self.service_fee = new_fee
    return True

@external
def set_min_stake(new_min_stake: uint256) -> bool:
    """
    Set a new minimum stake amount
    Returns success status
    """
    self._only_owner()
    self.min_stake = new_min_stake
    return True

# View functions
@view
@external
def get_user_predictions(user: address) -> DynArray[uint256, 100]:
    """
    Get all prediction IDs for a specific user
    Returns empty array if no predictions exist
    """
    if len(self.user_predictions[user]) == 0:
        return []
    return self.user_predictions[user]

@view
@external
def get_prediction_details(prediction_id: uint256) -> (address, String[500], String[200], uint256, uint256, bool, bool, String[500], bool):
    """
    Get detailed information about a specific prediction
    Returns tuple of prediction details
    """
    prediction: Prediction = self.predictions[prediction_id]
    assert prediction.predictor != empty(address), "Prediction does not exist"
    return (
        prediction.predictor,
        prediction.prediction_text,
        prediction.keywords,
        prediction.expiry_date,
        prediction.stake_amount,
        prediction.is_verified,
        prediction.is_correct,
        prediction.evidence,
        prediction.is_claimed
    )