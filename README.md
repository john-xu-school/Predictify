
## How To Use
Submit Prediction by filling out all prompts, include all possible keywords to increase your chances at getting the prompt correct! For demonstration, the time set for expiry is set to be valid within 2 minutes, so if you want a quick prediction result set your current date to today. Put your stake in (min 0.01 ETH) and submit your prediction. Wait a while until the transaction goes through and the contract receives your stake (there will be a confirmation prompt). Wait until your prediction displays as "Pending Verification" to run oracleTest.py to verify the news prediction. Refersh your website to claim your rewards if you got the prediction correct. 

## app.js
### Initialization and Setup
- Loads the contract ABI (Application Binary Interface) from a JSON file
- Connects to the Ethereum blockchain using Web3.js and Ethers.js libraries
- Sets up event listeners for user interactions
- Checks if MetaMask is installed and handles wallet connection

### Wallet Connection
- Prompts users to connect their MetaMask wallet
- Handles account changes and network changes
- Updates the UI to show wallet address and user data when connected

### User Dashboard
Once connected, the application shows:
- User's ETH balance
- Number of active predictions
- Success rate of past predictions
- A list of all the user's predictions

#### Creating Predictions
Users can create predictions by:
1. Entering a title, details, and keywords
2. Setting an expiry date
3. Staking ETH (putting up collateral)
4. Submitting the transaction to the blockchain

The system:
- Validates all form inputs
- Calculates gas costs for the transaction
- Sends the transaction with a 20% gas buffer
- Shows loading and success states

#### Viewing Predictions
- Displays all predictions made by the user
- Shows status badges (Active, Pending Verification, Correct, Incorrect)
- Includes details like keywords, expiry date, and stake amount
- For verified predictions, shows the verification result and evidence

#### Claiming Rewards
- Automatically checks for and claims pending rewards when loading user data
- Provides a button to manually claim rewards for correct predictions
- Handles the reward claim transaction and updates the UI accordingly

### Smart Contract Integration
- Uses both Web3.js and Ethers.js for blockchain interaction
- Calls contract methods like `create_prediction`, `get_prediction_details`, and `claim_reward`
- Handles transaction signing and confirmation
- Displays gas estimates and transaction costs


## oracleTest.py
### Setup and Initialization
- Loads environment variables for contract addresses, API keys, and blockchain connection details
- Initializes a Web3 connection to interact with the blockchain
- Loads the smart contract ABI and creates a contract instance
- Sets up NLP models for text analysis (sentence transformers and zero-shot classification)

### PredictionVerifier Class
This is the main class that handles all verification functionality:

#### Key Methods

1. **get_pending_predictions()**: 
   - Fetches all predictions from the blockchain that have expired but haven't been verified yet
   - Returns a list of pending predictions with their details

2. **get_news_articles()**: 
   - Queries a news API using keywords related to the prediction
   - Searches for relevant articles from the past few days

3. **scrape_additional_news()**: 
   - Scrapes additional news from sources like Reuters and BBC
   - Used as a backup if the news API doesn't return enough articles

4. **get_text_embedding()**: 
   - Converts text into numerical vectors using a BERT-based model
   - These embeddings capture the semantic meaning of the text

5. **verify_prediction()**: 
   - Compares prediction text with news articles using similarity measures
   - Uses both cosine similarity and zero-shot classification to determine if:
     - The prediction matches what happened in reality
     - There's enough evidence to make a determination
   - Returns a boolean result and supporting evidence

6. **submit_verification()**: 
   - Sends the verification result back to the blockchain
   - Creates, signs, and submits a transaction calling the `verify_prediction` function on the smart contract

### Verification Process

The verification happens in these steps:

1. The oracle periodically checks for pending predictions (predictions that have expired but haven't been verified)
2. For each pending prediction, it:
   - Searches for relevant news articles based on keywords
   - Analyzes the prediction text and news articles using NLP techniques
   - Determines if the prediction was correct based on similarity scores
   - Submits the result to the blockchain, including evidence
