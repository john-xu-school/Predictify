# oracle_backend.py - Backend service for verifying predictions against news

import os
import time
import json
import logging
import schedule
import requests
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('punkt_tab')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("oracle.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NewsOracle")

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize Web3
def initialize_web3():
    # Get provider URL from environment
    provider_url = os.getenv("WEB3_PROVIDER_URL")
    if not provider_url:
        raise ValueError("WEB3_PROVIDER_URL environment variable not set")
    
    logger.info(f"Connecting to provider URL: {provider_url}")
    
    # Initialize Web3 with provider
    w3 = Web3(Web3.HTTPProvider(provider_url))
    
    # Check connection
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Web3 provider at {provider_url}")
    
    # Get network details
    chain_id = w3.eth.chain_id
    try:
        block = w3.eth.get_block('latest')
        logger.info(f"Connected to network:")
        logger.info(f"  Chain ID: {chain_id}")
        logger.info(f"  Latest block: {block.number}")
        logger.info(f"  Network URL: {provider_url}")
    except Exception as e:
        logger.error(f"Error getting network details: {e}")
    
    return w3

# Load contract
def load_contract(w3):
    try:
        # Get contract address and ABI from environment
        contract_address = os.getenv("CONTRACT_ADDRESS")
        abi_file = os.getenv("CONTRACT_ABI_FILE", "n/PredictifyContract.json")
        
        if not contract_address:
            raise ValueError("CONTRACT_ADDRESS environment variable not set")
            
        # Ensure contract address is checksummed
        contract_address = Web3.to_checksum_address(contract_address)
        
        # Load ABI from file
        try:
            with open(abi_file, 'r') as f:
                contract_abi = json.load(f)
                logger.info(f"Loaded ABI from {abi_file}")
                logger.info(f"ABI contains {len(contract_abi)} entries")
        except Exception as e:
            raise ValueError(f"Failed to load contract ABI from {abi_file}: {e}")
        
        # Verify contract exists at address
        code = w3.eth.get_code(contract_address)
        if code == b'' or code == '0x' or code == b'0x':
            raise ValueError(f"No contract code found at address {contract_address}")
            
        # Create contract instance
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Try to call a view function to verify contract is accessible
        try:
            prediction_count = contract.functions.prediction_count().call()
            logger.info(f"Successfully connected to contract. Current prediction count: {prediction_count}")
        except Exception as e:
            logger.error(f"Failed to call contract function: {e}")
            raise ValueError(f"Contract initialization failed: {e}")
        
        logger.info(f"Contract loaded successfully at address: {contract_address}")
        return contract
        
    except Exception as e:
        logger.error(f"Failed to load contract: {e}")
        logger.error(f"Contract address: {contract_address}")
        logger.error(f"ABI file: {abi_file}")
        raise

# Class for news verification
class NewsVerifier:
    def __init__(self, w3, contract):
        self.w3 = w3
        self.contract = contract
        self.oracle_address = os.getenv("ORACLE_ADDRESS")
        self.oracle_private_key = os.getenv("ORACLE_PRIVATE_KEY")
        
        if not self.oracle_address or not self.oracle_private_key:
            raise ValueError("ORACLE_ADDRESS and ORACLE_PRIVATE_KEY must be set")
        
        # News API key
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def check_pending_predictions(self):
        """Check for predictions that need verification"""
        logger.info("Checking for pending predictions...")
        
        try:
            # Get total prediction count
            
            prediction_count = self.contract.functions.prediction_count().call()
            logger.info(f"Total predictions: {prediction_count}")
            
            # Check all predictions
            for i in range(prediction_count):
                # Get prediction details
                prediction = self.contract.functions.get_prediction_details(i+1).call()
                logger.info(f"Prediction details: {prediction}")
                
                # Extract prediction data
                predictor = prediction[0]
                prediction_text = prediction[1]
                logger.info(f"Prediction text: {prediction_text}")
                keywords = prediction[2]
                expiry_timestamp = prediction[3]
                is_verified = prediction[5]
                
                # Convert timestamp to datetime
                expiry_date = datetime.fromtimestamp(expiry_timestamp)
                now = datetime.now()
                
                logger.info(is_verified)
                logger.info(now)
                logger.info(expiry_date)
                
                # Check if prediction is expired but not verified
                if not is_verified and now < expiry_date:
                    logger.info(f"Processing prediction #{i+1}: {prediction_text[:50]}...")
                    self.verify_prediction(i+1, prediction_text, keywords)
        
        except Exception as e:
            logger.error(f"Error checking predictions: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def submit_verification_with_retry(self, prediction_id, is_correct, evidence, max_retries=3):
        """Submit verification with retries"""
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries}")
                return self.submit_verification(prediction_id, is_correct, evidence)
            except Exception as e:
                last_error = e
                if "is not in the chain after" in str(e):
                    logger.warning(f"Transaction timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(30)  # Wait 30 seconds before retry
                else:
                    # For other errors, don't retry
                    raise
        
        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed")
        raise last_error

    def verify_prediction(self, prediction_id, prediction_text, keywords):
        """Verify a prediction against news sources and handle rewards"""
        try:
            logger.info(f"Starting verification for prediction #{prediction_id}")
            logger.info(f"Prediction text: {prediction_text}")
            logger.info(f"Keywords: {keywords}")
            
            # Get prediction details first to check current state
            prediction = self.contract.functions.get_prediction_details(prediction_id).call()
            predictor_address = prediction[0]
            is_verified = prediction[5]
            
            if is_verified:
                logger.info(f"Prediction #{prediction_id} is already verified")
                return
            
            # Parse keywords and clean them
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            if not keyword_list:
                logger.warning(f"No valid keywords found for prediction #{prediction_id}")
                return
                
            logger.info(f"Processed keywords: {keyword_list}")
            
            # Get news articles related to keywords
            articles = self.fetch_news(keyword_list)
            
            if not articles:
                logger.warning(f"No articles found for prediction #{prediction_id}")
                return
            
            logger.info(f"Found {len(articles)} articles to analyze")
            
            # Process prediction text
            processed_prediction = self.process_text(prediction_text)
            if not processed_prediction:
                logger.warning(f"No valid text to process in prediction #{prediction_id}")
                return
                
            # Check if prediction matches any news
            is_correct, evidence = self.match_prediction_to_news(processed_prediction, articles)
            
            logger.info(f"Prediction #{prediction_id} verified: {'Correct' if is_correct else 'Incorrect'}")
            logger.info(f"Evidence: {evidence[:100]}...")
            
            # Submit verification to blockchain with retry
            #self.submit_verification_with_retry(prediction_id, is_correct, evidence)
            # If prediction is correct, help user claim reward
            if is_correct:
                logger.info(f"Prediction #{prediction_id} is correct. Attempting to claim reward for user...")
                try:
                    self.claim_reward_for_user(prediction_id, predictor_address)
                except Exception as e:
                    logger.error(f"Failed to claim reward for prediction #{prediction_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error verifying prediction #{prediction_id}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def claim_reward_for_user(self, prediction_id, predictor_address):
        """Helper method to claim reward for a correct prediction"""
        try:
            # Get prediction details to verify it's claimable
            prediction = self.contract.functions.get_prediction_details(prediction_id).call()
            #is_verified = prediction[5]
            #is_correct = prediction[6]
            is_claimed = prediction[8]
                
            if is_claimed:
                logger.warning(f"Reward for prediction #{prediction_id} has already been claimed")
                return
            
            logger.info(f"Claiming reward for prediction #{prediction_id}")
            
            
            # nonce = self.w3.eth.get_transaction_count(predictor_address)
            
            # # Get gas price
            # base_fee = self.w3.eth.get_block('latest').baseFeePerGas
            # priority_fee = self.w3.eth.max_priority_fee
            # max_fee_per_gas = base_fee * 2 + priority_fee
            
            # # Build transaction
            # tx = self.contract.functions.claim_reward(prediction_id).build_transaction({
            #     'from': self.oracle_address,
            #     'nonce': nonce,
            #     'gas': 300000,
            #     'maxFeePerGas': max_fee_per_gas,
            #     'maxPriorityFeePerGas': priority_fee,
            #     'chainId': self.w3.eth.chain_id,
            #     'type': 2
            # })
            
            # logger.info(f"Claim reward transaction built: {tx}")
            
            # # Sign and send transaction
            # signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.oracle_private_key)
            # tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # logger.info(f"Claim reward transaction sent: {tx_hash.hex()}")
            
            # # Wait for receipt
            # receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            # if receipt['status'] == 1:
            #     logger.info(f"Successfully claimed reward for prediction #{prediction_id}")
            #     logger.info(f"Transaction hash: {tx_hash.hex()}")
            # else:
            #     logger.error(f"Failed to claim reward: {receipt}")
            
            # Get nonce
            amount_in_eth = 0.001 + (prediction[4])
            logger.info(f"Amount in ETH: {prediction[4]}")
            amount_in_wei = self.w3.to_wei(amount_in_eth, 'ether')

            # Get the nonce for the sender address
            nonce = self.w3.eth.get_transaction_count(self.oracle_address)

            # Prepare transaction
            tx = {
                'nonce': nonce,
                'to': prediction[0],
                'value': amount_in_wei,
                'gas': 21000,  # Standard gas limit for ETH transfers
                'gasPrice': self.w3.eth.gas_price,
                'chainId': 11155111,
            }
            
            logger.info(f"Transaction built: {tx}")

            # Sign the transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.oracle_private_key)

            # Send the transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"Transaction sent! Hash: {self.w3.to_hex(tx_hash)}")
                
        except Exception as e:
            logger.error(f"Error claiming reward: {e}")
            raise

    def fetch_news(self, keywords):
        """Fetch news articles related to keywords"""
        articles = []
        
        # Use News API if key is available
        if self.news_api_key:
            articles.extend(self.fetch_from_news_api(keywords))
        
        # Web scraping from major news sources
        articles.extend(self.scrape_news_websites(keywords))
        
        return articles
    
    def fetch_from_news_api(self, keywords):
        """Fetch news from News API"""
        try:
            if not keywords:
                logger.warning("No keywords provided for News API search")
                return []

            # Prepare query - use AND instead of OR to make the search more specific
            query = " AND ".join(f'"{keyword.strip()}"' for keyword in keywords if keyword.strip())
            
            if not query:
                logger.warning("No valid keywords after processing")
                return []
                
            logger.info(f"Searching News API with query: {query}")
            
            # Set date range (last 24 hours)
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            # Make API request
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": self.news_api_key
            }
            
            logger.info(f"Making request to News API with parameters: {params}")
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", {}).get("name", "News API")
                    })
                
                logger.info(f"Found {len(articles)} articles from News API")
                return articles
            else:
                logger.error(f"News API error: {response.status_code} - {response.text}")
                if response.status_code == 401:
                    logger.error("Invalid API key or unauthorized access")
                elif response.status_code == 429:
                    logger.error("Rate limit exceeded")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def scrape_news_websites(self, keywords):
        """Scrape news from major websites"""
        articles = []
        
        # Define news sources to scrape
        news_sources = [
            {"name": "Reuters", "url": "https://www.reuters.com/", "selector": "article"},
            {"name": "BBC", "url": "https://www.bbc.com/news", "selector": "article"},
            {"name": "CNN", "url": "https://www.cnn.com/", "selector": "article.card"}
        ]
        
        for source in news_sources:
            try:
                # Get website content
                response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code != 200:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find articles
                article_elements = soup.select(source["selector"])
                
                for element in article_elements[:10]:  # Limit to first 10 articles
                    # Extract text content
                    text = element.get_text().strip()
                    
                    # Check if any keyword is in the text
                    if any(keyword.lower() in text.lower() for keyword in keywords):
                        # Find URL if available
                        url_element = element.find('a')
                        url = url_element.get('href') if url_element else ""
                        
                        # Normalize URL (handle relative URLs)
                        if url and not url.startswith('http'):
                            url = source["url"] + url if not url.startswith('/') else source["url"] + url
                        
                        articles.append({
                            "title": element.find('h2').get_text().strip() if element.find('h2') else "",
                            "content": text,
                            "url": url,
                            "source": source["name"]
                        })
            
            except Exception as e:
                logger.error(f"Error scraping {source['name']}: {e}")
        
        logger.info(f"Found {len(articles)} articles from web scraping")
        return articles
    
    def process_text(self, text):
        """Process text for NLP comparison"""
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words]
        
        # Lemmatize
        lemmatized = [self.lemmatizer.lemmatize(w) for w in filtered_tokens]
        
        return lemmatized
    
    def match_prediction_to_news(self, processed_prediction, articles):
        """Match processed prediction to news articles"""
        best_match_score = 0
        best_match_article = None
        
        for article in articles:
            # Combine title and content
            full_text = f"{article['title']} {article['content']}"
            
            # Process article text
            processed_article = self.process_text(full_text)
            
            # Calculate match score (percentage of prediction words in article)
            matches = sum(1 for word in processed_prediction if word in processed_article)
            score = matches / len(processed_prediction) if processed_prediction else 0
            
            if score > best_match_score:
                best_match_score = score
                best_match_article = article
        
        # Determine if prediction is correct (threshold: 60% match)
        is_correct = best_match_score >= 0.6
        
        # Prepare evidence
        if best_match_article:
            evidence = f"Source: {best_match_article['source']}\nTitle: {best_match_article['title']}\n"
            if best_match_article['url']:
                evidence += f"URL: {best_match_article['url']}\n"
            evidence += f"Match Score: {best_match_score:.2%}"
        else:
            evidence = "No matching news articles found."
        
        return is_correct, evidence
    
    def submit_verification(self, prediction_id, is_correct, evidence):
        """Submit verification result to blockchain"""
        try:
            logger.info(f"Submitting verification for prediction #{prediction_id}")
            logger.info(f"Is correct: {is_correct}")
            logger.info(f"Evidence: {evidence[:100]}...")

            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.oracle_address)
            
            # Get gas price with 20% increase to ensure faster mining
            base_fee = self.w3.eth.get_block('latest').baseFeePerGas
            priority_fee = self.w3.eth.max_priority_fee
            max_fee_per_gas = base_fee * 2 + priority_fee  # Double the base fee plus priority fee
            
            logger.info(f"Base fee: {base_fee}")
            logger.info(f"Priority fee: {priority_fee}")
            logger.info(f"Max fee per gas: {max_fee_per_gas}")
            
            # Build transaction
            tx = self.contract.functions.verify_prediction(
                prediction_id,
                is_correct,  # Use the actual verification result
                evidence
            ).build_transaction({
                'from': self.oracle_address,
                'nonce': nonce,
                'gas': 300000,  # Gas limit
                'maxFeePerGas': max_fee_per_gas,
                'maxPriorityFeePerGas': priority_fee,
                'chainId': self.w3.eth.chain_id,
                'type': 2  # EIP-1559 transaction
            })
            
            logger.info(f"Transaction built: {tx}")
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                tx,
                private_key=self.oracle_private_key
            )
            
            logger.info("Transaction signed successfully")
            
            # Send raw transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            
            # Wait for transaction receipt with increased timeout
            logger.info("Waiting for transaction confirmation...")
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=300,  # 5 minutes timeout
                poll_latency=5  # Check every 5 seconds
            )
            
            if receipt['status'] == 1:
                logger.info(f"Verification submitted successfully: TX hash {tx_hash.hex()}")
                logger.info(f"Gas used: {receipt['gasUsed']}")
                logger.info(f"Block number: {receipt['blockNumber']}")
                logger.info(f"Transaction index: {receipt['transactionIndex']}")
                return True
            else:
                logger.error(f"Transaction failed: {receipt}")
                return False
            
        except Exception as e:
            logger.error(f"Error submitting verification: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # If it's a timeout error, provide more context
            if "is not in the chain after" in str(e):
                logger.error("Transaction took too long to mine. Possible reasons:")
                logger.error("1. Network congestion")
                logger.error("2. Gas price too low")
                logger.error("3. Sepolia network issues")
                logger.error("You can check the transaction status at:")
                logger.error(f"https://sepolia.etherscan.io/tx/{tx_hash.hex()}")
            
            raise

# Main function
def main():
    try:
        # Initialize Web3 and contract
        w3 = initialize_web3()
        contract = load_contract(w3)
        
        try:
            code = w3.eth.get_code(contract.address)
            if code == b'' or code == '0x':
                logger.error("No contract code found at the specified address!")
                logger.error(contract.address)
            else:
                logger.info(f"Contract code found at {contract.address}")
        except Exception as e:
            logger.error(f"Failed to check contract code: {e}")

        
        # Create verifier
        verifier = NewsVerifier(w3, contract)
        
        # Schedule verification job
        schedule.every().hour.do(verifier.check_pending_predictions)
        
        # Run once immediately
        verifier.check_pending_predictions()
        
        # Keep running
        logger.info("Oracle service started. Running verification every hour.")
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
    except Exception as e:
        logger.error(f"Service error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())