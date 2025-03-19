# oracle_backend.py - Backend service for verifying predictions against news

import os
import time
import json
import logging
import schedule
import requests
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import geth_poa_middleware
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    
    # Initialize Web3 with provider
    w3 = Web3(Web3.HTTPProvider(provider_url))
    
    # Add middleware for compatibility with PoA chains like BSC
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    # Check connection
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Web3 provider at {provider_url}")
    
    logger.info(f"Connected to network: {w3.eth.chain_id}")
    return w3

# Load contract
# Load contract
def load_contract(w3):
    # Get contract address and ABI from environment
    contract_address = os.getenv("CONTRACT_ADDRESS")
    abi_file = os.getenv("CONTRACT_ABI_FILE", "contract_abi.json")
    
    if not contract_address:
        raise ValueError("CONTRACT_ADDRESS environment variable not set")
    
    # Load ABI from file
    try:
        with open(abi_file, 'r') as f:
            contract_abi = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load contract ABI from {abi_file}: {e}")
    
    # Create contract instance
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    logger.info(f"Contract loaded at address: {contract_address}")
    return contract

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
                prediction = self.contract.functions.get_prediction_details(i).call()
                
                # Extract prediction data
                predictor = prediction[0]
                prediction_text = prediction[1]
                keywords = prediction[2]
                expiry_timestamp = prediction[3]
                is_verified = prediction[5]
                
                # Convert timestamp to datetime
                expiry_date = datetime.fromtimestamp(expiry_timestamp)
                now = datetime.now()
                
                # Check if prediction is expired but not verified
                if not is_verified and now > expiry_date:
                    logger.info(f"Processing prediction #{i}: {prediction_text[:50]}...")
                    self.verify_prediction(i, prediction_text, keywords)
        
        except Exception as e:
            logger.error(f"Error checking predictions: {e}")
    
    def verify_prediction(self, prediction_id, prediction_text, keywords):
        """Verify a prediction against news sources"""
        try:
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(',')]
            
            # Get news articles related to keywords
            articles = self.fetch_news(keyword_list)
            
            if not articles:
                logger.warning(f"No articles found for prediction #{prediction_id}")
                return
            
            # Process prediction text
            processed_prediction = self.process_text(prediction_text)
            
            # Check if prediction matches any news
            is_correct, evidence = self.match_prediction_to_news(processed_prediction, articles)
            
            logger.info(f"Prediction #{prediction_id} verified: {'Correct' if is_correct else 'Incorrect'}")
            logger.info(f"Evidence: {evidence[:100]}...")
            
            # Submit verification to blockchain
            self.submit_verification(prediction_id, is_correct, evidence)
            
        except Exception as e:
            logger.error(f"Error verifying prediction #{prediction_id}: {e}")
    
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
            # Prepare query
            query = " OR ".join(keywords)
            
            # Set date range (last 24 hours)
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            # Make API request
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": self.news_api_key
            }
            
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
                logger.warning(f"News API error: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
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
            # Build transaction
            tx = self.contract.functions.verify_prediction(
                prediction_id,
                is_correct,
                evidence
            ).build_transaction({
                'from': self.oracle_address,
                'nonce': self.w3.eth.get_transaction_count(self.oracle_address),
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.oracle_private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Verification submitted: TX hash {tx_hash.hex()}")
            logger.info(f"Gas used: {receipt['gasUsed']}")
            
        except Exception as e:
            logger.error(f"Error submitting verification: {e}")

# Main function
def main():
    try:
        # Initialize Web3 and contract
        w3 = initialize_web3()
        contract = load_contract(w3)
        
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