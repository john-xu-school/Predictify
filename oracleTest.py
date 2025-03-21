import os
import time
import json
import schedule
import requests
import torch
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, pipeline
from bs4 import BeautifulSoup
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# Load environment variables
load_dotenv()

# Environment variables
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
ORACLE_PRIVATE_KEY = os.getenv('OWNER_PRIVATE_KEY')
WEB3_PROVIDER_URL = os.getenv('WEB3_PROVIDER_URL')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
VERIFICATION_THRESHOLD = float(os.getenv('VERIFICATION_THRESHOLD', '0.75'))  # Similarity threshold

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)  # For compatibility with PoA networks like BSC

# Load contract ABI
with open('n/PredictifyContract.json', 'r') as f:
    contract_abi = json.load(f)
    
print(CONTRACT_ADDRESS)
contract_address = Web3.to_checksum_address(CONTRACT_ADDRESS)

# Initialize contract
code = w3.eth.get_code(contract_address)
if code == b'' or code == '0x' or code == b'0x':
    raise ValueError(f"No contract code found at address {contract_address}")
    
# Create contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)
print(contract_address)

# Set up oracle account
oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
oracle_address = oracle_account.address

# Initialize NLP models
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class PredictionVerifier:
    def __init__(self):
        print(f"Initializing PredictionVerifier as {oracle_address}")
        
        # # Verify we are the oracle
        contract_oracle = contract.functions.oracle_address().call()
        if contract_oracle.lower() != oracle_address.lower():
            raise Exception(f"Oracle address mismatch: {contract_oracle} != {oracle_address}")
            
        #print(f"Connected to contract at {CONTRACT_ADDRESS}")
        print(f"Current oracle address: {contract_oracle}")
    
    def get_pending_predictions(self):
        """Get all predictions that have expired but not been verified"""
        print("Getting pending predictions...")
        prediction_count = contract.functions.prediction_count().call()
        pending_predictions = []
        
        print(f"Total prediction count: {prediction_count}")
        
        current_timestamp = int(datetime.now().timestamp()) + 86400  # Add one day (24 * 60 * 60 seconds)
        
        for pred_id in range(1, prediction_count + 1):
            try:
                pred_details = contract.functions.get_prediction_details(pred_id).call()
                # Unpack prediction details
                predictor, text, keywords, expiry_timestamp, stake_amount, is_verified, is_correct, evidence, is_claimed = pred_details
                
                # Check if prediction has expired but not been verified
                print(expiry_timestamp)
                print(int(datetime.now().timestamp()))
                if not is_verified and current_timestamp >= expiry_timestamp:
                    pending_predictions.append({
                        'id': pred_id,
                        'predictor': predictor,
                        'text': text,
                        'keywords': keywords,
                        'expiry_timestamp': expiry_timestamp,
                        'stake_amount': stake_amount
                    })
            except Exception as e:
                print(f"Error fetching prediction {pred_id}: {str(e)}")
        
        print(f"Found {len(pending_predictions)} pending predictions to verify")
        return pending_predictions
    
    def get_news_articles(self, keywords, days_back=3):
        """Fetch news articles based on keywords"""
        # Convert keywords to a list and clean them
        if isinstance(keywords, str):
            keyword_list = [k.strip() for k in keywords.split(',')]
        else:
            keyword_list = keywords
            
        # Combine keywords for API query
        query = ' OR '.join(keyword_list)
        
        # Calculate date range for search
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        print(f"Searching news from {from_date} to {to_date} for keywords: {query}")
        
        # News API request
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'apiKey': NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"News API error: {data.get('message', 'Unknown error')}")
                return []
                
            articles = data.get('articles', [])
            print(f"Found {len(articles)} news articles")
            return articles
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []
    
    def scrape_additional_news(self, keywords):
        """Scrape additional news sources for relevant articles"""
        articles = []
        keyword_str = ' '.join(keywords.split(','))
        
        # Example search URLs (would need to be extended for production)
        search_urls = [
            f"https://www.reuters.com/search/news?blob={keyword_str}",
            f"https://www.bbc.co.uk/search?q={keyword_str}&filter=news"
        ]
        
        for url in search_urls:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # The selectors below would need to be adjusted based on the actual website structure
                    # These are examples and would not work without modification
                    if 'reuters.com' in url:
                        news_elements = soup.select('article.search-result-indiv')
                        for element in news_elements[:5]:  # Limit to top 5 results
                            title_elem = element.select_one('h3.search-result-title')
                            desc_elem = element.select_one('div.search-result-snippet')
                            
                            if title_elem:
                                articles.append({
                                    'title': title_elem.text.strip(),
                                    'description': desc_elem.text.strip() if desc_elem else '',
                                    'url': 'https://www.reuters.com' + title_elem.a['href'] if title_elem.a else '',
                                    'source': {'name': 'Reuters'}
                                })
                    
                    elif 'bbc.co.uk' in url:
                        news_elements = soup.select('div.ssrcss-1mxwxkh-PromoContent')
                        for element in news_elements[:5]:
                            title_elem = element.select_one('h3')
                            desc_elem = element.select_one('p')
                            
                            if title_elem:
                                articles.append({
                                    'title': title_elem.text.strip(),
                                    'description': desc_elem.text.strip() if desc_elem else '',
                                    'url': 'https://www.bbc.co.uk' + title_elem.a['href'] if title_elem.a else '',
                                    'source': {'name': 'BBC News'}
                                })
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                
        print(f"Scraped {len(articles)} additional articles")
        return articles
    
    def get_text_embedding(self, text):
        """Get embedding vector for a text using BERT"""
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling - take average of all tokens
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Multiply by attention masks to avoid including padding in the average
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Final embedding vector
        embedding = sum_embeddings / sum_mask
        return embedding[0].numpy()
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def verify_prediction(self, prediction, articles):
        """Verify if prediction is correct based on news articles"""
        if not articles:
            print(f"No articles found for prediction {prediction['id']}")
            return False, "No relevant news articles found"
        
        # Get embedding for prediction text
        pred_embedding = self.get_text_embedding(prediction['text'])
        
        # Get embeddings for article titles and descriptions
        article_texts = [f"{a['title']}. {a.get('description', '')}" for a in articles]
        article_embeddings = [self.get_text_embedding(text) for text in article_texts]
        
        # Calculate similarities
        similarities = [self.cosine_similarity(pred_embedding, art_emb) for art_emb in article_embeddings]
        
        # Find best matching article
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        best_article = articles[max_sim_idx]
        
        # Generate evidence text
        evidence = f"Source: {best_article['source']['name']}. "
        evidence += f"Title: {best_article['title']}. "
        evidence += f"URL: {best_article['url']}"
        
        # Additional zero-shot classification for verification
        classification_result = zero_shot_classifier(
            prediction['text'],
            [article['title'] for article in articles[:5]],  # Use top 5 articles as candidates
            hypothesis_template="This article confirms that {}."
        )
        
        # Combine semantic similarity and classification confidence
        top_label_score = classification_result['scores'][0]
        combined_score = (max_similarity + top_label_score) / 2
        
        print(f"Prediction {prediction['id']} verification:")
        print(f"- Max similarity: {max_similarity:.4f}")
        print(f"- Classification score: {top_label_score:.4f}")
        print(f"- Combined score: {combined_score:.4f}")
        print(f"- Threshold: {VERIFICATION_THRESHOLD}")
        print(f"- Evidence: {evidence}")
        
        # Determine if prediction is correct
        is_correct = combined_score >= VERIFICATION_THRESHOLD
        
        return is_correct, evidence[:200]  # Truncate evidence to match contract limit
    
    def submit_verification(self, prediction_id, is_correct, evidence):
        """Submit verification result to the smart contract"""
        print(f"Submitting verification for prediction {prediction_id}: correct={is_correct}, evidence={evidence}")
        
        try:
            # Ensure is_correct is a proper boolean
            is_correct_bool = bool(is_correct)
            
            # Build transaction
            txn = contract.functions.verify_prediction(
                prediction_id,
                is_correct_bool,  # Convert to explicit boolean
                evidence
            ).build_transaction({
                'from': oracle_address,
                'nonce': w3.eth.get_transaction_count(oracle_address),
                'gas': 500000,
                'gasPrice': w3.eth.gas_price
            })
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(txn, ORACLE_PRIVATE_KEY)
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            print(f"Transaction sent: {tx_hash.hex()}")
            
            # Wait for transaction receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Transaction confirmed: {receipt.transactionHash.hex()}")
            print(f"Gas used: {receipt.gasUsed}")
            
            return True
        except Exception as e:
            print(f"Error submitting verification: {str(e)}")
            return False
    
    def process_pending_predictions(self):
        """Process all pending predictions"""
        pending_predictions = self.get_pending_predictions()
        
        for prediction in pending_predictions:
            print(f"\nProcessing prediction {prediction['id']}: {prediction['text']}")
            
            # Fetch news articles
            articles = self.get_news_articles(prediction['keywords'])
            
            # Add scraped articles if needed
            if len(articles) < 5:
                scraped_articles = self.scrape_additional_news(prediction['keywords'])
                articles.extend(scraped_articles)
            
            # Verify prediction
            is_correct, evidence = self.verify_prediction(prediction, articles)
            
            # Submit verification to blockchain
            success = self.submit_verification(prediction['id'], is_correct, evidence)
            
            if success:
                print(f"Successfully verified prediction {prediction['id']}")
            else:
                print(f"Failed to verify prediction {prediction['id']}")
            
            # Delay between processing predictions
            time.sleep(2)

def run_verification_job():
    """Run the verification job"""
    print("\n===== Starting verification job =====")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        verifier = PredictionVerifier()
        print("Processing pending predictions...")
        verifier.process_pending_predictions()
        print("===== Verification job completed =====\n")
    except Exception as e:
        print(f"Error in verification job: {str(e)}")

def main():
    print("Starting NewsOracle Prediction Verifier")
    print(f"Connected to network: {w3.eth.chain_id}")
    print(f"Oracle address: {oracle_address}")
    
    # Run immediately on startup
    run_verification_job()
    
    # Schedule to run every 4 hours
    schedule.every(4).hours.do(run_verification_job)
    
    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()