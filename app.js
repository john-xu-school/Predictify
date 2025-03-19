// App.js - Frontend logic for NewsOracle DApp

// Contract ABI - This would be generated from your compiled Vyper contract
let contractABI = []; // Include actual ABI here after compilation
const contractAddress = "0x83d98bb0a795AD2e765c2323c944011826d63Dc5"; // Deploy the contract and include its address here

// Global variables
let web3;
let contract;
let userAccount;
let predictions = [];

async function loadContractABI() {
    try {
        const response = await fetch('PredictifyContract.json');
        contractABI = await response.json();
        console.log(contractABI);
    }
    catch (error){
        console.log (error)
    }
}


// Initialize the application
async function init() {
    // Check if MetaMask is installed
    if (window.ethereum) {
        await loadContractABI();
        // Initialize Web3 instance
        web3 = new Web3(window.ethereum);
        
        // Check network
        const networkId = await web3.eth.net.getId();
        const networkName = getNetworkName(networkId);
        document.getElementById('network').textContent = networkName;
        
        // Initialize contract with signer
        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        contract = new ethers.Contract(contractAddress, contractABI, signer);

        
        // Setup event listeners
        setupEventListeners();
        
        // Check if user is already connected
        const accounts = await web3.eth.getAccounts();
        if (accounts.length > 0) {
            userAccount = accounts[0];
            onConnected();
        } else {
            showConnectPrompt();
        }
    } else {
        showError("MetaMask is not installed. Please install MetaMask to use this application.");
    }
}

// Setup event listeners
function setupEventListeners() {
    // Connect wallet buttons
    document.getElementById('connect-wallet').addEventListener('click', connectWallet);
    document.getElementById('connect-wallet-prompt').addEventListener('click', connectWallet);
    
    // Prediction form
    document.getElementById('prediction-form').addEventListener('submit', createPrediction);
    
    // Set minimum date for prediction expiry
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const minDate = tomorrow.toISOString().split('T')[0];
    document.getElementById('prediction-expiry').setAttribute('min', minDate);
}

// Connect wallet function
async function connectWallet() {
    try {
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
        userAccount = accounts[0];
        onConnected();
        
        // Setup account change listener
        window.ethereum.on('accountsChanged', (accounts) => {
            if (accounts.length === 0) {
                // User disconnected their wallet
                showConnectPrompt();
            } else {
                userAccount = accounts[0];
                onConnected();
            }
        });
        
        // Setup network change listener
        window.ethereum.on('chainChanged', () => {
            window.location.reload();
        });
    } catch (error) {
        console.error("Error connecting wallet:", error);
        showError("Failed to connect wallet. Please try again.");
    }
}

// Actions to take when wallet is connected
async function onConnected() {
    // Hide connect prompt, show dashboard
    document.getElementById('connect-prompt').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
    
    // Update wallet info
    document.getElementById('connect-wallet').classList.add('hidden');
    document.getElementById('wallet-info').classList.remove('hidden');
    document.getElementById('wallet-address').textContent = `${userAccount.substring(0, 6)}...${userAccount.substring(38)}`;
    
    // Load user data
    await loadUserData();
    await loadUserPredictions();
}

// Load user data (balance, active predictions, success rate)
async function loadUserData() {
    try {
        // Get ETH balance using ethers.js
        const balance = await contract.runner.provider.getBalance(userAccount);
        const ethBalance = ethers.formatEther(balance);
        document.getElementById('user-balance').textContent = `${parseFloat(ethBalance).toFixed(4)} ETH`;
        
        // Get user's predictions
        let predictionIds;
        try {
            // Call the contract function directly
            const result = await contract.get_user_predictions(userAccount);
            predictionIds = Array.isArray(result) ? result : [];
            // Convert BigInts to Numbers for easier handling
            predictionIds = predictionIds.map(id => Number(id));
        } catch (error) {
            console.log("No predictions found for user:", error);
            console.log(predictionIds); 
            predictionIds = [];
        }
        
        // Calculate active predictions and success rate
        let activePredictions = 0;
        let successfulPredictions = 0;
        let verifiedPredictions = 0;
        
        for (const id of predictionIds) {
            try {
                const prediction = await contract.get_prediction_details(id);
                if (!prediction[5]) { // not verified yet
                    activePredictions++;
                } else {
                    verifiedPredictions++;
                    if (prediction[6]) { // is correct
                        successfulPredictions++;
                    }
                }
            } catch (error) {
                console.error(`Error loading prediction ${id}:`, error);
                continue;
            }
        }
        
        document.getElementById('active-predictions').textContent = activePredictions;
        
        const successRate = verifiedPredictions > 0 
            ? Math.round((successfulPredictions / verifiedPredictions) * 100) 
            : 0;
        document.getElementById('success-rate').textContent = `${successRate}%`;
        
    } catch (error) {
        console.error("Error loading user data:", error);
        showError(`Failed to load user data: ${error.message}`);
    }
}

// Load user predictions
async function loadUserPredictions() {
    try {
        // Clear predictions list
        const predictionsList = document.getElementById('predictions-list');
        predictionsList.innerHTML = '';

        let predictionIds = [];
        try {
            // Call the contract function and convert result to array if needed
            const result = await contract.get_user_predictions(userAccount);
            console.log("User predictions:", result);
            predictionIds = Array.isArray(result) ? result : [];
            // Convert BigInts to Numbers for easier handling
            predictionIds = predictionIds.map(id => Number(id));
        } catch (error) {
            console.log("No predictions found for user:", error);
            predictionIds = [];
        }
        
        if (predictionIds.length === 0) {
            predictionsList.innerHTML = '<div class="col-span-1 md:col-span-2 p-6 bg-gray-50 rounded-lg text-center text-gray-500">No predictions yet. Create your first prediction above!</div>';
            return;
        }
        
        // Sort prediction IDs (newest first)
        predictionIds.sort((a, b) => b - a);
        
        // Clear existing predictions array
        predictions = [];
        
        // Load each prediction
        for (const id of predictionIds) {
            try {
                const p = await contract.get_prediction_details(id);
                
                const prediction = {
                    id: id,
                    predictor: p[0],
                    text: p[1],
                    keywords: p[2],
                    expiryDate: new Date(Number(p[3]) * 1000),
                    stakeAmount: ethers.formatEther(p[4]),
                    isVerified: p[5],
                    isCorrect: p[6],
                    evidence: p[7],
                    isClaimed: p[8]
                };
                
                predictions.push(prediction);
                
                // Create prediction card
                const card = createPredictionCard(prediction);
                predictionsList.appendChild(card);
            } catch (error) {
                console.error(`Error loading prediction ${id}:`, error);
                continue;
            }
        }
    } catch (error) {
        console.error("Error loading predictions:", error);
        showError(`Failed to load predictions: ${error.message}`);
    }
}

// Create a prediction card element
function createPredictionCard(prediction) {
    const now = new Date();
    const isExpired = prediction.expiryDate < now;
    
    // Create card element
    const card = document.createElement('div');
    card.className = 'prediction-card bg-white rounded-lg shadow overflow-hidden';
    
    // Determine status badge
    let statusBadge;
    if (!prediction.isVerified && !isExpired) {
        statusBadge = '<span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">Active</span>';
    } else if (!prediction.isVerified && isExpired) {
        statusBadge = '<span class="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-medium">Pending Verification</span>';
    } else if (prediction.isVerified && prediction.isCorrect) {
        statusBadge = '<span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">Correct</span>';
    } else {
        statusBadge = '<span class="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium">Incorrect</span>';
    }
    
    // Format expiry date
    const expiryFormatted = prediction.expiryDate.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
    
    // Create card content
    card.innerHTML = `
        <div class="p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-semibold">${prediction.text.substring(0, 50)}${prediction.text.length > 50 ? '...' : ''}</h3>
                ${statusBadge}
            </div>
            <div class="mb-4">
                <p class="text-gray-600"><strong>Keywords:</strong> ${prediction.keywords}</p>
                <p class="text-gray-600"><strong>Expires:</strong> ${expiryFormatted}</p>
                <p class="text-gray-600"><strong>Stake:</strong> ${prediction.stakeAmount} ETH</p>
            </div>
            ${prediction.isVerified ? `
                <div class="mt-4 pt-4 border-t border-gray-200">
                    <p class="text-gray-600"><strong>Verification:</strong> ${prediction.isCorrect ? 'Correct' : 'Incorrect'}</p>
                    <p class="text-gray-600"><strong>Evidence:</strong> ${prediction.evidence}</p>
                </div>
            ` : ''}
            ${prediction.isVerified && prediction.isCorrect && !prediction.isClaimed ? `
                <button class="claim-reward mt-4 px-4 py-2 gradient-bg text-white rounded-lg font-medium w-full" data-id="${prediction.id}">Claim Reward</button>
            ` : ''}
            ${prediction.isVerified && prediction.isCorrect && prediction.isClaimed ? `
                <div class="mt-4 px-4 py-2 bg-green-100 text-green-800 rounded-lg text-center">Reward Claimed</div>
            ` : ''}
        </div>
    `;
    
    // Add event listener for claim button if present
    const claimButton = card.querySelector('.claim-reward');
    if (claimButton) {
        claimButton.addEventListener('click', () => claimReward(prediction.id));
    }
    
    return card;
}

// Create a new prediction
async function createPrediction(event) {
    event.preventDefault();
    
    try {
        // Get form values
        const title = document.getElementById('prediction-title').value.trim();
        const details = document.getElementById('prediction-details').value.trim();
        const keywords = document.getElementById('prediction-keywords').value.trim();
        const expiryDateStr = document.getElementById('prediction-expiry').value;
        const stakeAmount = document.getElementById('prediction-stake').value;
        
        // Validate inputs
        if (!title || !details || !keywords || !expiryDateStr || !stakeAmount) {
            showError("Please fill in all fields.");
            return;
        }
        
        // Format prediction text
        const predictionText = `${title}: ${details}`;
        
        // Convert expiry date to timestamp
        const expiryDate = new Date(expiryDateStr);
        expiryDate.setHours(23, 59, 59, 999); // End of day
        const expiryTimestamp = Math.floor(expiryDate.getTime() / 1000);
        
        // Convert stake amount to wei
        const stakeAmountWei = ethers.parseEther(stakeAmount.toString());
        
        console.log("Creating prediction with:", {
            text: predictionText,
            keywords: keywords,
            expiryTimestamp: expiryTimestamp,
            stakeAmount: stakeAmountWei.toString()
        });
        
        // Show loading state
        showLoading("Creating prediction...");
        
        // Get gas estimate
        const gasEstimate = await contract.create_prediction.estimateGas(
            predictionText,
            keywords,
            expiryTimestamp,
            { value: stakeAmountWei }
        );
        
        // Get current gas price
        const feeData = await contract.runner.provider.getFeeData();
        const gasPrice = feeData.gasPrice;
        const estimatedCost = (gasEstimate * gasPrice);  // This is a BigInt multiplication
        
        console.log(`Estimated gas: ${gasEstimate.toString()}`);
        console.log(`Current gas price: ${ethers.formatUnits(gasPrice, 'gwei')} gwei`);
        console.log(`Estimated transaction cost: ${ethers.formatEther(estimatedCost)} ETH`);
        
        // Send transaction
        const tx = await contract.create_prediction(
            predictionText,
            keywords,
            expiryTimestamp,
            {
                value: stakeAmountWei,
                gasLimit: BigInt(Math.floor(Number(gasEstimate) * 1.2)) // Convert to BigInt after calculation
            }
        );
        
        console.log("Transaction hash:", tx.hash);
        
        // Wait for transaction receipt
        const receipt = await tx.wait();
        console.log("Transaction receipt:", receipt);
        
        hideLoading();
        showSuccess(`Prediction created successfully! Gas used: ${receipt.gasUsed}`);
        
        // Reset form
        document.getElementById('prediction-form').reset();
        
        // Reload user data and predictions
        loadUserData();
        loadUserPredictions();
        
    } catch (error) {
        console.error("Error creating prediction:", error);
        hideLoading();
        let errorMessage = "Failed to create prediction.\n";
        errorMessage += `Error: ${error.message}\n`;
        if (error.data) {
            errorMessage += `Error data: ${error.data}\n`;
        }
        if (error.transaction) {
            errorMessage += `Transaction: ${error.transaction}\n`;
        }
        showError(errorMessage);
    }
}

// Claim reward for a correct prediction
async function claimReward(predictionId) {
    try {
        showLoading("Claiming reward...");
        
        // Get gas estimate
        const gasEstimate = await contract.claim_reward.estimateGas(predictionId);
        
        // Get current gas price
        const feeData = await contract.runner.provider.getFeeData();
        const gasPrice = feeData.gasPrice;
        const estimatedCost = (gasEstimate * gasPrice);
        
        console.log(`Estimated gas: ${gasEstimate.toString()}`);
        console.log(`Current gas price: ${ethers.formatUnits(gasPrice, 'gwei')} gwei`);
        console.log(`Estimated transaction cost: ${ethers.formatEther(estimatedCost)} ETH`);
        
        // Send transaction
        const tx = await contract.claim_reward(
            predictionId,
            {
                gasLimit: BigInt(Math.floor(Number(gasEstimate) * 1.2)) // Add 20% buffer
            }
        );
        
        console.log("Transaction hash:", tx.hash);
        
        // Wait for transaction receipt
        const receipt = await tx.wait();
        console.log("Transaction receipt:", receipt);
        
        hideLoading();
        showSuccess(`Reward claimed successfully! Gas used: ${receipt.gasUsed}`);
        
        // Reload user data and predictions
        loadUserData();
        loadUserPredictions();
        
    } catch (error) {
        console.error("Error claiming reward:", error);
        hideLoading();
        let errorMessage = "Failed to claim reward.\n";
        errorMessage += `Error: ${error.message}\n`;
        if (error.data) {
            errorMessage += `Error data: ${error.data}\n`;
        }
        if (error.transaction) {
            errorMessage += `Transaction: ${error.transaction}\n`;
        }
        showError(errorMessage);
    }
}

// Utility functions
function getNetworkName(networkId) {
    const networks = {
        1: "Ethereum Mainnet",
        3: "Ropsten Testnet",
        4: "Rinkeby Testnet",
        5: "Goerli Testnet",
        42: "Kovan Testnet",
        56: "BSC Mainnet",
        97: "BSC Testnet",
        137: "Polygon Mainnet",
        80001: "Mumbai Testnet"
    };
    return networks[networkId] || `Network ID: ${networkId}`;
}

function showConnectPrompt() {
    document.getElementById('connect-prompt').classList.remove('hidden');
    document.getElementById('dashboard').classList.add('hidden');
    document.getElementById('wallet-info').classList.add('hidden');
    document.getElementById('connect-wallet').classList.remove('hidden');
}

function showError(message) {
    // Implement error notification
    alert(message); // Replace with a better UI notification system
}

function showSuccess(message) {
    // Implement success notification
    alert(message); // Replace with a better UI notification system
}

function showLoading(message) {
    // Implement loading overlay
    console.log("Loading:", message);
}

function hideLoading() {
    // Hide loading overlay
    console.log("Loading complete");
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);