// chainEnv/contracts/AggregatorAuction.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AggregatorAuction {
    enum State { CollectingOffers, Closed, ModelsUploading, AggregationComplete }
    
    State public auctionState;
    address public owner;
    address public aggregator;
    uint256 public deadline;
    uint256 public roundNumber;
    address[] public whitelist;
    
    bool private electionCompleted;
    
    // NUOVO: Tracking degli hash IPFS dei modelli
    mapping(address => string) public modelHashes;
    uint256 public modelsUploaded;
    string public globalModelHash;  // Hash del modello aggregato finale
    
    struct FLOffer {
        address node;
        uint256 computePower;
        uint256 bandwidth;
        uint256 reliability;
        uint256 dataSize;
        uint256 cost;
        bool submitted;
    }
    
    mapping(address => FLOffer) public offers;
    uint256 public totalOffers;
    
    // Eventi esistenti
    event OfferSubmitted(address indexed node, uint256 computePower, uint256 bandwidth, uint256 reliability, uint256 dataSize, uint256 cost);
    event AuctionClosed(uint256 roundNumber);
    event AggregatorElected(address indexed aggregator, uint256 roundNumber);
    
    // NUOVI: Eventi per comunicazione P2P
    event ModelUploaded(address indexed node, string ipfsHash, uint256 timestamp);
    event AllModelsReady(uint256 totalModels, uint256 roundNumber);
    event GlobalModelReady(string ipfsHash, uint256 roundNumber, address indexed aggregator);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier onlyWhitelisted() {
        require(isWhitelisted(msg.sender), "Not whitelisted");
        _;
    }
    
    modifier inState(State _state) {
        require(auctionState == _state, "Invalid state");
        _;
    }
    
    // NUOVO: Solo l'aggregatore può chiamare
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator can call");
        _;
    }
    
    constructor(address[] memory _whitelist, uint256 _timeoutSeconds, uint256 _roundNumber) {
        owner = msg.sender;
        whitelist = _whitelist;
        deadline = block.timestamp + _timeoutSeconds;
        roundNumber = _roundNumber;
        auctionState = State.CollectingOffers;
        electionCompleted = false;
        modelsUploaded = 0;
    }
    
    function submitOffer(
        uint256 computePower,
        uint256 bandwidth,
        uint256 reliability,
        uint256 dataSize,
        uint256 cost
    ) external onlyWhitelisted inState(State.CollectingOffers) {
        require(!offers[msg.sender].submitted, "Offer already submitted");
        require(block.timestamp <= deadline, "Deadline passed");
        
        offers[msg.sender] = FLOffer(
            msg.sender,
            computePower,
            bandwidth,
            reliability,
            dataSize,
            cost,
            true
        );
        totalOffers++;
        
        emit OfferSubmitted(msg.sender, computePower, bandwidth, reliability, dataSize, cost);
        
        if (totalOffers == whitelist.length || block.timestamp >= deadline) {
            closeAuction();
        }
    }
    
    function closeAuction() internal {
        require(auctionState == State.CollectingOffers, "Already closed");
        auctionState = State.Closed;
        emit AuctionClosed(roundNumber);
        electAggregator();
    }
    
    function electAggregator() internal {
        require(!electionCompleted, "Election already completed");  
        electionCompleted = true; 
        
        address bestNode;
        uint256 bestScore = 0;
        
        for (uint256 i = 0; i < whitelist.length; i++) {
            address node = whitelist[i];
            FLOffer memory offer = offers[node];
            
            if (offer.submitted) {
                uint256 score = (
                    offer.computePower * 30 +
                    offer.bandwidth * 25 +
                    offer.reliability * 20 +
                    offer.dataSize * 15 +
                    1000 * 10
                ) * 1e18 / (offer.cost + 1);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestNode = node;
                }
            }
        }
        
        aggregator = bestNode;
        
        // NUOVO: Transizione automatica allo stato ModelsUploading
        auctionState = State.ModelsUploading;
        
        emit AggregatorElected(aggregator, roundNumber);
    }
    
    // ==================== NUOVE FUNZIONI P2P ====================
    
    /**
     * @notice I nodi participant uploadano il loro hash IPFS dopo l'elezione
     * @param ipfsHash Hash IPFS del modello trainato
     */
    function submitModelHash(string calldata ipfsHash) 
        external 
        onlyWhitelisted 
        inState(State.ModelsUploading)
    {
        require(bytes(modelHashes[msg.sender]).length == 0, "Model hash already submitted");
        require(bytes(ipfsHash).length > 0, "Empty IPFS hash");
        
        modelHashes[msg.sender] = ipfsHash;
        modelsUploaded++;
        
        emit ModelUploaded(msg.sender, ipfsHash, block.timestamp);
        
        // Notifica quando tutti i participant hanno uploadato
        // (whitelist.length - 1 perché l'aggregatore non uploada prima)
        if (modelsUploaded >= whitelist.length - 1) {
            emit AllModelsReady(modelsUploaded, roundNumber);
        }
    }
    
    /**
     * @notice L'aggregatore registra il modello globale dopo l'aggregazione
     * @param ipfsHash Hash IPFS del modello aggregato
     */
    function submitGlobalModel(string calldata ipfsHash)
        external
        onlyAggregator
        inState(State.ModelsUploading)
    {
        require(bytes(ipfsHash).length > 0, "Empty IPFS hash");
        require(bytes(globalModelHash).length == 0, "Global model already submitted");
        
        globalModelHash = ipfsHash;
        auctionState = State.AggregationComplete;
        
        emit GlobalModelReady(ipfsHash, roundNumber, msg.sender);
    }
    
    /**
     * @notice Recupera tutti gli hash IPFS dei modelli participant
     * @return Array di hash IPFS (uno per nodo nella whitelist)
     */
    function getAllModelHashes() external view returns (string[] memory) {
        string[] memory hashes = new string[](whitelist.length);
        
        for (uint256 i = 0; i < whitelist.length; i++) {
            hashes[i] = modelHashes[whitelist[i]];
        }
        
        return hashes;
    }
    
    /**
     * @notice Recupera l'hash del modello di un nodo specifico
     * @param node Indirizzo del nodo
     * @return Hash IPFS del modello del nodo
     */
    function getModelHash(address node) external view returns (string memory) {
        return modelHashes[node];
    }
    
    /**
     * @notice Recupera l'hash del modello globale aggregato
     * @return Hash IPFS del modello globale
     */
    function getGlobalModelHash() external view returns (string memory) {
        require(bytes(globalModelHash).length > 0, "Global model not yet available");
        return globalModelHash;
    }
    
    /**
     * @notice Verifica se tutti i modelli participant sono stati uploadati
     * @return true se tutti hanno uploadato, false altrimenti
     */
    function areAllModelsUploaded() external view returns (bool) {
        // -1 perché l'aggregatore non uploada prima dell'aggregazione
        return modelsUploaded >= whitelist.length - 1;
    }
    
    /**
     * @notice Conta quanti modelli sono stati uploadati finora
     * @return Numero di modelli uploadati
     */
    function getUploadedModelsCount() external view returns (uint256) {
        return modelsUploaded;
    }
    
    // ==================== FUNZIONI ESISTENTI ====================
    
    function isWhitelisted(address node) public view returns (bool) {
        for (uint256 i = 0; i < whitelist.length; i++) {
            if (whitelist[i] == node) return true;
        }
        return false;
    }
    
    function forceCloseAuction() external onlyOwner {
        require(block.timestamp > deadline, "Deadline not reached");
        require(auctionState == State.CollectingOffers, "Already closed");
        closeAuction();
    }
    
    function getWhitelist() external view returns (address[] memory) {
        return whitelist;
    }
    
    function getAuctionInfo() external view returns (
        State state,
        address electedAggregator,
        uint256 round,
        uint256 totalSubmittedOffers,
        uint256 uploadedModels,
        uint256 whitelistSize,
        bool hasGlobalModel
    ) {
        return (
            auctionState,
            aggregator,
            roundNumber,
            totalOffers,
            modelsUploaded,
            whitelist.length,
            bytes(globalModelHash).length > 0
        );
    }
}