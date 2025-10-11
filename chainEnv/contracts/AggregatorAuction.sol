// chainEnv/contracts/AggregatorAuction.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AggregatorAuction {
    // ==================== ENUMS & STATE ====================
    
    enum State { 
        CollectingOffers,      // Fase 1: Raccolta offerte dai nodi
        Closed,                // Fase 2: Asta chiusa, elezione completata
        ModelsUploading,       // Fase 3: Nodi uploadano modelli su IPFS
        Aggregating,           // Fase 4: Aggregatore sta aggregando
        AggregationComplete    // Fase 5: Modello globale pronto
    }
    
    State public auctionState;
    address public owner;
    address public aggregator;
    uint256 public deadline;
    uint256 public roundNumber;
    address[] public whitelist;
    
    bool private electionCompleted;
    
    // ==================== IPFS TRACKING ====================
    
    // Hash IPFS dei modelli locali di ogni nodo
    mapping(address => string) public modelHashes;
    uint256 public modelsUploaded;
    
    // Hash IPFS del modello globale aggregato (NUOVO)
    string public globalModelHash;

    mapping(address => bool) public hasSubmittedModel;
    
    // ==================== OFFER STRUCTURE ====================
    
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
    
    // ==================== EVENTS ====================
    
    // Eventi esistenti
    event OfferSubmitted(
        address indexed node, 
        uint256 computePower, 
        uint256 bandwidth, 
        uint256 reliability, 
        uint256 dataSize, 
        uint256 cost
    );
    event AuctionClosed(uint256 roundNumber);
    event AggregatorElected(address indexed aggregator, uint256 roundNumber);
    
    // Eventi per comunicazione P2P
    event ModelUploaded(
        address indexed node, 
        string ipfsHash, 
        uint256 timestamp
    );
    event AllModelsReady(
        uint256 totalModels, 
        uint256 roundNumber
    );
    
    //  NUOVO: Evento per notificare disponibilità modello globale
    event GlobalModelReady(
        string ipfsHash, 
        uint256 roundNumber, 
        address indexed aggregator,
        uint256 timestamp
    );
    
    //  NUOVO: Evento per transizione di stato
    event StateChanged(
        State oldState, 
        State newState, 
        uint256 timestamp
    );
    
    // ==================== MODIFIERS ====================
    
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
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator can call");
        _;
    }
    
    // ==================== CONSTRUCTOR ====================
    
    constructor(
        address[] memory _whitelist, 
        uint256 _timeoutSeconds, 
        uint256 _roundNumber
    ) {
        owner = msg.sender;
        whitelist = _whitelist;
        deadline = block.timestamp + _timeoutSeconds;
        roundNumber = _roundNumber;
        auctionState = State.CollectingOffers;
        electionCompleted = false;
        modelsUploaded = 0;
    }
    
    // ==================== PHASE 1: OFFER SUBMISSION ====================
    
    /**
     * @notice I nodi whitelistati sottomettono le loro offerte per diventare aggregatori
     * @param computePower Potenza computazionale del nodo
     * @param bandwidth Larghezza di banda disponibile
     * @param reliability Score di affidabilità (0-100)
     * @param dataSize Quantità di dati disponibili
     * @param cost Costo richiesto per l'aggregazione
     */
    function submitOffer(
        uint256 computePower,
        uint256 bandwidth,
        uint256 reliability,
        uint256 dataSize,
        uint256 cost
    ) external onlyWhitelisted inState(State.CollectingOffers) {
        require(!offers[msg.sender].submitted, "Offer already submitted");
        require(block.timestamp <= deadline, "Deadline passed");
        require(cost > 0, "Cost must be greater than 0");
        
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
        
        emit OfferSubmitted(
            msg.sender, 
            computePower, 
            bandwidth, 
            reliability, 
            dataSize, 
            cost
        );
        
        // Auto-chiusura se tutti hanno sottomesso o scaduto il tempo
        if (totalOffers == whitelist.length || block.timestamp >= deadline) {
            closeAuction();
        }
    }
    
    /**
     * @notice Chiude l'asta e procede all'elezione dell'aggregatore
     */
    function closeAuction() internal {
        require(auctionState == State.CollectingOffers, "Already closed");
        
        State oldState = auctionState;
        auctionState = State.Closed;
        emit StateChanged(oldState, State.Closed, block.timestamp);
        emit AuctionClosed(roundNumber);
        
        electAggregator();
    }
    
    /**
     * @notice Forza la chiusura dell'asta se il deadline è scaduto
     */
    function forceCloseAuction() external onlyOwner {
        require(block.timestamp > deadline, "Deadline not reached");
        require(auctionState == State.CollectingOffers, "Already closed");
        closeAuction();
    }
    
    // ==================== PHASE 2: AGGREGATOR ELECTION ====================
    
    /**
     * @notice Elegge l'aggregatore basandosi su uno score multi-criterio
     * Formula: score = (computePower*30 + bandwidth*25 + reliability*20 + dataSize*15 + 1000*10) / cost
     */
    function electAggregator() internal {
        require(!electionCompleted, "Election already completed");  
        electionCompleted = true; 
        
        address bestNode;
        uint256 bestScore = 0;
        
        // Calcola score per ogni nodo che ha sottomesso un'offerta
        for (uint256 i = 0; i < whitelist.length; i++) {
            address node = whitelist[i];
            FLOffer memory offer = offers[node];
            
            if (offer.submitted) {
                // Score ponderato normalizzato per il costo
                uint256 score = (
                    offer.computePower * 30 +
                    offer.bandwidth * 25 +
                    offer.reliability * 20 +
                    offer.dataSize * 15 +
                    1000 * 10  // Bonus base
                ) * 1e18 / (offer.cost + 1);  // +1 per evitare divisione per zero
                
                if (score > bestScore) {
                    bestScore = score;
                    bestNode = node;
                }
            }
        }
        
        require(bestNode != address(0), "No valid aggregator found");
        aggregator = bestNode;
        
        // Transizione automatica allo stato di upload modelli
        State oldState = auctionState;
        auctionState = State.ModelsUploading;
        emit StateChanged(oldState, State.ModelsUploading, block.timestamp);
        
        emit AggregatorElected(aggregator, roundNumber);
    }
    
    // ==================== PHASE 3: MODEL UPLOADS ====================
    
    /**
     * @notice I nodi participant uploadano il loro hash IPFS dopo l'elezione
     * @param ipfsHash Hash IPFS del modello trainato localmente
     */
    function submitModelHash(string calldata ipfsHash) 
        external 
        onlyWhitelisted 
        inState(State.ModelsUploading)
    {
        require(msg.sender != aggregator, "Aggregator should not submit before aggregation");
        require(!hasSubmittedModel[msg.sender], "Model already submitted");
        require(bytes(ipfsHash).length > 0, "Empty IPFS hash");

        modelHashes[msg.sender] = ipfsHash;
        hasSubmittedModel[msg.sender] = true;
        modelsUploaded++;

        emit ModelUploaded(msg.sender, ipfsHash, block.timestamp);

        if (modelsUploaded >= whitelist.length - 1) {
            State oldState = auctionState;
            auctionState = State.Aggregating;
            emit StateChanged(oldState, State.Aggregating, block.timestamp);
            emit AllModelsReady(modelsUploaded, roundNumber);
        }
    }

    function getMissingUploads() external view returns (address[] memory) {
        uint256 missingCount = 0;

        // Count missing uploads
        for (uint256 i = 0; i < whitelist.length; i++) {
            address node = whitelist[i];
            if (node != aggregator && !hasSubmittedModel[node]) {
                missingCount++;
            }
        }

        // Build array of missing nodes
        address[] memory missing = new address[](missingCount);
        uint256 index = 0;

        for (uint256 i = 0; i < whitelist.length; i++) {
            address node = whitelist[i];
            if (node != aggregator && !hasSubmittedModel[node]) {
                missing[index] = node;
                index++;
            }
        }

        return missing;
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
     * @notice Verifica se tutti i modelli participant sono stati uploadati
     * @return true se tutti hanno uploadato, false altrimenti
     */
    function areAllModelsUploaded() external view returns (bool) {
        return modelsUploaded >= whitelist.length - 1;
    }
    
    /**
     * @notice Conta quanti modelli sono stati uploadati finora
     * @return Numero di modelli uploadati
     */
    function getUploadedModelsCount() external view returns (uint256) {
        return modelsUploaded;
    }
    
    // ==================== PHASE 4: GLOBAL MODEL SUBMISSION ====================
    
    /**
     *  MODIFICATO: L'aggregatore registra il modello globale dopo l'aggregazione
     * @param ipfsHash Hash IPFS del modello aggregato globale
     */
    function submitGlobalModel(string calldata ipfsHash)
        external
        onlyAggregator
    {
        require(
            auctionState == State.ModelsUploading || auctionState == State.Aggregating,
            "Cannot submit global model in current state"
        );
        require(bytes(ipfsHash).length > 0, "Empty IPFS hash");
        require(bytes(globalModelHash).length == 0, "Global model already submitted");
        
        globalModelHash = ipfsHash;
        
        State oldState = auctionState;
        auctionState = State.AggregationComplete;
        emit StateChanged(oldState, State.AggregationComplete, block.timestamp);
        
        //  NUOVO: Emetti evento per notificare i nodi
        emit GlobalModelReady(
            ipfsHash, 
            roundNumber, 
            msg.sender,
            block.timestamp
        );
    }
    
    /**
     *  NUOVO: Recupera l'hash del modello globale aggregato
     * @return Hash IPFS del modello globale
     */
    function getGlobalModelHash() external view returns (string memory) {
        require(bytes(globalModelHash).length > 0, "Global model not yet available");
        return globalModelHash;
    }
    
    /**
     *  NUOVO: Verifica se il modello globale è disponibile
     * @return true se il modello globale è stato uploadato
     */
    function isGlobalModelReady() external view returns (bool) {
        return bytes(globalModelHash).length > 0;
    }
    
    // ==================== UTILITY FUNCTIONS ====================
    
    /**
     * @notice Verifica se un indirizzo è nella whitelist
     * @param node Indirizzo da verificare
     * @return true se il nodo è whitelistato
     */
    function isWhitelisted(address node) public view returns (bool) {
        for (uint256 i = 0; i < whitelist.length; i++) {
            if (whitelist[i] == node) return true;
        }
        return false;
    }
    
    /**
     * @notice Ottiene la whitelist completa
     * @return Array di indirizzi whitelistati
     */
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
        bool hasGlobalModel,
        string memory globalModel,
        bool allModelsUploaded
    ) {
        bool allUploaded = (modelsUploaded >= whitelist.length - 1);
        
        return (
            auctionState,
            aggregator,
            roundNumber,
            totalOffers,
            modelsUploaded,
            whitelist.length,
            bytes(globalModelHash).length > 0,
            globalModelHash,
            allUploaded
        );
    }
    
    /**
     * NUOVO: Ottieni l'offerta di un nodo specifico
     * @param node Indirizzo del nodo
     * @return Offerta completa del nodo
     */
    function getOffer(address node) external view returns (FLOffer memory) {
        return offers[node];
    }
    
    /**
     *  NUOVO: Ottieni tutte le offerte sottomesse
     * @return Array di offerte
     */
    function getAllOffers() external view returns (FLOffer[] memory) {
        FLOffer[] memory allOffers = new FLOffer[](totalOffers);
        uint256 index = 0;
        
        for (uint256 i = 0; i < whitelist.length; i++) {
            if (offers[whitelist[i]].submitted) {
                allOffers[index] = offers[whitelist[i]];
                index++;
            }
        }
        
        return allOffers;
    }
    
    /**
     *  NUOVO: Tempo rimanente prima della scadenza
     * @return Secondi rimanenti (0 se scaduto)
     */
    function getTimeRemaining() external view returns (uint256) {
        if (block.timestamp >= deadline) {
            return 0;
        }
        return deadline - block.timestamp;
    }
    
    /**
     *  NUOVO: Verifica se l'asta è scaduta
     * @return true se il deadline è passato
     */
    function isExpired() external view returns (bool) {
        return block.timestamp > deadline;
    }
}