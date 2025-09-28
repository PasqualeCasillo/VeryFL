// SPDX-License-Identifier: UNLICENSE
pragma solidity ^0.8.0;

contract ModelRegistry {
    
    struct ModelVersion {
        string ipfsHash;        // IPFS hash del modello
        uint256 epoch;          // Numero epoca
        address uploader;       // Chi ha caricato il modello
        uint256 timestamp;      // Timestamp creazione
        bool isActive;          // Se il modello è attivo
        uint256 clientCount;    // Numero di client che hanno contribuito
    }
    
    // Mapping epoch -> ModelVersion
    mapping(uint256 => ModelVersion) public models;
    
    // Current active epoch
    uint256 public currentEpoch;
    
    // Owner of the contract (FL server)
    address public owner;
    
    // Events
    event ModelRegistered(uint256 indexed epoch, string ipfsHash, address uploader);
    event ModelActivated(uint256 indexed epoch);
    
    modifier onlyOwner {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        currentEpoch = 0;
    }
    
    function registerModel(
        uint256 epoch,
        string calldata ipfsHash,
        uint256 clientCount
    ) external onlyOwner {
        require(bytes(ipfsHash).length > 0, "IPFS hash cannot be empty");
        require(epoch >= currentEpoch, "Cannot register past epoch");
        
        models[epoch] = ModelVersion({
            ipfsHash: ipfsHash,
            epoch: epoch,
            uploader: msg.sender,
            timestamp: block.timestamp,
            isActive: false,
            clientCount: clientCount
        });
        
        emit ModelRegistered(epoch, ipfsHash, msg.sender);
    }
    
    function activateModel(uint256 epoch) external onlyOwner {
        require(models[epoch].timestamp > 0, "Model does not exist");
        
        // Deactivate previous model
        if (currentEpoch > 0) {
            models[currentEpoch].isActive = false;
        }
        
        // Activate new model
        models[epoch].isActive = true;
        currentEpoch = epoch;
        
        emit ModelActivated(epoch);
    }
    
    function getCurrentModel() external view returns (string memory) {
        require(currentEpoch > 0, "No active model");
        return models[currentEpoch].ipfsHash;
    }
    
    function getModel(uint256 epoch) external view returns (ModelVersion memory) {
        return models[epoch];
    }
}