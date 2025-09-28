// chainEnv/contracts/AggregatorAuction.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AggregatorAuction {
    enum State { CollectingOffers, Closed }
    
    State public auctionState;
    address public owner;
    address public aggregator;
    uint256 public deadline;
    uint256 public roundNumber;
    address[] public whitelist;
    
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
    
    event OfferSubmitted(address indexed node, uint256 computePower, uint256 bandwidth, uint256 reliability, uint256 dataSize, uint256 cost);
    event AuctionClosed(uint256 roundNumber);
    event AggregatorElected(address indexed aggregator, uint256 roundNumber);
    
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
    
    constructor(address[] memory _whitelist, uint256 _timeoutSeconds, uint256 _roundNumber) {
        owner = msg.sender;
        whitelist = _whitelist;
        deadline = block.timestamp + _timeoutSeconds;
        roundNumber = _roundNumber;
        auctionState = State.CollectingOffers;
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
        emit AggregatorElected(aggregator, roundNumber);
    }
    
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
}