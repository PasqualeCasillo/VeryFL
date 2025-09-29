# chainfl/auction_proxy.py
import logging
from chainfl.interact import chainProxy

logger = logging.getLogger(__name__)

# Import brownie globally like interact.py does
try:
    import brownie
    from brownie import network
    BROWNIE_AVAILABLE = True
    logger.info(f"Brownie module loaded in auction_proxy, network: {network.show_active()}")
except ImportError:
    brownie = None
    BROWNIE_AVAILABLE = False
    logger.info("Brownie not available in auction_proxy")

class AuctionChainProxy(chainProxy):
    def __init__(self):
        super().__init__()
        self.auction_contracts = {}
        
    def deploy_auction_contract(self, whitelist, timeout_seconds, round_number):
        """Deploy new auction contract for a round"""
        try:
            if BROWNIE_AVAILABLE and brownie:
                # Deploy actual smart contract
                logger.info(f"Deploying REAL auction contract for round {round_number}")
                
                # Access contracts from brownie project
                contracts = brownie.project.chainServer
                auction_contract = contracts.AggregatorAuction.deploy(
                    whitelist,
                    timeout_seconds,
                    round_number,
                    {'from': brownie.accounts[0]}
                )
                
                contract_address = auction_contract.address
                self.auction_contracts[round_number] = auction_contract
                
                logger.info(f"Real auction deployed at {contract_address}")
                return contract_address
            else:
                # Mock deployment for testing
                contract_address = f"0xauction{round_number:04d}"
                self.auction_contracts[round_number] = {
                    'address': contract_address,
                    'state': 'CollectingOffers',
                    'offers': {},
                    'aggregator': None
                }
                
                logger.info(f"Mock deployed auction contract at {contract_address}")
                return contract_address
                
        except Exception as e:
            logger.error(f"Failed to deploy auction contract: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def submit_offer(self, auction_address, node_address, computePower, 
                    bandwidth, reliability, dataSize, cost):
        """Submit offer to auction contract"""
        try:
            # Find contract by address
            contract = None
            round_num = None
            for rn, stored_contract in self.auction_contracts.items():
                if (hasattr(stored_contract, 'address') and stored_contract.address == auction_address) or \
                   (isinstance(stored_contract, dict) and stored_contract['address'] == auction_address):
                    contract = stored_contract
                    round_num = rn
                    break
                    
            if not contract:
                logger.error(f"Auction contract not found: {auction_address}")
                return False
                
            if hasattr(contract, 'submitOffer'):
                # Real blockchain call
                logger.info(f"Submitting REAL offer from {node_address}")
                
                tx = contract.submitOffer(
                    computePower, bandwidth, reliability, dataSize, cost,
                    {'from': node_address}
                )
                logger.info(f"Real offer submitted, tx: {tx.txid}")
                return True
            else:
                # Mock submission
                import random
                varied_cost = cost + random.randint(-20, 20)
                
                contract['offers'][node_address] = {
                    'computePower': computePower,
                    'bandwidth': bandwidth,
                    'reliability': reliability,
                    'dataSize': dataSize,
                    'cost': max(1, varied_cost)
                }
                
                # Trigger election when all offers received
                if len(contract['offers']) >= 5:
                    best_node = None
                    best_score = 0
                    
                    for addr, offer in contract['offers'].items():
                        score = (
                            offer['computePower'] * 30 +
                            offer['bandwidth'] * 25 + 
                            offer['reliability'] * 20 +
                            offer['dataSize'] * 15 +
                            1000 * 10
                        ) * 1e18 / (offer['cost'] + 1)
                        
                        if score > best_score:
                            best_score = score
                            best_node = addr
                            
                    contract['aggregator'] = best_node
                    contract['state'] = 'Closed'
                    logger.info(f"Mock election: {best_node} won with score {best_score}")
                    
                logger.info(f"Mock offer submitted for {node_address} with cost {varied_cost}")
                return True
                    
        except Exception as e:
            logger.error(f"Error submitting offer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def get_election_result(self, auction_address):
        """Get election result from auction contract"""
        try:
            # Find contract
            contract = None
            for round_num, stored_contract in self.auction_contracts.items():
                if (hasattr(stored_contract, 'address') and stored_contract.address == auction_address) or \
                   (isinstance(stored_contract, dict) and stored_contract['address'] == auction_address):
                    contract = stored_contract
                    break
                    
            if not contract:
                return None
                
            if hasattr(contract, 'aggregator'):
                # Real blockchain call
                logger.info(f"Reading REAL election result")
                aggregator = contract.aggregator()
                if aggregator != "0x0000000000000000000000000000000000000000":
                    logger.info(f"Real election result: {aggregator}")
                    return aggregator
            else:
                # Mock contract
                if contract['state'] == 'Closed' and contract['aggregator']:
                    return contract['aggregator']
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting election result: {e}")
            return None

# Create global instance
auction_chain_proxy = AuctionChainProxy()