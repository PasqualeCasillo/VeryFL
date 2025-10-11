# chainfl/auction_proxy.py
import logging
from chainfl.interact import chainProxy

logger = logging.getLogger(__name__)

try:
    import brownie
    from brownie import network
    BROWNIE_AVAILABLE = True
    logger.info(f"Brownie module loaded in auction_proxy, network: {network.show_active()}")
except ImportError:
    brownie = None
    BROWNIE_AVAILABLE = False
    logger.info("Brownie not available in auction_proxy")


class AuctionContract:
    """Wrapper uniforme per contratti reali e mock"""
    
    def __init__(self, contract, is_real=True):
        self.contract = contract
        self.is_real = is_real
    
    @property
    def address(self):
        if self.is_real:
            return self.contract.address
        else:
            return self.contract['address']
    
    def submit_offer(self, *args, **kwargs):
        if self.is_real:
            return self.contract.submitOffer(*args, **kwargs)
        else:
            return None
    
    def get_aggregator(self):
        if self.is_real:
            return self.contract.aggregator()
        else:
            return self.contract.get('aggregator')
    
    def get_state(self):
        if self.is_real:
            return self.contract.auctionState()
        else:
            return self.contract.get('state')


class AuctionChainProxy(chainProxy):
    def __init__(self):
        super().__init__()
        self.auction_contracts = {}
        
    def deploy_auction_contract(self, whitelist, timeout_seconds, round_number):
        """Deploy new auction contract for a round"""
        try:
            if BROWNIE_AVAILABLE and brownie:
                logger.info(f"Deploying REAL auction contract for round {round_number}")
                
                contracts = brownie.project.chainServer
                auction_contract = contracts.AggregatorAuction.deploy(
                    whitelist,
                    timeout_seconds,
                    round_number,
                    {'from': brownie.accounts[0]}
                )
                
                wrapped_contract = AuctionContract(auction_contract, is_real=True)
                self.auction_contracts[round_number] = wrapped_contract
                
                logger.info(f"Real auction deployed at {wrapped_contract.address}")
                return wrapped_contract.address
            else:
                contract_address = f"0xauction{round_number:04d}"
                mock_contract = {
                    'address': contract_address,
                    'state': 'CollectingOffers',
                    'offers': {},
                    'aggregator': None
                }
                
                wrapped_contract = AuctionContract(mock_contract, is_real=False)
                self.auction_contracts[round_number] = wrapped_contract
                
                logger.info(f"Mock deployed auction contract at {contract_address}")
                return contract_address
                
        except Exception as e:
            logger.error(f"Failed to deploy auction contract: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _find_contract_by_address(self, auction_address):
        """Find contract wrapper by address"""
        for round_num, wrapped_contract in self.auction_contracts.items():
            if wrapped_contract.address == auction_address:
                return wrapped_contract
        return None
    
    def verify_all_models_uploaded(self, auction_address):
        """
        Verify that all required nodes have uploaded their models.
        Returns tuple (all_uploaded: bool, missing_nodes: list)
        """
        try:
            wrapped_contract = self._find_contract_by_address(auction_address)
            
            if not wrapped_contract:
                logger.error(f"Auction contract not found: {auction_address}")
                return False, []
            
            if wrapped_contract.is_real:
                # Real blockchain verification
                missing = wrapped_contract.contract.getMissingUploads()
                all_uploaded = len(missing) == 0
                
                if not all_uploaded:
                    logger.warning(f"Missing uploads from {len(missing)} nodes")
                    for addr in missing:
                        logger.warning(f"  Node {addr[:10]}... has not uploaded")
                
                return all_uploaded, list(missing)
            else:
                # Mock verification
                expected_uploads = len(wrapped_contract.contract['offers']) - 1
                actual_uploads = len([
                    h for h in wrapped_contract.contract.get('model_hashes', {}).values() 
                    if h
                ])
                all_uploaded = actual_uploads >= expected_uploads
                
                return all_uploaded, []
                
        except Exception as e:
            logger.error(f"Error verifying uploads: {e}")
            return False, []
            
    def submit_offer(self, auction_address, node_address, computePower, 
                    bandwidth, reliability, dataSize, cost):
        """Submit offer to auction contract"""
        try:
            wrapped_contract = self._find_contract_by_address(auction_address)
            
            if not wrapped_contract:
                logger.error(f"Auction contract not found: {auction_address}")
                return False
            
            if wrapped_contract.is_real:
                logger.info(f"Submitting REAL offer from {node_address}")
                
                tx = wrapped_contract.submit_offer(
                    computePower, bandwidth, reliability, dataSize, cost,
                    {'from': node_address}
                )
                logger.info(f"Real offer submitted, tx: {tx.txid}")
                return True
            else:
                import random
                varied_cost = cost + random.randint(-20, 20)
                
                wrapped_contract.contract['offers'][node_address] = {
                    'computePower': computePower,
                    'bandwidth': bandwidth,
                    'reliability': reliability,
                    'dataSize': dataSize,
                    'cost': max(1, varied_cost)
                }
                
                if len(wrapped_contract.contract['offers']) >= 5:
                    best_node = None
                    best_score = 0
                    
                    for addr, offer in wrapped_contract.contract['offers'].items():
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
                    
                    wrapped_contract.contract['aggregator'] = best_node
                    wrapped_contract.contract['state'] = 'Closed'
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
            wrapped_contract = self._find_contract_by_address(auction_address)
            
            if not wrapped_contract:
                return None
            
            if wrapped_contract.is_real:
                logger.info(f"Reading REAL election result")
                aggregator = wrapped_contract.get_aggregator()
                if aggregator != "0x0000000000000000000000000000000000000000":
                    logger.info(f"Real election result: {aggregator}")
                    return aggregator
            else:
                if wrapped_contract.contract['state'] == 'Closed' and wrapped_contract.contract['aggregator']:
                    return wrapped_contract.contract['aggregator']
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting election result: {e}")
            return None


auction_chain_proxy = AuctionChainProxy()