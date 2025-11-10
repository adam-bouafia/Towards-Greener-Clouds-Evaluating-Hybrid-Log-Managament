"""
Blockchain Logger for Immutable Log Storage
Provides selective blockchain verification for sensitive logs
"""

import hashlib
import json
import logging
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from web3 import Web3
    from web3.exceptions import TransactionNotFound, TimeExhausted
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None

logger = logging.getLogger(__name__)


class BlockchainLogger:
    """
    Manages blockchain storage for sensitive log verification.
    Stores cryptographic hashes on-chain for immutability proof.
    """
    
    SENSITIVE_LEVELS = {'FATAL', 'CRITICAL', 'ERROR', 'SECURITY', 'ALERT'}
    SENSITIVE_KEYWORDS = {
        'security', 'breach', 'attack', 'intrusion', 'unauthorized',
        'fraud', 'malicious', 'exploit', 'vulnerability', 'injection',
        'authentication', 'authorization', 'password', 'token', 'credential',
        'failure', 'crash', 'panic', 'fatal', 'critical'
    }
    
    # Regex patterns for detecting sensitive data (PII, credentials, etc.)
    SENSITIVE_PATTERNS = {
        'ipv4': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # Long alphanumeric
        'jwt': re.compile(r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*'),
        'private_key': re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
    }
    
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        enabled: bool = False,
        sensitivity_threshold: float = 0.5,
        use_ml_detector: bool = False,
        ml_model_type: str = "distilbert",
        ml_model_path: Optional[str] = None
    ):
        self.enabled = enabled and WEB3_AVAILABLE
        self.sensitivity_threshold = sensitivity_threshold  # Score threshold (0.0-1.0)
        self.use_ml_detector = use_ml_detector
        self.ml_detector = None
        self.w3 = None
        self.contract = None
        self.account = None
        
        # Initialize ML detector if requested
        if use_ml_detector:
            try:
                from .ml_sensitivity_detector import MLSensitivityDetector
                self.ml_detector = MLSensitivityDetector(
                    model_type=ml_model_type,
                    model_path=ml_model_path,
                    confidence_threshold=sensitivity_threshold
                )
                logger.info(f"ML sensitivity detector enabled ({ml_model_type})")
            except Exception as e:
                logger.warning(f"Failed to load ML detector: {e}")
                logger.warning("Falling back to heuristic detection")
                self.use_ml_detector = False
        
        if not self.enabled:
            if enabled and not WEB3_AVAILABLE:
                logger.warning("Blockchain logging requested but web3 not installed")
            return
        
        if not all([rpc_url, contract_address, private_key]):
            logger.warning("Blockchain config incomplete, running in simulation mode")
            self.enabled = False
            return
        
        try:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                logger.error(f"Cannot connect to blockchain RPC: {rpc_url}")
                self.enabled = False
                return
            
            self.contract_address = Web3.to_checksum_address(contract_address)
            self.account = self.w3.eth.account.from_key(private_key)
            
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self._get_contract_abi()
            )
            
            logger.info(f"Blockchain logger initialized on {rpc_url}")
            logger.info(f"Contract: {self.contract_address}")
            logger.info(f"Account: {self.account.address}")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain logger: {e}")
            self.enabled = False
    
    def get_sensitivity_score(self, log: Dict[str, Any]) -> float:
        """
        Calculate weighted sensitivity score (0.0 to 1.0).
        
        Scoring breakdown:
        - Level severity: up to 0.4
        - Content keywords: up to 0.4
        - Component type: up to 0.2
        - PII/sensitive patterns: +0.3 boost
        
        Returns:
            Float score between 0.0 and 1.0
        """
        score = 0.0
        
        # 1. Level severity (max 0.4)
        level_scores = {
            'FATAL': 0.4, 'CRITICAL': 0.35, 'ALERT': 0.35,
            'SECURITY': 0.4, 'ERROR': 0.25, 'WARN': 0.1
        }
        level = log.get('Level', '').upper()
        score += level_scores.get(level, 0.0)
        
        # 2. Content keyword analysis (max 0.4)
        content = log.get('Content', '').lower()
        
        # High-risk keywords (0.3 each, capped at 0.4)
        high_risk = ['breach', 'attack', 'exploit', 'injection', 'unauthorized', 'hack']
        high_risk_count = sum(1 for kw in high_risk if kw in content)
        if high_risk_count > 0:
            score += min(0.4, high_risk_count * 0.3)
        else:
            # Medium-risk keywords (0.2)
            med_risk = ['fail', 'denied', 'invalid', 'timeout', 'refused']
            if any(kw in content for kw in med_risk):
                score += 0.2
        
        # Credential mentions (additional 0.2)
        if any(kw in content for kw in ['password', 'token', 'key', 'secret', 'credential']):
            score += 0.2
        
        # 3. Component weight (max 0.2)
        component = log.get('Component', '').lower()
        if any(kw in component for kw in ['security', 'auth', 'firewall']):
            score += 0.2
        elif any(kw in component for kw in ['payment', 'billing', 'admin']):
            score += 0.15
        
        # 4. Pattern detection bonus (adds 0.3 if PII found)
        if self._contains_sensitive_patterns(log):
            score += 0.3
            logger.debug(f"Detected sensitive pattern in log, boosting score by 0.3")
        
        return min(1.0, score)  # Cap at 1.0
    
    def _contains_sensitive_patterns(self, log: Dict[str, Any]) -> bool:
        """
        Check if log content contains sensitive data patterns.
        
        Detects:
        - IP addresses
        - Email addresses
        - Credit card numbers
        - SSNs
        - API keys / tokens
        - JWTs
        - Private keys
        
        Returns:
            True if any pattern matches
        """
        content = log.get('Content', '')
        
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                logger.debug(f"Detected {pattern_name} pattern in log")
                return True
        
        return False
    
    def is_sensitive(self, log: Dict[str, Any]) -> bool:
        """
        Determine if a log requires blockchain verification.
        
        Methods:
        1. ML-based (if enabled): Uses pre-trained model for smart detection
        2. Heuristic: Weighted scoring system (fallback or default)
        
        ML Detection (when use_ml_detector=True):
        - Uses DistilBERT/TinyBERT for semantic understanding
        - Zero-shot classification (no training needed)
        - Understands context beyond keywords
        
        Heuristic Detection (default):
        - Level severity (FATAL, CRITICAL, ERROR, etc.)
        - Content keywords (security, breach, attack, etc.)
        - Component type (security, auth, payment)
        - PII patterns (emails, IPs, credit cards)
        
        Returns:
            True if sensitive (ML confidence or score >= threshold)
        """
        if not self.enabled:
            return False
        
        # Use ML detector if enabled and loaded
        if self.use_ml_detector and self.ml_detector:
            try:
                is_sens, confidence = self.ml_detector.predict(log)
                if is_sens:
                    logger.debug(f"ML detector: sensitive (confidence: {confidence:.2f})")
                return is_sens
            except Exception as e:
                logger.error(f"ML detection failed, falling back to heuristics: {e}")
                # Fall through to heuristic detection
        
        # Heuristic detection (default or fallback)
        score = self.get_sensitivity_score(log)
        is_sens = score >= self.sensitivity_threshold
        
        if is_sens:
            logger.debug(f"Heuristic: sensitive (score: {score:.2f} >= {self.sensitivity_threshold})")
        
        return is_sens
    
    def compute_hash(self, log: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of log content for blockchain storage.
        """
        log_string = json.dumps(log, sort_keys=True, default=str)
        return hashlib.sha256(log_string.encode()).hexdigest()
    
    def store_hash(
        self,
        log: Dict[str, Any],
        backend: str,
        gas_limit: int = 100000
    ) -> Optional[str]:
        """
        Store log hash on blockchain for immutability proof.
        
        Returns:
            Transaction hash if successful, None otherwise
        """
        if not self.enabled:
            logger.debug("Blockchain disabled, simulating hash storage")
            return self._simulate_storage(log)
        
        try:
            content_hash = self.compute_hash(log)
            log_level = log.get('Level', 'UNKNOWN')
            
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            tx = self.contract.functions.storeProof(
                bytes.fromhex(content_hash),
                log_level,
                backend
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': self.w3.eth.chain_id
            })
            
            signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=120
            )
            
            if receipt['status'] == 1:
                logger.info(f"Stored hash {content_hash[:16]}... on blockchain")
                return tx_hash.hex()
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to store hash on blockchain: {e}")
            return None
    
    def verify_hash(
        self,
        log: Dict[str, Any],
        claimed_hash: str
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Verify log authenticity against blockchain record.
        
        Returns:
            (is_valid, timestamp, reporter_address)
        """
        if not self.enabled:
            return False, None, None
        
        try:
            actual_hash = self.compute_hash(log)
            
            if actual_hash != claimed_hash:
                return False, None, None
            
            result = self.contract.functions.verifyProof(
                bytes.fromhex(claimed_hash)
            ).call()
            
            is_valid, timestamp, reporter = result
            return is_valid, timestamp, reporter
            
        except Exception as e:
            logger.error(f"Failed to verify hash: {e}")
            return False, None, None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get blockchain logger statistics.
        """
        if not self.enabled:
            return {
                'enabled': False,
                'mode': 'simulation'
            }
        
        try:
            total_proofs = self.contract.functions.getTotalProofs().call()
            balance = self.w3.eth.get_balance(self.account.address)
            gas_price = self.w3.eth.gas_price
            
            return {
                'enabled': True,
                'mode': 'live',
                'network': self.w3.eth.chain_id,
                'contract': self.contract_address,
                'account': self.account.address,
                'balance_wei': balance,
                'balance_eth': self.w3.from_wei(balance, 'ether'),
                'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei'),
                'total_proofs': total_proofs
            }
        except Exception as e:
            logger.error(f"Failed to get blockchain stats: {e}")
            return {'enabled': True, 'mode': 'live', 'error': str(e)}
    
    def _simulate_storage(self, log: Dict[str, Any]) -> str:
        """
        Simulate blockchain storage for testing without real transactions.
        """
        content_hash = self.compute_hash(log)
        tx_hash = hashlib.sha256(
            f"{content_hash}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        return f"0x{tx_hash}"
    
    def _get_contract_abi(self) -> list:
        """
        Minimal ABI for log proof storage contract.
        """
        return [
            {
                "inputs": [
                    {"name": "_contentHash", "type": "bytes32"},
                    {"name": "_logLevel", "type": "string"},
                    {"name": "_backend", "type": "string"}
                ],
                "name": "storeProof",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_hash", "type": "bytes32"}
                ],
                "name": "verifyProof",
                "outputs": [
                    {"name": "isValid", "type": "bool"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "reporter", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getTotalProofs",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
