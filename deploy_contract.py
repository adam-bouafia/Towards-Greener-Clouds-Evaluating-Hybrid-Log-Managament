#!/usr/bin/env python3
"""
Deploy LogAuditContract to Polygon Amoy Testnet
"""

import os
import json
from web3 import Web3
from eth_account import Account
from solcx import compile_source, install_solc

# Smart Contract Source Code
CONTRACT_SOURCE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LogAuditContract {
    struct LogProof {
        bytes32 contentHash;
        uint256 timestamp;
        address submitter;
        string logLevel;
        string backend;
    }
    
    mapping(bytes32 => LogProof) public proofs;
    bytes32[] public proofHashes;
    
    event ProofStored(
        bytes32 indexed contentHash,
        uint256 timestamp,
        address submitter,
        string logLevel,
        string backend
    );
    
    function storeProof(
        bytes32 _contentHash,
        string memory _logLevel,
        string memory _backend
    ) external {
        require(proofs[_contentHash].timestamp == 0, "Proof already exists");
        
        proofs[_contentHash] = LogProof({
            contentHash: _contentHash,
            timestamp: block.timestamp,
            submitter: msg.sender,
            logLevel: _logLevel,
            backend: _backend
        });
        
        proofHashes.push(_contentHash);
        
        emit ProofStored(_contentHash, block.timestamp, msg.sender, _logLevel, _backend);
    }
    
    function verifyProof(bytes32 _contentHash)
        external
        view
        returns (
            bool exists,
            uint256 timestamp,
            address submitter
        )
    {
        LogProof memory proof = proofs[_contentHash];
        exists = proof.timestamp != 0;
        timestamp = proof.timestamp;
        submitter = proof.submitter;
    }
    
    function getTotalProofs() external view returns (uint256) {
        return proofHashes.length;
    }
    
    function getProofDetails(bytes32 _contentHash)
        external
        view
        returns (
            bytes32 contentHash,
            uint256 timestamp,
            address submitter,
            string memory logLevel,
            string memory backend
        )
    {
        LogProof memory proof = proofs[_contentHash];
        require(proof.timestamp != 0, "Proof does not exist");
        
        return (
            proof.contentHash,
            proof.timestamp,
            proof.submitter,
            proof.logLevel,
            proof.backend
        );
    }
}
'''

def main():
    print("=" * 80)
    print("🚀 Deploying LogAuditContract to Polygon Amoy Testnet")
    print("=" * 80)
    
    # Load credentials from environment
    rpc_url = os.environ.get("POLYGON_RPC_URL")
    private_key = os.environ.get("BLOCKCHAIN_PRIVATE_KEY")
    
    if not rpc_url or not private_key:
        print("❌ ERROR: Missing environment variables!")
        print("   Required:")
        print("   - POLYGON_RPC_URL")
        print("   - BLOCKCHAIN_PRIVATE_KEY")
        return
    
    # Connect to Polygon Amoy
    print("\n📡 Connecting to Polygon Amoy...")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    if not w3.is_connected():
        print(f"❌ Failed to connect to: {rpc_url}")
        return
    
    print(f"✅ Connected to Polygon Amoy")
    print(f"   Chain ID: {w3.eth.chain_id}")
    
    # Load account
    account = Account.from_key(private_key)
    print(f"\n👛 Wallet Address: {account.address}")
    
    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_matic = w3.from_wei(balance, 'ether')
    print(f"   Balance: {balance_matic} MATIC")
    
    if balance == 0:
        print("❌ Insufficient balance! Get test MATIC from https://faucet.polygon.technology/")
        return
    
    # Install Solidity compiler
    print("\n🔧 Installing Solidity compiler...")
    try:
        install_solc('0.8.20')
    except:
        pass  # Already installed
    
    # Compile contract
    print("📝 Compiling smart contract...")
    compiled_sol = compile_source(
        CONTRACT_SOURCE,
        output_values=['abi', 'bin'],
        solc_version='0.8.20'
    )
    
    contract_id, contract_interface = compiled_sol.popitem()
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']
    
    print("✅ Contract compiled successfully")
    
    # Deploy contract
    print("\n🚀 Deploying contract...")
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = w3.eth.gas_price
    
    print(f"   Gas Price: {w3.from_wei(gas_price, 'gwei')} Gwei")
    
    # Estimate gas
    constructor_txn = Contract.constructor().build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 2000000,  # Estimate
        'gasPrice': gas_price,
    })
    
    # Sign transaction
    signed_txn = account.sign_transaction(constructor_txn)
    
    # Send transaction
    print("   Sending deployment transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"   Transaction Hash: {tx_hash.hex()}")
    
    # Wait for receipt
    print("   Waiting for confirmation (this may take 30-60 seconds)...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    
    if tx_receipt.status == 1:
        contract_address = tx_receipt.contractAddress
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Contract Deployed!")
        print("=" * 80)
        print(f"\n📍 Contract Address: {contract_address}")
        print(f"   Transaction Hash: {tx_hash.hex()}")
        print(f"   Block Number: {tx_receipt.blockNumber}")
        print(f"   Gas Used: {tx_receipt.gasUsed}")
        
        gas_cost_matic = w3.from_wei(tx_receipt.gasUsed * gas_price, 'ether')
        print(f"   Gas Cost: {gas_cost_matic} MATIC")
        
        print("\n" + "=" * 80)
        print("🔧 NEXT STEP: Set Environment Variable")
        print("=" * 80)
        print(f"\nRun this command:")
        print(f"export BLOCKCHAIN_CONTRACT_ADDRESS=\"{contract_address}\"")
        
        print("\n" + "=" * 80)
        print("🔍 Verify on Block Explorer:")
        print("=" * 80)
        print(f"https://amoy.polygonscan.com/address/{contract_address}")
        
        # Save ABI to file
        abi_file = "contract_abi.json"
        with open(abi_file, 'w') as f:
            json.dump(abi, f, indent=2)
        print(f"\n💾 Contract ABI saved to: {abi_file}")
        
        # Save deployment info
        deployment_info = {
            "contract_address": contract_address,
            "transaction_hash": tx_hash.hex(),
            "block_number": tx_receipt.blockNumber,
            "deployer": account.address,
            "network": "Polygon Amoy Testnet",
            "chain_id": w3.eth.chain_id
        }
        
        info_file = "deployment_info.json"
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"💾 Deployment info saved to: {info_file}")
        
    else:
        print("\n❌ Deployment failed!")
        print(f"   Transaction Hash: {tx_hash.hex()}")
        print(f"   Status: {tx_receipt.status}")

if __name__ == "__main__":
    main()
