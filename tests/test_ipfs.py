import importlib.util, pytest

if importlib.util.find_spec("ipfshttpclient") is None:
    pytest.skip("ipfshttpclient not installed; skipping IPFS smoke test", allow_module_level=True)

import ipfshttpclient

log_data = "{...}" # Your log data

try:
    # The 'with' statement handles connection setup and teardown
    with ipfshttpclient.connect() as ipfs_client:
        result = ipfs_client.add_json(log_data)
        print(f"Successfully added log data to IPFS. CID: {result}")
except Exception as e:
    # This catches the connection error and logs it properly
    print(f"[IPFS WRITE FAILED] Could not connect to IPFS daemon or add data. Error: {e}")