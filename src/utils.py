import psutil
import time
 
simulated_mysql_load = 0.1
simulated_elk_load = 0.2
simulated_ipfs_load = 0.05

def get_system_state():
 
    cpu_percent = psutil.cpu_percent(interval=None)  # CPU utilization of the host
    memory_percent = psutil.virtual_memory().percent  # Memory utilization of the host
 
    net_io_counters = psutil.net_io_counters()
    simulated_network_traffic = (net_io_counters.bytes_sent + net_io_counters.bytes_recv) / (1024 * 1024) # MB
 
    global simulated_mysql_load, simulated_elk_load, simulated_ipfs_load
 
    simulated_mysql_load = max(0, min(1, simulated_mysql_load + (time.time() % 0.1 - 0.05)))
    simulated_elk_load = max(0, min(1, simulated_elk_load + (time.time() % 0.15 - 0.075)))
    simulated_ipfs_load = max(0, min(1, simulated_ipfs_load + (time.time() % 0.05 - 0.025)))
 
    return [
        cpu_percent,          # Host CPU utilization (0-100)
        memory_percent,       # Host Memory utilization (0-100)
        simulated_network_traffic, # Simulated network traffic (MB)
        simulated_mysql_load, # Simulated load/queue for MySQL (0-1)
        simulated_elk_load,   # Simulated load/queue for ELK (0-1)
        simulated_ipfs_load   # Simulated load/queue for IPFS (0-1)
    ]


