import csv
from pathlib import Path
from src.experiment import _aggregate

def test_aggregate_basic(tmp_path):
    # Create two mock per-log CSVs
    headers = [
        'log_id','router','dataset_name','destination','routing_latency_ms','backend_write_latency_ms',
        'total_latency_ms','success','energy_cpu_pkg_j','payload_bytes','sensitive','proc_cpu_pct','proc_rss_mb'
    ]
    data1 = [
        ['1','a2c','mock','mysql','1.0','2.0','3.0','True','0.01','10','False','5.0','50'],
        ['2','a2c','mock','elk','2.0','3.0','5.0','True','0.02','12','True','6.0','51'],
    ]
    data2 = [
        ['1','q_learning','mock','ipfs','1.5','2.5','4.0','False','0.03','11','False','5.5','52'],
    ]
    p1 = tmp_path / 'a2c_mock.csv'
    p2 = tmp_path / 'q_learning_mock.csv'
    for p, rows in [(p1,data1),(p2,data2)]:
        with open(p,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

    agg = _aggregate([p1,p2], 'mock', tmp_path)
    assert agg['rows'] == 3
    assert 'p95_total_latency_ms' in agg
    assert abs(agg['success_rate'] - (2/3)) < 1e-6
    # destination mix sums to 1 across mysql/elk/ipfs present
    s = agg['destination_mix_mysql'] + agg['destination_mix_elk'] + agg['destination_mix_ipfs']
    assert abs(s - 1.0) < 1e-6
    # compliance fields present (values predictable with this mock data)
    for k in [
        'sensitive_total','sensitive_to_ipfs','sensitive_coverage','sensitive_leakage',
        'leakage_rate','non_sensitive_ipfs_fraction','compliance_score'
    ]:
        assert k in agg
