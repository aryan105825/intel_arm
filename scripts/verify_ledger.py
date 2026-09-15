import json, hashlib, sys

PATH = "benchmarks/raw/compliance_ledger.jsonl"

def canonical(event):
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def chain_hash(prev, sig):
    return "sha256:" + hashlib.sha256(f"{prev}{sig}".encode()).hexdigest()

def main():
    try:
        with open(PATH) as f:
            records = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Ledger not found at {PATH}")
        return

    prev = "sha256:" + ("0" * 64)
    for i, rec in enumerate(records):
        sig = "sha256:" + hashlib.sha256(canonical(rec["event"]).encode()).hexdigest()
        expected = chain_hash(prev, sig)
        ok = (expected == rec["chain_hash"]) and (sig == rec["signature"])
        status = "OK " if ok else "TAMPERED"
        
        event_name = rec['event'].get('taxonomy', rec['event'].get('event', 'UNKNOWN'))
        print(f"[{i:03d}] {status}  event={event_name}  chain_hash={rec['chain_hash'][:20]}...")
        if not ok:
            print(f"       ^^^ chain breaks here. Everything after this point is now provably invalid.")
        prev = rec["chain_hash"]

if __name__ == "__main__":
    main()