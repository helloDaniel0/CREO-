# UDP Handover Notification

The example defines a 16-byte header with a protocol marker, version, message
type, sequence, and absolute handover time. The notifier sends before `tHO`,
retries at one RTT until acknowledged, and the receiver deduplicates messages,
returns an ACK, and invokes a callback at `tHO`.

```bash
./ns3 build creo-udp-handover-notification
./ns3 run 'creo-udp-handover-notification --baseRttMs=50 \
  --handoverTime=2 --leadTimeMs=100 --maxAttempts=3'
```

The callback is the integration point for the CREO+ stop/wait/resume state
machine; this example does not alter a TCP socket by itself.
