/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef CREO_DRL_UAPI_H
#define CREO_DRL_UAPI_H

#include <linux/types.h>

#define CREO_DRL_ABI_VERSION 1U
#define CREO_DRL_CAP_WINDOW 8U
#define CREO_DRL_STATE_MESSAGE_SIZE 172U
#define CREO_DRL_ACTION_MESSAGE_SIZE 56U

#define CREO_DRL_ACTION_FALLBACK 0U
#define CREO_DRL_ACTION_MODEL 1U

/* Kernel-to-user telemetry. All rate fields use bit/s and all RTTs use us. */
struct creo_drl_state_msg {
	__u32 abi_version;
	__u32 message_size;
	__u64 flow_id;
	__u64 sequence;
	__u64 timestamp_ns;
	__u64 throughput_bps;
	__u64 capacity_bps;
	__u64 pacing_rate_bps;
	__u64 last_action_sequence;
	__u32 srtt_us;
	__u32 min_rtt_us;
	__s32 rtt_gradient_us;
	__u32 loss_ppm;
	__u32 inflight_pkts;
	__u32 cwnd_pkts;
	__u32 bdp_pkts;
	__u32 app_limited;
	__u32 last_action_q10;
	__u32 last_action_source;
	__u32 daemon_connected;
	__u64 capacity_series_bps[CREO_DRL_CAP_WINDOW];
} __attribute__((packed));

/* User-to-kernel action. A zero target rate/cwnd asks the kernel to derive it. */
struct creo_drl_action_msg {
	__u32 abi_version;
	__u32 message_size;
	__u64 flow_id;
	__u64 state_sequence;
	__u64 produced_ns;
	__u64 target_rate_bps;
	__u32 action_index;
	__u32 action_q10;
	__u32 target_cwnd_pkts;
	__u32 valid_for_ms;
} __attribute__((packed));

#endif /* CREO_DRL_UAPI_H */
