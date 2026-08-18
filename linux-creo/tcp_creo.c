// SPDX-License-Identifier: GPL-2.0
/*
 * CREO connected-phase TCP congestion control for Linux.
 *
 * This is an out-of-tree congestion-control module.  It maps Linux TCP
 * delivery-rate samples into the CREO DRL state contract, exchanges telemetry
 * and actions with a user-space model daemon, and applies the action through
 * pacing rate and cwnd without blocking the TCP ACK path.
 */

#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/poll.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <net/inet_connection_sock.h>
#include <net/sock.h>
#include <net/tcp.h>

#include "creo_drl_uapi.h"

#define CREO_SCALE_Q10 1024U
#define CREO_CAP_WINDOW 7U
#define CREO_MIN_CWND 2U
#define CREO_DEFAULT_MIN_RTT_US 1000U
#define CREO_DEFAULT_RATE_BPS 10000000ULL
#define CREO_MAX_RATE_BPS_DEFAULT 10000000000ULL
#define CREO_DRL_STATE_RING_SIZE 256U
#define CREO_DRL_ACTION_SLOTS 256U
#define CREO_DRL_MIN_ACTION_Q10 (CREO_SCALE_Q10 / 2U)
#define CREO_DRL_MAX_ACTION_Q10 (CREO_SCALE_Q10 * 2U)
#define CREO_DRL_DEFAULT_VALID_MS 1000U
#define CREO_DRL_MAX_VALID_MS 5000U

/*
 * Same style as the ns-3 agent action space:
 * [0.61, 0.85, 0.95, 1.0, 1.03, 1.27, 1.67] in Q10 fixed point.
 */
static const u32 creo_action_table_q10[] = {
	625, 870, 973, 1024, 1055, 1300, 1710,
};

static unsigned int action_index = 3;
static unsigned int action_gain_q10;
static unsigned int probe_cycle_steps;
static unsigned int probe_gain_q10 = 1280;
static unsigned int cruise_gain_q10 = 799;
static unsigned int update_interval_us = 100000;
static unsigned long max_rate_bps = CREO_MAX_RATE_BPS_DEFAULT;
static bool control_cwnd = true;
static bool drl_enabled;
static bool debug;

module_param(action_index, uint, 0644);
MODULE_PARM_DESC(action_index,
		 "fixed CREO action index: 0=0.61, 1=0.85, 2=0.95, 3=1.0, 4=1.03, 5=1.27, 6=1.67");

module_param(action_gain_q10, uint, 0644);
MODULE_PARM_DESC(action_gain_q10,
		 "optional fixed-point model action override; zero uses action_index");

module_param(probe_cycle_steps, uint, 0644);
MODULE_PARM_DESC(probe_cycle_steps,
		 "enable one probe action every N model steps; zero disables cycling");

module_param(probe_gain_q10, uint, 0644);
MODULE_PARM_DESC(probe_gain_q10, "Q10 gain used by the periodic probe action");

module_param(cruise_gain_q10, uint, 0644);
MODULE_PARM_DESC(cruise_gain_q10, "Q10 gain used between periodic probes");

module_param(update_interval_us, uint, 0644);
MODULE_PARM_DESC(update_interval_us, "minimum microseconds between CREO model evaluations");

module_param(max_rate_bps, ulong, 0644);
MODULE_PARM_DESC(max_rate_bps, "maximum pacing rate, in bit/s");

module_param(control_cwnd, bool, 0644);
MODULE_PARM_DESC(control_cwnd, "also update snd_cwnd from the model pacing-rate action");

module_param(drl_enabled, bool, 0644);
MODULE_PARM_DESC(drl_enabled, "exchange per-flow state/actions with /dev/creo_drl");

module_param(debug, bool, 0644);
MODULE_PARM_DESC(debug, "print sampled CREO features and actions");

struct creo_features {
	u64 throughput_bps;
	u64 capacity_bps;
	u32 srtt_us;
	u32 min_rtt_us;
	s32 rtt_gradient_us;
	u32 loss_ppm;
	u32 inflight_pkts;
	u32 cwnd_pkts;
	u64 pacing_rate_bps;
	u32 bdp_pkts;
	u32 app_limited;
	u32 last_action_q10;
	u64 capacity_series_bps[CREO_CAP_WINDOW];
};

struct creo_action {
	u32 action_q10;
	u64 target_rate_bps;
	u32 target_cwnd_pkts;
};

struct creo {
	u64 last_update_us;
	u64 flow_id;
	u32 min_rtt_us;
	u32 last_rtt_us;
	u32 last_action_q10;
	u32 control_step;
	u32 cap_head;
	u32 last_model_sequence;
	u32 last_action_source;
	u64 cap_ring[CREO_CAP_WINDOW];
};

struct creo_drl_action_slot {
	u64 flow_id;
	u64 state_sequence;
	u64 expires_ns;
	u64 target_rate_bps;
	u32 action_index;
	u32 action_q10;
	u32 target_cwnd_pkts;
	bool valid;
};

static struct creo_drl_state_msg creo_state_ring[CREO_DRL_STATE_RING_SIZE];
static u32 creo_state_head;
static u32 creo_state_tail;
static u32 creo_state_count;
static DEFINE_SPINLOCK(creo_state_lock);
static DECLARE_WAIT_QUEUE_HEAD(creo_state_wait);

static struct creo_drl_action_slot creo_action_slots[CREO_DRL_ACTION_SLOTS];
static DEFINE_SPINLOCK(creo_action_lock);
static atomic_t creo_daemon_open = ATOMIC_INIT(0);

static atomic64_t creo_states_emitted = ATOMIC64_INIT(0);
static atomic64_t creo_states_dropped = ATOMIC64_INIT(0);
static atomic64_t creo_actions_received = ATOMIC64_INIT(0);
static atomic64_t creo_actions_applied = ATOMIC64_INIT(0);
static atomic64_t creo_fallback_actions = ATOMIC64_INIT(0);
static atomic64_t creo_next_flow_id = ATOMIC64_INIT(0);

static u64 creo_now_us(void)
{
	return div_u64(ktime_get_ns(), NSEC_PER_USEC);
}

static bool creo_drl_online(void)
{
	return READ_ONCE(drl_enabled) && atomic_read(&creo_daemon_open);
}

static struct creo_drl_action_slot *
creo_drl_find_action_locked(u64 flow_id, bool create, u64 now_ns)
{
	struct creo_drl_action_slot *candidate = NULL;
	u32 start = (u32)(flow_id ^ (flow_id >> 32)) &
		(CREO_DRL_ACTION_SLOTS - 1U);
	u32 i;

	for (i = 0; i < CREO_DRL_ACTION_SLOTS; i++) {
		struct creo_drl_action_slot *slot =
			&creo_action_slots[(start + i) &
					  (CREO_DRL_ACTION_SLOTS - 1U)];

		if (slot->valid && slot->flow_id == flow_id)
			return slot;
		if (create && !candidate &&
		    (!slot->valid || slot->expires_ns <= now_ns))
			candidate = slot;
	}

	/* Bounded storage: replacing the hashed slot is preferable to blocking. */
	return create ? (candidate ?: &creo_action_slots[start]) : NULL;
}

static int creo_drl_store_action(const struct creo_drl_action_msg *message)
{
	struct creo_drl_action_slot *slot;
	unsigned long flags;
	u32 valid_for_ms;
	u64 now_ns = ktime_get_ns();

	if (message->abi_version != CREO_DRL_ABI_VERSION ||
	    message->message_size != sizeof(*message) || !message->flow_id)
		return -EPROTO;
	if (message->action_q10 < CREO_DRL_MIN_ACTION_Q10 ||
	    message->action_q10 > CREO_DRL_MAX_ACTION_Q10 ||
	    message->action_index >= ARRAY_SIZE(creo_action_table_q10))
		return -ERANGE;

	valid_for_ms = message->valid_for_ms ?: CREO_DRL_DEFAULT_VALID_MS;
	valid_for_ms = min_t(u32, valid_for_ms, CREO_DRL_MAX_VALID_MS);

	spin_lock_irqsave(&creo_action_lock, flags);
	slot = creo_drl_find_action_locked(message->flow_id, true, now_ns);
	if (slot->valid && slot->flow_id == message->flow_id &&
	    message->state_sequence < slot->state_sequence) {
		spin_unlock_irqrestore(&creo_action_lock, flags);
		return -ESTALE;
	}

	slot->flow_id = message->flow_id;
	slot->state_sequence = message->state_sequence;
	slot->expires_ns = now_ns + (u64)valid_for_ms * NSEC_PER_MSEC;
	slot->target_rate_bps = message->target_rate_bps;
	slot->action_index = message->action_index;
	slot->action_q10 = message->action_q10;
	slot->target_cwnd_pkts = message->target_cwnd_pkts;
	slot->valid = true;
	spin_unlock_irqrestore(&creo_action_lock, flags);

	atomic64_inc(&creo_actions_received);
	return 0;
}

static bool creo_drl_take_action(u64 flow_id, u64 current_sequence,
				 struct creo_action *action,
				 u64 *action_sequence)
{
	struct creo_drl_action_slot *slot;
	unsigned long flags;
	u64 now_ns = ktime_get_ns();
	bool found = false;

	if (!creo_drl_online())
		return false;

	spin_lock_irqsave(&creo_action_lock, flags);
	slot = creo_drl_find_action_locked(flow_id, false, now_ns);
	if (slot && slot->valid) {
		if (slot->expires_ns <= now_ns) {
			slot->valid = false;
		} else if (slot->state_sequence <= current_sequence) {
			action->action_q10 = slot->action_q10;
			action->target_rate_bps = slot->target_rate_bps;
			action->target_cwnd_pkts = slot->target_cwnd_pkts;
			*action_sequence = slot->state_sequence;
			found = true;
		}
	}
	spin_unlock_irqrestore(&creo_action_lock, flags);

	if (found)
		atomic64_inc(&creo_actions_applied);
	return found;
}

static void creo_drl_remove_action(u64 flow_id)
{
	struct creo_drl_action_slot *slot;
	unsigned long flags;

	spin_lock_irqsave(&creo_action_lock, flags);
	slot = creo_drl_find_action_locked(flow_id, false, ktime_get_ns());
	if (slot)
		slot->valid = false;
	spin_unlock_irqrestore(&creo_action_lock, flags);
}

static void creo_drl_clear_actions(void)
{
	unsigned long flags;

	spin_lock_irqsave(&creo_action_lock, flags);
	memset(creo_action_slots, 0, sizeof(creo_action_slots));
	spin_unlock_irqrestore(&creo_action_lock, flags);
}

static void creo_drl_emit_state(const struct creo_features *features,
				const struct creo *ca, u64 sequence)
{
	struct creo_drl_state_msg message = { 0 };
	unsigned long flags;
	u32 i;

	if (!creo_drl_online())
		return;

	message.abi_version = CREO_DRL_ABI_VERSION;
	message.message_size = sizeof(message);
	message.flow_id = ca->flow_id;
	message.sequence = sequence;
	message.timestamp_ns = ktime_get_ns();
	message.throughput_bps = features->throughput_bps;
	message.capacity_bps = features->capacity_bps;
	message.pacing_rate_bps = features->pacing_rate_bps;
	message.last_action_sequence = ca->last_model_sequence;
	message.srtt_us = features->srtt_us;
	message.min_rtt_us = features->min_rtt_us;
	message.rtt_gradient_us = features->rtt_gradient_us;
	message.loss_ppm = features->loss_ppm;
	message.inflight_pkts = features->inflight_pkts;
	message.cwnd_pkts = features->cwnd_pkts;
	message.bdp_pkts = features->bdp_pkts;
	message.app_limited = features->app_limited;
	message.last_action_q10 = ca->last_action_q10;
	message.last_action_source = ca->last_action_source;
	message.daemon_connected = atomic_read(&creo_daemon_open);
	for (i = 0; i < CREO_CAP_WINDOW; i++)
		message.capacity_series_bps[i] = features->capacity_series_bps[i];

	spin_lock_irqsave(&creo_state_lock, flags);
	if (creo_state_count == CREO_DRL_STATE_RING_SIZE) {
		creo_state_tail = (creo_state_tail + 1U) % CREO_DRL_STATE_RING_SIZE;
		creo_state_count--;
		atomic64_inc(&creo_states_dropped);
	}
	creo_state_ring[creo_state_head] = message;
	creo_state_head = (creo_state_head + 1U) % CREO_DRL_STATE_RING_SIZE;
	creo_state_count++;
	spin_unlock_irqrestore(&creo_state_lock, flags);

	atomic64_inc(&creo_states_emitted);
	wake_up_interruptible(&creo_state_wait);
}

static int creo_drl_device_open(struct inode *inode, struct file *file)
{
	unsigned long flags;

	if (atomic_cmpxchg(&creo_daemon_open, 0, 1))
		return -EBUSY;

	spin_lock_irqsave(&creo_state_lock, flags);
	creo_state_head = 0;
	creo_state_tail = 0;
	creo_state_count = 0;
	spin_unlock_irqrestore(&creo_state_lock, flags);
	creo_drl_clear_actions();

	return nonseekable_open(inode, file);
}

static int creo_drl_device_release(struct inode *inode, struct file *file)
{
	creo_drl_clear_actions();
	atomic_set(&creo_daemon_open, 0);
	return 0;
}

static ssize_t creo_drl_device_read(struct file *file, char __user *buffer,
				    size_t length, loff_t *offset)
{
	struct creo_drl_state_msg message;
	unsigned long flags;
	int error;

	if (length < sizeof(message))
		return -EMSGSIZE;

	for (;;) {
		spin_lock_irqsave(&creo_state_lock, flags);
		if (creo_state_count) {
			message = creo_state_ring[creo_state_tail];
			creo_state_tail =
				(creo_state_tail + 1U) % CREO_DRL_STATE_RING_SIZE;
			creo_state_count--;
			spin_unlock_irqrestore(&creo_state_lock, flags);
			break;
		}
		spin_unlock_irqrestore(&creo_state_lock, flags);

		if (file->f_flags & O_NONBLOCK)
			return -EAGAIN;
		error = wait_event_interruptible(creo_state_wait,
					     READ_ONCE(creo_state_count) > 0);
		if (error)
			return error;
	}

	if (copy_to_user(buffer, &message, sizeof(message)))
		return -EFAULT;
	return sizeof(message);
}

static ssize_t creo_drl_device_write(struct file *file,
				     const char __user *buffer,
				     size_t length, loff_t *offset)
{
	struct creo_drl_action_msg message;
	int error;

	if (length != sizeof(message))
		return -EMSGSIZE;
	if (copy_from_user(&message, buffer, sizeof(message)))
		return -EFAULT;

	error = creo_drl_store_action(&message);
	return error ? error : sizeof(message);
}

static __poll_t creo_drl_device_poll(struct file *file, poll_table *wait)
{
	__poll_t mask = EPOLLOUT | EPOLLWRNORM;

	poll_wait(file, &creo_state_wait, wait);
	if (READ_ONCE(creo_state_count))
		mask |= EPOLLIN | EPOLLRDNORM;
	return mask;
}

static const struct file_operations creo_drl_fops = {
	.owner = THIS_MODULE,
	.open = creo_drl_device_open,
	.release = creo_drl_device_release,
	.read = creo_drl_device_read,
	.write = creo_drl_device_write,
	.poll = creo_drl_device_poll,
	.llseek = noop_llseek,
};

static struct miscdevice creo_drl_device = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "creo_drl",
	.fops = &creo_drl_fops,
	.mode = 0600,
};

static u32 creo_safe_mss(const struct tcp_sock *tp)
{
	return max_t(u32, tp->mss_cache, 1U);
}

static u64 creo_bytes_to_bits_rate(u64 bytes_per_sec)
{
	return bytes_per_sec * 8ULL;
}

static u64 creo_bits_to_bytes_rate(u64 bits_per_sec)
{
	return max_t(u64, div64_u64(bits_per_sec, 8ULL), 1ULL);
}

static u64 creo_clamp_bps(u64 rate_bps)
{
	u64 max_bps = max_t(u64, max_rate_bps, 1ULL);

	if (!rate_bps)
		rate_bps = CREO_DEFAULT_RATE_BPS;
	return clamp_t(u64, rate_bps, 1ULL, max_bps);
}

static void creo_record_capacity(struct creo *ca, u64 throughput_bps)
{
	if (!throughput_bps)
		return;
	ca->cap_ring[ca->cap_head++ % CREO_CAP_WINDOW] = throughput_bps;
}

static u64 creo_capacity_estimate(const struct creo *ca, u64 throughput_bps)
{
	u64 capacity_bps = max_t(u64, throughput_bps, CREO_DEFAULT_RATE_BPS);
	u32 i;

	for (i = 0; i < CREO_CAP_WINDOW; i++)
		capacity_bps = max(capacity_bps, ca->cap_ring[i]);

	return capacity_bps;
}

static u32 creo_bdp_packets(const struct tcp_sock *tp, u64 rate_bps, u32 rtt_us)
{
	u64 bytes;
	u32 mss = creo_safe_mss(tp);

	if (!rtt_us)
		rtt_us = CREO_DEFAULT_MIN_RTT_US;

	bytes = div64_u64(rate_bps * rtt_us, USEC_PER_SEC * 8ULL);
	return max_t(u32, div64_u64(bytes, mss), CREO_MIN_CWND);
}

static u32 creo_loss_ppm(const struct rate_sample *rs)
{
	u32 delivered;
	u32 losses;

	if (!rs || rs->delivered <= 0)
		return 0;

	delivered = rs->delivered;
	losses = max_t(int, rs->losses, 0);
	if (!losses)
		return 0;

	return min_t(u32,
		     div64_u64((u64)losses * 1000000ULL, delivered + losses),
		     1000000U);
}

static u64 creo_delivery_rate_bps(const struct tcp_sock *tp,
				  const struct rate_sample *rs)
{
	u64 bytes;

	if (!rs || rs->delivered <= 0 || rs->interval_us <= 0)
		return 0;

	bytes = (u64)rs->delivered * creo_safe_mss(tp);
	return div64_u64(bytes * USEC_PER_SEC * 8ULL, rs->interval_us);
}

static void creo_build_features(struct sock *sk, const struct rate_sample *rs,
				struct creo_features *features)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct creo *ca = inet_csk_ca(sk);
	u64 throughput_bps = creo_delivery_rate_bps(tp, rs);
	u32 rtt_us = 0;
	u32 i;

	memset(features, 0, sizeof(*features));

	if (rs && rs->rtt_us > 0)
		rtt_us = rs->rtt_us;
	else if (tp->srtt_us)
		rtt_us = tp->srtt_us >> 3;

	if (!rtt_us)
		rtt_us = ca->last_rtt_us ?: CREO_DEFAULT_MIN_RTT_US;

	if (!ca->min_rtt_us || rtt_us < ca->min_rtt_us)
		ca->min_rtt_us = rtt_us;

	creo_record_capacity(ca, throughput_bps);

	features->throughput_bps = throughput_bps;
	features->capacity_bps = creo_capacity_estimate(ca, throughput_bps);
	features->srtt_us = rtt_us;
	features->min_rtt_us = ca->min_rtt_us ?: rtt_us;
	features->rtt_gradient_us = (s32)rtt_us - (s32)ca->last_rtt_us;
	features->loss_ppm = creo_loss_ppm(rs);
	features->inflight_pkts = tcp_packets_in_flight(tp);
	features->cwnd_pkts = tcp_snd_cwnd(tp);
	features->pacing_rate_bps = creo_bytes_to_bits_rate(READ_ONCE(sk->sk_pacing_rate));
	features->bdp_pkts = creo_bdp_packets(tp, features->capacity_bps,
					       features->min_rtt_us);
	features->app_limited = rs && rs->is_app_limited;
	features->last_action_q10 = ca->last_action_q10 ?: CREO_SCALE_Q10;

	for (i = 0; i < CREO_CAP_WINDOW; i++)
		features->capacity_series_bps[i] = ca->cap_ring[i];

	ca->last_rtt_us = rtt_us;
}

/* Safe local policy used while the model daemon is absent or an action expires. */
static void creo_fallback_infer(const struct creo_features *features,
				const struct creo *ca,
				struct creo_action *action)
{
	u32 idx = min_t(u32, action_index, ARRAY_SIZE(creo_action_table_q10) - 1);

	memset(action, 0, sizeof(*action));
	if (probe_cycle_steps) {
		action->action_q10 = ca->control_step % probe_cycle_steps == 0 ?
			probe_gain_q10 : cruise_gain_q10;
		action->action_q10 = clamp_t(u32, action->action_q10,
					     CREO_SCALE_Q10 / 2,
					     CREO_SCALE_Q10 * 2);
	} else {
		action->action_q10 = action_gain_q10 ?
			clamp_t(u32, action_gain_q10, CREO_SCALE_Q10 / 2,
				CREO_SCALE_Q10 * 2) : creo_action_table_q10[idx];
	}

	/*
	 * Connected phase only: derive target rate from capacity estimate and
	 * the selected DRL action.  A deployed model may set target_cwnd_pkts as
	 * well; zero means "derive cwnd from BDP".
	 */
	action->target_rate_bps = div64_u64(features->capacity_bps *
					    max_t(u32, action->action_q10, 1U),
					    CREO_SCALE_Q10);
}

static void creo_apply_action(struct sock *sk,
			      const struct creo_features *features,
			      const struct creo_action *action)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct creo *ca = inet_csk_ca(sk);
	u64 target_bps = creo_clamp_bps(action->target_rate_bps);
	u64 target_bytes_per_sec = creo_bits_to_bytes_rate(target_bps);
	u32 target_cwnd = action->target_cwnd_pkts;

	WRITE_ONCE(sk->sk_pacing_rate, (unsigned long)target_bytes_per_sec);
	WRITE_ONCE(sk->sk_pacing_status, SK_PACING_NEEDED);

	if (control_cwnd) {
		if (!target_cwnd)
			target_cwnd = creo_bdp_packets(tp, target_bps,
						       features->min_rtt_us);
		target_cwnd = clamp_t(u32, target_cwnd, CREO_MIN_CWND,
				      max_t(u32, tp->snd_cwnd_clamp,
					    CREO_MIN_CWND));
		tcp_snd_cwnd_set(tp, target_cwnd);
	}

	ca->last_action_q10 = action->action_q10 ?: CREO_SCALE_Q10;

	if (debug) {
		pr_info("creo: tp=%llu cap=%llu rtt=%u minrtt=%u grad=%d loss=%u inflight=%u cwnd=%u action=%u target=%llu source=%u seq=%u\n",
			features->throughput_bps, features->capacity_bps,
			features->srtt_us, features->min_rtt_us,
			features->rtt_gradient_us, features->loss_ppm,
			features->inflight_pkts, tcp_snd_cwnd(tp),
			ca->last_action_q10, target_bps,
			ca->last_action_source, ca->last_model_sequence);
	}
}

static void creo_update(struct sock *sk, const struct rate_sample *rs)
{
	struct creo *ca = inet_csk_ca(sk);
	struct creo_features features;
	struct creo_action action = { 0 };
	u64 now_us = creo_now_us();
	u64 sequence;
	u64 action_sequence = 0;
	u64 flow_id;

	if (rs && (rs->delivered <= 0 || rs->interval_us <= 0))
		return;

	if (ca->last_update_us &&
	    now_us - ca->last_update_us < update_interval_us)
		return;

	sequence = (u64)ca->control_step + 1ULL;
	flow_id = ca->flow_id;
	creo_build_features(sk, rs, &features);
	if (creo_drl_take_action(flow_id, sequence, &action,
				 &action_sequence)) {
		if (!action.target_rate_bps)
			action.target_rate_bps =
				div64_u64(features.capacity_bps *
					  max_t(u32, action.action_q10, 1U),
					  CREO_SCALE_Q10);
		ca->last_model_sequence = min_t(u64, action_sequence, U32_MAX);
		ca->last_action_source = CREO_DRL_ACTION_MODEL;
	} else {
		creo_fallback_infer(&features, ca, &action);
		ca->last_model_sequence = 0;
		ca->last_action_source = CREO_DRL_ACTION_FALLBACK;
		atomic64_inc(&creo_fallback_actions);
	}
	creo_apply_action(sk, &features, &action);
	ca->control_step = (u32)sequence;
	ca->last_update_us = now_us;
	creo_drl_emit_state(&features, ca, sequence);
}

static void creo_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct creo *ca = inet_csk_ca(sk);

	memset(ca, 0, sizeof(*ca));
	ca->flow_id = (u64)atomic64_inc_return(&creo_next_flow_id);
	ca->last_action_q10 = CREO_SCALE_Q10;

	if (!READ_ONCE(sk->sk_pacing_rate))
		WRITE_ONCE(sk->sk_pacing_rate,
			   (unsigned long)creo_bits_to_bytes_rate(CREO_DEFAULT_RATE_BPS));
	WRITE_ONCE(sk->sk_pacing_status, SK_PACING_NEEDED);

	tp->snd_ssthresh = TCP_INFINITE_SSTHRESH;
}

static void creo_release(struct sock *sk)
{
	struct creo *ca = inet_csk_ca(sk);

	creo_drl_remove_action(ca->flow_id);
}

static u32 creo_ssthresh(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);

	return max_t(u32, tcp_snd_cwnd(tp) >> 1U, CREO_MIN_CWND);
}

static void creo_cong_control(struct sock *sk, u32 ack, int flag,
			      const struct rate_sample *rs)
{
	creo_update(sk, rs);
}

static struct tcp_congestion_ops tcp_creo __read_mostly = {
	.init = creo_init,
	.ssthresh = creo_ssthresh,
	.cong_control = creo_cong_control,
	.undo_cwnd = tcp_reno_undo_cwnd,
	.flags = TCP_CONG_NON_RESTRICTED,
	.owner = THIS_MODULE,
	.name = "creo",
	.release = creo_release,
};

static int __init creo_register(void)
{
	int error;

	BUILD_BUG_ON(sizeof(struct creo) > ICSK_CA_PRIV_SIZE);
	BUILD_BUG_ON(sizeof(struct creo_drl_state_msg) !=
		     CREO_DRL_STATE_MESSAGE_SIZE);
	BUILD_BUG_ON(sizeof(struct creo_drl_action_msg) !=
		     CREO_DRL_ACTION_MESSAGE_SIZE);

	error = misc_register(&creo_drl_device);
	if (error)
		return error;

	error = tcp_register_congestion_control(&tcp_creo);
	if (error) {
		misc_deregister(&creo_drl_device);
		return error;
	}

	pr_info("creo: registered connected-phase DRL TCP CC with /dev/%s\n",
		creo_drl_device.name);
	return 0;
}

static void __exit creo_unregister(void)
{
	tcp_unregister_congestion_control(&tcp_creo);
	misc_deregister(&creo_drl_device);
	pr_info("creo: unregistered states=%lld dropped=%lld actions=%lld applied=%lld fallback=%lld\n",
		atomic64_read(&creo_states_emitted),
		atomic64_read(&creo_states_dropped),
		atomic64_read(&creo_actions_received),
		atomic64_read(&creo_actions_applied),
		atomic64_read(&creo_fallback_actions));
}

module_init(creo_register);
module_exit(creo_unregister);

MODULE_AUTHOR("Yuanxin Yan / CREO");
MODULE_DESCRIPTION("CREO+ connected-phase online DRL TCP congestion control");
MODULE_LICENSE("GPL");
