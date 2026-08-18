/*
 * Copyright (c) 2026
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#ifndef TCP_LEOCC_H
#define TCP_LEOCC_H

#include "tcp-congestion-ops.h"
#include "windowed-filter.h"

#include "ns3/data-rate.h"
#include "ns3/traced-value.h"

#include <deque>

namespace ns3
{

/**
 * \ingroup congestionOps
 *
 * \brief An ns-3 implementation of the LeoCC SIGCOMM'25 sender.
 *
 * It implements Startup, Drain, Dynamic Cruise, periodic reconfiguration
 * adaptation, and ProbeRTT.  The simulation-version reconfiguration timer mirrors
 * the public LeoCC implementation used with LeoReplayer.
 */
class TcpLeoCC : public TcpCongestionOps
{
  public:
    static const uint8_t GAIN_CYCLE_LENGTH = 8;
    static const double PACING_GAIN_CYCLE[GAIN_CYCLE_LENGTH];

    static TypeId GetTypeId();

    TcpLeoCC();
    TcpLeoCC(const TcpLeoCC& sock);
    ~TcpLeoCC() override;

    std::string GetName() const override;
    void Init(Ptr<TcpSocketState> tcb) override;
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    bool HasCongControl() const override;
    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs) override;
    Ptr<TcpCongestionOps> Fork() override;

    /** Notify LeoCC of an RI detector event shared by the terminal. */
    void NotifyReconfiguration();

  private:
    enum LeoCcMode_t
    {
        LEOCC_STARTUP,
        LEOCC_DRAIN,
        LEOCC_DYNAMIC_CRUISE,
        LEOCC_PROBE_RTT,
    };

    typedef WindowedFilter<DataRate,
                           MaxFilter<DataRate>,
                           uint32_t,
                           uint32_t>
        MaxBandwidthFilter_t;
    void InitializeModel(Ptr<TcpSocketState> tcb);
    void EnterStartup();
    void EnterDrain();
    void EnterDynamicCruise();
    void EnterProbeRtt(Ptr<TcpSocketState> tcb, bool reconfiguration);
    void InitRoundCounting();
    void InitPacingRate(Ptr<TcpSocketState> tcb);
    void UpdateRound(const TcpRateOps::TcpRateConnection& rc, const TcpRateOps::TcpRateSample& rs);
    void UpdateBandwidthModel(const TcpRateOps::TcpRateSample& rs);
    void UpdateAckAggregation(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs);
    void UpdateRttModel(const Time& rtt);
    void CheckStartupExit(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs);
    void CheckDrain(Ptr<TcpSocketState> tcb);
    void CheckReconfiguration(Ptr<TcpSocketState> tcb);
    void CheckProbeRtt(Ptr<TcpSocketState> tcb);
    void ExitProbeRtt(Ptr<TcpSocketState> tcb);
    void UpdateCruisePhase();
    void UpdateTargetRate();
    void UpdateControlParameters(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs);
    void SetPacingRate(Ptr<TcpSocketState> tcb, double gain);
    void SetSendQuantum(Ptr<TcpSocketState> tcb);
    void SetCwnd(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs);
    uint32_t InFlight(Ptr<const TcpSocketState> tcb, double gain) const;
    uint32_t AckAggregationCwnd() const;
    uint64_t EwmaBitsPerSecond(uint64_t current, uint64_t sample, double alpha) const;
    Time EwmaTime(Time current, Time sample, double alpha) const;
    void PruneRttSamples(Time now);

    struct RttSample
    {
        Time timestamp;
        Time rtt;
    };

    uint32_t m_bandwidthWindowLength;
    Time m_rttWindowLength;
    Time m_rttCongestionThreshold;
    Time m_minRttFilterLength;
    Time m_probeRttDuration;
    Time m_reconfigurationInterval;
    Time m_reconfigurationOffset;
    Time m_bandwidthGuardInterval;
    double m_startupGain;
    double m_burstGain;
    double m_drainGain;
    double m_cWndGain;
    double m_moderateBwAlpha;
    double m_latestRttAlpha;
    double m_probeRttCwndGain;
    uint32_t m_minPipeCwndSegments;
    bool m_enableReconfiguration;
    Ptr<TcpSocketState> m_tcb;

    LeoCcMode_t m_mode;
    bool m_isInitialized;
    bool m_roundStart;
    bool m_hasSeenRtt;
    bool m_isPipeFilled;
    bool m_reconfigurationTriggered;
    bool m_packetConservation;
    bool m_lossRecovery;
    TcpSocketState::TcpCongState_t m_previousCongState;
    uint32_t m_roundCount;
    uint32_t m_cycleIndex;
    uint32_t m_fullBandwidthCount;
    uint64_t m_nextRoundDelivered;
    uint64_t m_delivered;
    uint32_t m_sendQuantum;
    uint32_t m_targetCWnd;
    uint32_t m_priorCWnd;
    uint32_t m_extraAcked[2];
    uint32_t m_extraAckedWinRtt;
    uint32_t m_extraAckedIndex;
    uint32_t m_ackEpochAcked;
    DataRate m_fullBandwidth;
    DataRate m_targetRate;
    DataRate m_moderateBw;
    DataRate m_roundMaxBw;
    DataRate m_reconfigurationMaxBw;
    DataRate m_recoveryBw;
    DataRate m_lastHealthyBw;
    Time m_latestRtt;
    Time m_minRtt;
    Time m_rttLow;
    Time m_rttHigh;
    Time m_startTime;
    Time m_nextReconfiguration;
    Time m_probeRttDoneStamp;
    Time m_minRttStamp;
    Time m_lastHealthyBwStamp;
    Time m_ackEpochTime;

    MaxBandwidthFilter_t m_maxBwFilter;
    std::deque<RttSample> m_rttSamples;

    TracedValue<DataRate> m_aggressiveBwTrace;
    TracedValue<DataRate> m_moderateBwTrace;
    TracedValue<DataRate> m_targetRateTrace;
    TracedValue<Time> m_rttLowTrace;
    TracedValue<Time> m_rttHighTrace;
    TracedValue<Time> m_latestRttTrace;
    TracedValue<uint32_t> m_modeTrace;
};

} // namespace ns3

#endif // TCP_LEOCC_H
