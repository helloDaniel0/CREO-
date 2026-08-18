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

#include "tcp-leocc.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpLeoCC");
NS_OBJECT_ENSURE_REGISTERED(TcpLeoCC);

const double TcpLeoCC::PACING_GAIN_CYCLE[TcpLeoCC::GAIN_CYCLE_LENGTH] = {
    1.25,
    0.75,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
};

TypeId
TcpLeoCC::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpLeoCC")
            .SetParent<TcpCongestionOps>()
            .SetGroupName("Internet")
            .AddConstructor<TcpLeoCC>()
            .AddAttribute("StartupGain",
                          "Pacing gain used during startup.",
                          DoubleValue(2.885),
                          MakeDoubleAccessor(&TcpLeoCC::m_startupGain),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("BurstGain",
                          "Aggressive pacing gain used in the first RTT of a LeoCC cruise cycle.",
                          DoubleValue(1.25),
                          MakeDoubleAccessor(&TcpLeoCC::m_burstGain),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("DrainGain",
                          "Drain pacing gain used in the second RTT of a LeoCC cruise cycle.",
                          DoubleValue(0.75),
                          MakeDoubleAccessor(&TcpLeoCC::m_drainGain),
                          MakeDoubleChecker<double>(0.1, 1.0))
            .AddAttribute("CwndGain",
                          "Multiplier applied to the estimated BDP when deriving target cwnd.",
                          DoubleValue(2.0),
                          MakeDoubleAccessor(&TcpLeoCC::m_cWndGain),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("BwWindowLength",
                          "Round-based window length for aggressive bandwidth estimation.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&TcpLeoCC::m_bandwidthWindowLength),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("RttWindowLength",
                          "Time window used to track LeoCC RTT bands.",
                          TimeValue(Seconds(1)),
                          MakeTimeAccessor(&TcpLeoCC::m_rttWindowLength),
                          MakeTimeChecker())
            .AddAttribute("RttCongestionThreshold",
                          "Filtered RTT increase above minRTT that selects the moderate bandwidth estimator.",
                          TimeValue(MilliSeconds(10)),
                          MakeTimeAccessor(&TcpLeoCC::m_rttCongestionThreshold),
                          MakeTimeChecker())
            .AddAttribute("MinRttFilterLength",
                          "Lifetime of LeoCC's minimum RTT estimate.",
                          TimeValue(Seconds(20)),
                          MakeTimeAccessor(&TcpLeoCC::m_minRttFilterLength),
                          MakeTimeChecker())
            .AddAttribute("ProbeRttDuration",
                          "Time spent at 0.5 BDP while refreshing minRTT.",
                          TimeValue(MilliSeconds(200)),
                          MakeTimeAccessor(&TcpLeoCC::m_probeRttDuration),
                          MakeTimeChecker())
            .AddAttribute("ReconfigurationInterval",
                          "Period of the simulation-version LeoCC reconfiguration signal.",
                          TimeValue(Seconds(15)),
                          MakeTimeAccessor(&TcpLeoCC::m_reconfigurationInterval),
                          MakeTimeChecker())
            .AddAttribute("ReconfigurationOffset",
                          "First reconfiguration time relative to flow initialization.",
                          TimeValue(Seconds(12)),
                          MakeTimeAccessor(&TcpLeoCC::m_reconfigurationOffset),
                          MakeTimeChecker())
            .AddAttribute("BandwidthGuardInterval",
                          "Hold a valid pre-RTO delivery-rate sample while ns-3 TCP recovers.",
                          TimeValue(Seconds(2)),
                          MakeTimeAccessor(&TcpLeoCC::m_bandwidthGuardInterval),
                          MakeTimeChecker())
            .AddAttribute("EnableReconfiguration",
                          "Enable periodic reconfiguration adaptation as in LeoReplayer.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpLeoCC::m_enableReconfiguration),
                          MakeBooleanChecker())
            .AddAttribute("ModerateBwAlpha",
                          "Filter gain of the per-round moderate bandwidth estimator.",
                          DoubleValue(0.6),
                          MakeDoubleAccessor(&TcpLeoCC::m_moderateBwAlpha),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("LatestRttAlpha",
                          "Filter gain of the latest RTT estimator.",
                          DoubleValue(0.6),
                          MakeDoubleAccessor(&TcpLeoCC::m_latestRttAlpha),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("ProbeRttCwndGain",
                          "BDP multiplier used during reconfiguration adaptation and ProbeRTT.",
                          DoubleValue(0.5),
                          MakeDoubleAccessor(&TcpLeoCC::m_probeRttCwndGain),
                          MakeDoubleChecker<double>(0.1, 1.0))
            .AddAttribute("MinPipeCwndSegments",
                          "Minimum congestion window in segments.",
                          UintegerValue(4),
                          MakeUintegerAccessor(&TcpLeoCC::m_minPipeCwndSegments),
                          MakeUintegerChecker<uint32_t>(2))
            .AddTraceSource("AggressiveBw",
                            "LeoCC aggressive bottleneck bandwidth estimate.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_aggressiveBwTrace),
                            "ns3::TracedValueCallback::DataRate")
            .AddTraceSource("ModerateBw",
                            "LeoCC moderate bottleneck bandwidth estimate.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_moderateBwTrace),
                            "ns3::TracedValueCallback::DataRate")
            .AddTraceSource("TargetRate",
                            "LeoCC selected target sending rate.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_targetRateTrace),
                            "ns3::TracedValueCallback::DataRate")
            .AddTraceSource("RttLow",
                            "LeoCC lower RTT band.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_rttLowTrace),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("RttHigh",
                            "LeoCC upper RTT band.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_rttHighTrace),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("LatestRttEstimate",
                            "LeoCC filtered latest RTT estimate.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_latestRttTrace),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("Mode",
                            "LeoCC mode: 0 Startup, 1 Drain, 2 Dynamic Cruise, 3 ProbeRTT.",
                            MakeTraceSourceAccessor(&TcpLeoCC::m_modeTrace),
                            "ns3::TracedValueCallback::Uint32");
    return tid;
}

TcpLeoCC::TcpLeoCC()
    : TcpCongestionOps(),
      m_bandwidthWindowLength(10),
      m_rttWindowLength(Seconds(1)),
      m_rttCongestionThreshold(MilliSeconds(10)),
      m_minRttFilterLength(Seconds(20)),
      m_probeRttDuration(MilliSeconds(200)),
      m_reconfigurationInterval(Seconds(15)),
      m_reconfigurationOffset(Seconds(12)),
      m_bandwidthGuardInterval(Seconds(2)),
      m_startupGain(2.885),
      m_burstGain(1.25),
      m_drainGain(0.75),
      m_cWndGain(2.0),
      m_moderateBwAlpha(0.6),
      m_latestRttAlpha(0.6),
      m_probeRttCwndGain(0.5),
      m_minPipeCwndSegments(4),
      m_enableReconfiguration(true),
      m_tcb(nullptr),
      m_mode(LEOCC_STARTUP),
      m_isInitialized(false),
      m_roundStart(false),
      m_hasSeenRtt(false),
      m_isPipeFilled(false),
      m_reconfigurationTriggered(false),
      m_packetConservation(false),
      m_lossRecovery(false),
      m_previousCongState(TcpSocketState::CA_OPEN),
      m_roundCount(0),
      m_cycleIndex(0),
      m_fullBandwidthCount(0),
      m_nextRoundDelivered(0),
      m_delivered(0),
      m_sendQuantum(0),
      m_targetCWnd(0),
      m_priorCWnd(0),
      m_extraAcked{0, 0},
      m_extraAckedWinRtt(0),
      m_extraAckedIndex(0),
      m_ackEpochAcked(0),
      m_fullBandwidth(0),
      m_targetRate(0),
      m_moderateBw(0),
      m_roundMaxBw(0),
      m_reconfigurationMaxBw(0),
      m_recoveryBw(0),
      m_lastHealthyBw(0),
      m_latestRtt(Time::Max()),
      m_minRtt(Time::Max()),
      m_rttLow(Time::Max()),
      m_rttHigh(Time::Max()),
      m_startTime(Seconds(0)),
      m_nextReconfiguration(Time::Max()),
      m_probeRttDoneStamp(Seconds(0)),
      m_minRttStamp(Seconds(0)),
      m_lastHealthyBwStamp(Seconds(0)),
      m_ackEpochTime(Seconds(0)),
      m_maxBwFilter(10, DataRate(0), 0),
      m_rttSamples(),
      m_aggressiveBwTrace(0),
      m_moderateBwTrace(0),
      m_targetRateTrace(0),
      m_rttLowTrace(Time::Max()),
      m_rttHighTrace(Time(0)),
      m_latestRttTrace(Time::Max()),
      m_modeTrace(LEOCC_STARTUP)
{
    NS_LOG_FUNCTION(this);
}

TcpLeoCC::TcpLeoCC(const TcpLeoCC& sock)
    : TcpCongestionOps(sock),
      m_bandwidthWindowLength(sock.m_bandwidthWindowLength),
      m_rttWindowLength(sock.m_rttWindowLength),
      m_rttCongestionThreshold(sock.m_rttCongestionThreshold),
      m_minRttFilterLength(sock.m_minRttFilterLength),
      m_probeRttDuration(sock.m_probeRttDuration),
      m_reconfigurationInterval(sock.m_reconfigurationInterval),
      m_reconfigurationOffset(sock.m_reconfigurationOffset),
      m_bandwidthGuardInterval(sock.m_bandwidthGuardInterval),
      m_startupGain(sock.m_startupGain),
      m_burstGain(sock.m_burstGain),
      m_drainGain(sock.m_drainGain),
      m_cWndGain(sock.m_cWndGain),
      m_moderateBwAlpha(sock.m_moderateBwAlpha),
      m_latestRttAlpha(sock.m_latestRttAlpha),
      m_probeRttCwndGain(sock.m_probeRttCwndGain),
      m_minPipeCwndSegments(sock.m_minPipeCwndSegments),
      m_enableReconfiguration(sock.m_enableReconfiguration),
      m_tcb(nullptr),
      m_mode(sock.m_mode),
      m_isInitialized(sock.m_isInitialized),
      m_roundStart(sock.m_roundStart),
      m_hasSeenRtt(sock.m_hasSeenRtt),
      m_isPipeFilled(sock.m_isPipeFilled),
      m_reconfigurationTriggered(sock.m_reconfigurationTriggered),
      m_packetConservation(sock.m_packetConservation),
      m_lossRecovery(sock.m_lossRecovery),
      m_previousCongState(sock.m_previousCongState),
      m_roundCount(sock.m_roundCount),
      m_cycleIndex(sock.m_cycleIndex),
      m_fullBandwidthCount(sock.m_fullBandwidthCount),
      m_nextRoundDelivered(sock.m_nextRoundDelivered),
      m_delivered(sock.m_delivered),
      m_sendQuantum(sock.m_sendQuantum),
      m_targetCWnd(sock.m_targetCWnd),
      m_priorCWnd(sock.m_priorCWnd),
      m_extraAcked{sock.m_extraAcked[0], sock.m_extraAcked[1]},
      m_extraAckedWinRtt(sock.m_extraAckedWinRtt),
      m_extraAckedIndex(sock.m_extraAckedIndex),
      m_ackEpochAcked(sock.m_ackEpochAcked),
      m_fullBandwidth(sock.m_fullBandwidth),
      m_targetRate(sock.m_targetRate),
      m_moderateBw(sock.m_moderateBw),
      m_roundMaxBw(sock.m_roundMaxBw),
      m_reconfigurationMaxBw(sock.m_reconfigurationMaxBw),
      m_recoveryBw(sock.m_recoveryBw),
      m_lastHealthyBw(sock.m_lastHealthyBw),
      m_latestRtt(sock.m_latestRtt),
      m_minRtt(sock.m_minRtt),
      m_rttLow(sock.m_rttLow),
      m_rttHigh(sock.m_rttHigh),
      m_startTime(sock.m_startTime),
      m_nextReconfiguration(sock.m_nextReconfiguration),
      m_probeRttDoneStamp(sock.m_probeRttDoneStamp),
      m_minRttStamp(sock.m_minRttStamp),
      m_lastHealthyBwStamp(sock.m_lastHealthyBwStamp),
      m_ackEpochTime(sock.m_ackEpochTime),
      m_maxBwFilter(sock.m_maxBwFilter),
      m_rttSamples(sock.m_rttSamples),
      m_aggressiveBwTrace(sock.m_aggressiveBwTrace),
      m_moderateBwTrace(sock.m_moderateBwTrace),
      m_targetRateTrace(sock.m_targetRateTrace),
      m_rttLowTrace(sock.m_rttLowTrace),
      m_rttHighTrace(sock.m_rttHighTrace),
      m_latestRttTrace(sock.m_latestRttTrace),
      m_modeTrace(sock.m_modeTrace)
{
    NS_LOG_FUNCTION(this);
}

TcpLeoCC::~TcpLeoCC()
{
}

std::string
TcpLeoCC::GetName() const
{
    return "TcpLeoCC";
}

void
TcpLeoCC::Init(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    m_tcb = tcb;
    if (!m_isInitialized)
    {
        InitializeModel(tcb);
    }
}

void
TcpLeoCC::NotifyReconfiguration()
{
    NS_LOG_FUNCTION(this);
    if (!m_tcb)
    {
        return;
    }
    if (!m_isInitialized)
    {
        InitializeModel(m_tcb);
    }
    EnterProbeRtt(m_tcb, true);
}

void
TcpLeoCC::InitializeModel(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    m_maxBwFilter = MaxBandwidthFilter_t(m_bandwidthWindowLength, DataRate(0), 0);
    m_rttSamples.clear();

    m_roundStart = false;
    m_hasSeenRtt = false;
    m_isPipeFilled = false;
    m_reconfigurationTriggered = false;
    m_packetConservation = false;
    m_lossRecovery = false;
    m_previousCongState = TcpSocketState::CA_OPEN;
    m_roundCount = 0;
    m_cycleIndex = 0;
    m_fullBandwidthCount = 0;
    m_nextRoundDelivered = 0;
    m_delivered = 0;
    m_sendQuantum = tcb->m_segmentSize;
    m_targetCWnd = tcb->m_cWnd;
    m_priorCWnd = tcb->m_cWnd;
    m_extraAcked[0] = 0;
    m_extraAcked[1] = 0;
    m_extraAckedWinRtt = 0;
    m_extraAckedIndex = 0;
    m_ackEpochAcked = 0;
    m_fullBandwidth = DataRate(0);
    m_targetRate = DataRate(0);
    m_moderateBw = DataRate(0);
    m_roundMaxBw = DataRate(0);
    m_reconfigurationMaxBw = DataRate(0);
    m_recoveryBw = DataRate(0);
    m_lastHealthyBw = DataRate(0);
    m_latestRtt = tcb->m_lastRtt.Get() > Seconds(0) ? tcb->m_lastRtt.Get() : Time::Max();
    m_minRtt = tcb->m_minRtt;
    m_rttLow = tcb->m_minRtt;
    m_rttHigh = tcb->m_lastRtt.Get() > Seconds(0) ? tcb->m_lastRtt.Get() : Time::Max();
    m_startTime = Simulator::Now();
    m_minRttStamp = m_startTime;
    m_lastHealthyBwStamp = m_startTime;
    m_ackEpochTime = m_startTime;
    m_probeRttDoneStamp = Seconds(0);
    m_nextReconfiguration =
        m_enableReconfiguration ? m_startTime + m_reconfigurationOffset : Time::Max();

    if (!tcb->m_pacing)
    {
        tcb->m_pacing = true;
    }

    InitRoundCounting();
    EnterStartup();
    InitPacingRate(tcb);

    m_isInitialized = true;
}

void
TcpLeoCC::EnterStartup()
{
    NS_LOG_FUNCTION(this);
    m_mode = LEOCC_STARTUP;
    m_modeTrace = m_mode;
}

void
TcpLeoCC::EnterDrain()
{
    NS_LOG_FUNCTION(this);
    m_mode = LEOCC_DRAIN;
    m_modeTrace = m_mode;
}

void
TcpLeoCC::EnterDynamicCruise()
{
    NS_LOG_FUNCTION(this);
    m_mode = LEOCC_DYNAMIC_CRUISE;
    m_cycleIndex = 0;
    m_modeTrace = m_mode;
}

void
TcpLeoCC::EnterProbeRtt(Ptr<TcpSocketState> tcb, bool reconfiguration)
{
    NS_LOG_FUNCTION(this << tcb << reconfiguration);
    if (m_mode == LEOCC_PROBE_RTT)
    {
        m_reconfigurationTriggered = m_reconfigurationTriggered || reconfiguration;
        return;
    }

    m_priorCWnd = tcb->m_cWnd;
    m_probeRttDoneStamp = Seconds(0);
    m_reconfigurationTriggered = reconfiguration;
    m_reconfigurationMaxBw = DataRate(0);
    m_mode = LEOCC_PROBE_RTT;
    m_modeTrace = m_mode;

    const uint32_t probeCwnd =
        std::max(InFlight(tcb, m_probeRttCwndGain), m_minPipeCwndSegments * tcb->m_segmentSize);
    m_targetCWnd = probeCwnd;
    tcb->m_cWnd = std::max(std::min(tcb->m_cWnd.Get(), probeCwnd),
                           m_minPipeCwndSegments * tcb->m_segmentSize);
    SetPacingRate(tcb, 1.0);
}

void
TcpLeoCC::InitRoundCounting()
{
    NS_LOG_FUNCTION(this);
    m_nextRoundDelivered = 0;
    m_roundStart = false;
    m_roundCount = 0;
}

void
TcpLeoCC::InitPacingRate(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    Time rtt = tcb->m_minRtt != Time::Max() ? tcb->m_minRtt : MilliSeconds(1);
    uint32_t initialWindow = std::max(tcb->m_cWnd.Get(),
                                      tcb->m_initialCWnd * tcb->m_segmentSize);
    uint64_t nominalBitsPerSecond =
        std::max<uint64_t>(initialWindow * 8 / std::max(rtt.GetSeconds(), 1e-6), 1);
    DataRate nominalBw(nominalBitsPerSecond);
    tcb->m_pacingRate = nominalBw;
    m_targetRate = nominalBw;
    m_targetRateTrace = nominalBw;
    m_aggressiveBwTrace = nominalBw;
    m_moderateBw = nominalBw;
    m_moderateBwTrace = nominalBw;
}

void
TcpLeoCC::PktsAcked(Ptr<TcpSocketState> tcb [[maybe_unused]],
                    uint32_t segmentsAcked [[maybe_unused]],
                    const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);
    if (rtt.IsZero())
    {
        return;
    }
    UpdateRttModel(rtt);
}

void
TcpLeoCC::UpdateRttModel(const Time& rtt)
{
    NS_LOG_FUNCTION(this << rtt);
    const Time now = Simulator::Now();

    if (!m_hasSeenRtt)
    {
        m_hasSeenRtt = true;
        m_latestRtt = rtt;
        m_minRtt = rtt;
        m_minRttStamp = now;
        m_rttLow = rtt;
        m_rttHigh = rtt;
    }
    else
    {
        m_latestRtt = EwmaTime(m_latestRtt, rtt, m_latestRttAlpha);
        if (m_minRtt == Time::Max() || rtt < m_minRtt)
        {
            m_minRtt = rtt;
            m_minRttStamp = now;
        }
    }

    m_rttSamples.push_back({now, rtt});
    PruneRttSamples(now);

    m_rttLow = Time::Max();
    m_rttHigh = Time(0);
    for (const auto& sample : m_rttSamples)
    {
        m_rttLow = std::min(m_rttLow, sample.rtt);
        m_rttHigh = std::max(m_rttHigh, sample.rtt);
    }

    if (m_rttSamples.empty())
    {
        m_rttLow = rtt;
        m_rttHigh = rtt;
    }

    m_rttLowTrace = m_minRtt;
    m_rttHighTrace = m_rttHigh;
    m_latestRttTrace = m_latestRtt;
}

void
TcpLeoCC::UpdateRound(const TcpRateOps::TcpRateConnection& rc, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << rs);
    if (rs.m_priorDelivered >= m_nextRoundDelivered)
    {
        if (!m_lossRecovery && m_roundMaxBw != DataRate(0))
        {
            if (m_moderateBw == DataRate(0))
            {
                m_moderateBw = m_roundMaxBw;
            }
            else
            {
                m_moderateBw = DataRate(EwmaBitsPerSecond(m_moderateBw.GetBitRate(),
                                                          m_roundMaxBw.GetBitRate(),
                                                          m_moderateBwAlpha));
            }
            m_moderateBwTrace = m_moderateBw;
        }
        m_roundMaxBw = DataRate(0);
        m_nextRoundDelivered = rc.m_delivered;
        m_roundCount++;
        m_roundStart = true;
    }
    else
    {
        m_roundStart = false;
    }
}

void
TcpLeoCC::UpdateBandwidthModel(const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << rs);
    if (rs.m_deliveryRate == 0)
    {
        return;
    }

    const DataRate currentBest = m_maxBwFilter.GetBest();
    const uint64_t healthyReference =
        std::max(currentBest.GetBitRate(), m_lastHealthyBw.GetBitRate());
    const bool guardExpired =
        (Simulator::Now() - m_lastHealthyBwStamp) > m_bandwidthGuardInterval;
    if (healthyReference == 0 || guardExpired ||
        rs.m_deliveryRate.GetBitRate() >= static_cast<uint64_t>(0.75 * healthyReference))
    {
        m_lastHealthyBw = rs.m_deliveryRate;
        m_lastHealthyBwStamp = Simulator::Now();
    }

    if (m_lossRecovery && m_recoveryBw != DataRate(0))
    {
        if (rs.m_deliveryRate.GetBitRate() >=
            static_cast<uint64_t>(0.75 * m_recoveryBw.GetBitRate()))
        {
            m_lossRecovery = false;
        }
        else
        {
            m_aggressiveBwTrace = m_recoveryBw;
            return;
        }
    }

    if (rs.m_deliveryRate >= m_maxBwFilter.GetBest() || !rs.m_isAppLimited)
    {
        m_maxBwFilter.Update(rs.m_deliveryRate, m_roundCount);
    }

    if (rs.m_deliveryRate > m_roundMaxBw)
    {
        m_roundMaxBw = rs.m_deliveryRate;
    }
    if (m_mode == LEOCC_PROBE_RTT && rs.m_deliveryRate > m_reconfigurationMaxBw)
    {
        m_reconfigurationMaxBw = rs.m_deliveryRate;
    }

    m_aggressiveBwTrace = m_maxBwFilter.GetBest();
}

void
TcpLeoCC::UpdateAckAggregation(Ptr<TcpSocketState> tcb,
                               const TcpRateOps::TcpRateSample& rs)
{
    if (!rs.m_ackedSacked || rs.m_delivered < 0)
    {
        return;
    }

    if (m_roundStart)
    {
        m_extraAckedWinRtt = std::min<uint32_t>(31, m_extraAckedWinRtt + 1);
        if (m_extraAckedWinRtt >= 5)
        {
            m_extraAckedWinRtt = 0;
            m_extraAckedIndex = m_extraAckedIndex ? 0 : 1;
            m_extraAcked[m_extraAckedIndex] = 0;
        }
    }

    const double epochSeconds = (Simulator::Now() - m_ackEpochTime).GetSeconds();
    uint64_t expectedAcked =
        static_cast<uint64_t>(m_maxBwFilter.GetBest().GetBitRate() * epochSeconds / 8.0);
    if (m_ackEpochAcked <= expectedAcked || m_ackEpochAcked + rs.m_ackedSacked >= (1U << 20))
    {
        m_ackEpochAcked = 0;
        m_ackEpochTime = Simulator::Now();
        expectedAcked = 0;
    }

    m_ackEpochAcked += rs.m_ackedSacked;
    const uint32_t extraAcked = std::min<uint32_t>(
        m_ackEpochAcked - static_cast<uint32_t>(expectedAcked),
        tcb->m_cWnd.Get());
    m_extraAcked[m_extraAckedIndex] =
        std::max(m_extraAcked[m_extraAckedIndex], extraAcked);
}

void
TcpLeoCC::CheckStartupExit(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (m_mode != LEOCC_STARTUP || !m_roundStart || rs.m_isAppLimited)
    {
        return;
    }

    DataRate best = m_maxBwFilter.GetBest();
    if (best == DataRate(0))
    {
        return;
    }

    if (m_fullBandwidth == DataRate(0) || best.GetBitRate() >= m_fullBandwidth.GetBitRate() * 1.25)
    {
        m_fullBandwidth = best;
        m_fullBandwidthCount = 0;
        return;
    }

    m_fullBandwidthCount++;
    if (m_fullBandwidthCount >= 3)
    {
        m_isPipeFilled = true;
        EnterDrain();
        tcb->m_ssThresh = InFlight(tcb, 1.0);
    }
}

void
TcpLeoCC::CheckDrain(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    if (m_mode == LEOCC_DRAIN && tcb->m_bytesInFlight <= InFlight(tcb, 1.0))
    {
        EnterDynamicCruise();
    }
}

void
TcpLeoCC::CheckReconfiguration(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    const Time now = Simulator::Now();
    if (!m_enableReconfiguration || m_reconfigurationInterval.IsZero() ||
        m_nextReconfiguration == Time::Max() || now < m_nextReconfiguration)
    {
        return;
    }

    while (m_nextReconfiguration <= now)
    {
        m_nextReconfiguration += m_reconfigurationInterval;
    }
    EnterProbeRtt(tcb, true);
}

void
TcpLeoCC::CheckProbeRtt(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    const Time now = Simulator::Now();

    if (m_mode != LEOCC_PROBE_RTT && m_isPipeFilled &&
        (now - m_minRttStamp) >= m_minRttFilterLength)
    {
        EnterProbeRtt(tcb, false);
    }
    if (m_mode != LEOCC_PROBE_RTT)
    {
        return;
    }

    const uint32_t probeCwnd =
        std::max(InFlight(tcb, m_probeRttCwndGain), m_minPipeCwndSegments * tcb->m_segmentSize);
    if (m_probeRttDoneStamp.IsZero() && tcb->m_bytesInFlight <= probeCwnd)
    {
        m_probeRttDoneStamp = now + m_probeRttDuration;
    }
    if (!m_probeRttDoneStamp.IsZero() && now >= m_probeRttDoneStamp)
    {
        ExitProbeRtt(tcb);
    }
}

void
TcpLeoCC::ExitProbeRtt(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    const bool wasReconfiguration = m_reconfigurationTriggered;
    DataRate seed = m_reconfigurationMaxBw;
    if (seed == DataRate(0))
    {
        seed = m_moderateBw != DataRate(0) ? m_moderateBw : m_maxBwFilter.GetBest();
    }

    tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), m_priorCWnd);
    m_minRtt = m_latestRtt;
    m_minRttStamp = Simulator::Now();
    m_probeRttDoneStamp = Seconds(0);
    m_reconfigurationTriggered = false;

    if (wasReconfiguration)
    {
        m_maxBwFilter = MaxBandwidthFilter_t(m_bandwidthWindowLength, seed, m_roundCount);
        m_fullBandwidth = seed;
        m_fullBandwidthCount = 0;
        m_isPipeFilled = true;
        m_moderateBw = seed;
        m_roundMaxBw = DataRate(0);
        m_targetRate = seed;
        EnterDynamicCruise();
    }
    else
    {
        EnterDynamicCruise();
    }
}

void
TcpLeoCC::UpdateCruisePhase()
{
    NS_LOG_FUNCTION(this);
    if (m_mode == LEOCC_DYNAMIC_CRUISE && m_roundStart)
    {
        m_cycleIndex = (m_cycleIndex + 1) % GAIN_CYCLE_LENGTH;
    }
}

void
TcpLeoCC::UpdateTargetRate()
{
    NS_LOG_FUNCTION(this);
    if (m_lossRecovery && m_recoveryBw != DataRate(0))
    {
        m_targetRate = m_recoveryBw;
        m_targetRateTrace = m_targetRate;
        return;
    }

    DataRate aggressive = m_maxBwFilter.GetBest();
    DataRate moderate = m_moderateBw;

    if (m_lastHealthyBw != DataRate(0) &&
        (Simulator::Now() - m_lastHealthyBwStamp) <= m_bandwidthGuardInterval &&
        m_lastHealthyBw > aggressive)
    {
        aggressive = m_lastHealthyBw;
    }

    if (aggressive == DataRate(0))
    {
        aggressive = moderate;
    }
    if (moderate == DataRate(0))
    {
        moderate = aggressive;
    }
    if (aggressive != DataRate(0) && moderate.GetBitRate() < aggressive.GetBitRate() / 2)
    {
        moderate = DataRate(aggressive.GetBitRate() / 2);
    }

    bool useModerate = false;
    if (m_mode == LEOCC_DYNAMIC_CRUISE && !m_reconfigurationTriggered && m_hasSeenRtt &&
        m_minRtt != Time::Max() && m_latestRtt != Time::Max())
    {
        useModerate = m_latestRtt >= (m_minRtt + m_rttCongestionThreshold);
    }

    m_targetRate = useModerate ? moderate : aggressive;
    m_targetRateTrace = m_targetRate;
}

uint32_t
TcpLeoCC::InFlight(Ptr<const TcpSocketState> tcb, double gain) const
{
    Time baseRtt = m_minRtt != Time::Max() ? m_minRtt : tcb->m_minRtt;
    if (baseRtt == Time::Max() || m_targetRate == DataRate(0))
    {
        return std::max(tcb->m_cWnd.Get(), m_minPipeCwndSegments * tcb->m_segmentSize);
    }

    double estimatedBdp = m_targetRate.GetBitRate() * baseRtt.GetSeconds() / 8.0;
    return static_cast<uint32_t>(gain * estimatedBdp) + 3 * m_sendQuantum;
}

uint32_t
TcpLeoCC::AckAggregationCwnd() const
{
    if (!m_isPipeFilled)
    {
        return 0;
    }

    const uint64_t maxAggregationBytes = m_maxBwFilter.GetBest().GetBitRate() / (10 * 8);
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::max(m_extraAcked[0], m_extraAcked[1]),
        maxAggregationBytes));
}

void
TcpLeoCC::SetPacingRate(Ptr<TcpSocketState> tcb, double gain)
{
    NS_LOG_FUNCTION(this << tcb << gain);
    if (m_targetRate == DataRate(0))
    {
        return;
    }
    uint64_t targetRate = std::max<uint64_t>(static_cast<uint64_t>(gain * m_targetRate.GetBitRate()), 1);
    uint64_t maxPacingRate = tcb->m_maxPacingRate.GetBitRate();
    if (maxPacingRate > 0)
    {
        targetRate = std::min(targetRate, maxPacingRate);
    }
    tcb->m_pacingRate = DataRate(std::max<uint64_t>(targetRate, 1));
}

void
TcpLeoCC::SetSendQuantum(Ptr<TcpSocketState> tcb)
{
    m_sendQuantum = std::max<uint32_t>(tcb->m_segmentSize, 1);
}

void
TcpLeoCC::SetCwnd(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (rs.m_bytesLoss > 0)
    {
        tcb->m_cWnd = std::max<int64_t>(
            static_cast<int64_t>(tcb->m_cWnd.Get()) - static_cast<int64_t>(rs.m_bytesLoss),
            tcb->m_segmentSize);
    }
    if (!rs.m_ackedSacked)
    {
        return;
    }

    if (m_packetConservation)
    {
        tcb->m_cWnd =
            std::max(tcb->m_cWnd.Get(), tcb->m_bytesInFlight.Get() + rs.m_ackedSacked);
        return;
    }

    uint32_t minPipeCwnd = m_minPipeCwndSegments * tcb->m_segmentSize;
    double cwndGain = m_cWndGain;
    if (m_mode == LEOCC_STARTUP || m_mode == LEOCC_DRAIN)
    {
        cwndGain = m_startupGain;
    }
    else if (m_mode == LEOCC_PROBE_RTT)
    {
        cwndGain = m_probeRttCwndGain;
    }
    const uint32_t ackAggregation = m_mode == LEOCC_PROBE_RTT ? 0 : AckAggregationCwnd();
    m_targetCWnd = std::max(InFlight(tcb, cwndGain) + ackAggregation, minPipeCwnd);

    if (!m_isPipeFilled)
    {
        if (tcb->m_cWnd < m_targetCWnd || m_delivered < tcb->m_initialCWnd * tcb->m_segmentSize)
        {
            tcb->m_cWnd = tcb->m_cWnd.Get() + rs.m_ackedSacked;
        }
    }
    else
    {
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get() + rs.m_ackedSacked, m_targetCWnd);
    }

    if (m_mode == LEOCC_PROBE_RTT)
    {
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get(), m_targetCWnd);
    }

    tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), minPipeCwnd);
}

void
TcpLeoCC::UpdateControlParameters(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);

    double pacingGain = m_startupGain;
    if (m_mode == LEOCC_DRAIN)
    {
        pacingGain = 1.0 / m_startupGain;
    }
    else if (m_mode == LEOCC_DYNAMIC_CRUISE)
    {
        pacingGain = PACING_GAIN_CYCLE[m_cycleIndex];
        if (m_cycleIndex == 0)
        {
            const bool congestionLikely =
                m_hasSeenRtt && m_minRtt != Time::Max() && m_latestRtt != Time::Max() &&
                m_latestRtt >= (m_minRtt + m_rttCongestionThreshold);
            pacingGain = congestionLikely ? 1.05 : m_burstGain;
        }
        else if (m_cycleIndex == 1)
        {
            pacingGain = m_drainGain;
        }
    }
    else if (m_mode == LEOCC_PROBE_RTT)
    {
        pacingGain = 1.0;
    }

    SetPacingRate(tcb, pacingGain);
    SetSendQuantum(tcb);
    SetCwnd(tcb, rs);
}

void
TcpLeoCC::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);

    if (newState == TcpSocketState::CA_OPEN && !m_isInitialized)
    {
        InitializeModel(tcb);
    }
    else if (newState == TcpSocketState::CA_OPEN &&
             m_previousCongState >= TcpSocketState::CA_RECOVERY)
    {
        // Linux LeoCC restores prior_cwnd when leaving Recovery/Loss. ns-3 does
        // not emit CA_EVENT_COMPLETE_CWR when it leaves CA_LOSS after an RTO.
        tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), m_priorCWnd);
        m_packetConservation = false;
    }
    else if (newState == TcpSocketState::CA_LOSS)
    {
        m_priorCWnd = std::max(m_priorCWnd, tcb->m_cWnd.Get());
        m_recoveryBw = std::max(m_targetRate, m_maxBwFilter.GetBest());
        m_lossRecovery = m_recoveryBw != DataRate(0);
        m_roundStart = true;
    }
    else if (newState == TcpSocketState::CA_RECOVERY)
    {
        m_priorCWnd = std::max(m_priorCWnd, tcb->m_cWnd.Get());
        tcb->m_cWnd =
            tcb->m_bytesInFlight.Get() + std::max(tcb->m_lastAckedSackedBytes, tcb->m_segmentSize);
        m_packetConservation = true;
    }

    m_previousCongState = newState;
}

void
TcpLeoCC::CwndEvent(Ptr<TcpSocketState> tcb,
                    const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);
    if (event == TcpSocketState::CA_EVENT_COMPLETE_CWR)
    {
        m_packetConservation = false;
        tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), m_priorCWnd);
    }
}

uint32_t
TcpLeoCC::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    (void)bytesInFlight;
    m_priorCWnd = std::max(m_priorCWnd, tcb->m_cWnd.Get());
    return tcb->m_ssThresh;
}

bool
TcpLeoCC::HasCongControl() const
{
    return true;
}

void
TcpLeoCC::CongControl(Ptr<TcpSocketState> tcb,
                      const TcpRateOps::TcpRateConnection& rc,
                      const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);

    if (!m_isInitialized)
    {
        InitializeModel(tcb);
    }

    m_delivered = rc.m_delivered;
    if (rs.m_deliveryRate != DataRate(0))
    {
        UpdateRound(rc, rs);
        UpdateBandwidthModel(rs);
        UpdateAckAggregation(tcb, rs);
        CheckReconfiguration(tcb);
        CheckStartupExit(tcb, rs);
        CheckDrain(tcb);
        CheckProbeRtt(tcb);
        UpdateCruisePhase();
        UpdateTargetRate();
    }

    UpdateControlParameters(tcb, rs);
}

Ptr<TcpCongestionOps>
TcpLeoCC::Fork()
{
    return CreateObject<TcpLeoCC>(*this);
}

uint64_t
TcpLeoCC::EwmaBitsPerSecond(uint64_t current, uint64_t sample, double alpha) const
{
    return static_cast<uint64_t>((1.0 - alpha) * static_cast<double>(current) +
                                 alpha * static_cast<double>(sample));
}

Time
TcpLeoCC::EwmaTime(Time current, Time sample, double alpha) const
{
    int64_t currentUs = current.GetMicroSeconds();
    int64_t sampleUs = sample.GetMicroSeconds();
    int64_t filtered =
        static_cast<int64_t>((1.0 - alpha) * static_cast<double>(currentUs) +
                             alpha * static_cast<double>(sampleUs));
    return MicroSeconds(filtered);
}

void
TcpLeoCC::PruneRttSamples(Time now)
{
    while (!m_rttSamples.empty() && (now - m_rttSamples.front().timestamp) > m_rttWindowLength)
    {
        m_rttSamples.pop_front();
    }
}

} // namespace ns3
